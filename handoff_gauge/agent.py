"""TriageAgent: a Facade that runs guardrails, concurrent samples, the confidence estimator
and the Decider inside one OpenTelemetry trace."""

import asyncio
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from handoff_gauge import config, tracing
from handoff_gauge.decider import CalibrationProfile, Decider, Decision
from handoff_gauge.estimators import ConfidenceEstimator, Estimate
from handoff_gauge.guardrails import (
    GuardrailResult,
    TriageContext,
    build_input_chain,
    build_output_chain,
)
from handoff_gauge.llm import ChatBackend, GroqBackend, JsonlCache, LLMClient, LLMSample
from handoff_gauge.prompt import build_messages
from handoff_gauge.ratelimit import TokenRateLimiter


@dataclass
class TriageResult:
    decision: Decision
    redacted_text: str
    notes: list[str]
    guardrails: list[GuardrailResult]
    samples: list[LLMSample]
    estimate: Estimate | None
    spans: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        d = self.decision
        return {
            "decision": d.decision,
            "intent": d.intent,
            "calibrated_confidence": round(d.calibrated_confidence, 4),
            "raw_score": round(d.raw_score, 4),
            "threshold": d.threshold,
            "reasons": d.reasons,
            "notes": self.notes,
            "redacted_text": self.redacted_text,
            "guardrails": [asdict(g) for g in self.guardrails],
            "samples": [
                {"seed": s.seed, "content": s.content, "cache_hit": s.cache_hit, "error": s.error}
                for s in self.samples
            ],
            "spans": self.spans,
        }


class TriageAgent:
    def __init__(
        self,
        client: LLMClient,
        decider: Decider,
        estimator: ConfidenceEstimator,
        *,
        n_samples: int = config.N_SAMPLES,
        temperature: float = config.TEMPERATURE,
        timeout: float = 60.0,
    ) -> None:
        self.client = client
        self.timeout = timeout
        self.decider = decider
        self.estimator = estimator
        self.n_samples = n_samples
        self.temperature = temperature
        self.input_chain = build_input_chain()
        self.output_chain = build_output_chain()

    async def triage(self, text: str) -> TriageResult:
        tracer = tracing.tracer()
        with tracer.start_as_current_span("invoke_agent handoff-gauge") as root:
            root.set_attributes(
                {
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": "handoff-gauge",
                    "handoff.n_samples": self.n_samples,
                    "handoff.message_length": len(text),
                }
            )
            ctx = TriageContext(text=text)
            results: list[GuardrailResult] = []
            estimate: Estimate | None = None
            try:
                results += self.input_chain.handle(ctx)
                if not _tripped(results):
                    # Retries and rate-limit waits must not keep a request open indefinitely;
                    # a timeout lands in the handler below and hands the case off.
                    async with asyncio.timeout(self.timeout):
                        ctx.samples = await self.client.sample_n(
                            build_messages(ctx.text), self.n_samples, temperature=self.temperature
                        )
                    results += self.output_chain.handle(ctx)
                if not _tripped(results):
                    estimate = self._estimate(ctx)
            except Exception as exc:  # any failure hands off; it never becomes an auto decision
                root.record_exception(exc)
                results.append(
                    GuardrailResult(
                        "agent_error", "runtime", tripped=True,
                        reason=f"The agent failed ({type(exc).__name__}), so a human takes it.",
                    )
                )
            with tracer.start_as_current_span("decision") as span:
                decision = self.decider.decide(estimate, _tripped(results))
                span.set_attributes(
                    {
                        "handoff.decision": decision.decision,
                        "handoff.intent": decision.intent or "",
                        "handoff.calibrated_confidence": decision.calibrated_confidence,
                        "handoff.threshold": -1.0 if decision.threshold is None
                        else decision.threshold,
                    }
                )
            root.set_attribute("handoff.decision", decision.decision)
            trace_id = root.get_span_context().trace_id
        return TriageResult(
            decision=decision,
            redacted_text=ctx.text,
            notes=ctx.notes,
            guardrails=results,
            samples=ctx.samples,
            estimate=estimate,
            spans=tracing.spans_as_json(trace_id),
        )

    def _estimate(self, ctx: TriageContext) -> Estimate:
        with tracing.tracer().start_as_current_span(f"estimate {self.estimator.name}") as span:
            estimate = self.estimator.estimate(ctx.answers)
            span.set_attributes(
                {
                    "handoff.estimator": self.estimator.name,
                    "handoff.raw_score": estimate.raw_score,
                    "handoff.agreement": estimate.agreement,
                    "handoff.verbalized": estimate.verbalized,
                }
            )
        return estimate


def _tripped(results: list[GuardrailResult]) -> list[GuardrailResult]:
    return [r for r in results if r.tripped]


def build_agent(
    *,
    backend: ChatBackend | None = None,
    cache_path: Path | None = config.CACHE_PATH,
    persist_cache: bool = True,
    profile_path: Path = config.CALIBRATION_PATH,
    profile: CalibrationProfile | None = None,
    n_samples: int = config.N_SAMPLES,
) -> TriageAgent:
    """The production wiring: Groq backend, shared cache, rate limiter, calibrated decider.
    Pass `profile` only in tests; otherwise the exported calibration file must exist."""
    if backend is None:
        config.load_env_file()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set (put it in .env.local)")
        backend = GroqBackend(api_key)
    profile = profile or CalibrationProfile.load(profile_path)
    client = LLMClient(
        backend,
        cache=JsonlCache(cache_path, persist=persist_cache),
        limiter=TokenRateLimiter(config.REQUESTS_PER_MINUTE, config.TOKENS_PER_MINUTE),
    )
    return TriageAgent(client, Decider(profile), profile.build_estimator(), n_samples=n_samples)
