"""Replay cached LLM samples for a data split and score them the way the live agent does.

Used by scripts/calibrate.py and the regression test. Nothing here calls an API: a cache
miss becomes an error sample, which counts as disagreement.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from handoff_gauge import config
from handoff_gauge.estimators import features
from handoff_gauge.guardrails import TriageContext, build_input_chain, build_output_chain
from handoff_gauge.llm import JsonlCache, LLMClient, NoNetworkBackend
from handoff_gauge.prompt import build_messages


@dataclass(frozen=True)
class Scored:
    id: str
    label: str
    intent: str | None
    agreement: float
    verbalized: float
    input_tripped: bool
    output_tripped: bool
    cache_misses: int

    @property
    def correct(self) -> bool:
        return not self.output_tripped and self.intent == self.label


def load_split(name: str, data_dir: Path = config.DATA_DIR) -> list[dict]:
    path = data_dir / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def replay(
    rows: list[dict], *, n_samples: int = config.N_SAMPLES, cache_path: Path = config.CACHE_PATH
) -> list[Scored]:
    client = LLMClient(NoNetworkBackend(), cache=JsonlCache(cache_path, persist=False))
    input_chain, output_chain = build_input_chain(), build_output_chain()
    scored = []
    for row in rows:
        ctx = TriageContext(text=row["text"])
        input_tripped = any(r.tripped for r in input_chain.handle(ctx))
        # Samples exist for every message (collected after redaction), so the estimator is
        # scored on all of them; an input trip only changes the end-to-end decision.
        ctx.samples = await client.sample_n(build_messages(ctx.text), n_samples)
        output_tripped = any(r.tripped for r in output_chain.handle(ctx))
        intent, agreement, verbalized = features(ctx.answers)
        scored.append(
            Scored(
                id=row["id"],
                label=row["label"],
                intent=None if output_tripped else intent,
                agreement=0.0 if output_tripped else agreement,
                verbalized=0.0 if output_tripped else verbalized,
                input_tripped=input_tripped,
                output_tripped=output_tripped,
                cache_misses=sum(not s.cache_hit for s in ctx.samples),
            )
        )
    return scored
