# Tracing conventions

OpenTelemetry Python SDK, following the GenAI semantic conventions (`gen_ai.*`).

| Span | Name | Key attributes |
|---|---|---|
| Root | `invoke_agent handoff-gauge` | `gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name` |
| Guardrail | `guardrail <name>` | `guardrail.name`, `guardrail.stage` (input/output), `guardrail.tripped`, `guardrail.reason` |
| LLM sample | `chat <model>` | `gen_ai.operation.name=chat`, `gen_ai.provider.name=groq`, `gen_ai.request.model`, `gen_ai.request.temperature`, `gen_ai.request.seed`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `handoff.cache_hit` |
| Estimate | `estimate <strategy>` | `handoff.estimator`, `handoff.raw_score`, `handoff.agreement` |
| Decision | `decision` | `handoff.decision`, `handoff.calibrated_confidence`, `handoff.threshold`, `handoff.intent` |

- Never put raw message text or PII into attributes: only redacted text, lengths and labels.
- The API returns the request's spans as JSON (`tracing.spans_as_json(trace_id)`), which the
  UI draws as a waterfall. Times are milliseconds relative to the root span.
- Project-specific attributes use the `handoff.` prefix, and standard ones keep their semconv names.
