# Guardrail conventions

- Each guardrail is a `Guardrail` subclass with a stable snake_case `name` and one method,
  `check(ctx) -> GuardrailResult`. The chain (`GuardrailChain`) runs the links in order and stops
  at the first trip (Chain of Responsibility).
- A trip returns `GuardrailResult(tripped=True, reason="...")`. The reason is shown to the human
  reviewer, so write it as a sentence about the message, e.g. "Message tries to override the
  agent's instructions." Every trip forces a handoff.
- A guardrail may transform the context without tripping (PII redaction replaces emails,
  phone numbers and card numbers with `[EMAIL]`, `[PHONE]`, `[CARD]` and records a note). The
  LLM only ever sees the redacted text.
- Input order: `pii_redaction` -> `empty_message` -> `prompt_injection` -> `off_topic`. Redaction
  runs first so later links and the trace never hold raw PII.
- Output order (per decision): `output_schema` (pydantic) -> `allowed_label`.
- Each guardrail runs inside its own span `guardrail <name>` (see tracing.md).
- Adding a guardrail: write the class, add it to the chain in `build_input_chain()` /
  `build_output_chain()`, add a test that trips it and one that passes it, and add an
  adversarial message to `data/adversarial.jsonl` if it guards against an attack.
- Heuristics are deliberately simple (regex and keyword lists). Their limits are listed in the README.
