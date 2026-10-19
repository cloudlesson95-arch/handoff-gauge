---
description: Add an evaluation case (adversarial or labelled) and check how the agent handles it
argument-hint: <message text> [expected: handoff | <intent label>]
allowed-tools: Read, Edit, Bash(uv run pytest *), Bash(uv run python scripts/triage_once.py *)
---

Add this evaluation case: $ARGUMENTS

1. Decide the kind:
   - an attack or out-of-scope message (PII, prompt injection, off-topic, empty): append it to
     `data/adversarial.jsonl` as `{"id": "adv-NN", "text": ..., "kind": ..., "expected": "handoff"}`;
   - a labelled banking message: the label must be one of `config.INTENTS`. Add it to
     `tests/test_guardrails.py` or `tests/test_agent.py` as a fixture. Never add it to the
     calibration or test split, which are fixed by seed.
2. Add or extend a test that asserts the expected outcome with `FakeBackend`.
3. Run `uv run pytest -q`. If the case shows a gap in a guardrail, say so and propose the
   smallest fix. Don't fix it silently.
4. If a key is available, run `uv run python scripts/triage_once.py "<text>"` and report the
   decision, the reasons and the guardrail spans.
