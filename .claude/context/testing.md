# Testing and verification

The checks below are independent of the agent that wrote the code: a test written by the same
agent as the code can share its blind spots, so the math is also checked by properties and
mutation testing.

| Layer | Tool | Where |
|---|---|---|
| Lint | ruff (`uv run ruff check .`) | hook after each edit, CI |
| Unit tests | pytest + pytest-asyncio, `FakeBackend` (no network) | `tests/test_*.py` |
| Properties | Hypothesis: maps stay in [0,1] and are monotonic, the threshold picker meets its target on its own data, redaction never leaks | `tests/test_properties.py` |
| Regression gate | replays `data/llm_cache.jsonl`, fails if ECE rises or precision drops below target | `tests/test_regression.py` (marked `slow`) |
| Coverage | `pytest --cov --cov-branch`, fail-under gate | CI |
| Mutation | mutmut over `guardrails.py` and `calibration.py` | CI (Linux only) |

## Conventions

- Fast tests (everything not marked `slow`) must finish in a few seconds: the edit hook runs them.
- Concurrency tests use `FakeBackend(delay=...)` and assert on its recorded peak concurrency,
  not on wall-clock time alone.
- Name tests after the behaviour: `test_injection_trip_forces_handoff`, not `test_guard2`.
- When mutmut reports a surviving mutant, add a test that kills it and record it in
  `docs/verification.md`.
