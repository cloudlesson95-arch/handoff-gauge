# handoff-gauge: project rules for coding agents

Handoff Gauge triages customer-banking messages. It runs a guardrail chain, samples an LLM
several times concurrently, turns the samples into a calibrated confidence, and then either
handles the message automatically or hands it to a human review queue, giving the reason.

## Context modules

- Architecture, data flow and design patterns: @context/architecture.md
- Tests and verification: @context/testing.md
- Guardrail conventions: @context/guardrails.md
- Tracing conventions: @context/tracing.md

## Rules

- Python 3.12 through uv only: `uv add`, `uv sync`, `uv run ...`. Never install into the global
  interpreter. Runtime dependencies stay small: the live API must not need scikit-learn, numpy,
  MLflow or matplotlib (those are dev-only, for the offline pipeline).
- After any Python change, `uv run ruff check .` and `uv run pytest -q` must pass. The
  PostToolUse hook runs ruff and the fast tests after each edit: fix what it reports first.
- Tests never call the network. Use `FakeBackend` from `handoff_gauge.llm` or replay
  `data/llm_cache.jsonl`.
- A change to the calibration math, the threshold picker or a guardrail ships with a test
  that fails without it (a Hypothesis property for math).
- Secrets: `GROQ_API_KEY` lives in `.env.local` (git-ignored) and in the host's environment
  settings, and is read on the server only. Never print a key's value. `.env.example` lists names only.
- Frontend: Next.js 16 App Router at the repository root. Its APIs differ from older
  versions, so read the guide in `node_modules/next/dist/docs/` before using one (see AGENTS.md).
- Name the design pattern in the docstring when you add a Strategy, a guardrail link or an adapter.
- Commit messages: `<type>: <what it does>`, type one of feat, fix, docs, test, build, ci.
- Keep it small: no new dependency, layer or config option without a concrete need.
