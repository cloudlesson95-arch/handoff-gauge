# Handoff Gauge

A ticket-triage agent that uses calibrated confidence to decide when to hand off to a human.

Live demo: **https://handoff-gauge.vercel.app**

Handoff Gauge sorts customer-banking messages into ten intents. Each message runs through a
chain of guardrails, then the model is sampled three times in parallel. The answers become one
calibrated confidence. Above a threshold chosen for 90% precision, the agent handles the
message itself. Below it, or when any guardrail trips, the message goes to a human review
queue with the reason. Every decision comes with its OpenTelemetry trace.

| Result on 60 held-out messages | |
|---|---|
| Expected calibration error (stated confidence → calibrated) | 0.070 → 0.026 |
| Brier score | 0.092 → 0.078 |
| Precision on auto-handled messages | 0.923 (48 of 52; 90% interval 0.84–0.97) |
| Coverage (share handled automatically) | 87% |
| Accuracy of the majority answer, for reference | 0.883 |

## What it shows

- **Calibrated human handoff.** Three confidence estimators (verbalized, self-consistency over
  N samples, and a logistic combination) are each calibrated with isotonic regression and Platt
  scaling. The combination is picked by cross-validation on the calibration split only. The
  threshold is the lowest whose 90% Wilson lower bound of precision reaches the target, so it
  holds on new data rather than only on the data it was fitted on.
- **Guardrails** as a Chain of Responsibility: PII redaction (cards, phones, emails, keeping
  dates and amounts), empty message, prompt injection, off-topic, output schema and allowed label.
  Any trip forces a handoff, and a failed or invalid sample counts against the majority.
- **OpenTelemetry tracing** following the GenAI semantic conventions: one span per guardrail,
  model call (`gen_ai.*`: model, tokens, temperature, seed) and decision. The API returns the
  spans and the UI draws them as a waterfall.
- **MLflow tracking.** Every estimator × calibration method combination is logged as a run with
  parameters, metrics, the exported map and the reliability diagram.
- **asyncio concurrency.** Samples fan out with `asyncio.gather` behind a semaphore, a
  token-aware sliding-window rate limiter, bounded retries with backoff and jitter, a time
  limit, and a JSONL response cache keyed by model, prompt, temperature, seed and max tokens.
- **Design patterns**, named in the code: Strategy (`estimators.py`), Chain of Responsibility
  (`guardrails.py`), Adapter (`GroqBackend` in `llm.py`), Facade (`TriageAgent`).
- **A shared coding-agent setup** in `.claude/`: project rules, imported context modules,
  slash commands for planning, adding eval cases, calibrating and reviewing diffs, a reviewer
  subagent, and a hook that lints and runs the fast tests after every edit. See
  [docs/agent-workflow.md](docs/agent-workflow.md).
- **Verification of generated code** with tools that don't rely on the agent's own tests:
  Hypothesis properties, mutation testing, a branch-coverage gate and a replayed regression
  gate. [docs/verification.md](docs/verification.md) lists the defects found and how each was fixed.

## Architecture

```
browser ── Next.js (App Router, Tailwind) ── /api/py/* rewrite ──▶ FastAPI (api/index.py)
  │  Triage · Review queue (localStorage) · Calibration          │  demo guard: 5/min per IP,
  │                                                              │  daily cap, 500 chars,
  │                                                              │  cached-example fallback
  ▼                                                              ▼
TriageAgent (Facade), one OpenTelemetry trace per message
  ├─ input guardrails ─ pii_redaction → empty_message → prompt_injection → off_topic
  ├─ LLMClient ─ 3 × chat(gpt-oss-20b, JSON) via asyncio.gather
  │              semaphore · token rate limiter · retries · JSONL cache · GroqBackend (Adapter)
  ├─ output guardrails ─ output_schema → allowed_label
  ├─ estimator (Strategy) ─ verbalized | self_consistency | combined
  └─ Decider ─ calibration.json map + threshold → auto | handoff, reasons, spans

offline (uv, dev dependencies):
  prepare_data.py → collect.py (540 cached calls) → calibrate.py ─▶ calibration.json
                                                    (scikit-learn,   reports/calibration.{json,png}
                                                     MLflow runs)
```

The live function needs no scikit-learn: the fitted map is exported as points or
coefficients and applied in pure Python.

## Run it locally

Needs [uv](https://docs.astral.sh/uv/), Node.js 20+ and a Groq API key (free tier). Without a
key, the demo serves the precomputed examples.

```
uv sync
npm install
```

Put the key in `.env.local` (git-ignored; `.env.example` lists the name):

```
GROQ_API_KEY=...
```

Start the API and the web app in two terminals:

```
uv run uvicorn api.index:app --port 8000
npm run dev
```

Open http://localhost:3000. A single message from the command line:

```
uv run python scripts/triage_once.py "I was charged twice for my card"
```

Checks:

```
uv run ruff check .
uv run pytest -q
uv run pytest -q -m "not slow" --cov --cov-branch
npm run build
```

## Re-calibrate

```
uv run python scripts/prepare_data.py
uv run python scripts/collect.py
uv run python scripts/calibrate.py
uv run python scripts/make_examples.py --live
```

`collect.py` resumes from `data/llm_cache.jsonl`, so only new prompts call the API. Progress
goes to `logs/collect.log`. `calibrate.py` prints every estimator and method, logs them to the
MLflow experiment `calibration` (`mlflow.db` with artifacts in `mlruns/`; browse with
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`), and rewrites `calibration.json`
and `reports/`. Then `uv run pytest -q -m slow` replays the cache through the new profile.
In Claude Code, `/calibrate` runs the same steps and compares against the committed run.

## Limits

- **Small dataset**: 120 messages to fit, 60 to test, 10 of Banking77's 77 intents. The 90%
  interval on auto-handled precision (0.84–0.97) still includes values below the target.
- **Best-effort rate limit**: the per-IP limit and daily cap are kept in memory, so each
  serverless instance counts on its own. The 500-character limit and 3 samples per request
  bound the cost of any single call.
- **Browser-only review queue**: it lives in localStorage and holds the redacted text only.
  Nothing is stored on the server.
- **Self-consistency costs N calls**: every message costs three model calls. With three
  samples, agreement has only three possible values, so on its own it can't separate right
  from wrong finely enough; it is combined with the stated confidence.
- **Heuristic guardrails**: regex and keyword lists catch the obvious attacks, not a
  determined adversary.
- **Model drift**: the calibration belongs to one model, prompt and temperature. If any of
  them changes, re-calibrate.

## What was learned

- A threshold chosen where precision only just meets the target on the fitting data misses it
  on new data. The first run did exactly that (0.900 fitted, 0.883 on test). Requiring the
  lower confidence bound to meet the target fixed it without lowering the target.
- The model's stated confidence was over-confident (ECE 0.070). Agreement between samples is
  a coarse signal; the two together, calibrated, did best.
- Most errors are between intents that overlap in meaning, such as a pending transfer and a
  transfer that hasn't reached the recipient. Those are exactly where samples disagree, so the
  handoff lands on the cases a human should see.
- Property tests only find what their generators produce. The first redaction property used
  only formats the regex already handled; a separate review pass and wider generators found
  the gaps.
- A failure has to count as a vote against the answer. Without that rule, one confident
  sample and two errors could look like a confident decision.

## Data and attribution

Messages come from **Banking77** by PolyAI (Casanueva et al., "Efficient Intent Detection with
Dual Sentence Encoders", 2020), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source:
<https://github.com/PolyAI-LDN/task-specific-datasets>. This project uses a 180-message subset
of 10 intents; see [data/README.md](data/README.md). The 8 adversarial messages are synthetic.

Model: `openai/gpt-oss-20b` on Groq, JSON mode, temperature 1.0, 3 samples per message.
