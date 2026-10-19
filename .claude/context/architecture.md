# Architecture

## Request flow

```
message
  -> input guardrails (Chain of Responsibility): PII redaction -> injection -> off-topic
       a trip stops the chain and forces a handoff with its reason
  -> N concurrent LLM samples (asyncio.gather behind a semaphore and a token-aware rate limiter)
  -> output guardrails: schema validation -> allowed-label check
  -> confidence estimator (Strategy): verbalized | self-consistency | combined logistic
  -> Decider: calibration map (isotonic or Platt points from calibration.json) + threshold
  -> decision: auto | handoff, intent, calibrated confidence, reasons, spans
```

## Modules (`handoff_gauge/`)

| Module | Role | Pattern |
|---|---|---|
| `config.py` | Intent labels, model name, file paths, defaults (N samples, temperature) | |
| `llm.py` | `ChatBackend` protocol, `GroqBackend` (httpx, JSON mode), `FakeBackend`, `LLMClient` with semaphore, retries with backoff and the JSONL cache keyed by prompt+seed | Adapter |
| `ratelimit.py` | `TokenRateLimiter`: async sliding-window limit on requests and tokens per minute | |
| `guardrails.py` | `Guardrail` links, `GuardrailChain`, input and output guardrails | Chain of Responsibility |
| `estimators.py` | `ConfidenceEstimator` strategies that turn samples into (intent, raw score) | Strategy |
| `calibration.py` | Pure-Python calibration maps, ECE, Brier, reliability bins, threshold picker | |
| `decider.py` | Applies the calibration map and threshold, returns a `Decision` | |
| `tracing.py` | OpenTelemetry provider, in-memory exporter, spans as JSON | |
| `agent.py` | `TriageAgent`: wires the pipeline together inside one trace | Facade |

## Outside the package

- `scripts/`: offline pipeline: `prepare_data.py` (Banking77 subset), `collect.py` (cached
  LLM samples), `calibrate.py` (fit maps with scikit-learn, log to MLflow, export
  `calibration.json`, `reports/`), `make_examples.py`, `triage_once.py`.
- `api/index.py`: FastAPI app, served as a Vercel Python function under `/api/py/*`;
  locally by uvicorn on port 8000 behind the Next.js rewrite.
- `app/`: Next.js pages: Triage, Review queue (localStorage), Calibration.
- `data/`: committed dataset subset, adversarial messages, LLM cache, cached examples.

## Invariants

- Any guardrail trip means handoff. Nothing downstream can turn it back into auto.
- The live path needs no scikit-learn: maps are exported as points and applied in pure Python.
- The calibration split is only used for fitting, and the test split only for reporting.
