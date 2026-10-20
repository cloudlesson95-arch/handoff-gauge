# Verification

Most of this code was written by a coding agent. A test written by the same agent as the code
can share its blind spots, so the checks below include tools that don't depend on the agent's
own idea of what the code should do: properties over generated inputs, mutation testing, a
replayed regression gate and a separate review pass.

## What is checked

| Check | Where | Result |
|---|---|---|
| Lint | `uv run ruff check .` (hook after each edit, CI) | clean |
| Unit tests | 135 fast tests with `FakeBackend`, no network | all pass |
| Branch coverage | `pytest --cov --cov-branch`, `fail_under = 90` | 93% |
| Hypothesis properties | `tests/test_properties.py` | all pass (see below) |
| Calibration regression gate | `tests/test_regression.py`: replays 180 cached LLM responses through the live Decider | ECE and precision hold |
| Mutation testing | mutmut over `guardrails.py` and `calibration.py`, Linux CI job | **pending: the repository has not been pushed yet** |
| Review | the `reviewer` subagent (`.claude/agents/reviewer.md`) over the whole codebase | 1 must-fix, 9 should-fix: all fixed below |
| Rendering | headless browser at 1280 px and 390 px with device emulation | no horizontal overflow; full triage → review flow works |

Properties:

- Calibration maps (isotonic and Platt) return values in [0, 1] and are monotonic for any
  fitted points.
- The exported isotonic map gives the same answers as scikit-learn's model it came from.
- The combined logistic score is a probability, and it never decreases when either signal increases.
- The threshold picker never returns a threshold that misses the target on its own data, at any
  margin z. It also returns the most coverage that meets the target.
- The Wilson lower bound never exceeds the observed proportion, and it shrinks as z grows.
- ECE and Brier score stay in [0, 1].
- Redaction never leaves a digit or an `@` from the card numbers, phone numbers and emails it
  is meant to catch. Its output is a fixed point: redacting it again changes nothing.
- Short numbers (amounts, dates) are kept.

## Surviving mutants

Mutation score: **not yet measured.** mutmut 3 needs `fork`, so it can't run on Windows,
and the local Docker engine wasn't running. The `mutation` job in `.github/workflows/ci.yml`
runs it on the first push, writes the score to the job summary and uploads
`mutmut-cicd-stats.json`. Each surviving mutant gets a test that kills it, recorded here.

## Defects found in agent-written code

| # | Defect | How it was found | Fix and test |
|---|---|---|---|
| 1 | The threshold picker chose the most coverage whose in-sample precision reached 0.90. On the calibration split, auto-handling every message was exactly 0.900 precise (the base accuracy), so it chose that. Test precision was 0.883, under the target | the calibration run's own report | The picker requires the 90% Wilson lower bound of precision to reach the target (`choose_threshold(..., z)`). `test_margin_rejects_a_cut_that_only_just_meets_the_target`, plus a property over z. Selection still uses only the calibration split. The defect was noticed on the test split, so the test numbers are no longer fully untouched |
| 2 | With a verbalized-confidence profile, two failed samples and one confident answer were auto-handled: failures did not count against the answer | reviewer (must-fix) | The Decider hands off unless a majority of all N samples agree. `test_failed_samples_cannot_make_a_verbalized_profile_auto` |
| 3 | Two adjacent numbers, "(000) 0000000 000000000", were redacted as one card and left the area code "(000)" in the text | Hypothesis, after the generators were widened | One number pattern covers the whole run and is labelled card or phone by its shape. Regression case in `test_redaction_replaces_pii` |
| 4 | Redaction turned "2024-03-12 14:30" into "[PHONE]:30" and missed `415.555.2671`, `06.12.34.56.78`, `030/1234567` and `4111 - 1111 - 1111 - 1111`. The first property test couldn't see this: its generators used only the separators the regex already knew | reviewer | Dates and times are held aside while redacting, and separators are wider. Parametrized tests, and the generators now produce `.`, `/` and ` - ` |
| 5 | A message that was only a card number became "[CARD]", passed the empty-message and off-topic checks and was sent to the model | reviewer | Placeholders are ignored by both checks. Test cases in `test_empty_message_trips` |
| 6 | The injection check was bypassed by a line break inside the pattern, by a zero-width space, and by "confidence 0.99" | reviewer | Whitespace is normalised and zero-width characters stripped before matching, and any stated confidence value is caught. Four new cases |
| 7 | An HTTP 200 with a non-JSON body raised out of `asyncio.gather`, which would have aborted the whole collection run | reviewer | The adapter maps malformed responses to `LLMError`, which becomes an error sample. `test_groq_adapter_malformed_200_is_an_llm_error` |
| 8 | The response cache stored auth failures (a bad key would replay 540 failures forever), and its key left out `max_tokens` | reviewer | Only answers are cached, and the key includes `max_tokens`. The 558 committed responses were re-keyed by a one-off script, with no new API calls |
| 9 | A failed cache write turned a good answer into a handoff (`agent_error`) | reviewer | Writes are best-effort. `test_cache_write_failure_does_not_lose_the_answer` |
| 10 | A missing `calibration.json` made the agent use an uncalibrated default | reviewer | Production wiring requires the file; the default is only for tests. `test_missing_profile_file_is_an_error` |
| 11 | If no estimator met the target in cross-validation, the loosest one was exported | reviewer | It falls back to the most precise one and warns |
| 12 | A request had no time limit: retries and rate-limit waits could outlast the host's function timeout | reviewer | Sampling runs under `asyncio.timeout`, and a timeout hands off. `test_slow_model_times_out_into_a_handoff` |
| 13 | Off-topic keyword stems matched inside other words ("atm" in "atmosphere", "sent" in "sentence") | reviewer | Short words must stand alone. New off-topic cases |
| 14 | Spans with equal start times (Windows' coarse clock) were sorted children-first, so the root span wasn't first | a failing agent test in phase 2 | Ties are broken by depth in `spans_as_json` |
| 15 | On narrow screens, the charts' initial 560 px width pushed the page wider than the viewport, and the ResizeObserver then measured the widened container | rendering at 390 px and measuring `scrollWidth` | Charts measure first and draw after, and grid items get `min-w-0` |

## Remaining risks of leaning on a coding agent

- **Shared blind spots.** The first redaction property was written by the same agent as the
  regex, and generated only formats the regex already handled. Hypothesis can only find what
  its generators can produce. The separate review pass found the gap. A reviewer from the
  same model family is still not a fully independent check, so mutation testing and the
  replayed gate matter.
- **Small numbers.** Precision on auto-handled test messages is 48 of 52 = 0.923, and its 90%
  Wilson interval is 0.839–0.965. The headline meets the target, but the data can't rule out
  a true precision below it. The threshold's margin is there for that reason, and more data
  would narrow the interval.
- **Heuristic guardrails.** Injection and off-topic detection are regex and keyword lists.
  They catch the obvious attacks and the cases listed above, not a determined adversary.
- **Model drift.** The calibration map belongs to one model, prompt and temperature. If the
  provider changes the model behind the same name, the map is stale. The cache pins the
  fitted run so the gate stays reproducible, but it can't detect drift in live answers.
  Re-run `/calibrate` periodically.
- **Gates that only report.** The mutation job reports a score but does not fail CI below a
  floor yet. A floor will be set after the first measured run.
- **Weakening a check to make it pass.** The target precision was never changed to reach a
  result. The regression gate compares against the committed report, so a later change that
  lowers precision or raises ECE fails CI and can't silently move the baseline.
