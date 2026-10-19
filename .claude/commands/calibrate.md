---
description: Re-run LLM sample collection and calibration, then compare against the last run
allowed-tools: Read, Bash(uv run python scripts/collect.py *), Bash(uv run python scripts/calibrate.py *), Bash(uv run pytest *), Bash(git diff *)
---

Re-calibrate handoff-gauge.

1. Run `uv run python scripts/collect.py` (cached: only new prompts call the API) and check
   that the last line of `logs/collect.log` reports 0 failed calls or names the fallbacks.
2. Run `uv run python scripts/calibrate.py`. It logs one MLflow run per estimator and
   calibration method to the `calibration` experiment and rewrites `calibration.json` and
   `reports/calibration.json`.
3. Compare with the committed version (`git diff calibration.json reports/calibration.json`):
   ECE before and after, Brier score, the threshold, precision on auto-handled cases and coverage.
4. Run `uv run pytest -q -m slow` (regression gate). If precision fell below the target
   or ECE rose, stop and report. Don't change the target to make it pass.
5. Summarise the change in a table and say whether it is safe to commit.
