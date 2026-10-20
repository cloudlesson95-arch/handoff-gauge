"""Calibration regression gate: replays the cached LLM responses for the test split through
the live Decider with the exported calibration.json and compares against the committed report.
No API key needed. A missing calibration file or report fails the gate; it is never skipped."""

import asyncio
import json

import pytest

from handoff_gauge import config
from handoff_gauge.calibration import expected_calibration_error
from handoff_gauge.decider import CalibrationProfile, Decider
from handoff_gauge.estimators import Estimate
from handoff_gauge.evaluation import load_split, replay

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def replayed():
    profile = CalibrationProfile.load()
    report = json.loads((config.REPORTS_DIR / "calibration.json").read_text(encoding="utf-8"))
    scored = asyncio.run(replay(load_split("test")))
    estimator, decider = profile.build_estimator(), Decider(profile)
    decisions = []
    for s in scored:
        raw = estimator.score(s.agreement, s.verbalized) if s.intent else 0.0
        decisions.append(decider.decide(Estimate(s.intent, raw, s.agreement, s.verbalized), []))
    return profile, report, scored, decisions


def test_every_sample_is_replayed_from_the_cache(replayed) -> None:
    _, _, scored, _ = replayed
    assert sum(s.cache_misses for s in scored) == 0


def test_ece_does_not_rise(replayed) -> None:
    _, report, scored, decisions = replayed
    probs = [d.calibrated_confidence for d in decisions]
    ece = expected_calibration_error(probs, [s.correct for s in scored])
    assert ece <= report["headline"]["ece_after"] + 1e-9


def test_auto_handled_precision_meets_target(replayed) -> None:
    profile, report, scored, decisions = replayed
    auto = [s for s, d in zip(scored, decisions, strict=True) if d.decision == "auto"]
    assert auto, "nothing would be handled automatically"
    precision = sum(s.correct for s in auto) / len(auto)
    assert precision >= profile.target_precision
    assert len(auto) == report["headline"]["n_auto"]
