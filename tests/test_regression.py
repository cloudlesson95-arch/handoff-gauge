"""Calibration regression gate: replays the cached LLM responses for the test split with the
exported calibration.json and compares against the committed report. No API key needed."""

import json

import pytest

from handoff_gauge import config
from handoff_gauge.calibration import expected_calibration_error
from handoff_gauge.decider import CalibrationProfile
from handoff_gauge.evaluation import load_split, replay

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not config.CALIBRATION_PATH.exists(), reason="not calibrated yet"),
]


@pytest.fixture(scope="module")
def replayed():
    import asyncio

    profile = CalibrationProfile.load()
    report = json.loads((config.REPORTS_DIR / "calibration.json").read_text(encoding="utf-8"))
    scored = asyncio.run(replay(load_split("test")))
    estimator = profile.build_estimator()
    probs = [
        profile.map.apply(estimator.score(s.agreement, s.verbalized) if s.intent else 0.0)
        for s in scored
    ]
    return profile, report, scored, probs


def test_every_sample_is_replayed_from_the_cache(replayed) -> None:
    _, _, scored, _ = replayed
    assert sum(s.cache_misses for s in scored) == 0


def test_ece_does_not_rise(replayed) -> None:
    _, report, scored, probs = replayed
    ece = expected_calibration_error(probs, [s.correct for s in scored])
    assert ece <= report["headline"]["ece_after"] + 1e-9


def test_auto_handled_precision_meets_target(replayed) -> None:
    profile, _, scored, probs = replayed
    assert profile.threshold is not None, "no threshold reaches the target precision"
    auto = [s for s, p in zip(scored, probs, strict=True) if p >= profile.threshold]
    assert auto, "nothing would be handled automatically"
    precision = sum(s.correct for s in auto) / len(auto)
    assert precision >= profile.target_precision
