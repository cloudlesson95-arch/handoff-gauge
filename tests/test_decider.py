import json

from handoff_gauge.calibration import IsotonicMap, PlattMap
from handoff_gauge.decider import CalibrationProfile, Decider
from handoff_gauge.estimators import Estimate
from handoff_gauge.guardrails import GuardrailResult

CONFIDENT = Estimate("card_arrival", raw_score=1.0, agreement=1.0, verbalized=0.95)


def decider(threshold: float | None = 0.8) -> Decider:
    return Decider(CalibrationProfile(map=IsotonicMap([0.0, 1.0], [0.0, 0.9]), threshold=threshold))


def test_auto_when_calibrated_confidence_reaches_threshold() -> None:
    d = decider().decide(CONFIDENT, [])
    assert d.decision == "auto"
    assert d.calibrated_confidence == 0.9 and d.raw_score == 1.0
    assert "at or above" in d.reasons[0]


def test_handoff_below_threshold() -> None:
    d = decider().decide(Estimate("card_arrival", 0.5, 0.5, 0.9), [])
    assert d.decision == "handoff"
    assert "below the threshold" in d.reasons[0]


def test_any_trip_forces_handoff_even_with_full_confidence() -> None:
    trip = GuardrailResult("prompt_injection", "input", tripped=True, reason="Bad.")
    d = decider(threshold=0.0).decide(CONFIDENT, [trip])
    assert d.decision == "handoff"
    assert d.reasons == ["Guardrail prompt_injection: Bad."]


def test_handoff_without_estimate_or_threshold() -> None:
    assert decider().decide(None, []).decision == "handoff"
    assert decider().decide(Estimate(None, 0.0, 0.0, 0.0), []).decision == "handoff"
    assert decider(threshold=None).decide(CONFIDENT, []).decision == "handoff"


def test_profile_defaults_when_file_missing(tmp_path) -> None:
    profile = CalibrationProfile.load(tmp_path / "missing.json")
    assert profile.estimator == "self_consistency" and profile.threshold == 1.0


def test_profile_loads_exported_file(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "estimator": "combined",
        "estimator_params": {"bias": -2.0, "w_agreement": 3.0, "w_verbalized": 1.0},
        "map": PlattMap(2.0, -1.0).to_dict(),
        "threshold": 0.7,
        "target_precision": 0.9,
    }))
    profile = CalibrationProfile.load(path)
    assert profile.threshold == 0.7
    assert profile.build_estimator().name == "combined"
    assert profile.map.apply(0.5) == PlattMap(2.0, -1.0).apply(0.5)
