"""The Decider: calibrated confidence plus threshold, or any guardrail trip, decides between
handling a case automatically and handing it to a human."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from handoff_gauge import config
from handoff_gauge.calibration import CalibrationMap, IdentityMap, map_from_dict
from handoff_gauge.estimators import ConfidenceEstimator, Estimate, build_estimator
from handoff_gauge.guardrails import GuardrailResult


@dataclass(frozen=True)
class Decision:
    decision: Literal["auto", "handoff"]
    intent: str | None
    calibrated_confidence: float
    raw_score: float
    threshold: float | None
    reasons: list[str]


@dataclass
class CalibrationProfile:
    """What calibrate.py exports: the estimator, its calibration map and the threshold."""

    estimator: str = "self_consistency"
    estimator_params: dict[str, Any] = field(default_factory=dict)
    map: CalibrationMap = field(default_factory=IdentityMap)
    threshold: float | None = 1.0  # uncalibrated default: auto only when all samples agree
    target_precision: float = config.TARGET_PRECISION

    @classmethod
    def load(cls, path: Path = config.CALIBRATION_PATH) -> "CalibrationProfile":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            estimator=data["estimator"],
            estimator_params=data.get("estimator_params", {}),
            map=map_from_dict(data["map"]),
            threshold=data["threshold"],
            target_precision=data.get("target_precision", config.TARGET_PRECISION),
        )

    def build_estimator(self) -> ConfidenceEstimator:
        return build_estimator(self.estimator, self.estimator_params)


class Decider:
    def __init__(self, profile: CalibrationProfile) -> None:
        self.profile = profile

    def decide(self, estimate: Estimate | None, trips: Sequence[GuardrailResult]) -> Decision:
        raw = estimate.raw_score if estimate is not None else 0.0
        calibrated = self.profile.map.apply(raw) if estimate is not None else 0.0
        intent = estimate.intent if estimate is not None else None
        threshold = self.profile.threshold

        def handoff(*reasons: str) -> Decision:
            return Decision("handoff", intent, calibrated, raw, threshold, list(reasons))

        if trips:
            return handoff(*(f"Guardrail {t.name}: {t.reason}" for t in trips))
        if estimate is None or intent is None:
            return handoff("No usable answer from the model.")
        if threshold is None:
            return handoff(
                "No threshold reaches the target precision, so every case goes to a human."
            )
        if calibrated >= threshold:
            reason = (
                f"Calibrated confidence {calibrated:.2f} is at or above "
                f"the threshold {threshold:.2f}."
            )
            return Decision("auto", intent, calibrated, raw, threshold, [reason])
        return handoff(
            f"Calibrated confidence {calibrated:.2f} is below the threshold {threshold:.2f}."
        )
