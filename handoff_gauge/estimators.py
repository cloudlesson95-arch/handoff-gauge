"""Confidence estimators as Strategies.

Each strategy turns the N sampled answers into one raw score for the majority intent. The
Decider then maps the raw score through a calibration map fitted for that strategy.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from handoff_gauge.answers import Answer, majority_intent
from handoff_gauge.calibration import sigmoid


@dataclass(frozen=True)
class Estimate:
    intent: str | None
    raw_score: float
    agreement: float
    verbalized: float


def features(answers: Sequence[Answer | None]) -> tuple[str | None, float, float]:
    """Majority intent, its share of all samples (invalid ones count against it), and the
    mean stated confidence of the samples that chose it."""
    intent = majority_intent(answers)
    if intent is None or not answers:
        return None, 0.0, 0.0
    chosen = [a.confidence for a in answers if a is not None and a.intent == intent]
    return intent, len(chosen) / len(answers), sum(chosen) / len(chosen)


class ConfidenceEstimator(ABC):
    name: str

    def estimate(self, answers: Sequence[Answer | None]) -> Estimate:
        intent, agreement, verbalized = features(answers)
        score = self.score(agreement, verbalized) if intent is not None else 0.0
        return Estimate(intent, score, agreement, verbalized)

    @abstractmethod
    def score(self, agreement: float, verbalized: float) -> float: ...

    def params(self) -> dict[str, Any]:
        return {}


class VerbalizedConfidence(ConfidenceEstimator):
    """The confidence the model states for its own answer."""

    name = "verbalized"

    def score(self, agreement: float, verbalized: float) -> float:
        return verbalized


class SelfConsistency(ConfidenceEstimator):
    """The share of samples that agree with the majority intent."""

    name = "self_consistency"

    def score(self, agreement: float, verbalized: float) -> float:
        return agreement


class CombinedLogistic(ConfidenceEstimator):
    """A logistic model over both signals; weights are fitted offline on the calibration split."""

    name = "combined"

    def __init__(self, bias: float = -4.0, w_agreement: float = 4.0, w_verbalized: float = 2.0):
        self.bias = bias
        self.w_agreement = w_agreement
        self.w_verbalized = w_verbalized

    def score(self, agreement: float, verbalized: float) -> float:
        z = self.bias + self.w_agreement * agreement + self.w_verbalized * verbalized
        return sigmoid(z)

    def params(self) -> dict[str, Any]:
        return {
            "bias": self.bias,
            "w_agreement": self.w_agreement,
            "w_verbalized": self.w_verbalized,
        }


ESTIMATORS: dict[str, type[ConfidenceEstimator]] = {
    cls.name: cls for cls in (VerbalizedConfidence, SelfConsistency, CombinedLogistic)
}


def build_estimator(name: str, params: dict[str, Any] | None = None) -> ConfidenceEstimator:
    if name not in ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}; choose from {sorted(ESTIMATORS)}")
    return ESTIMATORS[name](**(params or {}))
