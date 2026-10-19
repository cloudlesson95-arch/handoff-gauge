"""Calibration maps and metrics in pure Python, so the live API needs no scikit-learn.

Maps are fitted offline (scripts/calibrate.py) and exported as points or coefficients.
"""

import bisect
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class CalibrationMap(ABC):
    kind: str

    @abstractmethod
    def apply(self, score: float) -> float: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


class IdentityMap(CalibrationMap):
    kind = "identity"

    def apply(self, score: float) -> float:
        return min(max(score, 0.0), 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


class IsotonicMap(CalibrationMap):
    """Piecewise-linear interpolation between fitted (x, y) points, clipped at both ends."""

    kind = "isotonic"

    def __init__(self, xs: Sequence[float], ys: Sequence[float]) -> None:
        if not xs or len(xs) != len(ys):
            raise ValueError("isotonic map needs matching, non-empty x and y points")
        if any(b < a for a, b in zip(xs, xs[1:], strict=False)):
            raise ValueError("x points must be sorted")
        if any(b < a for a, b in zip(ys, ys[1:], strict=False)):
            raise ValueError("y points must be non-decreasing")
        self.xs = [float(x) for x in xs]
        self.ys = [min(max(float(y), 0.0), 1.0) for y in ys]

    def apply(self, score: float) -> float:
        xs, ys = self.xs, self.ys
        if score <= xs[0]:
            return ys[0]
        if score >= xs[-1]:
            return ys[-1]
        i = bisect.bisect_right(xs, score)
        x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (score - x0) / (x1 - x0)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "x": self.xs, "y": self.ys}


class PlattMap(CalibrationMap):
    """Logistic map sigmoid(a * score + b); monotonic increasing when a >= 0."""

    kind = "platt"

    def __init__(self, a: float, b: float) -> None:
        self.a = float(a)
        self.b = float(b)

    def apply(self, score: float) -> float:
        z = self.a * score + self.b
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "a": self.a, "b": self.b}


def map_from_dict(data: dict[str, Any]) -> CalibrationMap:
    kind = data.get("kind")
    if kind == "identity":
        return IdentityMap()
    if kind == "isotonic":
        return IsotonicMap(data["x"], data["y"])
    if kind == "platt":
        return PlattMap(data["a"], data["b"])
    raise ValueError(f"unknown calibration map kind {kind!r}")


# --- metrics ------------------------------------------------------------------------------


def _check(probs: Sequence[float], correct: Sequence[bool]) -> None:
    if len(probs) != len(correct):
        raise ValueError("probs and correct must have the same length")


def brier_score(probs: Sequence[float], correct: Sequence[bool]) -> float:
    _check(probs, correct)
    if not probs:
        return 0.0
    return sum((p - float(c)) ** 2 for p, c in zip(probs, correct, strict=True)) / len(probs)


def reliability_bins(
    probs: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> list[dict[str, float]]:
    """Equal-width bins over [0, 1]; the last bin includes 1.0. Empty bins are kept (count 0)."""
    _check(probs, correct)
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "count": 0, "conf": 0.0, "acc": 0.0}
            for i in range(n_bins)]
    for p, c in zip(probs, correct, strict=True):
        b = bins[min(int(p * n_bins), n_bins - 1)]
        b["count"] += 1
        b["conf"] += p
        b["acc"] += float(c)
    for b in bins:
        if b["count"]:
            b["conf"] /= b["count"]
            b["acc"] /= b["count"]
    return bins


def expected_calibration_error(
    probs: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Count-weighted mean gap between confidence and accuracy over equal-width bins."""
    if not probs:
        return 0.0
    bins = reliability_bins(probs, correct, n_bins)
    return sum(b["count"] * abs(b["acc"] - b["conf"]) for b in bins) / len(probs)


# --- threshold ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float | None  # None: no threshold reaches the target, so always hand off
    precision: float
    coverage: float
    n_auto: int


def operating_points(
    scores: Sequence[float], correct: Sequence[bool]
) -> list[ThresholdChoice]:
    """For every distinct score t (high to low): precision and coverage of `score >= t`."""
    _check(scores, correct)
    pairs = sorted(zip(scores, correct, strict=True), key=lambda pc: -pc[0])
    points: list[ThresholdChoice] = []
    hits = 0
    for i, (score, ok) in enumerate(pairs):
        hits += int(ok)
        is_last_of_tie = i + 1 == len(pairs) or pairs[i + 1][0] != score
        if is_last_of_tie:
            n = i + 1
            points.append(ThresholdChoice(score, hits / n, n / len(pairs), n))
    return points


def choose_threshold(
    scores: Sequence[float], correct: Sequence[bool], target_precision: float
) -> ThresholdChoice:
    """Lowest threshold whose auto-handled cases (score >= threshold) reach the target
    precision, i.e. the most coverage that still meets the target."""
    meeting = [p for p in operating_points(scores, correct) if p.precision >= target_precision]
    if not meeting:
        return ThresholdChoice(None, 0.0, 0.0, 0)
    return min(meeting, key=lambda p: p.threshold if p.threshold is not None else math.inf)
