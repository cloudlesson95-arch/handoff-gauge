"""Fit calibration maps per estimator, pick the threshold, log every run to MLflow and export
calibration.json (for the live agent) and reports/calibration.{json,png} (for the web UI).

Selection uses only the calibration split (5-fold cross-validation); the test split is
used for reporting. "ECE before" is the model's own stated confidence, uncalibrated.

Usage: uv run python scripts/calibrate.py [--target 0.90]
"""

import argparse
import asyncio
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib
import mlflow
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from handoff_gauge import config
from handoff_gauge.calibration import (
    CalibrationMap,
    IdentityMap,
    IsotonicMap,
    PlattMap,
    brier_score,
    choose_threshold,
    expected_calibration_error,
    operating_points,
    reliability_bins,
)
from handoff_gauge.estimators import (
    CombinedLogistic,
    ConfidenceEstimator,
    SelfConsistency,
    VerbalizedConfidence,
)
from handoff_gauge.evaluation import Scored, load_split, replay

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRACKING_URI = f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = (config.ROOT / "mlruns").as_uri()
EXPERIMENT = "calibration"
Z = config.PRECISION_BOUND_Z
ESTIMATORS = ("verbalized", "self_consistency", "combined")
METHODS = ("identity", "isotonic", "platt")


def fit_estimator(name: str, train: Sequence[Scored]) -> ConfidenceEstimator:
    if name == "verbalized":
        return VerbalizedConfidence()
    if name == "self_consistency":
        return SelfConsistency()
    x = np.array([[s.agreement, s.verbalized] for s in train])
    y = np.array([s.correct for s in train])
    if len(set(y)) < 2:
        return CombinedLogistic()
    lr = LogisticRegression().fit(x, y)
    return CombinedLogistic(
        bias=float(lr.intercept_[0]),
        w_agreement=float(lr.coef_[0][0]),
        w_verbalized=float(lr.coef_[0][1]),
    )


def raw_scores(estimator: ConfidenceEstimator, items: Sequence[Scored]) -> list[float]:
    return [estimator.score(s.agreement, s.verbalized) if s.intent else 0.0 for s in items]


def fit_map(method: str, scores: Sequence[float], correct: Sequence[bool]) -> CalibrationMap:
    if method == "identity":
        return IdentityMap()
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, correct)
        return IsotonicMap(iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist())
    rate = min(max(float(np.mean(correct)), 1e-3), 1 - 1e-3)
    if len(set(correct)) < 2:
        return PlattMap(0.0, math.log(rate / (1 - rate)))
    lr = LogisticRegression(C=1e4).fit(np.reshape(scores, (-1, 1)), correct)
    a = float(lr.coef_[0][0])
    if a <= 0:  # no usable signal: a flat map at the base rate keeps it monotonic
        return PlattMap(0.0, math.log(rate / (1 - rate)))
    return PlattMap(a, float(lr.intercept_[0]))


@dataclass
class Fitted:
    estimator: ConfidenceEstimator
    cmap: CalibrationMap
    threshold: float | None

    def calibrated(self, items: Sequence[Scored]) -> list[float]:
        return [self.cmap.apply(x) for x in raw_scores(self.estimator, items)]


def fit(name: str, method: str, train: Sequence[Scored], target: float) -> Fitted:
    estimator = fit_estimator(name, train)
    correct = [s.correct for s in train]
    cmap = fit_map(method, raw_scores(estimator, train), correct)
    calibrated = [cmap.apply(x) for x in raw_scores(estimator, train)]
    return Fitted(estimator, cmap, choose_threshold(calibrated, correct, target, Z).threshold)


def auto_stats(probs: Sequence[float], items: Sequence[Scored], threshold: float | None,
               end_to_end: bool = False) -> dict[str, float]:
    auto = [
        s for p, s in zip(probs, items, strict=True)
        if threshold is not None and p >= threshold and not (end_to_end and s.input_tripped)
    ]
    hits = sum(s.correct for s in auto)
    return {
        "n_auto": len(auto),
        "precision": hits / len(auto) if auto else 0.0,
        "coverage": len(auto) / len(items),
        "handoff_rate": 1 - len(auto) / len(items),
    }


def cross_validate(name: str, method: str, cal: Sequence[Scored], target: float) -> dict:
    """Held-out calibrated scores and auto decisions over 5 folds of the calibration split."""
    probs: list[float] = []
    held: list[Scored] = []
    auto: list[bool] = []
    for train_idx, test_idx in KFold(5, shuffle=True, random_state=0).split(cal):
        fitted = fit(name, method, [cal[i] for i in train_idx], target)
        part = [cal[i] for i in test_idx]
        p = fitted.calibrated(part)
        probs += p
        held += part
        auto += [fitted.threshold is not None and x >= fitted.threshold for x in p]
    picked = [s for s, a in zip(held, auto, strict=True) if a]
    correct = [s.correct for s in held]
    return {
        "cv_brier": brier_score(probs, correct),
        "cv_ece": expected_calibration_error(probs, correct),
        "cv_precision": sum(s.correct for s in picked) / len(picked) if picked else 0.0,
        "cv_coverage": len(picked) / len(held),
    }


def plot(report: dict, path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    left.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect")
    for key, color, label in (("before", "#d97706", "stated confidence"),
                              ("after", "#2563eb", "calibrated")):
        bins = [b for b in report["reliability"][key] if b["count"]]
        left.plot([b["conf"] for b in bins], [b["acc"] for b in bins], "o-", color=color,
                  label=f"{label} (ECE {report['headline'][f'ece_{key}']:.3f})")
    left.set(xlabel="confidence", ylabel="accuracy", title="Reliability (test split)",
             xlim=(0, 1), ylim=(0, 1.02))
    left.legend(loc="upper left", fontsize=8)
    curve = report["curve"]
    right.plot([p["coverage"] for p in curve], [p["precision"] for p in curve], "o-",
               color="#2563eb")
    right.axhline(report["target_precision"], color="#999", ls="--", lw=1, label="target")
    right.set(xlabel="coverage (share handled automatically)", ylabel="precision",
              title="Precision vs coverage (test split)", xlim=(0, 1.02), ylim=(0, 1.02))
    right.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(target: float) -> None:
    cal = asyncio.run(replay(load_split("calibration")))
    test = asyncio.run(replay(load_split("test")))
    misses = sum(s.cache_misses for s in cal + test)
    if misses:
        print(f"warning: {misses} samples missing from the cache count as invalid")
    test_correct = [s.correct for s in test]
    stated = [s.verbalized for s in test]

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT)

    rows = []
    for name in ESTIMATORS:
        for method in METHODS:
            fitted = fit(name, method, cal, target)
            raw_test = raw_scores(fitted.estimator, test)
            probs = fitted.calibrated(test)
            row = {
                "estimator": name,
                "method": method,
                "ece_raw": expected_calibration_error(raw_test, test_correct),
                "ece": expected_calibration_error(probs, test_correct),
                "brier_raw": brier_score(raw_test, test_correct),
                "brier": brier_score(probs, test_correct),
                "threshold": fitted.threshold,
                **cross_validate(name, method, cal, target),
                **{f"test_{k}": v for k, v in auto_stats(probs, test, fitted.threshold).items()},
                "_fitted": fitted,
                "_probs": probs,
            }
            rows.append(row)

    candidates = [r for r in rows if r["method"] != "identity"]
    meeting = [r for r in candidates if r["cv_precision"] >= target] or candidates
    chosen = max(meeting, key=lambda r: (r["cv_coverage"], -r["cv_brier"]))
    fitted: Fitted = chosen["_fitted"]
    probs = chosen["_probs"]
    e2e = auto_stats(probs, test, fitted.threshold, end_to_end=True)

    report = {
        "model": config.MODEL,
        "n_samples": config.N_SAMPLES,
        "temperature": config.TEMPERATURE,
        "target_precision": target,
        "precision_bound_z": Z,
        "splits": {"calibration": len(cal), "test": len(test)},
        "chosen": {"estimator": chosen["estimator"], "method": chosen["method"],
                   "threshold": fitted.threshold},
        "headline": {
            "accuracy": sum(test_correct) / len(test),
            "ece_before": expected_calibration_error(stated, test_correct),
            "ece_after": chosen["ece"],
            "brier_before": brier_score(stated, test_correct),
            "brier_after": chosen["brier"],
            "precision_auto": chosen["test_precision"],
            "coverage": chosen["test_coverage"],
            "handoff_rate": chosen["test_handoff_rate"],
            "n_auto": chosen["test_n_auto"],
            "end_to_end_coverage": e2e["coverage"],
            "end_to_end_precision": e2e["precision"],
        },
        "reliability": {
            "before": reliability_bins(stated, test_correct),
            "after": reliability_bins(probs, test_correct),
        },
        "curve": [
            {"threshold": p.threshold, "precision": p.precision, "coverage": p.coverage}
            for p in operating_points(probs, test_correct)
        ],
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    profile = {
        "estimator": chosen["estimator"],
        "estimator_params": fitted.estimator.params(),
        "method": chosen["method"],
        "map": fitted.cmap.to_dict(),
        "threshold": fitted.threshold,
        "target_precision": target,
        "precision_bound_z": Z,
        "model": config.MODEL,
        "n_samples": config.N_SAMPLES,
        "temperature": config.TEMPERATURE,
        "fitted_on": f"calibration split ({len(cal)} messages)",
    }

    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.CALIBRATION_PATH.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    report_json = config.REPORTS_DIR / "calibration.json"
    report_png = config.REPORTS_DIR / "calibration.png"
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(report, report_png)

    for r in rows:
        with mlflow.start_run(run_name=f"{r['estimator']}-{r['method']}"):
            mlflow.set_tag("chosen", str(r is chosen).lower())
            mlflow.log_params({
                "estimator": r["estimator"], "method": r["method"], "model": config.MODEL,
                "n_samples": config.N_SAMPLES, "temperature": config.TEMPERATURE,
                "target_precision": target, "precision_bound_z": Z,
                "n_calibration": len(cal), "n_test": len(test),
                **{f"param_{k}": v for k, v in r["_fitted"].estimator.params().items()},
            })
            mlflow.log_metrics({
                k: float(v) for k, v in r.items()
                if not k.startswith("_") and isinstance(v, int | float) and k != "threshold"
            } | {"threshold": -1.0 if r["threshold"] is None else r["threshold"],
                 "accuracy": report["headline"]["accuracy"]})
            mlflow.log_dict(r["_fitted"].cmap.to_dict(), "map.json")
            if r is chosen:
                mlflow.log_artifact(str(report_png))
                mlflow.log_artifact(str(report_json))
                mlflow.log_artifact(str(config.CALIBRATION_PATH))

    print(f"{'estimator':<17}{'method':<10}{'ECE raw->cal':>16}{'Brier raw->cal':>17}"
          f"{'CV prec':>9}{'CV cov':>8}{'test prec':>11}{'test cov':>10}")
    for r in rows:
        print(f"{r['estimator']:<17}{r['method']:<10}{r['ece_raw']:>8.3f} ->{r['ece']:>6.3f}"
              f"{r['brier_raw']:>9.3f} ->{r['brier']:>6.3f}{r['cv_precision']:>9.2f}"
              f"{r['cv_coverage']:>8.2f}{r['test_precision']:>11.2f}{r['test_coverage']:>10.2f}")
    h = report["headline"]
    threshold = "none (always hand off)" if fitted.threshold is None else f"{fitted.threshold:.3f}"
    print(f"\nchosen: {chosen['estimator']} + {chosen['method']} (by cross-validation on the "
          f"calibration split)")
    print(f"accuracy of the majority intent (test): {h['accuracy']:.3f}")
    print(f"ECE before (stated confidence): {h['ece_before']:.3f} -> after: {h['ece_after']:.3f}")
    print(f"Brier before: {h['brier_before']:.3f} -> after: {h['brier_after']:.3f}")
    print(f"threshold: {threshold} (target precision {target:.2f})")
    print(f"precision on auto-handled test cases: {h['precision_auto']:.3f} "
          f"({h['n_auto']} of {len(test)})")
    print(f"coverage: {h['coverage']:.3f}, handoff rate: {h['handoff_rate']:.3f} "
          f"(end to end with input guardrails: coverage {h['end_to_end_coverage']:.3f})")
    print(f"logged {len(rows)} runs to MLflow experiment '{EXPERIMENT}' ({TRACKING_URI})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=config.TARGET_PRECISION)
    main(parser.parse_args().target)
