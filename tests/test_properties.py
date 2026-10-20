"""Hypothesis properties for the parts where a wrong answer is silent: calibration maps, the
threshold picker and PII redaction."""

import math
import re

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from sklearn.isotonic import IsotonicRegression

from handoff_gauge.calibration import (
    IsotonicMap,
    PlattMap,
    brier_score,
    choose_threshold,
    expected_calibration_error,
    wilson_lower_bound,
)
from handoff_gauge.estimators import CombinedLogistic
from handoff_gauge.guardrails import redact

unit = st.floats(0.0, 1.0, allow_nan=False)
anywhere = st.floats(-10.0, 10.0, allow_nan=False)


@st.composite
def isotonic_maps(draw) -> IsotonicMap:
    xs = sorted(draw(st.lists(anywhere, min_size=1, max_size=12)))
    ys = sorted(draw(st.lists(st.floats(-0.5, 1.5), min_size=len(xs), max_size=len(xs))))
    return IsotonicMap(xs, ys)


platt_maps = st.builds(PlattMap, st.floats(0.0, 1e3), st.floats(-1e3, 1e3))


@given(st.one_of(isotonic_maps(), platt_maps), anywhere, anywhere)
def test_calibrated_scores_stay_in_unit_interval_and_are_monotonic(cmap, x1, x2) -> None:
    lo, hi = min(x1, x2), max(x1, x2)
    y_lo, y_hi = cmap.apply(lo), cmap.apply(hi)
    assert 0.0 <= y_lo <= 1.0 and 0.0 <= y_hi <= 1.0
    assert y_lo <= y_hi + 1e-12


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.tuples(unit, st.booleans()), min_size=2, max_size=60),
    st.lists(unit, max_size=20),
)
def test_exported_isotonic_map_matches_scikit_learn(data, probes) -> None:
    x = [d[0] for d in data]
    y = [float(d[1]) for d in data]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)
    exported = IsotonicMap(iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist())
    for p in probes + x:
        assert math.isclose(exported.apply(p), float(iso.predict(np.array([p]))[0]), abs_tol=1e-9)


@given(st.floats(-20, 20), st.floats(0, 20), st.floats(0, 20), unit, unit, unit)
def test_combined_score_is_a_probability_and_increasing(bias, wa, wv, a, v, delta) -> None:
    est = CombinedLogistic(bias, wa, wv)
    s = est.score(a, v)
    assert 0.0 <= s <= 1.0
    assert est.score(min(a + delta, 1.0), v) >= s
    assert est.score(a, min(v + delta, 1.0)) >= s


scored_data = st.lists(st.tuples(unit, st.booleans()), min_size=1, max_size=80)


def precision_at(data, t: float) -> float:
    picked = [ok for s, ok in data if s >= t]
    return sum(picked) / len(picked)


def bound_at(data, t: float, z: float) -> float:
    picked = [ok for s, ok in data if s >= t]
    return wilson_lower_bound(sum(picked), len(picked), z)


@given(scored_data, st.floats(0.5, 1.0), st.sampled_from([0.0, 1.2816]) | st.floats(0.0, 3.0))
def test_threshold_picker_meets_the_target_on_its_own_data(data, target, z) -> None:
    scores = [s for s, _ in data]
    correct = [ok for _, ok in data]
    choice = choose_threshold(scores, correct, target, z)
    if choice.threshold is None:
        # nothing reaches the target at any cut-off
        assert all(bound_at(data, t, z) < target for t in set(scores))
        return
    assert precision_at(data, choice.threshold) >= target
    assert bound_at(data, choice.threshold, z) >= target
    assert choice.precision == precision_at(data, choice.threshold)
    assert choice.n_auto == sum(s >= choice.threshold for s in scores)
    # and it is the most coverage that meets the target: no lower cut-off also does
    assert all(bound_at(data, t, z) < target for t in set(scores) if t < choice.threshold)


@given(st.integers(0, 500), st.integers(1, 500), st.floats(0.0, 3.0), st.floats(0.0, 3.0))
def test_wilson_bound_is_below_the_proportion_and_shrinks_with_z(hits, n, z1, z2) -> None:
    hits = min(hits, n)
    lo, hi = sorted((z1, z2))
    assert 0.0 <= wilson_lower_bound(hits, n, hi) <= wilson_lower_bound(hits, n, lo) + 1e-12
    assert wilson_lower_bound(hits, n, lo) <= hits / n + 1e-12


@given(scored_data)
def test_ece_and_brier_are_bounded(data) -> None:
    probs = [s for s, _ in data]
    correct = [ok for _, ok in data]
    assert 0.0 <= expected_calibration_error(probs, correct) <= 1.0
    assert 0.0 <= brier_score(probs, correct) <= 1.0


# --- redaction --------------------------------------------------------------------------

digits = st.text("0123456789", min_size=1, max_size=1)


@st.composite
def card_numbers(draw) -> str:
    n = draw(st.integers(13, 19))
    number = "".join(draw(digits) for _ in range(n))
    sep = draw(st.sampled_from(["", " ", "-", "."]))
    return sep.join(number[i:i + 4] for i in range(0, n, 4))


@st.composite
def phone_numbers(draw) -> str:
    n = draw(st.integers(9, 12))
    number = "".join(draw(digits) for _ in range(n))
    prefix = draw(st.sampled_from(["", "+", "+44 ", "(0", "0"]))
    sep = draw(st.sampled_from(["", " ", "-"]))
    body = sep.join(number[i:i + 3] for i in range(0, n, 3))
    return prefix + body if not prefix.startswith("(") else f"(0{body[:2]}) {body[2:]}"


emails = st.from_regex(r"[a-z][a-z0-9._+-]{0,10}@[a-z][a-z0-9-]{0,8}\.[a-z]{2,4}", fullmatch=True)
# Surrounding text has no digits or @, so any digit or address left after redaction leaked.
filler = st.text(st.sampled_from("abcdefghij XYZ,.!?'-\n"), max_size=20)


@settings(max_examples=200)
@given(st.lists(st.one_of(card_numbers(), phone_numbers(), emails), min_size=1, max_size=4),
       st.lists(filler, min_size=5, max_size=5))
def test_redaction_never_leaks_what_it_should_catch(pii, fillers) -> None:
    text = fillers[0] + "".join(f" {p} {f}" for p, f in zip(pii, fillers[1:], strict=False))
    redacted, counts = redact(text)
    assert not re.search(r"\d", redacted), redacted
    assert "@" not in redacted
    assert sum(counts.values()) >= 1
    assert redact(redacted) == (redacted, {})  # nothing left that a pattern would catch


@given(st.text(st.sampled_from("abcdefghij XYZ,.!?£$€:/0123456789"), max_size=60))
def test_short_numbers_are_kept(text) -> None:
    assume(not re.search(r"\d(?:[ .()+-]{0,2}\d){8}", text))  # fewer than 9 digits in a row
    assert redact(text) == (text, {})
