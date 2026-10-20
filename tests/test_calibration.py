import pytest

from handoff_gauge.calibration import (
    IdentityMap,
    IsotonicMap,
    PlattMap,
    brier_score,
    choose_threshold,
    expected_calibration_error,
    map_from_dict,
    operating_points,
    reliability_bins,
    wilson_lower_bound,
)


def test_isotonic_interpolates_and_clips() -> None:
    m = IsotonicMap([0.2, 0.6, 1.0], [0.1, 0.5, 0.9])
    assert m.apply(0.0) == 0.1
    assert m.apply(0.4) == pytest.approx(0.3)
    assert m.apply(0.6) == pytest.approx(0.5)
    assert m.apply(2.0) == 0.9


def test_isotonic_rejects_bad_points() -> None:
    with pytest.raises(ValueError):
        IsotonicMap([], [])
    with pytest.raises(ValueError):
        IsotonicMap([0.5, 0.2], [0.1, 0.2])
    with pytest.raises(ValueError):
        IsotonicMap([0.2, 0.5], [0.6, 0.2])


def test_platt_is_a_sigmoid() -> None:
    m = PlattMap(a=4.0, b=-2.0)
    assert m.apply(0.5) == pytest.approx(0.5)
    assert m.apply(1.0) > 0.85
    assert PlattMap(a=1000.0, b=0).apply(-5) >= 0.0  # no overflow


@pytest.mark.parametrize(
    "m", [IdentityMap(), IsotonicMap([0.0, 1.0], [0.2, 0.8]), PlattMap(3.0, -1.0)]
)
def test_maps_round_trip_through_dict(m) -> None:
    rebuilt = map_from_dict(m.to_dict())
    assert [rebuilt.apply(x / 10) for x in range(11)] == [m.apply(x / 10) for x in range(11)]


def test_unknown_map_kind() -> None:
    with pytest.raises(ValueError):
        map_from_dict({"kind": "magic"})


def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    probs = [0.75] * 4
    correct = [True, True, True, False]
    assert expected_calibration_error(probs, correct) == pytest.approx(0.0)


def test_ece_known_value() -> None:
    # bin [0.9, 1.0]: conf 0.95, acc 0.5 -> gap 0.45 over 2 of 4 items; bin [0.1, 0.2): gap 0.15
    probs = [0.95, 0.95, 0.15, 0.15]
    correct = [True, False, False, False]
    assert expected_calibration_error(probs, correct) == pytest.approx((2 * 0.45 + 2 * 0.15) / 4)


def test_reliability_bins_put_one_in_the_last_bin() -> None:
    bins = reliability_bins([1.0, 0.0], [True, False], n_bins=5)
    assert bins[-1]["count"] == 1 and bins[0]["count"] == 1


def test_brier_score() -> None:
    assert brier_score([1.0, 0.0], [True, False]) == 0.0
    assert brier_score([0.5, 0.5], [True, False]) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        brier_score([0.5], [])


def test_operating_points_group_ties() -> None:
    points = operating_points([0.9, 0.9, 0.5], [True, False, True])
    assert [(p.threshold, p.n_auto) for p in points] == [(0.9, 2), (0.5, 3)]
    assert points[0].precision == 0.5


def test_choose_threshold_takes_the_most_coverage_meeting_target() -> None:
    scores = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [True, True, True, True, False, False]
    choice = choose_threshold(scores, correct, 0.9)
    assert choice.threshold == 0.7
    assert choice.precision == 1.0
    assert choice.coverage == pytest.approx(4 / 6)


def test_wilson_lower_bound() -> None:
    assert wilson_lower_bound(9, 10, 0.0) == 0.9
    assert wilson_lower_bound(9, 10, 1.2816) == pytest.approx(0.7175, abs=1e-3)
    assert wilson_lower_bound(90, 100, 1.2816) > wilson_lower_bound(9, 10, 1.2816)
    assert wilson_lower_bound(0, 0, 1.0) == 0.0


def test_margin_rejects_a_cut_that_only_just_meets_the_target() -> None:
    # Auto-handling everything is exactly 90% precise here: it meets the target only in-sample.
    scores = [0.9] * 50 + [0.5] * 50
    correct = [True] * 50 + [True] * 40 + [False] * 10
    assert choose_threshold(scores, correct, 0.9).threshold == 0.5
    with_margin = choose_threshold(scores, correct, 0.9, z=1.2816)
    assert with_margin.threshold == 0.9
    assert with_margin.n_auto == 50 and with_margin.hits == 50


def test_choose_threshold_returns_none_when_target_unreachable() -> None:
    choice = choose_threshold([0.9, 0.8], [False, True], 0.9)
    assert choice.threshold is None and choice.coverage == 0.0
