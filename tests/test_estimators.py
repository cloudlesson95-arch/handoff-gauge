import pytest

from handoff_gauge.answers import Answer, majority_intent
from handoff_gauge.estimators import (
    CombinedLogistic,
    SelfConsistency,
    VerbalizedConfidence,
    build_estimator,
    features,
)


def a(intent: str, confidence: float) -> Answer:
    return Answer(intent=intent, confidence=confidence)


MIXED = [a("card_arrival", 0.9), a("card_arrival", 0.7), a("card_delivery_estimate", 0.95)]


def test_features_use_all_samples_as_denominator() -> None:
    intent, agreement, verbalized = features([*MIXED, None])
    assert intent == "card_arrival"
    assert agreement == pytest.approx(2 / 4)
    assert verbalized == pytest.approx(0.8)


def test_verbalized_strategy_uses_stated_confidence() -> None:
    est = VerbalizedConfidence().estimate(MIXED)
    assert est.intent == "card_arrival"
    assert est.raw_score == pytest.approx(0.8)


def test_self_consistency_strategy_uses_agreement() -> None:
    assert SelfConsistency().estimate(MIXED).raw_score == pytest.approx(2 / 3)
    unanimous = [a("pending_transfer", 0.5)] * 3
    assert SelfConsistency().estimate(unanimous).raw_score == 1.0


def test_combined_strategy_is_increasing_in_both_signals() -> None:
    combined = CombinedLogistic(bias=-3.0, w_agreement=2.0, w_verbalized=1.5)
    base = combined.score(0.5, 0.5)
    assert 0.0 < base < 1.0
    assert combined.score(0.9, 0.5) > base
    assert combined.score(0.5, 0.9) > base


@pytest.mark.parametrize("answers", [[], [None, None, None]])
def test_no_valid_answer_scores_zero(answers) -> None:
    for strategy in (VerbalizedConfidence(), SelfConsistency(), CombinedLogistic()):
        est = strategy.estimate(answers)
        assert est.intent is None and est.raw_score == 0.0


def test_majority_tie_breaks_by_confidence_then_name() -> None:
    assert majority_intent([a("b_label", 0.9), a("a_label", 0.4)]) == "b_label"
    assert majority_intent([a("b_label", 0.5), a("a_label", 0.5)]) == "a_label"


def test_build_estimator_round_trips_params() -> None:
    combined = CombinedLogistic(bias=-1.0, w_agreement=2.0, w_verbalized=3.0)
    rebuilt = build_estimator("combined", combined.params())
    assert rebuilt.score(0.4, 0.6) == combined.score(0.4, 0.6)
    with pytest.raises(ValueError, match="unknown estimator"):
        build_estimator("magic")
