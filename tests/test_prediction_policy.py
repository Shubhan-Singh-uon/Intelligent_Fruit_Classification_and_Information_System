"""Tests for src/ml/prediction_policy.py: every outcome, the rule order, and the boundaries."""
from src.fruit_classifier.schemas import ModelOutput, PredictionStatus
from src.ml.prediction_policy import (
    REASON_FAR_FROM_TRAINING_DATA,
    REASON_INCONSISTENT_VIEWS,
    REASON_LOW_CONFIDENCE,
    apply_policy,
    invalid_input_result,
)

VERSION = "M4_PC1-test"
CONF, DIST = 0.90, 50   # pass explicitly so the tests do not move when config.py is retuned


def make_output(apple=0.99, distance=20.0, tta=True):
    banana = 1.0 - apple
    return ModelOutput(
        predicted_class="Apple" if apple >= banana else "Banana",
        apple_probability=apple,
        banana_probability=banana,
        confidence=max(apple, banana),
        feature_distance=distance,
        tta_consistent=tta,
    )


def run(output):
    return apply_policy(output, VERSION, confidence_threshold=CONF, distance_threshold=DIST)


def test_accepted():
    result = run(make_output(apple=0.99))
    assert result.status is PredictionStatus.ACCEPTED
    assert result.predicted_class == "Apple"
    assert result.reason_code is None
    assert result.should_show_fruit_info


def test_accepted_banana_maps_to_lowercase_model_label():
    result = run(make_output(apple=0.02))
    assert result.status is PredictionStatus.ACCEPTED
    assert result.predicted_class == "Banana"
    assert result.model_label == "banana"      # matches FRUIT.model_label


def test_low_confidence_needs_review_with_tentative_class_and_no_fruit_info():
    result = run(make_output(apple=0.48))       # banana 0.52, like the wireframe example
    assert result.status is PredictionStatus.NEEDS_REVIEW
    assert result.reason_code == REASON_LOW_CONFIDENCE
    assert result.predicted_class == "Banana"   # tentative prediction is still shown
    assert result.apple_probability == 0.48 and result.banana_probability == 0.52
    assert not result.should_show_fruit_info


def test_inconsistent_views_need_review():
    result = run(make_output(apple=0.99, tta=False))
    assert result.status is PredictionStatus.NEEDS_REVIEW
    assert result.reason_code == REASON_INCONSISTENT_VIEWS
    assert not result.should_show_fruit_info


def test_far_from_training_data_is_rejected_even_when_model_is_confident():
    result = run(make_output(apple=0.999, distance=80.0))
    assert result.status is PredictionStatus.REJECTED
    assert result.reason_code == REASON_FAR_FROM_TRAINING_DATA
    assert result.predicted_class is None       # must not confirm a fruit
    assert result.model_label is None
    assert result.confidence == 0.999           # still recorded for the history table
    assert not result.should_show_fruit_info


def test_rejection_takes_priority_over_low_confidence():
    result = run(make_output(apple=0.42, distance=80.0))   # 58% Banana but far away: wireframe C
    assert result.status is PredictionStatus.REJECTED


def test_boundaries_are_inclusive_for_accept():
    at_distance = run(make_output(apple=0.99, distance=DIST))
    assert at_distance.status is PredictionStatus.ACCEPTED
    just_over = run(make_output(apple=0.99, distance=DIST + 0.01))
    assert just_over.status is PredictionStatus.REJECTED

    at_confidence = run(make_output(apple=CONF))
    assert at_confidence.status is PredictionStatus.ACCEPTED
    just_under = run(make_output(apple=CONF - 0.001))
    assert just_under.status is PredictionStatus.NEEDS_REVIEW


def test_invalid_input_has_no_prediction_fields():
    result = invalid_input_result("unreadable_file", VERSION)
    assert result.status is PredictionStatus.INVALID_INPUT
    assert result.reason_code == "unreadable_file"
    for field in ("predicted_class", "apple_probability", "banana_probability", "confidence", "fruit_id"):
        assert getattr(result, field) is None
    assert not result.should_show_fruit_info


def test_status_values_are_the_strings_stored_in_the_database():
    assert [s.value for s in PredictionStatus] == ["Accepted", "Needs Review", "Rejected", "Invalid Input"]
