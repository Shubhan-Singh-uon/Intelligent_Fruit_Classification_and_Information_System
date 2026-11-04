"""
File: prediction_policy.py
Project: Intelligent Fruit Classification and Information System
Description:
    Applies the agreed acceptance, needs-review, and rejection
    rules to model outputs, including configured confidence
    thresholds and any implemented unsupported-input checks.

Author: Shubhan Singh
Responsible Pillar: Pillar 2 - Machine Learning & Evaluation
Maintainers: Shubhan Singh

Note:
    The model's predicted class and probabilities are distinct
    from the final policy decision. Confidence alone does not
    establish prediction correctness.

Decision rule (checked in this order):
    1. feature_distance > DISTANCE_REJECT_THRESHOLD   -> REJECTED      (far from both classes: an
                                                                        out-of-distribution signal, not proof
                                                                        that the image is not a fruit)
    2. confidence < CONFIDENCE_THRESHOLD              -> NEEDS_REVIEW  (the model is torn between classes)
    3. predictions disagree under test-time views     -> NEEDS_REVIEW  (unstable prediction)
    4. otherwise                                      -> ACCEPTED

    The accept condition is exactly the ML notebook's combined signal (confidence,
    feature distance AND TTA consistency). The notebook only decided accept vs "unable to
    classify"; splitting the non-accepted images into REJECTED and NEEDS_REVIEW by which
    signal failed is the project's own addition, so it needs team sign-off.
"""
from src.fruit_classifier.config import CONFIDENCE_THRESHOLD, DISTANCE_REJECT_THRESHOLD
from src.fruit_classifier.schemas import ModelOutput, PredictionResult, PredictionStatus

# Reason codes, stored in PREDICTION_HISTORY.rejection_reason and used by the UI to pick a message.
REASON_FAR_FROM_TRAINING_DATA = "far_from_training_data"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_INCONSISTENT_VIEWS = "inconsistent_under_transforms"


def apply_policy(
    output: ModelOutput,
    model_version: str,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    distance_threshold: float = DISTANCE_REJECT_THRESHOLD,
) -> PredictionResult:
    """Turn raw model signals into one of the outcomes A, B or C.

    The thresholds are parameters (defaulting to config.py) so tests and threshold
    sweeps can vary them without editing shared configuration.
    """
    common = dict(
        model_version=model_version,
        apple_probability=output.apple_probability,
        banana_probability=output.banana_probability,
        confidence=output.confidence,
        feature_distance=output.feature_distance,
        tta_consistent=output.tta_consistent,
    )

    if output.feature_distance > distance_threshold:
        # Outcome C: the application must not confirm a fruit, so predicted_class stays None.
        return PredictionResult(
            status=PredictionStatus.REJECTED,
            reason_code=REASON_FAR_FROM_TRAINING_DATA,
            **common,
        )

    if output.confidence < confidence_threshold:
        return PredictionResult(
            status=PredictionStatus.NEEDS_REVIEW,
            predicted_class=output.predicted_class,
            reason_code=REASON_LOW_CONFIDENCE,
            **common,
        )

    if not output.tta_consistent:
        return PredictionResult(
            status=PredictionStatus.NEEDS_REVIEW,
            predicted_class=output.predicted_class,
            reason_code=REASON_INCONSISTENT_VIEWS,
            **common,
        )

    return PredictionResult(
        status=PredictionStatus.ACCEPTED,
        predicted_class=output.predicted_class,
        **common,
    )


def invalid_input_result(reason_code: str, model_version: str) -> PredictionResult:
    """Outcome D. No inference ran, so there are no probabilities or confidence."""
    return PredictionResult(
        status=PredictionStatus.INVALID_INPUT,
        model_version=model_version,
        reason_code=reason_code,
    )
