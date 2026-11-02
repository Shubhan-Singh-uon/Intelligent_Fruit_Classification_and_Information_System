"""Tests for src/services/prediction_service.py using a fake model and fake repositories.

No torch, no database file: these check the workflow rules (what is shown, what is saved, how
technical failures are reported). test_prediction_service_integration.py runs the same service
against the real repository code and the real SQL schema.
"""
import io
import sqlite3

import pytest
from PIL import Image

from src.fruit_classifier.exceptions import DatabaseUnavailableError, InferenceError
from src.fruit_classifier.schemas import ModelOutput, PredictionStatus
from src.services.prediction_service import (
    ISSUE_FRUIT_INFO_UNAVAILABLE,
    ISSUE_HISTORY_NOT_SAVED,
    PredictionService,
)

VERSION = "M4_PC1-test"


# ------------------------------------------------------------------------------ 
# Doubles
# ------------------------------------------------------------------------------ 
def model_output(apple=0.99, distance=20.0, tta=True):
    banana = 1.0 - apple
    return ModelOutput(
        predicted_class="Apple" if apple >= banana else "Banana",
        apple_probability=apple,
        banana_probability=banana,
        confidence=max(apple, banana),
        feature_distance=distance,
        tta_consistent=tta,
    )


class FakeModel:
    model_version = VERSION

    def __init__(self, output=None, error=None):
        self.output, self.error, self.calls = output, error, 0

    def predict(self, image):
        self.calls += 1
        if self.error:
            raise self.error
        assert image.size == (128, 128) and image.mode == "RGB"   # preprocessing really ran
        return self.output


APPLE = {"fruit_id": 1, "fruit_name": "Apple", "model_label": "apple", "description": "An apple."}
BANANA = {"fruit_id": 2, "fruit_name": "Banana", "model_label": "banana", "description": "A banana."}
SRC_A = {"source_name": "AFCD apple", "source_url": "https://example.org/a", "access_date": "2026-10-16"}
SRC_B = {"source_name": "Harvard apples", "source_url": "https://example.org/b", "access_date": "2026-10-16"}


class FakeFruitRepo:
    def __init__(self, fail=False, fruits=(APPLE, BANANA)):
        self.fail, self.calls = fail, 0
        self.fruits = {f["model_label"]: f for f in fruits}

    def _tick(self):
        self.calls += 1
        if self.fail:
            raise sqlite3.OperationalError("no such table: FRUIT")

    def get_fruit_by_model_label(self, label):
        self._tick()
        return self.fruits.get(label)

    def get_nutrition_by_fruit_id(self, fruit_id):
        self._tick()
        return {"nutrition_id": 1, "fruit_id": fruit_id, "serving_basis": "100g", "energy_kcal": 57.8, **SRC_A}

    def get_health_benefits_by_fruit_id(self, fruit_id):
        self._tick()
        return [
            {"benefit_description": "Fibre.", **SRC_B},
            {"benefit_description": "Pectin.", **SRC_B},   # same source twice
        ]

    def get_uses_by_fruit_id(self, fruit_id):
        self._tick()
        return [{"use_id": 1, "fruit_id": fruit_id, "use_description": "Eat Fresh"}]

    def get_fun_facts_by_fruit_id(self, fruit_id):
        self._tick()
        return [{"fact_text": "Rose family.", **SRC_A}]    # source already used by nutrition


class FakePredictionRepo:
    def __init__(self, fail=False, history=()):
        self.fail, self.saved, self.history = fail, [], list(history)

    def save_prediction(self, **kwargs):
        if self.fail:
            raise sqlite3.OperationalError("database is locked")
        self.saved.append(kwargs)

    def get_prediction_history(self):
        if self.fail:
            raise sqlite3.OperationalError("no such table: PREDICTION_HISTORY")
        return self.history


def upload(fmt="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", (300, 200), (200, 30, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


def make_service(output=None, error=None, fruit_repo=None, prediction_repo=None, **kwargs):
    model = FakeModel(output, error)
    fruit_repo = fruit_repo or FakeFruitRepo()
    prediction_repo = prediction_repo or FakePredictionRepo()
    service = PredictionService(model, fruit_repo, prediction_repo, **kwargs)
    return service, model, fruit_repo, prediction_repo


# ------------------------------------------------------------------------------ 
# Outcomes
# ------------------------------------------------------------------------------
def test_accepted_returns_fruit_information_and_saves_a_linked_row():
    service, _, _, repo = make_service(model_output(apple=0.99))
    response = service.classify(upload())

    assert response.prediction.status is PredictionStatus.ACCEPTED
    assert response.prediction.fruit_id == 1
    assert response.history_saved and response.issues == ()

    info = response.fruit_info
    assert info.fruit["fruit_name"] == "Apple"
    assert info.nutrition["energy_kcal"] == 57.8
    assert [b["benefit_description"] for b in info.health_benefits] == ["Fibre.", "Pectin."]
    assert [u["use_description"] for u in info.uses] == ["Eat Fresh"]
    assert [f["fact_text"] for f in info.fun_facts] == ["Rose family."]
    # Two distinct source URLs even though three rows cite them, in first-used order.
    assert [s["source_url"] for s in info.sources] == [SRC_A["source_url"], SRC_B["source_url"]]

    assert repo.saved == [
        dict(
            fruit_id=1, predicted_class="Apple", apple_probability=0.99, banana_probability=pytest.approx(0.01),
            confidence=0.99, prediction_status="accepted", rejection_reason=None, model_version=VERSION,
        )
    ]


def test_banana_is_looked_up_by_lowercase_label():
    service, *_ = make_service(model_output(apple=0.01))
    response = service.classify(upload())
    assert response.prediction.predicted_class == "Banana"
    assert response.fruit_info.fruit["fruit_name"] == "Banana"


def test_needs_review_shows_no_fruit_information_but_is_saved():
    service, _, fruit_repo, repo = make_service(model_output(apple=0.45))   # 55% Banana
    response = service.classify(upload())

    assert response.prediction.status is PredictionStatus.NEEDS_REVIEW
    assert response.prediction.predicted_class == "Banana"      # tentative
    assert response.fruit_info is None
    assert fruit_repo.calls == 0                                # the database was not even queried
    assert response.history_saved
    row = repo.saved[0]
    assert row["prediction_status"] == "needs_review" and row["fruit_id"] is None
    assert row["rejection_reason"] == "low_confidence"


def test_rejected_confirms_no_fruit_but_keeps_the_numbers():
    service, _, fruit_repo, repo = make_service(model_output(apple=0.999, distance=120.0))
    response = service.classify(upload())

    assert response.prediction.status is PredictionStatus.REJECTED
    assert response.fruit_info is None and fruit_repo.calls == 0
    row = repo.saved[0]
    assert row["prediction_status"] == "rejected" and row["predicted_class"] is None
    assert row["fruit_id"] is None and row["apple_probability"] == 0.999


@pytest.mark.parametrize(
    "data, reason",
    [(b"", "empty_file"), (b"not an image" * 20, "unreadable_file"), (upload("GIF"), "unsupported_format")],
)
def test_invalid_input_never_reaches_the_model_and_has_no_probabilities(data, reason):
    service, model, fruit_repo, repo = make_service(model_output())
    response = service.classify(data)

    assert response.prediction.status is PredictionStatus.INVALID_INPUT
    assert response.prediction.reason_code == reason
    assert model.calls == 0 and fruit_repo.calls == 0 and response.fruit_info is None
    row = repo.saved[0]
    assert row["prediction_status"] == "invalid_input" and row["rejection_reason"] == reason
    for field in ("fruit_id", "predicted_class", "apple_probability", "banana_probability", "confidence"):
        assert row[field] is None


def test_accepts_a_file_like_upload():
    service, *_ = make_service(model_output())
    assert service.classify(io.BytesIO(upload())).prediction.status is PredictionStatus.ACCEPTED


# ------------------------------------------------------------------------------
# Technical failures
# ------------------------------------------------------------------------------
def test_history_save_failure_still_shows_the_accepted_prediction_and_info():
    service, *_ = make_service(model_output(), prediction_repo=FakePredictionRepo(fail=True))
    response = service.classify(upload())

    assert response.prediction.status is PredictionStatus.ACCEPTED
    assert response.fruit_info is not None
    assert response.history_saved is False
    assert response.issues == (ISSUE_HISTORY_NOT_SAVED,)


def test_fruit_info_failure_keeps_the_prediction_and_still_saves_history():
    service, _, _, repo = make_service(model_output(), fruit_repo=FakeFruitRepo(fail=True))
    response = service.classify(upload())

    assert response.prediction.status is PredictionStatus.ACCEPTED
    assert response.fruit_info is None and response.prediction.fruit_id is None
    assert response.issues == (ISSUE_FRUIT_INFO_UNAVAILABLE,)
    assert response.history_saved and repo.saved[0]["fruit_id"] is None


def test_label_missing_from_the_database_is_reported_not_crashed():
    service, *_ = make_service(model_output(), fruit_repo=FakeFruitRepo(fruits=(BANANA,)))
    response = service.classify(upload())
    assert response.fruit_info is None and response.issues == (ISSUE_FRUIT_INFO_UNAVAILABLE,)


def test_both_database_steps_failing_still_returns_the_prediction():
    service, *_ = make_service(
        model_output(), fruit_repo=FakeFruitRepo(fail=True), prediction_repo=FakePredictionRepo(fail=True)
    )
    response = service.classify(upload())
    assert response.prediction.status is PredictionStatus.ACCEPTED
    assert response.issues == (ISSUE_FRUIT_INFO_UNAVAILABLE, ISSUE_HISTORY_NOT_SAVED)


def test_invalid_input_with_database_down_is_still_an_invalid_input_outcome():
    service, *_ = make_service(model_output(), prediction_repo=FakePredictionRepo(fail=True))
    response = service.classify(b"garbage" * 10)
    assert response.prediction.status is PredictionStatus.INVALID_INPUT
    assert response.issues == (ISSUE_HISTORY_NOT_SAVED,)


def test_model_crash_is_a_technical_failure_not_an_outcome():
    service, _, _, repo = make_service(error=RuntimeError("boom"))
    with pytest.raises(InferenceError):
        service.classify(upload())
    assert repo.saved == []


# ------------------------------------------------------------------------------
# Save Policy
# ------------------------------------------------------------------------------
def test_statuses_left_out_of_the_save_policy_are_not_stored_and_raise_no_issue():
    service, _, _, repo = make_service(
        model_output(distance=120.0), save_statuses=("accepted", "needs_review")
    )
    response = service.classify(upload())
    assert response.prediction.status is PredictionStatus.REJECTED
    assert repo.saved == [] and response.history_saved is False and response.issues == ()


# ------------------------------------------------------------------------------
# Prediction History
# ------------------------------------------------------------------------------
def history_row(prediction_id, status="accepted"):
    return {"prediction_id": prediction_id, "prediction_status": status, "predicted_class": "Apple"}


def test_history_is_returned_as_plain_dicts_newest_first():
    repo = FakePredictionRepo(history=[history_row(1), history_row(3), history_row(2)])
    service, *_ = make_service(prediction_repo=repo)
    assert [r["prediction_id"] for r in service.get_history()] == [3, 2, 1]


def test_history_failure_is_reported_as_database_unavailable():
    service, *_ = make_service(prediction_repo=FakePredictionRepo(fail=True))
    with pytest.raises(DatabaseUnavailableError):
        service.get_history()


def test_history_summary_matches_the_wireframe_tiles():
    records = [history_row(i, s) for i, s in enumerate(
        ["accepted"] * 9 + ["needs_review"] * 2 + ["rejected", "invalid_input"], start=1)]
    assert PredictionService.summarise_history(records) == {
        "total": 13, "accepted": 9, "needs_review": 2, "rejected_or_invalid": 2,
    }
    assert PredictionService.summarise_history([]) == {
        "total": 0, "accepted": 0, "needs_review": 0, "rejected_or_invalid": 0,
    }
