"""
File: test_hasnain.py
Project: Intelligent Fruit Classification and Information System
Description:
    small tests for the prediction service. a fake model is use so
    the test is fast and a new test databse is made every time

Author: Hasnain Younas
Responsible Pillar: Pillar 4 - Application & System Integration

Run: python -m pytest hasnain
"""
import io
import sqlite3
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.repositories import database
from src.fruit_classifier.exceptions import InferenceError, DatabaseUnavailableError
from src.fruit_classifier.schemas import ModelOutput, PredictionStatus
from hasnain.prediction_service import PredictionService, get_history, get_summary

DB_FOLDER = Path(__file__).resolve().parents[1] / "src" / "database"
APP_CODE = "from hasnain.pages import run_app\nrun_app()"


class FakeModel:
    model_version = "fake-v1"

    def __init__(self, output):
        self.output = output

    def predict(self, image):
        return self.output


class BrokenModel:
    model_version = "broken-v1"

    def predict(self, image):
        raise RuntimeError("model crash")


def make_test_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    connection = sqlite3.connect(path)
    connection.executescript((DB_FOLDER / "schema.sql").read_text(encoding="utf-8"))
    connection.executescript((DB_FOLDER / "seed_fruit.sql").read_text(encoding="utf-8"))
    connection.close()
    monkeypatch.setattr(database, "DATABASE_PATH", path)


def make_photo():
    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), "red").save(buffer, "PNG")
    return buffer.getvalue()


def model_output(apple, distance=10.0, same=True):
    return ModelOutput(
        predicted_class="Apple" if apple >= 0.5 else "Banana",
        apple_probability=apple,
        banana_probability=1 - apple,
        confidence=max(apple, 1 - apple),
        feature_distance=distance,
        tta_consistent=same,
    )


def test_accepted_apple_show_info_and_save(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    service = PredictionService(FakeModel(model_output(0.99)))
    response = service.classify(make_photo())
    assert response.prediction.status == PredictionStatus.ACCEPTED
    assert response.fruit_info.fruit["fruit_name"] == "Apple"
    assert len(response.fruit_info.sources) > 0
    assert response.history_saved == True
    assert get_history()[0]["fruit_id"] == 1


def test_needs_review_and_rejected_have_no_info(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    review = PredictionService(FakeModel(model_output(0.6))).classify(make_photo())
    rejected = PredictionService(FakeModel(model_output(0.99, distance=500))).classify(make_photo())
    assert review.prediction.status == PredictionStatus.NEEDS_REVIEW
    assert rejected.prediction.status == PredictionStatus.REJECTED
    assert review.fruit_info == None
    assert rejected.fruit_info == None
    assert len(get_history()) == 2


def test_broken_file_is_invalid(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    response = PredictionService(FakeModel(model_output(0.99))).classify(b"this is not a image")
    assert response.prediction.status == PredictionStatus.INVALID_INPUT
    assert response.prediction.reason_code == "unreadable_file"
    assert get_history()[0]["prediction_status"] == PredictionStatus.INVALID_INPUT.value


def test_no_databse_still_give_result(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "empty.db")
    response = PredictionService(FakeModel(model_output(0.99))).classify(make_photo())
    assert response.prediction.status == PredictionStatus.ACCEPTED
    assert "history_not_saved" in response.issues
    assert "fruit_info_unavailable" in response.issues


def test_photo_change_when_turn_need_review(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    response = PredictionService(FakeModel(model_output(0.99, same=False))).classify(make_photo())
    assert response.prediction.status == PredictionStatus.NEEDS_REVIEW
    assert response.prediction.reason_code == "inconsistent_under_transforms"
    assert response.fruit_info == None


def test_model_crash_is_a_error(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    with pytest.raises(InferenceError):
        PredictionService(BrokenModel()).classify(make_photo())
    assert len(get_history()) == 0


def test_no_databse_history_give_error(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "empty.db")
    with pytest.raises(DatabaseUnavailableError):
        get_history()


def test_summary_count_all_result(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    PredictionService(FakeModel(model_output(0.99))).classify(make_photo())
    PredictionService(FakeModel(model_output(0.01))).classify(make_photo())
    PredictionService(FakeModel(model_output(0.6))).classify(make_photo())
    PredictionService(FakeModel(model_output(0.99, distance=500))).classify(make_photo())
    PredictionService(FakeModel(model_output(0.99))).classify(b"this is not a image")
    summary = get_summary(get_history())
    assert summary["total"] == 5
    assert summary[PredictionStatus.ACCEPTED.value] == 2
    assert summary[PredictionStatus.NEEDS_REVIEW.value] == 1
    assert summary[PredictionStatus.REJECTED.value] == 1
    assert summary[PredictionStatus.INVALID_INPUT.value] == 1
    assert summary["apples"] == 1
    assert summary["bananas"] == 1
    assert summary["average_confidence"] == pytest.approx((0.99 + 0.99 + 0.6 + 0.99) / 4)


def test_summary_of_empty_history():
    summary = get_summary([])
    assert summary["total"] == 0
    assert summary["average_confidence"] == None


def test_home_button_open_classify_page(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    app = AppTest.from_string(APP_CODE, default_timeout=60).run()
    assert app.title[0].value == "Intelligent Fruit Classification and Information System"
    app.button[0].click().run()
    assert app.title[0].value == "Classify Fruit"
    assert len(app.exception) == 0


def test_history_page_show_summary(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    PredictionService(FakeModel(model_output(0.99))).classify(make_photo())
    app = AppTest.from_string(APP_CODE, default_timeout=60).run()
    app.sidebar.radio[0].set_value("Prediction History").run()
    assert len(app.exception) == 0
    assert app.metric[0].value == "1"
    assert app.metric[1].value == "1"
    assert len(app.dataframe) == 1


def test_history_page_without_databse(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "empty.db")
    app = AppTest.from_string(APP_CODE, default_timeout=60).run()
    app.sidebar.radio[0].set_value("Prediction History").run()
    assert len(app.exception) == 0
    assert "can not be load" in app.error[0].value


def test_group_history_page_get_newest_first_and_tiles(tmp_path, monkeypatch):
    make_test_db(tmp_path, monkeypatch)
    service = PredictionService(FakeModel(model_output(0.99)))
    service.classify(make_photo())
    service.classify(b"this is not a image")
    history = service.get_history()
    assert history[0]["prediction_status"] == PredictionStatus.INVALID_INPUT.value
    assert PredictionService.summarise_history(history) == {"total": 2, "accepted": 1, "needs_review": 0, "rejected_or_invalid": 1}


def test_history_page_work_with_old_databse(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    old_schema = (DB_FOLDER / "schema.sql").read_text(encoding="utf-8").replace("created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,", "")
    connection = sqlite3.connect(path)
    connection.executescript(old_schema)
    connection.executescript((DB_FOLDER / "seed_fruit.sql").read_text(encoding="utf-8"))
    connection.close()
    monkeypatch.setattr(database, "DATABASE_PATH", path)
    PredictionService(FakeModel(model_output(0.99))).classify(make_photo())
    app = AppTest.from_string(APP_CODE, default_timeout=60).run()
    app.sidebar.radio[0].set_value("Prediction History").run()
    assert len(app.exception) == 0
    assert len(app.dataframe) == 1
