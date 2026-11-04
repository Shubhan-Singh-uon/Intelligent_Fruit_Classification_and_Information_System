"""Runs PredictionService against the real repository code and the project's real SQL files.

Each test gets a throwaway SQLite file built from src/database/schema.sql and seed_fruit.sql,
so nothing touches your real fruit_classifier.db. The model is still a fake (no torch needed).
"""
import io
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from src.fruit_classifier.exceptions import DatabaseUnavailableError
from src.fruit_classifier.schemas import ModelOutput, PredictionStatus
from src.services.prediction_service import (
    ISSUE_FRUIT_INFO_UNAVAILABLE,
    ISSUE_HISTORY_NOT_SAVED,
    PredictionService,
)

database = pytest.importorskip("src.repositories.database")
SRC_DIR = Path(database.__file__).resolve().parent.parent


class FakeModel:
    model_version = "M4_PC1-test"

    def __init__(self, output):
        self.output = output

    def predict(self, image):
        return self.output


def output(apple=0.99, distance=20.0, tta=True):
    banana = 1.0 - apple
    return ModelOutput("Apple" if apple >= banana else "Banana", apple, banana, max(apple, banana), distance, tta)


def upload():
    buffer = io.BytesIO()
    Image.new("RGB", (200, 150), (200, 30, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    schema, seed = SRC_DIR / "database" / "schema.sql", SRC_DIR / "database" / "seed_fruit.sql"
    if not (schema.is_file() and seed.is_file()):
        pytest.skip("schema.sql / seed_fruit.sql not found")
    path = tmp_path / "test.db"
    connection = sqlite3.connect(path)
    connection.executescript(schema.read_text(encoding="utf-8"))
    connection.executescript(seed.read_text(encoding="utf-8"))
    connection.commit()
    connection.close()
    monkeypatch.setattr(database, "DATABASE_PATH", path)   # get_connection() reads this at call time
    return path


def service_for(model_output):
    return PredictionService(FakeModel(model_output))      # real repositories (the defaults)


def rows(path, sql="SELECT * FROM PREDICTION_HISTORY ORDER BY prediction_id"):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in connection.execute(sql)]
    finally:
        connection.close()


def test_accepted_apple_loads_real_fruit_information_and_links_the_history_row(db_path):
    response = service_for(output(apple=0.99)).classify(upload())
    info = response.fruit_info

    assert response.prediction.status is PredictionStatus.ACCEPTED and response.issues == ()
    assert info.fruit["fruit_name"] == "Apple" and info.nutrition["serving_basis"] == "100g"
    assert info.nutrition["energy_kcal"] == pytest.approx(57.8)
    assert info.health_benefits and info.uses and info.fun_facts and info.sources
    assert all({"source_name", "source_url", "access_date"} <= set(s) for s in info.sources)
    assert len({s["source_url"] for s in info.sources}) == len(info.sources)    # de-duplicated

    (row,) = rows(db_path)
    assert row["fruit_id"] == 1 and row["predicted_class"] == "Apple" and row["prediction_status"] == "accepted"
    assert row["model_version"] == "M4_PC1-test" and row["rejection_reason"] is None


def test_accepted_banana(db_path):
    response = service_for(output(apple=0.01)).classify(upload())
    assert response.fruit_info.fruit["fruit_name"] == "Banana"
    assert response.fruit_info.nutrition["potassium_mg"] == pytest.approx(348)
    assert rows(db_path)[0]["fruit_id"] == 2


def test_every_outcome_is_stored_and_history_reads_back_newest_first(db_path):
    attempts = [
        (output(apple=0.99), upload()),                  # accepted
        (output(apple=0.45), upload()),                  # needs review
        (output(apple=0.99, distance=120.0), upload()),  # rejected
        (output(), b"not an image" * 20),                # invalid input
    ]
    for model_output, data in attempts:
        service_for(model_output).classify(data)

    history = service_for(output()).get_history()
    assert [r["prediction_status"] for r in history] == ["invalid_input", "rejected", "needs_review", "accepted"]
    assert PredictionService.summarise_history(history) == {
        "total": 4, "accepted": 1, "needs_review": 1, "rejected_or_invalid": 2,
    }
    invalid = history[0]
    assert invalid["rejection_reason"] == "unreadable_file" and invalid["confidence"] is None
    assert history[1]["predicted_class"] is None and history[1]["rejection_reason"] == "far_from_training_data"


def test_database_that_was_never_initialised_does_not_hide_the_prediction(tmp_path, monkeypatch):
    empty = tmp_path / "empty.db"          # sqlite3.connect() silently creates an empty file
    monkeypatch.setattr(database, "DATABASE_PATH", empty)

    service = service_for(output())
    response = service.classify(upload())
    assert response.prediction.status is PredictionStatus.ACCEPTED
    assert response.fruit_info is None
    assert response.issues == (ISSUE_FRUIT_INFO_UNAVAILABLE, ISSUE_HISTORY_NOT_SAVED)

    with pytest.raises(DatabaseUnavailableError):
        service.get_history()


def test_history_rows_carry_a_utc_timestamp_filled_in_by_the_database(db_path):
    service_for(output()).classify(upload())
    (record,) = service_for(output()).get_history()
    assert datetime.strptime(record["created_at"], "%Y-%m-%d %H:%M:%S")   # needs the created_at column in schema.sql
