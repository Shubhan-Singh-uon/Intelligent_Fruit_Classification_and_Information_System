import sqlite3
import pytest

import src.repositories.database as database
from src.repositories.prediction_repository import (
    save_prediction,
    get_prediction_history,
)


# ============================================
# TEST DATABASE SETUP
# ============================================

@pytest.fixture
def test_database(tmp_path, monkeypatch):
    # Create a temporary database so tests do not modify the real application database.
    test_db_path = tmp_path / "test_fruit_classifier.db"

    # Redirect repository database connections to the temporary database.
    monkeypatch.setattr(database, "DATABASE_PATH", test_db_path)

    connection = sqlite3.connect(test_db_path)
    connection.execute("PRAGMA foreign_keys = ON;")

    # Create the minimum FRUIT table required by the prediction-history foreign key.
    connection.execute("""
        CREATE TABLE FRUIT (
            fruit_id INTEGER PRIMARY KEY,
            fruit_name TEXT NOT NULL,
            model_label TEXT NOT NULL
        )
    """)

    # Add Apple and Banana records used by the prediction tests.
    connection.execute("""
        INSERT INTO FRUIT (fruit_id, fruit_name, model_label)
        VALUES (1, 'Apple', 'apple'), (2, 'Banana', 'banana')
    """)

    # Create a temporary prediction-history table matching the application schema.
    connection.execute("""
        CREATE TABLE PREDICTION_HISTORY (
            prediction_id INTEGER PRIMARY KEY,
            fruit_id INTEGER NULL,
            predicted_class TEXT NULL,
            apple_probability REAL NULL,
            banana_probability REAL NULL,
            confidence REAL NULL,
            prediction_status TEXT NOT NULL,
            rejection_reason TEXT NULL,
            model_version TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (fruit_id) REFERENCES FRUIT(fruit_id)
        )
    """)

    connection.commit()
    connection.close()

    return test_db_path


# ============================================
# PREDICTION STORAGE TESTS
# ============================================

# Verify that an accepted prediction is stored with all expected values.
def test_save_accepted_prediction(test_database):
    save_prediction(
        1,
        "Apple",
        0.97,
        0.03,
        0.97,
        "accepted",
        None,
        "EfficientNetB0_FT_v1"
    )

    history = get_prediction_history()

    assert len(history) == 1

    prediction = history[0]

    assert prediction["fruit_id"] == 1
    assert prediction["predicted_class"] == "Apple"
    assert prediction["apple_probability"] == 0.97
    assert prediction["banana_probability"] == 0.03
    assert prediction["confidence"] == 0.97
    assert prediction["prediction_status"] == "accepted"
    assert prediction["rejection_reason"] is None
    assert prediction["model_version"] == "EfficientNetB0_FT_v1"


# Verify that a rejected prediction can be stored without an assigned fruit.
def test_save_rejected_prediction(test_database):
    save_prediction(
        None,
        None,
        0.08,
        0.12,
        0.12,
        "rejected",
        "far_from_training_data",
        "EfficientNetB0_FT_v1"
    )

    history = get_prediction_history()

    assert len(history) == 1

    prediction = history[0]

    assert prediction["fruit_id"] is None
    assert prediction["predicted_class"] is None
    assert prediction["apple_probability"] == 0.08
    assert prediction["banana_probability"] == 0.12
    assert prediction["confidence"] == 0.12
    assert prediction["prediction_status"] == "rejected"
    assert prediction["rejection_reason"] == "far_from_training_data"
    assert prediction["model_version"] == "EfficientNetB0_FT_v1"


# Verify that invalid input can be recorded even when prediction values are unavailable.
def test_save_invalid_input_prediction(test_database):
    save_prediction(
        None,
        None,
        None,
        None,
        None,
        "invalid_input",
        "unreadable_image",
        "EfficientNetB0_FT_v1"
    )

    history = get_prediction_history()

    assert len(history) == 1

    prediction = history[0]

    assert prediction["fruit_id"] is None
    assert prediction["predicted_class"] is None
    assert prediction["apple_probability"] is None
    assert prediction["banana_probability"] is None
    assert prediction["confidence"] is None
    assert prediction["prediction_status"] == "invalid_input"
    assert prediction["rejection_reason"] == "unreadable_image"
    assert prediction["model_version"] == "EfficientNetB0_FT_v1"


# Verify that a low-confidence prediction can be stored for later review.
def test_save_needs_review_prediction(test_database):
    save_prediction(
        1,
        "Apple",
        0.60,
        0.40,
        0.60,
        "needs_review",
        "low_confidence",
        "EfficientNetB0_FT_v1"
    )

    history = get_prediction_history()

    assert len(history) == 1

    prediction = history[0]

    assert prediction["fruit_id"] == 1
    assert prediction["predicted_class"] == "Apple"
    assert prediction["apple_probability"] == 0.60
    assert prediction["banana_probability"] == 0.40
    assert prediction["confidence"] == 0.60
    assert prediction["prediction_status"] == "needs_review"
    assert prediction["rejection_reason"] == "low_confidence"
    assert prediction["model_version"] == "EfficientNetB0_FT_v1"


# ============================================
# PREDICTION HISTORY TEST
# ============================================

# Verify that prediction history is returned with the newest prediction first.
def test_prediction_history_newest_first(test_database):
    save_prediction(
        1, "Apple", 0.97, 0.03, 0.97,
        "accepted", None, "EfficientNetB0_FT_v1"
    )

    save_prediction(
        2, "Banana", 0.05, 0.95, 0.95,
        "accepted", None, "EfficientNetB0_FT_v1"
    )

    history = get_prediction_history()

    assert len(history) == 2

    assert history[0]["predicted_class"] == "Banana"
    assert history[1]["predicted_class"] == "Apple"

    assert history[0]["prediction_id"] > history[1]["prediction_id"]