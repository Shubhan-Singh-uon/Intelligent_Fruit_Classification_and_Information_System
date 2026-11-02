"""
File: prediction_repository.py
Project: Intelligent Fruit Classification and Information System
Description:
    Persists inference-attempt records in PREDICTION_HISTORY
    and provides database operations for retrieving prediction
    history where required by the application.

Author: Abeeha Mariam
Responsible Pillar: Pillar 3 - Database & Information System
Maintainers: Abeeha Mariam
"""
from src.repositories.database import get_connection



def save_prediction(fruit_id,
                    predicted_class,
                    apple_probability,
                    banana_probability,
                    confidence,
                    prediction_status,
                    rejection_reason,
                    model_version):
# Save each prediction attempt to PREDICTION_HISTORY.
# This includes accepted, rejected, and invalid-input predictions.
# Nullable fields allow rejected or invalid predictions to be recorded
# even when no fruit class or probability values are available.
# created_at is not listed here: SQLite fills it in automatically.
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute(
    """
    INSERT INTO PREDICTION_HISTORY (
        fruit_id,
        predicted_class,
        apple_probability,
        banana_probability,
        confidence,
        prediction_status,
        rejection_reason,
        model_version
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (fruit_id,
    predicted_class,
    apple_probability,
    banana_probability,
    confidence,
    prediction_status,
    rejection_reason,
    model_version))
    connection.commit()
    connection.close()

def get_prediction_history():
# Retrieve all stored prediction attempts from PREDICTION_HISTORY, newest first.
# This allows the application to access previously recorded prediction results
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("""SELECT * FROM PREDICTION_HISTORY ORDER BY prediction_id DESC""")
    prediction_history=cursor.fetchall()
    connection.close()
    return prediction_history  