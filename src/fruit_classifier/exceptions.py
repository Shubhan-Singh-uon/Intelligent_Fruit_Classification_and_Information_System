"""
File: exceptions.py
Project: Intelligent Fruit Classification and Information System
Description:
    Defines application-specific exceptions for controlled handling
    of input validation, model inference, and database failures.

Author: Project Team
Responsible Area: Shared/Core
Maintainers: Project Team

Note:
    Exception definitions must remain consistent across all pillars.
"""

class FruitClassifierError(Exception):
    """Base class for every application-specific error."""


class InvalidImageError(FruitClassifierError):
    """The uploaded file cannot be read or processed (prediction outcome D).

    `reason_code` is a short machine-readable string that is stored in
    PREDICTION_HISTORY.rejection_reason and lets the UI show a specific message.
    Codes: empty_file, file_too_large, unreadable_file, unsupported_format.
    """

    def __init__(self, reason_code: str, message: str = ""):
        super().__init__(message or reason_code)
        self.reason_code = reason_code


class ModelUnavailableError(FruitClassifierError):
    """The trained model or its supporting artefacts cannot be loaded.

    This is a technical failure, NOT a prediction outcome.
    """


class InferenceError(FruitClassifierError):
    """The model failed unexpectedly while classifying a readable image.

    This is a technical failure, NOT a prediction outcome.
    """


class DatabaseUnavailableError(FruitClassifierError):
    """The SQLite database could not be read (for example it was never initialised).

    Raised only where there is nothing sensible to show without the database
    (the prediction_history page). Classification degrades gracefully instead of raising.
    """
