"""
File: prediction_service.py
Project: Intelligent Fruit Classification and Information System
Description:
    Coordinates the inference workflow by connecting image
    preprocessing, model inference, prediction policy, fruit
    information retrieval, and prediction-history persistence.

Author(s): Group 8
Integration Lead: Hasnain Younas
ML Contributor: Shubhan Singh
Database Contributor: Abeeha Mariam
Responsible Area: Shared/Core Integration
Maintainers: Hasnain Younas, Shubhan Singh, Abeeha Mariam
"""

import logging
import sqlite3
from dataclasses import replace
from typing import Any, Dict, List, Optional

from src.fruit_classifier.config import SAVE_TO_HISTORY_STATUSES
from src.fruit_classifier.exceptions import (
    DatabaseUnavailableError,
    InferenceError,
    InvalidImageError,
)
from src.fruit_classifier.schemas import (
    ClassificationResponse,
    FruitInformation,
    PredictionResult,
    PredictionStatus,
)
from src.ml.prediction_policy import apply_policy, invalid_input_result
from src.preprocessing.image_preprocessor import preprocess_image
from src.repositories import fruit_repository, prediction_repository

logger = logging.getLogger(__name__)

# Codes used in ClassificationResponse.issues. The UI maps them to messages.
ISSUE_HISTORY_NOT_SAVED = "history_not_saved"            # prediction is valid, but it could not be stored
ISSUE_FRUIT_INFO_UNAVAILABLE = "fruit_info_unavailable"  # accepted, but the fruit details could not be loaded


class PredictionService:
    """Runs the full classification workflow for one upload at a time.

    Holds no per-request state, so one instance can serve every Streamlit session.

    model_service       object with `.model_version` and `.predict(pil_image) -> ModelOutput`
                        (src.ml.model_service.ModelService).
    fruit_repo,         the repository modules (or test doubles with the same functions).
    prediction_repo
    save_statuses       which outcomes are stored in PREDICTION_HISTORY (config.py decides by default).
    """

    def __init__(
        self,
        model_service,
        fruit_repo=fruit_repository,
        prediction_repo=prediction_repository,
        save_statuses=SAVE_TO_HISTORY_STATUSES,
    ):
        self._model = model_service
        self._fruit_repo = fruit_repo
        self._prediction_repo = prediction_repo
        self._save_statuses = frozenset(save_statuses)

    # ------------------------------------------------------------------------------ 
    # Classify
    # ------------------------------------------------------------------------------
    def classify(self, upload) -> ClassificationResponse:
        """Classify one uploaded file (bytes, or a file-like object such as a Streamlit UploadedFile).

        Raises InferenceError if the model fails on a readable image. Everything else, including
        unreadable files, unsupported images and database problems, comes back in the response.
        """
        issues: List[str] = []

        result = self._predict(upload)

        fruit_info: Optional[FruitInformation] = None
        if result.should_show_fruit_info:
            fruit_info = self._load_fruit_information(result.model_label, issues)
            if fruit_info is not None:
                # fruit_id links the history row to the confirmed fruit. Needs-review and rejected
                # attempts are not linked, because no fruit was confirmed.
                result = replace(result, fruit_id=fruit_info.fruit["fruit_id"])

        history_saved = self._save_history(result, issues)
        return ClassificationResponse(
            prediction=result,
            fruit_info=fruit_info,
            history_saved=history_saved,
            issues=tuple(issues),
        )

    def _predict(self, upload) -> PredictionResult:
        model_version = self._model.model_version

        try:
            image = preprocess_image(upload)
        except InvalidImageError as error:
            # Outcome D: no inference runs - no probabilities.
            return invalid_input_result(error.reason_code, model_version)

        try:
            output = self._model.predict(image)
        except Exception as error:   # any model failure is technical
            logger.exception("Model inference failed")
            raise InferenceError("The model failed while classifying the image.") from error

        return apply_policy(output, model_version)

    # ------------------------------------------------------------------------------ 
    # Fruit Information
    # ------------------------------------------------------------------------------ 
    def _load_fruit_information(self, model_label: str, issues: List[str]) -> Optional[FruitInformation]:
        """Fetch everything the Classify page shows for an accepted fruit.

        Returns None (and records an issue) if the database cannot supply it; the accepted
        prediction itself is still shown.
        """
        try:
            fruit = self._fruit_repo.get_fruit_by_model_label(model_label)
            if fruit is None:
                logger.error("No FRUIT row for model label %r; check seed_fruit.sql", model_label)
                issues.append(ISSUE_FRUIT_INFO_UNAVAILABLE)
                return None

            fruit_id = fruit["fruit_id"]
            nutrition = self._fruit_repo.get_nutrition_by_fruit_id(fruit_id)
            health_benefits = self._fruit_repo.get_health_benefits_by_fruit_id(fruit_id)
            uses = self._fruit_repo.get_uses_by_fruit_id(fruit_id)
            fun_facts = self._fruit_repo.get_fun_facts_by_fruit_id(fruit_id)
        except sqlite3.Error:
            logger.exception("Could not load fruit information for %r", model_label)
            issues.append(ISSUE_FRUIT_INFO_UNAVAILABLE)
            return None

        nutrition = dict(nutrition) if nutrition is not None else None
        health_benefits = tuple(dict(row) for row in health_benefits)
        fun_facts = tuple(dict(row) for row in fun_facts)

        return FruitInformation(
            fruit=dict(fruit),
            nutrition=nutrition,
            health_benefits=health_benefits,
            uses=tuple(dict(row) for row in uses),
            fun_facts=fun_facts,
            sources=_collect_sources(nutrition, health_benefits, fun_facts),
        )

    # ------------------------------------------------------------------------------ 
    # Prediction History
    # ------------------------------------------------------------------------------ 
    def _save_history(self, result: PredictionResult, issues: List[str]) -> bool:
        """Store the attempt. Returns True only if a row was written.

        A failure never hides the prediction: it is logged and reported as an issue.
        """
        if result.status.value not in self._save_statuses:
            return False   # not attempted, by policy: no issue is raised

        try:
            self._prediction_repo.save_prediction(
                fruit_id=result.fruit_id,
                predicted_class=result.predicted_class,
                # Round to 4 decimal places to prevent float precision errors 
                apple_probability=round(result.apple_probability, 4) if result.apple_probability is not None else None,
                banana_probability=round(result.banana_probability, 4) if result.banana_probability is not None else None,
                confidence=round(result.confidence, 4) if result.confidence is not None else None,
                prediction_status=result.status.value,
                rejection_reason=result.reason_code,
                model_version=result.model_version,
            )
        except sqlite3.Error:
            logger.exception("Could not save prediction history")
            issues.append(ISSUE_HISTORY_NOT_SAVED)
            return False
        return True

    def get_history(self) -> List[Dict[str, Any]]:
        """All stored attempts as plain dicts, newest first (for the Prediction History page).

        Raises DatabaseUnavailableError if the database cannot be read.
        """
        try:
            rows = self._prediction_repo.get_prediction_history()
        except sqlite3.Error as error:
            logger.exception("Could not read prediction history")
            raise DatabaseUnavailableError("Prediction history could not be read.") from error

        records = [dict(row) for row in rows]
        # The table has no timestamp column yet, so the auto-incrementing id is the only ordering.
        records.sort(key=lambda record: record["prediction_id"], reverse=True)
        return records

    @staticmethod
    def summarise_history(records: List[Dict[str, Any]]) -> Dict[str, int]:
        """The four tiles at the top of the History page, from the records get_history() returned."""
        counts = {status.value: 0 for status in PredictionStatus}
        for record in records:
            if record["prediction_status"] in counts:
                counts[record["prediction_status"]] += 1
        return {
            "total": len(records),
            "accepted": counts[PredictionStatus.ACCEPTED.value],
            "needs_review": counts[PredictionStatus.NEEDS_REVIEW.value],
            "rejected_or_invalid": counts[PredictionStatus.REJECTED.value]
            + counts[PredictionStatus.INVALID_INPUT.value],
        }


def _collect_sources(nutrition, health_benefits, fun_facts):
    """One entry per distinct source URL, in the order first used, for the Sources section."""
    sources, seen = [], set()
    rows = ([nutrition] if nutrition else []) + list(health_benefits) + list(fun_facts)
    for row in rows:
        url = row.get("source_url")
        if url and url not in seen:
            seen.add(url)
            sources.append(
                {
                    "source_name": row["source_name"],
                    "source_url": url,
                    "access_date": row["access_date"],
                }
            )
    return tuple(sources)


#def build_prediction_service(**kwargs) -> PredictionService:
    """Load the trained model and return a ready service.

    Raises ModelUnavailableError if the model files are missing or inconsistent, so the app
    can show a clear technical error on start-up instead of failing on the first upload.
    The import is local so that the rest of this module (and its tests) work without torch.
    """
#    from src.ml.model_service import load_model_service

#    return PredictionService(load_model_service(), **kwargs)
