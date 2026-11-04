"""
File: prediction_service.py
Project: Intelligent Fruit Classification and Information System
Description:
    this file connect all the parts togather. it check the image
    run the model and the policy then get the fruit information
    from databse and save the result in the history table

Author: Hasnain Younas
Responsible Pillar: Pillar 4 - Application & System Integration
"""
import sqlite3
from dataclasses import replace

from src.fruit_classifier.config import SAVE_TO_HISTORY_STATUSES
from src.fruit_classifier.exceptions import InvalidImageError, InferenceError, DatabaseUnavailableError
from src.fruit_classifier.schemas import ClassificationResponse, FruitInformation, PredictionStatus
from src.preprocessing.image_preprocessor import preprocess_image
from src.ml.model_service import load_model_service
from src.ml.prediction_policy import apply_policy, invalid_input_result
from src.repositories import fruit_repository, prediction_repository

# issue codes that the ui turn into messages
HISTORY_NOT_SAVED = "history_not_saved"
FRUIT_INFO_MISSING = "fruit_info_missing"


def to_dicts(rows):
    # sqlite rows are change to normal dict so the ui dont need the databse
    result = []
    for row in rows:
        result.append(dict(row))
    return result


class PredictionService:

    def __init__(self, model_service=None):
        if model_service == None:
            model_service = load_model_service()
        self.model_service = model_service

    def classify(self, uploaded_file):
        version = self.model_service.model_version
        issues = []

        try:
            image = preprocess_image(uploaded_file)
        except InvalidImageError as error:
            result = invalid_input_result(error.reason_code, version)
            saved = self.save_history(result, issues)
            return ClassificationResponse(prediction=result, history_saved=saved, issues=tuple(issues))

        try:
            output = self.model_service.predict(image)
        except Exception as error:
            raise InferenceError("the model fail when it classify the image") from error

        result = apply_policy(output, version)

        # fruit information is only show when the result is accepted
        fruit_info = None
        if result.status == PredictionStatus.ACCEPTED:
            try:
                fruit_info = self.get_fruit_info(result.model_label)
            except sqlite3.Error:
                fruit_info = None
            if fruit_info == None:
                issues.append(FRUIT_INFO_MISSING)
            else:
                # result is frozen so we make a copy with the fruit id for the history
                result = replace(result, fruit_id=fruit_info.fruit["fruit_id"])

        saved = self.save_history(result, issues)
        return ClassificationResponse(prediction=result, fruit_info=fruit_info, history_saved=saved, issues=tuple(issues))

    def get_fruit_info(self, model_label):
        fruit = fruit_repository.get_fruit_by_model_label(model_label)
        if fruit == None:
            return None

        fruit_id = fruit["fruit_id"]
        nutrition = fruit_repository.get_nutrition_by_fruit_id(fruit_id)
        benefits = to_dicts(fruit_repository.get_health_benefits_by_fruit_id(fruit_id))
        uses = to_dicts(fruit_repository.get_uses_by_fruit_id(fruit_id))
        facts = to_dicts(fruit_repository.get_fun_facts_by_fruit_id(fruit_id))

        if nutrition != None:
            nutrition = dict(nutrition)

        # make one list of sources and dont add the same link two times
        rows = benefits + facts
        if nutrition != None:
            rows = [nutrition] + rows
        sources = []
        links = []
        for row in rows:
            if row["source_url"] not in links:
                links.append(row["source_url"])
                sources.append({
                    "source_name": row["source_name"],
                    "source_url": row["source_url"],
                    "access_date": row["access_date"],
                })

        return FruitInformation(
            fruit=dict(fruit),
            nutrition=nutrition,
            health_benefits=tuple(benefits),
            uses=tuple(uses),
            fun_facts=tuple(facts),
            sources=tuple(sources),
        )

    def save_history(self, result, issues):
        # config have names like Needs_Review but the status value is Needs Review
        status_name = result.status.value.replace(" ", "_")
        if status_name not in SAVE_TO_HISTORY_STATUSES:
            return False
        try:
            prediction_repository.save_prediction(
                result.fruit_id,
                result.predicted_class,
                result.apple_probability,
                result.banana_probability,
                result.confidence,
                result.status.value,
                result.reason_code,
                result.model_version,
            )
            return True
        except sqlite3.Error:
            issues.append(HISTORY_NOT_SAVED)
            return False


def get_history():
    # history page call this one. it dont need the model
    try:
        rows = prediction_repository.get_prediction_history()
    except sqlite3.Error as error:
        raise DatabaseUnavailableError("could not read the prediction history") from error
    return to_dicts(rows)


def get_summary(history):
    summary = {
        "total": len(history),
        PredictionStatus.ACCEPTED.value: 0,
        PredictionStatus.NEEDS_REVIEW.value: 0,
        PredictionStatus.REJECTED.value: 0,
        PredictionStatus.INVALID_INPUT.value: 0,
        "apples": 0,
        "bananas": 0,
        "average_confidence": None,
    }
    scores = []
    for item in history:
        status = item["prediction_status"]
        if status in summary:
            summary[status] = summary[status] + 1
        if status == PredictionStatus.ACCEPTED.value:
            if item["predicted_class"] == "Apple":
                summary["apples"] = summary["apples"] + 1
            elif item["predicted_class"] == "Banana":
                summary["bananas"] = summary["bananas"] + 1
        if item["confidence"] != None:
            scores.append(item["confidence"])
    if len(scores) > 0:
        summary["average_confidence"] = sum(scores) / len(scores)
    return summary
