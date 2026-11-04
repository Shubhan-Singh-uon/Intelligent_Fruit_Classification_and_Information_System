"""Shared doubles for the UI tests. Streamlit's AppTest runs page code in the same process, so tests
set module-level state here and the page scripts read it."""
import io

from PIL import Image

from src.fruit_classifier.exceptions import DatabaseUnavailableError
from src.fruit_classifier.schemas import (
    ClassificationResponse,
    FruitInformation,
    PredictionResult,
    PredictionStatus,
)
from src.services.prediction_service import PredictionService


def jpeg_bytes(color=(200, 40, 40)):
    buffer = io.BytesIO()
    Image.new("RGB", (240, 180), color).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeUpload:
    """Looks like Streamlit's UploadedFile as far as the pages are concerned."""
    def __init__(self, data=None, name="fruit.jpg", file_id="file-1"):
        self._data = jpeg_bytes() if data is None else data
        self.name, self.file_id, self.size = name, file_id, len(self._data)

    def getvalue(self):
        return self._data

SRC = {"source_name": "Source <b>One</b>", "source_url": "https://example.org/one", "access_date": "2026-10-16"}

def fruit_info(**overrides):
    base = dict(
        fruit={"fruit_id": 1, "fruit_name": "Apple", "model_label": "apple", "description": "An apple is a fruit."},
        nutrition={"serving_basis": "100g", "energy_kcal": 57.8, "protein_g": 0.2, "carbohydrate_g": 12.7,
                   "sugars_g": 12.4, "fibre_g": 2.5, "fat_g": 0, "potassium_mg": 115, "vitamin_c_mg": 7, **SRC},
        health_benefits=({"benefit_description": "Supports digestion.", **SRC},),
        uses=({"use_description": "Eat Fresh"}, {"use_description": "Baking"}),
        fun_facts=({"fact_text": "Rose family.", **SRC},),
        sources=({"source_name": SRC["source_name"], "source_url": SRC["source_url"], "access_date": "2026-10-16"},),
    )
    base.update(overrides)
    return FruitInformation(**base)

def response(status, **kw):
    fields = dict(model_version="M4_PC1-test")
    if status is not PredictionStatus.INVALID_INPUT:
        fields.update(apple_probability=0.97, banana_probability=0.03, confidence=0.97,
                      feature_distance=20.0, tta_consistent=True)
    fields.update(kw.pop("prediction", {}))
    info = kw.pop("fruit_info", None)
    return ClassificationResponse(
        prediction=PredictionResult(status=status, **fields), fruit_info=info,
        history_saved=kw.pop("history_saved", True), issues=tuple(kw.pop("issues", ())),
    )

ACCEPTED = lambda **kw: response(PredictionStatus.ACCEPTED, prediction=dict(predicted_class="Apple", fruit_id=1),
                                 fruit_info=kw.pop("fruit_info", fruit_info()), **kw)
NEEDS_REVIEW = lambda **kw: response(PredictionStatus.NEEDS_REVIEW, prediction=dict(
    predicted_class="Banana", apple_probability=0.48, banana_probability=0.52, confidence=0.52,
    reason_code=kw.pop("reason", "low_confidence")), **kw)
REJECTED = lambda **kw: response(PredictionStatus.REJECTED, prediction=dict(
    reason_code="far_from_training_data", feature_distance=88.0, apple_probability=0.42, banana_probability=0.58,
    confidence=0.58), **kw)
INVALID = lambda reason="unreadable_file", **kw: response(PredictionStatus.INVALID_INPUT,
                                                         prediction=dict(reason_code=reason), **kw)

class FakeClassifyService:
    """Returns a preset response (or raises a preset error) and counts calls."""

    def __init__(self):
        self.response, self.error, self.calls = None, None, 0

    def classify(self, upload):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response

class FakeHistoryService:
    def __init__(self, records=(), error=None):
        self.records, self.error = list(records), error

    def get_history(self):
        if self.error:
            raise self.error
        return self.records

    summarise_history = staticmethod(PredictionService.summarise_history)

# Module-level state the page scripts read
service = FakeClassifyService()
current_upload = None

def fake_file_uploader(*args, **kwargs):
    return current_upload
