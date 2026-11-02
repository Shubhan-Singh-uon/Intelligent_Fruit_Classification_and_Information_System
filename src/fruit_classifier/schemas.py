"""
File: schemas.py
Project: Intelligent Fruit Classification and Information System
Description:
    Defines the shared data structures and field conventions used
    to exchange prediction results and related information between
    preprocessing, ML inference, services, repositories, and the UI.

Author(s): Group 8
Responsible Area: Shared/Core
Maintainers: Group 8

Note:
    Changes to shared data contracts require team agreement.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class PredictionStatus(str, Enum):
    """The four agreed prediction outcomes.

    The string value is what is written to PREDICTION_HISTORY.prediction_status.
    Technical failures (model missing, database down) are NOT statuses; they are
    exceptions (see exceptions.py).
    """

    ACCEPTED = "accepted"            # A
    NEEDS_REVIEW = "needs_review"    # B
    REJECTED = "rejected"            # C
    INVALID_INPUT = "invalid_input"  # D


@dataclass(frozen=True)
class ModelOutput:
    """Raw signals from one inference run, before any accept/review/reject decision.

    Produced by ml/model_service.py, consumed by ml/prediction_policy.py.
    """

    predicted_class: str        # "Apple" or "Banana": the class with the higher probability
    apple_probability: float
    banana_probability: float
    confidence: float           # max(apple_probability, banana_probability)
    feature_distance: float     # Mahalanobis distance to the nearest class centroid
    tta_consistent: bool        # True if the class is unchanged under all test-time views


@dataclass(frozen=True)
class PredictionResult:
    """Final result handed to the service layer, the UI and the repository.

    Fields map one-to-one onto the PREDICTION_HISTORY columns where a column
    exists: fruit_id, predicted_class, apple_probability, banana_probability,
    confidence, prediction_status (<- status.value), rejection_reason
    (<- reason_code) and model_version.

    What each outcome contains:
      ACCEPTED       predicted_class, both probabilities, confidence
      NEEDS_REVIEW   predicted_class (tentative), both probabilities, confidence, reason_code
      REJECTED       probabilities and confidence (for the record); predicted_class is None
                     because the application must not confirm a fruit; reason_code
      INVALID_INPUT  no inference ran, so every prediction field is None; reason_code
    """

    status: PredictionStatus
    model_version: str
    predicted_class: Optional[str] = None
    apple_probability: Optional[float] = None
    banana_probability: Optional[float] = None
    confidence: Optional[float] = None
    reason_code: Optional[str] = None
    # Diagnostics only (no database column). Useful for threshold tuning and an "advanced" UI view.
    feature_distance: Optional[float] = None
    tta_consistent: Optional[bool] = None
    # Filled in by the service layer after the FRUIT lookup (use dataclasses.replace).
    fruit_id: Optional[int] = None

    @property
    def model_label(self) -> Optional[str]:
        """Lower-case label that matches FRUIT.model_label ('apple' / 'banana')."""
        return self.predicted_class.lower() if self.predicted_class else None

    @property
    def should_show_fruit_info(self) -> bool:
        """Fruit information is shown for accepted predictions only."""
        return self.status is PredictionStatus.ACCEPTED


@dataclass(frozen=True)
class FruitInformation:
    """Everything the Classify page shows for an accepted prediction.

    All values are plain dicts (not sqlite3.Row), so the UI does not depend on the database.
    Keys are the column names of the repository queries, e.g. nutrition["energy_kcal"],
    health_benefits[i]["benefit_description"], fun_facts[i]["fact_text"], uses[i]["use_description"].
    """

    fruit: Dict[str, Any]                          # FRUIT row: fruit_id, fruit_name, model_label, description
    nutrition: Optional[Dict[str, Any]]            # None if the fruit has no nutrition record
    health_benefits: Tuple[Dict[str, Any], ...]
    uses: Tuple[Dict[str, Any], ...]
    fun_facts: Tuple[Dict[str, Any], ...]
    sources: Tuple[Dict[str, Any], ...]            # de-duplicated: source_name, source_url, access_date


@dataclass(frozen=True)
class ClassificationResponse:
    """What PredictionService.classify() hands to the UI.

    `prediction` is the outcome. 
    `issues` lists technical problems that did NOT stop the prediction from being shown, as codes 
    defined in services/prediction_service.py (for example "history_not_saved"); the UI turns 
    them into messages. Technical failures that do stop classification are raised as exceptions instead 
    (see exceptions.py).
    """

    prediction: PredictionResult
    fruit_info: Optional[FruitInformation] = None   # accepted predictions only
    history_saved: bool = False                     # false with no issue code 
    issues: Tuple[str, ...] = ()
