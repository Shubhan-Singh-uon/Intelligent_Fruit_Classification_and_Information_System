"""
File: messages.py
Project: Intelligent Fruit Classification and Information System
Description:
    All UI wording for prediction outcomes, reason codes and technical
    issues, in one place so the Classify and History pages stay consistent.
    Unsupported images (outcome C) and unreadable files (outcome D) deliberately
    have different messages.

Author: Hasnain Younas
Responsible Pillar: Pillar 4 - Application & System Integration
"""
from src.fruit_classifier.config import MAX_UPLOAD_BYTES
from src.ml.prediction_policy import (
    REASON_FAR_FROM_TRAINING_DATA,
    REASON_INCONSISTENT_VIEWS,
    REASON_LOW_CONFIDENCE,
)
from src.services.prediction_service import ISSUE_FRUIT_INFO_UNAVAILABLE, ISSUE_HISTORY_NOT_SAVED

MAX_MB = MAX_UPLOAD_BYTES // (1024 * 1024)

# -------------------------------------------------------------------------------------
# Outcome B: Needs Review
# -------------------------------------------------------------------------------------
NEEDS_REVIEW_BANNER = "Prediction is uncertain. Please try a clearer image."
NEEDS_REVIEW_DETAIL = {
    REASON_LOW_CONFIDENCE: (
        "The model is not confident enough to make a final decision. "
        "Try uploading a clearer image or a different angle."
    ),
    REASON_INCONSISTENT_VIEWS: (
        "The prediction changed when the image was slightly flipped, rotated or brightened, "
        "so it is not reliable. Try uploading a clearer image or a different angle."
    ),
}

# -------------------------------------------------------------------------------------
# Outcome C: Rejected
# -------------------------------------------------------------------------------------
REJECTED_BANNER = "Image Rejected: It does not sufficiently resemble the supported fruit images."
REJECTED_DETAIL = (
    "This image does not look like the apple and banana images, this system was built to recognise "
    "(it may be a different fruit, an object, or unclear). No fruit has been identified."
)

# -------------------------------------------------------------------------------------
# Outcome D: Invalid Input
# -------------------------------------------------------------------------------------
INVALID_BANNER = "Image could not be Processed."
INVALID_DETAIL = {
    "empty_file": "The file is empty. Please upload a valid image.",
    "file_too_large": f"The file is larger than {MAX_MB} MB. Please upload a smaller image.",
    "unreadable_file": "The file is corrupted or unreadable, or it is not an image. Please upload a valid JPG or PNG.",
    "unsupported_format": "This image format is not supported. Please upload a JPG or PNG.",
}
INVALID_FALLBACK = "The file could not be processed. Please upload a valid JPG or PNG."

# -------------------------------------------------------------------------------------
# Technical Issues (NOT Outcomes)
# -------------------------------------------------------------------------------------
ISSUE_MESSAGES = {
    ISSUE_HISTORY_NOT_SAVED: (
        "This result could not be saved to your prediction history. The prediction itself is still valid."
    ),
    ISSUE_FRUIT_INFO_UNAVAILABLE: "Fruit information could not be loaded from the database right now.",
}
INFERENCE_ERROR = (
    "Something went wrong inside the classifier. This is a system problem, not a problem with your image. "
    "Please try again."
)

# -------------------------------------------------------------------------------------
# Short wording for the History Table
# -------------------------------------------------------------------------------------
STATUS_LABEL = {
    "accepted": ("Accepted", "ok"),
    "needs_review": ("Needs Review", "warn"),
    "rejected": ("Rejected", "bad"),
    "invalid_input": ("Invalid", "bad"),
}
REASON_SHORT = {
    REASON_LOW_CONFIDENCE: "Low Confidence",
    REASON_INCONSISTENT_VIEWS: "Unstable Prediction",
    REASON_FAR_FROM_TRAINING_DATA: "Unlike Supported Images",
    "empty_file": "Empty file",
    "file_too_large": "File too large",
    "unreadable_file": "Unreadable file",
    "unsupported_format": "Unsupported format",
}
