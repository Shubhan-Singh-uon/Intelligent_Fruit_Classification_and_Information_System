"""
File: config.py
Project: Intelligent Fruit Classification and Information System
Description:
    Defines shared configuration values, including application paths,
    model settings, database location, and runtime parameters.

Author(s): Group 8
Responsible Area: Shared/Core
Maintainers: Group 8

Note:
    Changes to shared configuration must be agreed upon and reviewed
    by the team before implementation.
"""
from pathlib import Path

# --------------------------------------------------------------------------- 
# Paths
# --------------------------------------------------------------------------- 
# This file lives at <root>/src/fruit_classifier/config.py, so the root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ASSETS_DIR = PROJECT_ROOT / "assets"   # pictures used by the UI, e.g. the Home page header

# Files in artifacts/ (see artifacts/README.md):
#   M4_PC1.pt                  trained weights (state_dict) from results/checkpoints/
#   final_model_selection.json deployment record from the collation notebook
#   m4_ood_stats.npz           class centroids + inverse covariance, made by export_ood_stats.py
MODEL_CHECKPOINT_PATH = ARTIFACTS_DIR / "M4_PC1.pt"
MODEL_SELECTION_PATH = ARTIFACTS_DIR / "final_model_selection.json"
OOD_STATS_PATH = ARTIFACTS_DIR / "m4_ood_stats.npz"

# --------------------------------------------------------------------------- 
# Model and preprocessing contract
# --------------------------------------------------------------------------- 
# Output index -> class name. The ML notebook uses ImageFolder order: 0 = Apple, 1 = Banana.
# FRUIT.model_label in the database is the lower-case form ('apple', 'banana').
CLASS_NAMES = ("Apple", "Banana")

IMAGE_SIZE = (128, 128)
PREPROCESSING_VERSION = "PP_v1: EXIF-upright, RGB, bicubic resize 128x128, pixels 0-1"

# Upload validation 
ALLOWED_IMAGE_FORMATS = ("JPEG", "MPO", "PNG", "WEBP", "BMP", "GIF", "TIFF", "HEIF")
UPLOAD_EXTENSIONS = ("jpg", "jpeg", "jpe", "jfif", "png", "webp", "bmp", "gif", "tif", "tiff", "heic", "heif")
SUPPORTED_FORMATS_LABEL = "JPG, PNG, WEBP, BMP, GIF, TIFF, HEIC"   # shown to the user
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


# --------------------------------------------------------------------------- 
# Prediction policy thresholds
# --------------------------------------------------------------------------- 
# Confidence gate, taken from the ML notebook (CONFIDENCE_THRESHOLD).
CONFIDENCE_THRESHOLD = 0.90

# Feature-distance gate: an image farther than this from BOTH class centroids is
# treated as unsupported (outcome C).
#
# The notebook's configured value (95th percentile of TRAINING
# distances, 35.73 for M4_PC1) rejects about two thirds of genuine test images,
# so it cannot be used as-is. 
DISTANCE_REJECT_THRESHOLD = 61.69

# --------------------------------------------------------------------------- 
# Prediction history
# --------------------------------------------------------------------------- 
# Which outcomes are written to PREDICTION_HISTORY. 
SAVE_TO_HISTORY_STATUSES = ("accepted", "needs_review", "rejected", "invalid_input")