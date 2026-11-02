"""
File: model_service.py
Project: Intelligent Fruit Classification and Information System
Description:
    Loads the selected trained model, performs inference on
    preprocessed images, and returns the candidate class and
    associated class probabilities.

Author: Shubhan Singh
Responsible Pillar: Pillar 2 - Machine Learning & Evaluation
Maintainers: Shubhan Singh

Note:
    The model is PyTorch (EfficientNetB0, fine-tuned; run M4_PC1). The checkpoint is a bare
    state_dict, so the architecture below must stay identical to the ML notebook's
    EfficientNetWrapper. Besides the two class probabilities, inference returns the two other
    reliability signals from the notebook's Section 18: Mahalanobis feature distance and
    test-time-augmentation (TTA) consistency. The decision (accept / needs review / reject) 
    is made in prediction_policy.py.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import nn
from torchvision import models
from torchvision.transforms import InterpolationMode

from src.fruit_classifier.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    MODEL_CHECKPOINT_PATH,
    MODEL_SELECTION_PATH,
    OOD_STATS_PATH,
    PREPROCESSING_VERSION,
)
from src.fruit_classifier.exceptions import ModelUnavailableError
from src.fruit_classifier.schemas import ModelOutput

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Test-time views, identical to TTA_TRANSFORMS in the notebook (Section 17), minus "original".
_TTA_VIEWS = (
    lambda img: TF.hflip(img),
    lambda img: TF.rotate(img, 10, interpolation=InterpolationMode.BILINEAR),
    lambda img: TF.rotate(img, -10, interpolation=InterpolationMode.BILINEAR),
    lambda img: TF.adjust_brightness(img, 1.25),
)


class EfficientNetWrapper(nn.Module):
    """EfficientNetB0 with a new 2-class head (copy of the ML notebook's class).

    Input is a raw 0-1 tensor; ImageNet mean/std standardisation happens inside the model.
    Attribute names (features, pool, classifier, mean, std) must not change: they are the
    keys of the saved state_dict.
    """

    def __init__(self, dropout=0.30, num_classes=2):
        super().__init__()
        # weights=None: the real weights come from our checkpoint, so nothing is downloaded.
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.pool = backbone.avgpool
        in_features = backbone.classifier[1].in_features
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def extract_features(self, x):
        """Penultimate (pre-classifier) feature vector, shape (N, 1280)."""
        x = (x - self.mean) / self.std
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.classifier(self.extract_features(x))


def mahalanobis_min_distance(feats, centroids, cov_inv):
    """Distance from each feature vector to its NEAREST class centroid (shared covariance).

    Same maths as `mahalanobis_distance` in the ML notebook (Section 16).
    feats: (N, D); centroids: (K, D); cov_inv: (D, D). Returns (N,).
    """
    per_class = []
    for centroid in centroids:
        diff = feats - centroid
        squared = np.einsum("ij,ij->i", diff @ cov_inv, diff)
        per_class.append(np.sqrt(np.maximum(squared, 0)))
    return np.stack(per_class, axis=1).min(axis=1)


class ModelService:
    """Loads the deployed model once and measures reliability signals for single images."""

    def __init__(
        self,
        checkpoint_path=MODEL_CHECKPOINT_PATH,
        selection_path=MODEL_SELECTION_PATH,
        ood_stats_path=OOD_STATS_PATH,
    ):
        checkpoint_path, selection_path, ood_stats_path = (
            Path(checkpoint_path),
            Path(selection_path),
            Path(ood_stats_path),
        )
        for path in (checkpoint_path, selection_path, ood_stats_path):
            if not path.is_file():
                raise ModelUnavailableError(
                    f"Required model file not found: {path}. See artifacts/README.md."
                )

        selection = self._read_selection(selection_path)
        if selection.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ModelUnavailableError(
                "Preprocessing mismatch: the model was trained with "
                f"'{selection.get('preprocessing_version')}' but the app implements '{PREPROCESSING_VERSION}'."
            )
        if Path(selection["checkpoint_path"]).name != checkpoint_path.name:
            raise ModelUnavailableError(
                f"final_model_selection.json names {Path(selection['checkpoint_path']).name} "
                f"but {checkpoint_path.name} was supplied."
            )
        # Stored in PREDICTION_HISTORY.model_version, e.g. "M4_PC1-d7e2e21cf167".
        self.model_version = f"{selection['recommended_checkpoint_run']}-{selection['config_hash']}"

        self._model = self._load_model(checkpoint_path)
        self._centroids, self._cov_inv = self._load_ood_stats(ood_stats_path)

    # -----------------------------------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------------------------------
    @staticmethod
    def _read_selection(path):
        try:
            selection = json.loads(path.read_text(encoding="utf-8"))
            for key in ("recommended_checkpoint_run", "config_hash", "checkpoint_path"):
                selection[key]
            return selection
        except (OSError, ValueError, KeyError) as exc:
            raise ModelUnavailableError(f"Could not read {path.name}: {exc!r}") from exc

    @staticmethod
    def _load_model(path):
        try:
            model = EfficientNetWrapper(num_classes=len(CLASS_NAMES))
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)      # strict: any architecture mismatch fails loudly
        except Exception as exc:              # corrupt file, wrong keys, wrong torch version...
            raise ModelUnavailableError(f"Could not load model weights from {path.name}: {exc!r}") from exc
        return model.eval()

    @staticmethod
    def _load_ood_stats(path):
        try:
            with np.load(path) as stats:
                centroids = stats["centroids"].astype(np.float64)
                cov_inv = stats["cov_inv"].astype(np.float64)
        except (OSError, ValueError, KeyError) as exc:
            raise ModelUnavailableError(f"Could not read {path.name}: {exc!r}") from exc
        if centroids.shape != (len(CLASS_NAMES), 1280) or cov_inv.shape != (1280, 1280):
            raise ModelUnavailableError(
                f"{path.name} has unexpected shapes {centroids.shape} / {cov_inv.shape}."
            )
        return centroids, cov_inv

    # -----------------------------------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def predict(self, image):
        """Run inference on one PP_v1 image (128x128 RGB PIL image from image_preprocessor).

        Returns a ModelOutput. The first view is the image itself; the other four are the
        notebook's test-time views, and `tta_consistent` is True when all five views predict
        the same class.
        """
        if image.size != IMAGE_SIZE or image.mode != "RGB":
            raise ValueError(f"Expected a {IMAGE_SIZE} RGB image, got {image.size} {image.mode}.")

        views = [image] + [make_view(image) for make_view in _TTA_VIEWS]
        batch = torch.stack([TF.to_tensor(view) for view in views])   # float32, 0-1, (5, 3, H, W)

        features = self._model.extract_features(batch)
        probs = torch.softmax(self._model.classifier(features), dim=1).numpy()

        original = probs[0]
        predicted_index = int(original.argmax())
        distance = mahalanobis_min_distance(
            features[:1].numpy().astype(np.float64), self._centroids, self._cov_inv
        )[0]

        # float() matters: sqlite3 cannot bind numpy.float32.
        return ModelOutput(
            predicted_class=CLASS_NAMES[predicted_index],
            apple_probability=float(original[CLASS_NAMES.index("Apple")]),
            banana_probability=float(original[CLASS_NAMES.index("Banana")]),
            confidence=float(original.max()),
            feature_distance=float(distance),
            tta_consistent=bool(len(set(probs.argmax(axis=1).tolist())) == 1),
        )


def load_model_service():
    """Build the ModelService. In Streamlit, wrap the call in @st.cache_resource so the
    model is loaded once per server, not on every rerun."""
    return ModelService()
