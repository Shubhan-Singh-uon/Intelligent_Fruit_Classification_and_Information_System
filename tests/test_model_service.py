"""Tests for src/ml/model_service.py.

These use a randomly initialised network and synthetic statistics, so they check the plumbing
(architecture parity, loading, signal calculation, database-safe types), NOT the real model's
accuracy. The real checkpoint still needs a manual check with known Apple, Banana and
unsupported images once it is in artifacts/.
"""
import json
import sqlite3

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402
from torchvision import models  # noqa: E402

from src.fruit_classifier.config import CLASS_NAMES, IMAGE_SIZE, PREPROCESSING_VERSION  # noqa: E402
from src.fruit_classifier.exceptions import ModelUnavailableError  # noqa: E402
from src.fruit_classifier.schemas import ModelOutput, PredictionStatus  # noqa: E402
from src.ml.model_service import (  # noqa: E402
    EfficientNetWrapper,
    ModelService,
    mahalanobis_min_distance,
)
from src.ml.prediction_policy import apply_policy  # noqa: E402
from src.preprocessing.image_preprocessor import preprocess_image  # noqa: E402


class NotebookEfficientNetWrapper(nn.Module):
    """Verbatim copy of the training notebook's class (Section 9), used as the reference."""

    def __init__(self, dropout=0.30, unfreeze_from=None, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        in_features = backbone.classifier[1].in_features
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


@pytest.fixture(scope="module")
def artefacts(tmp_path_factory):
    """A complete, valid artifacts/ folder built from a random-weight reference model."""
    folder = tmp_path_factory.mktemp("artifacts")
    torch.manual_seed(0)
    reference = NotebookEfficientNetWrapper(pretrained=False).eval()
    torch.save(reference.state_dict(), folder / "M4_PC1.pt")

    selection = {
        "selected_model": "M4",
        "recommended_checkpoint_run": "M4_PC1",
        "checkpoint_path": "results/checkpoints/M4_PC1.pt",
        "config_hash": "d7e2e21cf167",
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    (folder / "final_model_selection.json").write_text(json.dumps(selection))

    rng = np.random.RandomState(1)
    np.savez(folder / "m4_ood_stats.npz",
             centroids=rng.randn(2, 1280).astype(np.float32),
             cov_inv=np.eye(1280, dtype=np.float32))
    return folder, reference


@pytest.fixture(scope="module")
def service(artefacts):
    folder, _ = artefacts
    return ModelService(folder / "M4_PC1.pt", folder / "final_model_selection.json", folder / "m4_ood_stats.npz")


def sample_image(seed=0):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 256, (IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8))


def test_architecture_keys_and_shapes_match_the_notebook_class():
    ours = EfficientNetWrapper().state_dict()
    theirs = NotebookEfficientNetWrapper(pretrained=False).state_dict()
    assert list(ours.keys()) == list(theirs.keys())
    assert all(ours[k].shape == theirs[k].shape for k in ours)


def test_loads_and_reports_traceable_model_version(service):
    assert service.model_version == "M4_PC1-d7e2e21cf167"


def test_logits_match_the_notebook_class_exactly(artefacts, service):
    _, reference = artefacts
    batch = torch.rand(3, 3, *IMAGE_SIZE)
    with torch.inference_mode():
        assert torch.allclose(service._model(batch), reference(batch), atol=1e-6)


def test_predict_returns_valid_database_safe_values(service):
    out = service.predict(sample_image())
    assert out.predicted_class in CLASS_NAMES
    assert abs(out.apple_probability + out.banana_probability - 1.0) < 1e-5
    assert out.confidence == max(out.apple_probability, out.banana_probability)
    assert out.feature_distance >= 0
    assert isinstance(out.tta_consistent, bool)
    # sqlite3 cannot bind numpy.float32, so these must be plain Python floats.
    for value in (out.apple_probability, out.banana_probability, out.confidence, out.feature_distance):
        assert type(value) is float


def test_feature_distance_matches_manual_calculation(artefacts, service):
    image = sample_image(3)
    out = service.predict(image)
    from torchvision.transforms import functional as TF
    with torch.inference_mode():
        feats = service._model.extract_features(TF.to_tensor(image).unsqueeze(0)).numpy().astype(np.float64)
    expected = min(np.linalg.norm(feats[0] - c) for c in service._centroids)   # cov_inv is the identity
    assert out.feature_distance == pytest.approx(expected, rel=1e-4)


def test_predict_is_deterministic(service):
    image = sample_image(5)
    assert service.predict(image) == service.predict(image)


def test_mahalanobis_picks_nearest_centroid():
    centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    feats = np.array([[1.0, 0.0], [9.0, 0.0]])
    assert mahalanobis_min_distance(feats, centroids, np.eye(2)).tolist() == [1.0, 1.0]


def test_rejects_wrongly_sized_input(service):
    with pytest.raises(ValueError):
        service.predict(Image.new("RGB", (200, 200)))
    with pytest.raises(ValueError):
        service.predict(Image.new("L", IMAGE_SIZE))


@pytest.mark.parametrize("missing", ["M4_PC1.pt", "final_model_selection.json", "m4_ood_stats.npz"])
def test_missing_artefact_is_a_technical_failure(artefacts, tmp_path, missing):
    folder, _ = artefacts
    for name in ("M4_PC1.pt", "final_model_selection.json", "m4_ood_stats.npz"):
        if name != missing:
            (tmp_path / name).write_bytes((folder / name).read_bytes())
    with pytest.raises(ModelUnavailableError):
        ModelService(tmp_path / "M4_PC1.pt", tmp_path / "final_model_selection.json", tmp_path / "m4_ood_stats.npz")


def test_preprocessing_version_mismatch_is_refused(artefacts, tmp_path):
    folder, _ = artefacts
    selection = json.loads((folder / "final_model_selection.json").read_text())
    selection["preprocessing_version"] = "PP_v2: something else"
    (tmp_path / "final_model_selection.json").write_text(json.dumps(selection))
    (tmp_path / "M4_PC1.pt").write_bytes((folder / "M4_PC1.pt").read_bytes())
    (tmp_path / "m4_ood_stats.npz").write_bytes((folder / "m4_ood_stats.npz").read_bytes())
    with pytest.raises(ModelUnavailableError, match="Preprocessing mismatch"):
        ModelService(tmp_path / "M4_PC1.pt", tmp_path / "final_model_selection.json", tmp_path / "m4_ood_stats.npz")


def test_corrupt_checkpoint_is_a_technical_failure(artefacts, tmp_path):
    folder, _ = artefacts
    (tmp_path / "M4_PC1.pt").write_bytes(b"not a checkpoint")
    (tmp_path / "final_model_selection.json").write_bytes((folder / "final_model_selection.json").read_bytes())
    (tmp_path / "m4_ood_stats.npz").write_bytes((folder / "m4_ood_stats.npz").read_bytes())
    with pytest.raises(ModelUnavailableError):
        ModelService(tmp_path / "M4_PC1.pt", tmp_path / "final_model_selection.json", tmp_path / "m4_ood_stats.npz")


def test_end_to_end_upload_to_database_row(service):
    """upload bytes -> preprocess -> model -> policy -> row that fits PREDICTION_HISTORY."""
    import io
    buffer = io.BytesIO()
    sample_image(7).resize((640, 480)).save(buffer, format="JPEG")

    result = apply_policy(service.predict(preprocess_image(buffer.getvalue())), service.model_version)
    assert result.status in set(PredictionStatus) - {PredictionStatus.INVALID_INPUT}

    connection = sqlite3.connect(":memory:")
    connection.execute(
        """CREATE TABLE PREDICTION_HISTORY (
            prediction_id INTEGER PRIMARY KEY, fruit_id INTEGER NULL, predicted_class TEXT NULL,
            apple_probability REAL NULL, banana_probability REAL NULL, confidence REAL NULL,
            prediction_status TEXT NOT NULL, rejection_reason TEXT NULL, model_version TEXT NOT NULL)"""
    )
    connection.execute(
        "INSERT INTO PREDICTION_HISTORY (fruit_id, predicted_class, apple_probability, banana_probability,"
        " confidence, prediction_status, rejection_reason, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (result.fruit_id, result.predicted_class, result.apple_probability, result.banana_probability,
         result.confidence, result.status.value, result.reason_code, result.model_version),
    )
    status, version = connection.execute("SELECT prediction_status, model_version FROM PREDICTION_HISTORY").fetchone()
    assert status == result.status.value and version == "M4_PC1-d7e2e21cf167"
