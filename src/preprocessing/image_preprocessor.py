"""
File: image_preprocessor.py
Project: Intelligent Fruit Classification and Information System
Description:
    Validates uploaded images and applies the resizing, colour
    conversion, normalisation, and other preprocessing required
    by the selected trained classification model.

Author: Shubhan Singh
Responsible Pillar: Pillar 2 - Machine Learning & Evaluation
Maintainers: Wajahat Manzoor, Shubhan Singh

Note:
    Inference preprocessing must remain consistent with the
    selected model's training and evaluation pipeline.
"""
import io

from PIL import Image, ImageOps

from src.fruit_classifier.config import (
    ALLOWED_IMAGE_FORMATS,
    IMAGE_SIZE,
    MAX_UPLOAD_BYTES,
)
from src.fruit_classifier.exceptions import InvalidImageError


def _read_bytes(source):
    """Accept raw bytes or a file-like object (e.g. a Streamlit UploadedFile)."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "getvalue"):   # BytesIO / Streamlit UploadedFile
        return source.getvalue()
    if hasattr(source, "read"):
        return source.read()
    raise InvalidImageError("unreadable_file", "Unsupported upload object.")


def preprocess_image(source):
    """Validate an upload and return the PP_v1 image the model was trained on.

    Steps (identical to `load_resized` in the ML notebook, Section 3.1):
        JPEG reduced-size decode (draft) -> EXIF upright -> RGB -> bicubic resize to 128x128.

    The returned object is a 128x128 RGB PIL image. Conversion to a 0-1 tensor happens in
    ml/model_service.py because the test-time views are applied to the PIL image first.
    ImageNet mean/std standardisation is done inside the model, not here.

    Raises InvalidImageError with one of these reason codes:
        empty_file, file_too_large, unreadable_file, unsupported_format
    """
    data = _read_bytes(source)

    if len(data) == 0:
        raise InvalidImageError("empty_file", "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidImageError(
            "file_too_large", f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_IMAGE_FORMATS:
            raise InvalidImageError(
                "unsupported_format",
                f"Unsupported image format: {img.format}. Allowed: {', '.join(ALLOWED_IMAGE_FORMATS)}.",
            )
        # draft() is a JPEG-only speed-up that decodes at reduced size. Training used it, and it
        # changes pixel values slightly, so inference must use it too to match PP_v1 exactly.
        img.draft("RGB", IMAGE_SIZE)
        img.load()                       # forces a full decode, so truncated files fail here
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
    except InvalidImageError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        # UnidentifiedImageError is a subclass of OSError.
        raise InvalidImageError("unreadable_file", "The file could not be read as an image.") from exc
