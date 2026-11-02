"""Tests for src/preprocessing/image_preprocessor.py."""
import io

import numpy as np
import pytest
from PIL import Image, ImageOps

from src.fruit_classifier.config import IMAGE_SIZE, MAX_UPLOAD_BYTES
from src.fruit_classifier.exceptions import InvalidImageError
from src.preprocessing.image_preprocessor import preprocess_image


def encode(image, fmt="JPEG", **save_kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def noisy_image(size=(640, 480), seed=0):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8))


def notebook_load_resized(data, size=IMAGE_SIZE):
    """Verbatim logic of `load_resized` from the training notebook (Section 3.1)."""
    img = Image.open(io.BytesIO(data))
    img.draft("RGB", size)
    img.load()
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB").resize(size, Image.BICUBIC)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_output_is_128x128_rgb(fmt):
    result = preprocess_image(encode(noisy_image(), fmt))
    assert result.size == IMAGE_SIZE and result.mode == "RGB"


def test_matches_the_training_pipeline_pixel_for_pixel():
    # Large JPEG so that draft() actually changes the decode path.
    data = encode(noisy_image(size=(2400, 1800)), "JPEG", quality=90)
    expected = np.asarray(notebook_load_resized(data))
    actual = np.asarray(preprocess_image(data))
    assert np.array_equal(actual, expected)


def test_png_with_alpha_becomes_rgb():
    rgba = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
    result = preprocess_image(encode(rgba, "PNG"))
    assert result.mode == "RGB"


def test_exif_orientation_is_applied():
    # 200x100 landscape image with a red square at the top-left. Orientation 6 means the file must be
    # rotated 90 degrees clockwise to be upright, which moves the red square to the top-right.
    base = Image.new("RGB", (200, 100), "white")
    base.paste(Image.new("RGB", (40, 40), "red"), (0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6
    result = preprocess_image(encode(base, "JPEG", quality=100, exif=exif))
    red_corner = np.asarray(result)[10, 115].astype(int)
    white_corner = np.asarray(result)[10, 8].astype(int)
    assert red_corner[0] > 200 and red_corner[1] < 80
    assert white_corner.min() > 200


def test_accepts_file_like_objects():
    data = encode(noisy_image(), "PNG")
    assert preprocess_image(io.BytesIO(data)).size == IMAGE_SIZE


def test_empty_file():
    with pytest.raises(InvalidImageError) as error:
        preprocess_image(b"")
    assert error.value.reason_code == "empty_file"


def test_file_too_large():
    with pytest.raises(InvalidImageError) as error:
        preprocess_image(b"\x00" * (MAX_UPLOAD_BYTES + 1))
    assert error.value.reason_code == "file_too_large"


def test_random_bytes_are_unreadable():
    with pytest.raises(InvalidImageError) as error:
        preprocess_image(b"this is not an image" * 50)
    assert error.value.reason_code == "unreadable_file"


def test_truncated_jpeg_is_unreadable():
    data = encode(noisy_image(size=(800, 600)), "JPEG")
    with pytest.raises(InvalidImageError) as error:
        preprocess_image(data[: len(data) // 2])
    assert error.value.reason_code == "unreadable_file"


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "WEBP"])
def test_valid_image_in_other_format_is_unsupported_not_unreadable(fmt):
    with pytest.raises(InvalidImageError) as error:
        preprocess_image(encode(noisy_image(size=(64, 64)), fmt))
    assert error.value.reason_code == "unsupported_format"
