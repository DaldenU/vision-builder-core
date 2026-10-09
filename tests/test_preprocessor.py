"""Unit tests for ImagePreprocessor module."""

import numpy as np
import pytest

from vision_builder.preprocessor import ImagePreprocessor, PreprocessingConfig


def test_invalid_normalization_method() -> None:
    config = PreprocessingConfig(normalize_method="invalid_method")
    with pytest.raises(ValueError, match="Unsupported normalize_method"):
        ImagePreprocessor(config)


def test_invalid_input_frames() -> None:
    preprocessor = ImagePreprocessor()
    with pytest.raises(ValueError, match="non-null numpy ndarray"):
        preprocessor.process(None)  # type: ignore

    with pytest.raises(ValueError, match="Invalid frame dimensions"):
        preprocessor.process(np.array([]))

    with pytest.raises(ValueError, match="Invalid frame dimensions"):
        preprocessor.process(np.zeros((10, 10, 3, 2)))


def test_grayscale_and_rgba_conversion() -> None:
    preprocessor = ImagePreprocessor(
        PreprocessingConfig(target_width=320, target_height=240, to_channel_first=True)
    )

    # Grayscale 2D
    gray_frame = np.ones((200, 200), dtype=np.uint8) * 120
    out_gray = preprocessor.process(gray_frame)
    assert out_gray.shape == (3, 240, 320)

    # Grayscale 3D (H, W, 1)
    gray_3d = np.ones((200, 200, 1), dtype=np.uint8) * 120
    out_gray_3d = preprocessor.process(gray_3d)
    assert out_gray_3d.shape == (3, 240, 320)

    # RGBA 4-channel
    rgba_frame = np.ones((100, 150, 4), dtype=np.uint8) * 200
    out_rgba = preprocessor.process(rgba_frame)
    assert out_rgba.shape == (3, 240, 320)


def test_minmax_normalization() -> None:
    config = PreprocessingConfig(
        target_width=100,
        target_height=100,
        normalize_method="minmax",
        preserve_aspect_ratio=False,
        to_channel_first=False,
    )
    preprocessor = ImagePreprocessor(config)
    frame = np.full((50, 50, 3), 255, dtype=np.uint8)
    out = preprocessor.process(frame)

    assert out.shape == (100, 100, 3)
    assert np.allclose(out, 1.0)
    assert out.dtype == np.float32


def test_standardize_normalization() -> None:
    config = PreprocessingConfig(
        target_width=64,
        target_height=64,
        normalize_method="standardize",
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        preserve_aspect_ratio=False,
        to_channel_first=False,
    )
    preprocessor = ImagePreprocessor(config)
    frame = np.full(
        (32, 32, 3), 255, dtype=np.uint8
    )  # 255/255 = 1.0 -> (1.0 - 0.5)/0.5 = 1.0
    out = preprocessor.process(frame)
    assert np.allclose(out, 1.0)


def test_letterbox_padding() -> None:
    preprocessor = ImagePreprocessor(
        PreprocessingConfig(
            target_width=200, target_height=200, preserve_aspect_ratio=True
        )
    )
    # Wide image
    wide_img = np.ones((50, 100, 3), dtype=np.uint8) * 50
    padded, scale, (pad_w, pad_h) = preprocessor.letterbox(wide_img, (200, 200))

    assert padded.shape == (200, 200, 3)
    assert scale == 2.0
    assert pad_w == 0
    assert pad_h == 50  # (200 - 100)/2 = 50
