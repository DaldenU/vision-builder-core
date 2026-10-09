"""Scientific image preprocessing module for Vision Builder.

Provides vectorized tensor transformations, dynamic letterboxing,
color-space normalization, and dimensional verification for edge inference.
"""

from typing import List, Tuple
import cv2
import numpy as np
from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Configuration parameters for scientific frame preprocessing."""

    target_width: int = Field(default=640, ge=32, le=4096)
    target_height: int = Field(default=640, ge=32, le=4096)
    preserve_aspect_ratio: bool = Field(default=True)
    normalize_method: str = Field(
        default="standardize",
        description="Normalization mode: 'standardize', 'minmax', or 'none'",
    )
    mean: List[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])
    to_channel_first: bool = Field(
        default=True,
        description="Convert (H, W, C) layout to (C, H, W) PyTorch/ONNX tensor format",
    )


class ImagePreprocessor:
    """Scientific preprocessor converting raw vision feeds to normalized tensors."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        """Initialize preprocessor with configuration."""
        self.config = config or PreprocessingConfig()
        if self.config.normalize_method not in {"standardize", "minmax", "none"}:
            raise ValueError(
                f"Unsupported normalize_method: {self.config.normalize_method}"
            )

    def letterbox(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        pad_value: int = 114,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Resize image with aspect-ratio preservation and padding (letterboxing).

        Args:
            image: Source image numpy array (H, W, C) or (H, W).
            target_size: Tuple of (target_width, target_height).
            pad_value: Pixel padding intensity for letterbox borders.

        Returns:
            Tuple of (padded_image, scale_factor, (pad_w, pad_h)).
        """
        shape = image.shape[:2]  # [height, width]
        target_w, target_h = target_size

        scale = min(target_w / shape[1], target_h / shape[0])
        new_w = int(round(shape[1] * scale))
        new_h = int(round(shape[0] * scale))

        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        if (shape[1], shape[0]) != (new_w, new_h):
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image.copy()

        if image.ndim == 3:
            padded = np.full(
                (target_h, target_w, image.shape[2]),
                pad_value,
                dtype=image.dtype,
            )
            padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w, :] = resized
        else:
            padded = np.full((target_h, target_w), pad_value, dtype=image.dtype)
            padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        return padded, scale, (pad_w, pad_h)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Execute full preprocessing pipeline on a single frame.

        Args:
            frame: Input image array.

        Returns:
            Preprocessed tensor array (float32).
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a non-null numpy ndarray")
        if frame.size == 0 or frame.ndim not in {2, 3}:
            raise ValueError(
                f"Invalid frame dimensions: shape={getattr(frame, 'shape', None)}"
            )

        working = frame.copy()

        # Convert grayscale to RGB if required
        if working.ndim == 2:
            working = cv2.cvtColor(working, cv2.COLOR_GRAY2RGB)
        elif working.shape[2] == 1:
            working = cv2.cvtColor(working, cv2.COLOR_GRAY2RGB)
        elif working.shape[2] == 4:
            working = cv2.cvtColor(working, cv2.COLOR_BGRA2RGB)
        elif working.shape[2] == 3:
            # Assume BGR if opencv default, convert to RGB
            working = cv2.cvtColor(working, cv2.COLOR_BGR2RGB)

        target_size = (self.config.target_width, self.config.target_height)

        if self.config.preserve_aspect_ratio:
            working, _, _ = self.letterbox(working, target_size)
        else:
            working = cv2.resize(working, target_size, interpolation=cv2.INTER_LINEAR)

        working = working.astype(np.float32)

        if self.config.normalize_method == "minmax":
            working = working / 255.0
        elif self.config.normalize_method == "standardize":
            working = working / 255.0
            mean = np.array(self.config.mean, dtype=np.float32)
            std = np.array(self.config.std, dtype=np.float32)
            working = (working - mean) / std

        if self.config.to_channel_first:
            # (H, W, C) -> (C, H, W)
            working = np.transpose(working, (2, 0, 1))

        return np.asarray(working, dtype=np.float32)
