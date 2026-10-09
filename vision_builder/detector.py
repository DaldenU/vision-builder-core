"""Scientific computer vision anomaly and defect detection engine.

Implements edge-optimized inspection algorithms, contour morphology,
defect scoring, and severity quantification for industrial CV-Ops.
"""

from enum import Enum
import time
from typing import List
import cv2
import numpy as np
from pydantic import BaseModel, Field


class DefectSeverity(str, Enum):
    """Classification of defect severity level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BoundingBox(BaseModel):
    """Normalized or absolute spatial bounding coordinates."""

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(ge=0.0)
    y_max: float = Field(ge=0.0)

    @property
    def width(self) -> float:
        """Calculate box width."""
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        """Calculate box height."""
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        """Calculate box area."""
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union (IoU) with another bounding box."""
        inter_x1 = max(self.x_min, other.x_min)
        inter_y1 = max(self.y_min, other.y_min)
        inter_x2 = min(self.x_max, other.x_max)
        inter_y2 = min(self.y_max, other.y_max)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area


class DefectItem(BaseModel):
    """Defect detection instance with category, score, and spatial localization."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    severity: DefectSeverity = DefectSeverity.MEDIUM


class DetectionResult(BaseModel):
    """Comprehensive outcome of a frame inspection pass."""

    frame_id: str
    timestamp: float
    defects: List[DefectItem] = Field(default_factory=list)
    inference_time_ms: float = Field(ge=0.0)
    status: str = Field(description="'PASS' or 'DEFECT_DETECTED'")

    @property
    def is_defective(self) -> bool:
        """Check if any defects were detected."""
        return len(self.defects) > 0


class DetectorConfig(BaseModel):
    """Parameters governing defect detection and sensitivity."""

    confidence_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    min_defect_area_px: float = Field(default=25.0, ge=1.0)
    critical_area_threshold_px: float = Field(default=400.0, ge=1.0)
    edge_gradient_threshold: int = Field(default=120, ge=1, le=255)
    default_defect_label: str = Field(default="surface_crack")


class DefectDetector:
    """Edge inference module detecting anomalies in operational frames."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        """Initialize detector with configuration."""
        self.config = config or DetectorConfig()

    def _determine_severity(self, area: float, confidence: float) -> DefectSeverity:
        """Map defect area and confidence to severity rating."""
        if area >= self.config.critical_area_threshold_px and confidence >= 0.80:
            return DefectSeverity.CRITICAL
        if area >= self.config.critical_area_threshold_px * 0.5:
            return DefectSeverity.HIGH
        if confidence >= 0.65:
            return DefectSeverity.MEDIUM
        return DefectSeverity.LOW

    def detect(self, frame: np.ndarray, frame_id: str = "frame_0") -> DetectionResult:
        """Execute anomaly detection on a frame image.

        Args:
            frame: Input image array (H, W, 3) or (H, W).
            frame_id: Identifier string for tracking.

        Returns:
            DetectionResult with detected defect entities and inference metrics.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame must be a valid non-empty numpy array")

        start_time = time.perf_counter()

        # Convert to single-channel 8-bit grayscale for gradient and contour inspection
        if frame.ndim == 3:
            if frame.dtype != np.uint8:
                # If normalized float in [0, 1] or standardized
                norm_frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                if norm_frame.shape[0] in {1, 3}:  # (C, H, W)
                    norm_frame = np.transpose(norm_frame, (1, 2, 0))
                gray = cv2.cvtColor(norm_frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            if frame.dtype != np.uint8:
                gray = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                gray = frame.copy()

        # Scientific edge and anomaly analysis:
        # Gaussian smoothing to suppress sensor noise, followed by Canny edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            blurred,
            threshold1=self.config.edge_gradient_threshold // 2,
            threshold2=self.config.edge_gradient_threshold,
        )

        # Morphological dilation to group fracture edges into cohesive defect contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        defects: List[DefectItem] = []
        img_h, img_w = gray.shape[:2]
        total_pixels = float(img_h * img_w)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_defect_area_px:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            bbox = BoundingBox(
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + w),
                y_max=float(y + h),
            )

            # Anomaly confidence score from salience ratio and contour density
            salience_ratio = min(1.0, (area / total_pixels) * 50.0 + 0.50)
            confidence = round(float(salience_ratio), 3)

            if confidence >= self.config.confidence_threshold:
                severity = self._determine_severity(area, confidence)
                defects.append(
                    DefectItem(
                        label=self.config.default_defect_label,
                        confidence=confidence,
                        bbox=bbox,
                        severity=severity,
                    )
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        status = "DEFECT_DETECTED" if defects else "PASS"

        return DetectionResult(
            frame_id=frame_id,
            timestamp=time.time(),
            defects=defects,
            inference_time_ms=round(elapsed_ms, 2),
            status=status,
        )
