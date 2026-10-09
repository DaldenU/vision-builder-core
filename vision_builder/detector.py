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
    classify_defects: bool = Field(
        default=True, description="Enable morphological shape classification"
    )
    max_area_ratio: float = Field(
        default=0.15,
        ge=0.01,
        le=1.0,
        description="Reject macro objects exceeding this frame area fraction",
    )
    max_dimension_ratio: float = Field(
        default=0.75,
        ge=0.05,
        le=1.0,
        description="Reject boxes spanning wider than this dimension fraction",
    )
    min_contrast_delta: float = Field(
        default=12.0,
        ge=0.0,
        le=255.0,
        description="Minimum contrast delta between anomaly and neighborhood",
    )
    reject_straight_edges: bool = Field(
        default=True,
        description="Reject straight collinear edges like borders or tracks",
    )
    boundary_margin_px: int = Field(
        default=2,
        ge=0,
        description="Ignore contours clipped at frame border margins",
    )
    sensitivity: str = Field(
        default="medium",
        description="Operational sensitivity preset: 'low', 'medium', or 'high'",
    )


class DefectDetector:
    """Edge inference module detecting anomalies in operational frames."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        """Initialize detector with configuration and sensitivity presets."""
        self.config = config or DetectorConfig()

        # Apply industry-standard sensitivity presets if configured
        if self.config.sensitivity == "low":
            if self.config.edge_gradient_threshold == 120:
                self.config.edge_gradient_threshold = 160
            if self.config.min_defect_area_px == 25.0:
                self.config.min_defect_area_px = 60.0
            if self.config.min_contrast_delta == 12.0:
                self.config.min_contrast_delta = 25.0
            if self.config.confidence_threshold == 0.50:
                self.config.confidence_threshold = 0.55
        elif self.config.sensitivity == "high":
            if self.config.edge_gradient_threshold == 120:
                self.config.edge_gradient_threshold = 80
            if self.config.min_defect_area_px == 25.0:
                self.config.min_defect_area_px = 15.0
            if self.config.min_contrast_delta == 12.0:
                self.config.min_contrast_delta = 8.0
            if self.config.confidence_threshold == 0.50:
                self.config.confidence_threshold = 0.40

    def _determine_severity(
        self, area: float, confidence: float, label: str = "surface_crack"
    ) -> DefectSeverity:
        """Map defect area, confidence, and type to severity rating."""
        if label == "surface_crack" and (
            area >= self.config.critical_area_threshold_px or confidence >= 0.85
        ):
            return DefectSeverity.CRITICAL
        if area >= self.config.critical_area_threshold_px and confidence >= 0.75:
            return DefectSeverity.CRITICAL
        if area >= self.config.critical_area_threshold_px * 0.5 or confidence >= 0.70:
            return DefectSeverity.HIGH
        if confidence >= 0.55:
            return DefectSeverity.MEDIUM
        return DefectSeverity.LOW

    def _classify_morphology(
        self,
        contour: np.ndarray,
        gray: np.ndarray,
        total_pixels: float,
        img_w: int,
        img_h: int,
    ) -> tuple[str, float, BoundingBox] | None:
        """Classify contour geometry into defect categories or reject non-defects.

        Uses morphological tortuosity, rotated bounding box elongation, polygon
        approximation, and local radiometric contrast delta to filter out regular
        workpiece boundaries, conveyor guides, and natural scene edges.
        """
        area = cv2.contourArea(contour)
        if area < self.config.min_defect_area_px or area > (
            total_pixels * self.config.max_area_ratio
        ):
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if (
            w > img_w * self.config.max_dimension_ratio
            or h > img_h * self.config.max_dimension_ratio
        ):
            return None

        # Ignore contours clipped at frame margins
        m = self.config.boundary_margin_px
        if x <= m or y <= m or (x + w) >= (img_w - m) or (y + h) >= (img_h - m):
            return None

        peri = cv2.arcLength(contour, True)
        if peri <= 0.0:
            return None

        circ = min(1.0, (4.0 * np.pi * area) / (peri**2))

        # Rotated bounding box for true physical elongation and orientation
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        ar_rotated = max(rw / max(1e-3, rh), rh / max(1e-3, rw))

        # Polygon approximation for straightness analysis
        poly = cv2.approxPolyDP(contour, 0.02 * peri, True)
        extent = area / float(max(1, w * h))

        # Reject straight machine guide rails running across the frame
        if (
            self.config.reject_straight_edges
            and len(poly) <= 3
            and ar_rotated >= 4.0
            and (w >= img_w * 0.70 or h >= img_h * 0.70)
        ):
            return None

        # Reject macro square workpieces / components
        if (
            len(poly) == 4
            and extent >= 0.80
            and area >= (total_pixels * 0.04)
            and ar_rotated <= 1.3
        ):
            return None

        # Compute radiometric contrast against local neighborhood using fast ROI crop
        pad = 8
        y1 = max(0, y - pad)
        y2 = min(img_h, y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img_w, x + w + pad)

        roi = gray[y1:y2, x1:x2]
        roi_cnt = contour - np.array([[[x1, y1]]])

        roi_mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.drawContours(roi_mask, [roi_cnt], -1, 255, -1)
        mean_inside = cv2.mean(roi, mask=roi_mask)[0]

        k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        roi_dil = cv2.dilate(roi_mask, k5, iterations=2)
        roi_bg = cv2.bitwise_xor(roi_dil, roi_mask)
        mean_bg = cv2.mean(roi, mask=roi_bg)[0]
        contrast_delta = abs(mean_inside - mean_bg)

        if contrast_delta < self.config.min_contrast_delta:
            return None

        bbox = BoundingBox(
            x_min=float(x),
            y_min=float(y),
            x_max=float(x + w),
            y_max=float(y + h),
        )

        if not self.config.classify_defects:
            conf = min(
                0.99,
                0.40
                + min(0.30, (contrast_delta / 255.0) * 0.40)
                + min(0.30, (area / self.config.critical_area_threshold_px) * 0.30),
            )
            return self.config.default_defect_label, round(float(conf), 3), bbox

        # Morphological shape classification
        if ar_rotated >= 2.5 and circ <= 0.40:
            label = "surface_crack"
            conf = min(
                0.99,
                0.45
                + min(0.25, (ar_rotated / 15.0) * 0.25)
                + min(0.30, (contrast_delta / 255.0) * 0.30),
            )
        elif (
            circ >= 0.48
            and area < self.config.critical_area_threshold_px * 6.0
            and ar_rotated < 2.5
        ):
            label = "void_pit"
            conf = min(
                0.99,
                0.45
                + min(0.25, circ * 0.25)
                + min(0.30, (contrast_delta / 255.0) * 0.30),
            )
        elif contrast_delta >= self.config.min_contrast_delta * 1.5:
            label = "surface_blemish"
            conf = min(
                0.95,
                0.42
                + min(0.40, (contrast_delta / 255.0) * 0.40)
                + min(0.18, (area / self.config.critical_area_threshold_px) * 0.20),
            )
        else:
            return None

        return label, round(float(conf), 3), bbox

    def _apply_nms(
        self, defects: List[DefectItem], iou_threshold: float = 0.40
    ) -> List[DefectItem]:
        """Perform Non-Maximum Suppression to eliminate duplicate overlapping boxes."""
        if not defects:
            return []
        sorted_defects = sorted(defects, key=lambda d: d.confidence, reverse=True)
        keep: List[DefectItem] = []
        for d in sorted_defects:
            if not any(d.bbox.iou(kept.bbox) > iou_threshold for kept in keep):
                keep.append(d)
        return keep

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

        # Morphological dilation to bridge subtle micro-fracture gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Retrieve both external and internal contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        defects: List[DefectItem] = []
        img_h, img_w = gray.shape[:2]
        total_pixels = float(img_h * img_w)

        for cnt in contours:
            classification = self._classify_morphology(
                contour=cnt,
                gray=gray,
                total_pixels=total_pixels,
                img_w=img_w,
                img_h=img_h,
            )
            if classification is None:
                continue

            label, confidence, bbox = classification
            if confidence >= self.config.confidence_threshold:
                severity = self._determine_severity(
                    area=bbox.area,
                    confidence=confidence,
                    label=label,
                )
                defects.append(
                    DefectItem(
                        label=label,
                        confidence=confidence,
                        bbox=bbox,
                        severity=severity,
                    )
                )

        # Suppress duplicate concentric bounding boxes
        defects = self._apply_nms(defects)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        status = "DEFECT_DETECTED" if defects else "PASS"

        return DetectionResult(
            frame_id=frame_id,
            timestamp=time.time(),
            defects=defects,
            inference_time_ms=round(elapsed_ms, 2),
            status=status,
        )
