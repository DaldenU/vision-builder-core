"""Unit tests for DefectDetector module."""

import numpy as np
import pytest

from vision_builder.detector import (
    BoundingBox,
    DefectDetector,
    DefectItem,
    DefectSeverity,
    DetectorConfig,
)


def test_bounding_box_geometry() -> None:
    box1 = BoundingBox(x_min=10, y_min=10, x_max=30, y_max=30)
    assert box1.width == 20
    assert box1.height == 20
    assert box1.area == 400

    box2 = BoundingBox(x_min=20, y_min=20, x_max=40, y_max=40)
    # Intersection: [20, 20] to [30, 30] -> area 100
    # Union: 400 + 400 - 100 = 700
    iou = box1.iou(box2)
    assert abs(iou - (100 / 700)) < 1e-4

    # Non-overlapping box
    box3 = BoundingBox(x_min=100, y_min=100, x_max=120, y_max=120)
    assert box1.iou(box3) == 0.0


def test_detector_invalid_input() -> None:
    detector = DefectDetector()
    with pytest.raises(ValueError, match="valid non-empty numpy array"):
        detector.detect(None)  # type: ignore

    with pytest.raises(ValueError, match="valid non-empty numpy array"):
        detector.detect(np.array([]))


def test_clean_frame_detection_pass() -> None:
    detector = DefectDetector()
    clean_frame = np.full((300, 300, 3), 200, dtype=np.uint8)
    result = detector.detect(clean_frame, frame_id="clean_01")

    assert result.frame_id == "clean_01"
    assert result.status == "PASS"
    assert not result.is_defective
    assert len(result.defects) == 0
    assert result.inference_time_ms >= 0.0


def test_defective_frame_detected() -> None:
    detector = DefectDetector(
        DetectorConfig(confidence_threshold=0.4, min_defect_area_px=15.0)
    )
    frame = np.full((400, 400, 3), 220, dtype=np.uint8)
    # Draw significant dark anomaly
    frame[150:200, 150:250] = 10

    result = detector.detect(frame, frame_id="defective_01")
    assert result.status == "DEFECT_DETECTED"
    assert result.is_defective
    assert len(result.defects) >= 1

    first_defect = result.defects[0]
    assert isinstance(first_defect, DefectItem)
    assert first_defect.confidence >= 0.4
    assert first_defect.severity in {
        DefectSeverity.LOW,
        DefectSeverity.MEDIUM,
        DefectSeverity.HIGH,
        DefectSeverity.CRITICAL,
    }


def test_normalized_float_input_frame() -> None:
    detector = DefectDetector()
    # Normalized frame (C, H, W)
    float_frame = np.ones((3, 200, 200), dtype=np.float32) * 0.8
    result = detector.detect(float_frame, frame_id="float_01")
    assert result.status == "PASS"
