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


def test_morphological_classification_crack() -> None:
    detector = DefectDetector()
    frame = np.full((300, 300, 3), 200, dtype=np.uint8)
    import cv2

    # Draw an elongated fracture crack
    cv2.line(frame, (50, 50), (220, 230), (20, 20, 20), 4)

    result = detector.detect(frame, frame_id="crack_01")
    assert result.status == "DEFECT_DETECTED"
    assert any(d.label == "surface_crack" for d in result.defects)


def test_morphological_classification_pit() -> None:
    detector = DefectDetector()
    frame = np.full((300, 300, 3), 200, dtype=np.uint8)
    import cv2

    # Draw a compact circular void/pit
    cv2.circle(frame, (150, 150), 12, (20, 20, 20), -1)

    result = detector.detect(frame, frame_id="pit_01")
    assert result.status == "DEFECT_DETECTED"
    assert any(d.label == "void_pit" for d in result.defects)


def test_straight_edge_rejection() -> None:
    detector = DefectDetector(DetectorConfig(reject_straight_edges=True))
    frame = np.full((300, 300, 3), 200, dtype=np.uint8)
    import cv2

    # Draw a straight conveyor guide border
    cv2.line(frame, (20, 150), (280, 150), (120, 120, 120), 2)

    result = detector.detect(frame, frame_id="guide_01")
    assert result.status == "PASS"
    assert len(result.defects) == 0


def test_detector_sensitivity_presets() -> None:
    det_low = DefectDetector(DetectorConfig(sensitivity="low"))
    assert det_low.config.edge_gradient_threshold == 160
    assert det_low.config.min_defect_area_px == 60.0
    assert det_low.config.min_contrast_delta == 25.0
    assert det_low.config.confidence_threshold == 0.55

    det_high = DefectDetector(DetectorConfig(sensitivity="high"))
    assert det_high.config.edge_gradient_threshold == 80
    assert det_high.config.min_defect_area_px == 15.0
    assert det_high.config.min_contrast_delta == 8.0
    assert det_high.config.confidence_threshold == 0.40


def test_custom_label_unclassified() -> None:
    detector = DefectDetector(
        DetectorConfig(
            classify_defects=False,
            default_defect_label="custom_anomaly",
            confidence_threshold=0.3,
        )
    )
    frame = np.full((300, 300, 3), 200, dtype=np.uint8)
    import cv2

    cv2.circle(frame, (150, 150), 15, (20, 20, 20), -1)
    result = detector.detect(frame, frame_id="custom_01")
    assert result.status == "DEFECT_DETECTED"
    assert result.defects[0].label == "custom_anomaly"
