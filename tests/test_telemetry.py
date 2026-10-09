"""Unit tests for TelemetryTracker module."""

import numpy as np

from vision_builder.detector import DetectionResult
from vision_builder.telemetry import TelemetryMetrics, TelemetryTracker


def test_empty_telemetry_tracker() -> None:
    tracker = TelemetryTracker()
    metrics = tracker.get_metrics()

    assert isinstance(metrics, TelemetryMetrics)
    assert metrics.total_frames == 0
    assert metrics.defective_frames == 0
    assert metrics.defect_rate_pct == 0.0
    assert metrics.avg_latency_ms == 0.0
    assert metrics.throughput_fps == 0.0


def test_telemetry_recording_and_percentiles() -> None:
    tracker = TelemetryTracker()

    # Record 10 latencies: 10, 20, 30, ... 100
    for i in range(1, 11):
        tracker.record_latency(latency_ms=float(i * 10), has_defect=(i > 8))

    metrics = tracker.get_metrics()
    assert metrics.total_frames == 10
    assert metrics.defective_frames == 2
    assert metrics.defect_rate_pct == 20.0
    assert metrics.min_latency_ms == 10.0
    assert metrics.max_latency_ms == 100.0
    assert metrics.avg_latency_ms == 55.0
    assert metrics.p95_latency_ms == pytest_approx_p95(100.0)


def pytest_approx_p95(expected: float) -> float:
    # helper for checking percentiles
    data = np.array([i * 10 for i in range(1, 11)], dtype=np.float64)
    return round(float(np.percentile(data, 95)), 2)


def test_record_result_object() -> None:
    tracker = TelemetryTracker()
    res = DetectionResult(
        frame_id="f1",
        timestamp=100.0,
        defects=[],
        inference_time_ms=12.5,
        status="PASS",
    )
    tracker.record_result(res)

    metrics = tracker.get_metrics()
    assert metrics.total_frames == 1
    assert metrics.defective_frames == 0
    assert metrics.avg_latency_ms == 12.5


def test_reset_telemetry() -> None:
    tracker = TelemetryTracker()
    tracker.record_latency(25.0, has_defect=True)
    assert tracker.get_metrics().total_frames == 1

    tracker.reset()
    assert tracker.get_metrics().total_frames == 0
    assert tracker.to_dict()["total_frames"] == 0
