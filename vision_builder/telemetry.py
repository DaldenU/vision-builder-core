"""Scientific telemetry tracking module for real-time edge CV-Ops monitoring.

Calculates rolling statistical percentiles (p50, p95), frame-rate throughput,
and anomaly frequency distributions to ensure industrial SLA compliance.
"""

from collections import deque
import threading
import time
from typing import Any, Dict
import numpy as np
from pydantic import BaseModel, Field

from vision_builder.detector import DetectionResult


class TelemetryMetrics(BaseModel):
    """Aggregated operational metrics for edge vision node telemetry."""

    total_frames: int = Field(ge=0)
    defective_frames: int = Field(ge=0)
    defect_rate_pct: float = Field(ge=0.0, le=100.0)
    avg_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    min_latency_ms: float = Field(ge=0.0)
    max_latency_ms: float = Field(ge=0.0)
    throughput_fps: float = Field(ge=0.0)
    runtime_seconds: float = Field(ge=0.0)


class TelemetryTracker:
    """Thread-safe telemetry aggregator tracking latency and defect metrics."""

    def __init__(self, max_history_size: int = 1000) -> None:
        """Initialize tracker with rolling history buffer size."""
        self.max_history_size = max_history_size
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=max_history_size)
        self._total_frames: int = 0
        self._defective_frames: int = 0
        self._start_time: float = time.perf_counter()
        self._last_update_time: float = self._start_time

    def record_result(self, result: DetectionResult) -> None:
        """Record the outcome of a frame detection pass.

        Args:
            result: DetectionResult object from the detector.
        """
        self.record_latency(
            latency_ms=result.inference_time_ms,
            has_defect=result.is_defective,
        )

    def record_latency(self, latency_ms: float, has_defect: bool = False) -> None:
        """Directly record latency sample and defect flag.

        Args:
            latency_ms: Inference or processing latency in milliseconds.
            has_defect: Boolean flag indicating if defect was present.
        """
        with self._lock:
            self._latencies.append(float(latency_ms))
            self._total_frames += 1
            if has_defect:
                self._defective_frames += 1
            self._last_update_time = time.perf_counter()

    def get_metrics(self) -> TelemetryMetrics:
        """Compute aggregated statistical metrics.

        Returns:
            TelemetryMetrics data object.
        """
        with self._lock:
            if self._total_frames == 0 or not self._latencies:
                return TelemetryMetrics(
                    total_frames=0,
                    defective_frames=0,
                    defect_rate_pct=0.0,
                    avg_latency_ms=0.0,
                    p95_latency_ms=0.0,
                    min_latency_ms=0.0,
                    max_latency_ms=0.0,
                    throughput_fps=0.0,
                    runtime_seconds=0.0,
                )

            lat_array = np.array(self._latencies, dtype=np.float64)
            avg_lat = float(np.mean(lat_array))
            p95_lat = float(np.percentile(lat_array, 95))
            min_lat = float(np.min(lat_array))
            max_lat = float(np.max(lat_array))

            runtime = max(0.001, time.perf_counter() - self._start_time)
            fps = float(self._total_frames / runtime)
            defect_pct = float((self._defective_frames / self._total_frames) * 100.0)

            return TelemetryMetrics(
                total_frames=self._total_frames,
                defective_frames=self._defective_frames,
                defect_rate_pct=round(defect_pct, 2),
                avg_latency_ms=round(avg_lat, 2),
                p95_latency_ms=round(p95_lat, 2),
                min_latency_ms=round(min_lat, 2),
                max_latency_ms=round(max_lat, 2),
                throughput_fps=round(fps, 2),
                runtime_seconds=round(runtime, 2),
            )

    def reset(self) -> None:
        """Reset all counters and timing records."""
        with self._lock:
            self._latencies.clear()
            self._total_frames = 0
            self._defective_frames = 0
            self._start_time = time.perf_counter()
            self._last_update_time = self._start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert current telemetry metrics to dictionary."""
        return self.get_metrics().model_dump()
