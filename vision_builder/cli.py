"""Command Line Interface for Vision Builder Core.

Provides benchmarking utilities, synthetic edge stream simulation,
and LLM incident reporting export for CV-Ops pipelines.
"""

import argparse
import json
import sys
from typing import List
import cv2
import numpy as np

from vision_builder import (
    DefectDetector,
    ImagePreprocessor,
    IncidentReportBuilder,
    TelemetryTracker,
    __version__,
)
from vision_builder.detector import DetectorConfig
from vision_builder.preprocessor import PreprocessingConfig


def generate_synthetic_frame(
    width: int = 640, height: int = 480, inject_defect: bool = False
) -> np.ndarray:
    """Generate a synthetic manufacturing surface frame."""
    frame = np.full((height, width, 3), 180, dtype=np.uint8)
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if inject_defect:
        cv2.line(
            frame,
            (width // 4, height // 3),
            (width // 2, (height // 3) + 80),
            (20, 20, 20),
            thickness=4,
        )
        cv2.circle(
            frame,
            (int(width * 0.7), int(height * 0.6)),
            radius=15,
            color=(30, 30, 30),
            thickness=-1,
        )
    return frame


def run_benchmark(iterations: int = 100, inject_rate: float = 0.2) -> int:
    """Run performance benchmark over synthetic video stream."""
    print(f"--- Running Vision Builder Benchmark ({iterations} iterations) ---")
    preprocessor = ImagePreprocessor(PreprocessingConfig())
    detector = DefectDetector(DetectorConfig())
    telemetry = TelemetryTracker()
    report_builder = IncidentReportBuilder("Benchmarking_Cell_1")

    last_report = None
    for i in range(iterations):
        inject = (i % int(1.0 / inject_rate)) == 0 if inject_rate > 0 else False
        frame = generate_synthetic_frame(inject_defect=inject)

        # Preprocessing pass
        _ = preprocessor.process(frame)

        # Edge detection pass
        result = detector.detect(frame, frame_id=f"frame_{i}")
        telemetry.record_result(result)

        if result.is_defective:
            last_report = report_builder.build_report(
                result, telemetry=telemetry.get_metrics()
            )

    metrics = telemetry.get_metrics()
    print("Benchmark Completed Successfully:")
    print(f"  Total Frames Processed : {metrics.total_frames}")
    print(f"  Defective Frames Flagged: {metrics.defective_frames}")
    print(f"  Defect Rate            : {metrics.defect_rate_pct}%")
    print(f"  Average Latency        : {metrics.avg_latency_ms} ms")
    print(f"  P95 Latency            : {metrics.p95_latency_ms} ms")
    print(f"  Throughput             : {metrics.throughput_fps} FPS")

    if last_report:
        print("\nSample LLM Prompt Generated:")
        print(f"  Urgency: {last_report.urgency}")
        print(f"  Actions: {', '.join(last_report.recommended_actions)}")
    return 0


def main(argv: List[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="vision-builder",
        description="Vision Builder Core CV-Ops processing and benchmarking CLI",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    bench_parser = subparsers.add_parser(
        "benchmark", help="Run synthetic stream processing benchmark"
    )
    bench_parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of synthetic frames to simulate",
    )
    bench_parser.add_argument(
        "--defect-rate",
        type=float,
        default=0.2,
        help="Fraction of frames containing synthetic anomalies (0.0 - 1.0)",
    )

    report_parser = subparsers.add_parser(
        "sample-report", help="Generate a sample LLM incident prompt JSON"
    )
    report_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to write JSON report to",
    )

    parsed = parser.parse_args(argv)

    if parsed.command == "benchmark":
        return run_benchmark(
            iterations=parsed.iterations, inject_rate=parsed.defect_rate
        )
    elif parsed.command == "sample-report":
        frame = generate_synthetic_frame(inject_defect=True)
        detector = DefectDetector()
        res = detector.detect(frame, "sample_frame_0")
        report = IncidentReportBuilder("Alpha_Cell").build_report(res)
        json_output = json.dumps(report.model_dump(), indent=2)
        if parsed.output:
            with open(parsed.output, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"Wrote sample report to {parsed.output}")
        else:
            print(json_output)
        return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
