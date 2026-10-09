"""Command Line Interface for Vision Builder Core.

Provides benchmarking utilities, synthetic edge stream simulation,
real-time video and webcam processing, and LLM incident reporting export.
"""

import argparse
import json
import os
import sys
from typing import List, Optional
import cv2
import numpy as np

from vision_builder import (
    DefectDetector,
    ImagePreprocessor,
    IncidentReportBuilder,
    TelemetryTracker,
    __version__,
)
from vision_builder.detector import DetectionResult, DetectorConfig
from vision_builder.preprocessor import PreprocessingConfig
from vision_builder.telemetry import TelemetryMetrics


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
    return np.asarray(frame, dtype=np.uint8)


def annotate_frame(
    frame: np.ndarray,
    result: DetectionResult,
    telemetry: TelemetryMetrics,
) -> np.ndarray:
    """Overlay defect bounding boxes, severity labels, and telemetry HUD."""
    annotated = frame.copy()

    # Severity color mapping (BGR)
    color_map = {
        "critical": (0, 0, 240),
        "high": (0, 140, 255),
        "medium": (0, 215, 255),
        "low": (255, 200, 0),
    }

    for defect in result.defects:
        b = defect.bbox
        x1, y1 = int(b.x_min), int(b.y_min)
        x2, y2 = int(b.x_max), int(b.y_max)
        color = color_map.get(defect.severity.value, (0, 0, 255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        sev_label = defect.severity.value.upper()
        label = f"{defect.label.upper()} [{sev_label}: {defect.confidence:.2f}]"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - 20)),
            (x1 + w + 6, max(20, y1)),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Heads-Up Display (HUD) overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, 10), (330, 115), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    status_color = (0, 0, 255) if result.is_defective else (0, 230, 0)
    cv2.putText(
        annotated,
        "VISION BUILDER EDGE CV-OPS",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Status: {result.status}",
        (20, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        status_color,
        2,
        cv2.LINE_AA,
    )
    stat1 = (
        f"Latency: {result.inference_time_ms:.1f}ms | "
        f"FPS: {telemetry.throughput_fps:.1f}"
    )
    cv2.putText(
        annotated,
        stat1,
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    stat2 = (
        f"Frames: {telemetry.total_frames} | "
        f"Defects: {telemetry.defective_frames} ({telemetry.defect_rate_pct:.1f}%)"
    )
    cv2.putText(
        annotated,
        stat2,
        (20, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return np.asarray(annotated, dtype=np.uint8)


def create_synthetic_conveyor_video(
    output_path: str = "demo_conveyor.mp4", num_frames: int = 120
) -> str:
    """Generate a synthetic manufacturing conveyor video for demonstration."""
    width, height = 640, 480
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), 160, dtype=np.uint8)
        offset = (i * 8) % 80
        for y in range(0, height, 80):
            cv2.line(frame, (0, y + offset), (width, y + offset), (140, 140, 140), 2)

        part_x = int((i * 12) % (width + 200)) - 100
        part_y = height // 2 - 60
        part_w, part_h = 120, 120

        cv2.rectangle(
            frame,
            (part_x, part_y),
            (part_x + part_w, part_y + part_h),
            (220, 220, 220),
            -1,
        )
        cv2.rectangle(
            frame,
            (part_x, part_y),
            (part_x + part_w, part_y + part_h),
            (80, 80, 80),
            2,
        )

        # Inject surface flaw on specific conveyor cycles
        if 35 <= (i % 70) <= 55:
            cx = part_x + 30
            cy = part_y + 25
            cv2.line(frame, (cx, cy), (cx + 50, cy + 55), (20, 20, 20), 4)
            cv2.circle(frame, (cx + 35, cy + 35), 8, (30, 30, 30), -1)

        writer.write(frame)

    writer.release()
    return output_path


def process_video_stream(
    source: str = "demo",
    output_path: Optional[str] = None,
    display: bool = False,
    max_frames: Optional[int] = None,
    export_report: Optional[str] = None,
) -> int:
    """Process video file or webcam stream with live anomaly detection."""
    temp_generated_video: Optional[str] = None

    if source.lower() == "demo":
        temp_generated_video = "demo_conveyor_temp.mp4"
        print("Generating synthetic manufacturing conveyor video...")
        create_synthetic_conveyor_video(temp_generated_video, num_frames=120)
        video_src: str | int = temp_generated_video
    elif source.lower() in {"webcam", "0", "camera"}:
        print("Opening live webcam device 0...")
        video_src = 0
    else:
        video_src = source

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"Error: Unable to open video source '{source}'")
        if temp_generated_video and os.path.exists(temp_generated_video):
            os.remove(temp_generated_video)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer: Optional[cv2.VideoWriter] = None
    if output_path:
        fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Recording annotated inspection video to '{output_path}'...")

    preprocessor = ImagePreprocessor(
        PreprocessingConfig(target_width=width, target_height=height)
    )
    detector = DefectDetector(
        DetectorConfig(confidence_threshold=0.45, min_defect_area_px=20.0)
    )
    telemetry = TelemetryTracker()
    report_builder = IncidentReportBuilder("Manufacturing_Cell_Live")

    frame_count = 0
    last_report = None

    print(f"Processing video stream: {source} (Press 'q' in preview window to stop)...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            if max_frames and frame_count > max_frames:
                break

            # Preprocessing and Edge Inference pass
            _ = preprocessor.process(frame)
            result = detector.detect(frame, frame_id=f"frame_{frame_count}")
            telemetry.record_result(result)

            if result.is_defective:
                last_report = report_builder.build_report(
                    result, telemetry=telemetry.get_metrics()
                )

            # Draw HUD and bounding boxes
            annotated = annotate_frame(frame, result, telemetry.get_metrics())

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("Vision Builder Edge CV-Ops Stream", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("Stream stopped by user.")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        if temp_generated_video and os.path.exists(temp_generated_video):
            try:
                os.remove(temp_generated_video)
            except OSError:
                pass

    metrics = telemetry.get_metrics()
    print("\n--- Video Processing Complete ---")
    print(f"  Frames Analyzed       : {metrics.total_frames}")
    print(f"  Defective Frames      : {metrics.defective_frames}")
    print(f"  Defect Rate           : {metrics.defect_rate_pct}%")
    print(f"  Average Frame Latency : {metrics.avg_latency_ms} ms")
    print(f"  Effective Throughput  : {metrics.throughput_fps} FPS")

    if export_report and last_report:
        with open(export_report, "w", encoding="utf-8") as f:
            json.dump(last_report.model_dump(), f, indent=2)
        print(f"  Saved LLM Incident Report to '{export_report}'")

    return 0


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

        _ = preprocessor.process(frame)
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

    video_parser = subparsers.add_parser(
        "process-video",
        help="Process video file or webcam stream with live anomaly detection",
    )
    video_parser.add_argument(
        "--source",
        type=str,
        default="demo",
        help="Source path: file path, 'webcam' / 0, or 'demo' for synthetic conveyor",
    )
    video_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save annotated output video (.mp4)",
    )
    video_parser.add_argument(
        "--display",
        action="store_true",
        help="Display live GUI window with annotated stream and HUD",
    )
    video_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before stopping",
    )
    video_parser.add_argument(
        "--export-report",
        type=str,
        default=None,
        help="Optional path to export final LLM JSON incident report",
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
    elif parsed.command == "process-video":
        return process_video_stream(
            source=parsed.source,
            output_path=parsed.output,
            display=parsed.display,
            max_frames=parsed.max_frames,
            export_report=parsed.export_report,
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
