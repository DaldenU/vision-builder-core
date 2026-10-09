"""Unit tests for the Vision Builder CLI."""

import json
from pathlib import Path
import numpy as np
import pytest

from vision_builder.cli import (
    annotate_frame,
    create_synthetic_conveyor_video,
    generate_synthetic_frame,
    main,
    process_video_stream,
    run_benchmark,
)
from vision_builder.detector import (
    BoundingBox,
    DefectItem,
    DefectSeverity,
    DetectionResult,
)
from vision_builder.telemetry import TelemetryMetrics


def test_generate_synthetic_frame() -> None:
    clean = generate_synthetic_frame(width=100, height=80, inject_defect=False)
    assert clean.shape == (80, 100, 3)

    defect = generate_synthetic_frame(width=100, height=80, inject_defect=True)
    assert defect.shape == (80, 100, 3)


def test_run_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    code = run_benchmark(iterations=10, inject_rate=0.5)
    assert code == 0
    captured = capsys.readouterr().out
    assert "Benchmark Completed Successfully" in captured
    assert "Total Frames Processed : 10" in captured


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr().out
    assert "0.1.0" in captured


def test_cli_benchmark_command(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["benchmark", "--iterations", "5", "--defect-rate", "0.2"])
    assert ret == 0
    captured = capsys.readouterr().out
    assert "Total Frames Processed : 5" in captured


def test_cli_sample_report_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_file = tmp_path / "sample_report.json"
    ret = main(["sample-report", "--output", str(report_file)])
    assert ret == 0
    assert report_file.exists()

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "incident_id" in data
    assert "llm_prompt_context" in data

    # Test without output file (prints to stdout)
    ret2 = main(["sample-report"])
    assert ret2 == 0
    out2 = capsys.readouterr().out
    assert "incident_id" in out2


def test_annotate_frame() -> None:
    frame = np.full((120, 160, 3), 150, dtype=np.uint8)
    defects = [
        DefectItem(
            label="surface_flaw",
            confidence=0.88,
            bbox=BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50),
            severity=DefectSeverity.HIGH,
        )
    ]
    res = DetectionResult(
        frame_id="f1",
        timestamp=100.0,
        defects=defects,
        inference_time_ms=2.5,
        status="DEFECT_DETECTED",
    )
    metrics = TelemetryMetrics(
        total_frames=1,
        defective_frames=1,
        defect_rate_pct=100.0,
        avg_latency_ms=2.5,
        p95_latency_ms=2.5,
        min_latency_ms=2.5,
        max_latency_ms=2.5,
        throughput_fps=50.0,
        runtime_seconds=0.02,
    )
    annotated = annotate_frame(frame, res, metrics)
    assert annotated.shape == (120, 160, 3)


def test_create_synthetic_conveyor_video(tmp_path: Path) -> None:
    vid_file = tmp_path / "test_conveyor.mp4"
    path = create_synthetic_conveyor_video(str(vid_file), num_frames=10)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_process_video_stream_demo(tmp_path: Path) -> None:
    out_video = tmp_path / "out.mp4"
    out_report = tmp_path / "report.json"
    ret = process_video_stream(
        source="demo",
        output_path=str(out_video),
        display=False,
        max_frames=15,
        export_report=str(out_report),
    )
    assert ret == 0
    assert out_video.exists()


def test_cli_process_video_command(tmp_path: Path) -> None:
    out_video = tmp_path / "cli_out.mp4"
    ret = main(
        [
            "process-video",
            "--source",
            "demo",
            "--max-frames",
            "10",
            "--output",
            str(out_video),
        ]
    )
    assert ret == 0
    assert out_video.exists()


def test_process_video_invalid_source() -> None:
    ret = process_video_stream(source="non_existent_path_404.mp4")
    assert ret == 1
