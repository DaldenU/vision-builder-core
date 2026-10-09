"""Unit tests for the Vision Builder CLI."""

import json
from pathlib import Path
import pytest

from vision_builder.cli import generate_synthetic_frame, main, run_benchmark


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
