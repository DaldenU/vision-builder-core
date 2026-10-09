"""Unit tests for IncidentReportBuilder module."""

from vision_builder.detector import (
    BoundingBox,
    DefectItem,
    DefectSeverity,
    DetectionResult,
)
from vision_builder.report_builder import IncidentReport, IncidentReportBuilder
from vision_builder.telemetry import TelemetryMetrics


def test_incident_report_generation_for_critical_defect() -> None:
    builder = IncidentReportBuilder(facility_zone="Manufacturing_Line_03")

    defects = [
        DefectItem(
            label="critical_fracture",
            confidence=0.95,
            bbox=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=120),
            severity=DefectSeverity.CRITICAL,
        ),
        DefectItem(
            label="minor_scratch",
            confidence=0.60,
            bbox=BoundingBox(x_min=150, y_min=160, x_max=170, y_max=180),
            severity=DefectSeverity.LOW,
        ),
    ]

    result = DetectionResult(
        frame_id="frame_crit_09",
        timestamp=1700000000.0,
        defects=defects,
        inference_time_ms=8.4,
        status="DEFECT_DETECTED",
    )

    telemetry = TelemetryMetrics(
        total_frames=100,
        defective_frames=5,
        defect_rate_pct=5.0,
        avg_latency_ms=9.1,
        p95_latency_ms=14.2,
        min_latency_ms=6.0,
        max_latency_ms=25.0,
        throughput_fps=65.2,
        runtime_seconds=1.5,
    )

    report = builder.build_report(
        result=result,
        telemetry=telemetry,
        operator_notes="Automated alert triggered by optical camera 2.",
    )

    assert isinstance(report, IncidentReport)
    assert report.facility_zone == "Manufacturing_Line_03"
    assert report.urgency == "IMMEDIATE_ACTION_REQUIRED"
    assert report.max_severity == "critical"
    assert report.defect_count == 2
    assert len(report.recommended_actions) >= 2
    assert (
        "Halt affected production conveyor immediately." in report.recommended_actions
    )
    assert "payload" in report.llm_prompt_context
    assert (
        report.llm_prompt_context["payload"]["urgency"] == "IMMEDIATE_ACTION_REQUIRED"
    )


def test_incident_report_clean_pass() -> None:
    builder = IncidentReportBuilder("Clean_Zone")
    result = DetectionResult(
        frame_id="clean_pass_0",
        timestamp=1700000000.0,
        defects=[],
        inference_time_ms=5.0,
        status="PASS",
    )
    report = builder.build_report(result)

    assert report.defect_count == 0
    assert report.max_severity == "low"
    assert report.urgency == "INFORMATIONAL"
