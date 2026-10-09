"""Incident report builder bridging computer vision telemetry and LLM context.

Transforms high-frequency computer vision telemetry and defect detections
into semantically rich, structured prompts and reports for automated LLM
reasoning, root-cause analysis, and operator alert dispatch.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
import uuid
from pydantic import BaseModel

from vision_builder.detector import DefectItem, DefectSeverity, DetectionResult
from vision_builder.telemetry import TelemetryMetrics


class IncidentReport(BaseModel):
    """Structured incident report formatted for operators and LLM ingestion."""

    incident_id: str
    timestamp: str
    facility_zone: str
    urgency: str
    defect_count: int
    max_severity: str
    detected_defects: List[Dict[str, Any]]
    telemetry_summary: Dict[str, Any]
    llm_prompt_context: Dict[str, Any]
    summary_text: str
    recommended_actions: List[str]


class IncidentReportBuilder:
    """Builder class constructing LLM-ready incident reports from CV observations."""

    def __init__(self, facility_zone: str = "Assembly_Line_Alpha") -> None:
        """Initialize builder with facility zone identifier."""
        self.facility_zone = facility_zone

    def _determine_highest_severity(self, defects: List[DefectItem]) -> DefectSeverity:
        """Identify maximum severity level in detected defect collection."""
        if not defects:
            return DefectSeverity.LOW

        severity_rank = {
            DefectSeverity.LOW: 1,
            DefectSeverity.MEDIUM: 2,
            DefectSeverity.HIGH: 3,
            DefectSeverity.CRITICAL: 4,
        }
        return max(defects, key=lambda d: severity_rank.get(d.severity, 1)).severity

    def build_report(
        self,
        result: DetectionResult,
        telemetry: TelemetryMetrics | None = None,
        operator_notes: str = "",
    ) -> IncidentReport:
        """Generate a complete incident report for a detection result.

        Args:
            result: DetectionResult from DefectDetector.
            telemetry: Optional aggregated TelemetryMetrics.
            operator_notes: Supplementary human or sensor notes.

        Returns:
            IncidentReport model populated with LLM prompt context and recommendations.
        """
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        iso_timestamp = datetime.now(timezone.utc).isoformat()
        max_severity = self._determine_highest_severity(result.defects)

        defect_entries: List[Dict[str, Any]] = [
            {
                "label": item.label,
                "confidence": item.confidence,
                "severity": item.severity.value,
                "bounding_box": item.bbox.model_dump(),
            }
            for item in result.defects
        ]

        telemetry_dict = (
            telemetry.model_dump()
            if telemetry
            else {
                "inference_time_ms": result.inference_time_ms,
                "status": result.status,
            }
        )

        urgency_map = {
            DefectSeverity.CRITICAL: "IMMEDIATE_ACTION_REQUIRED",
            DefectSeverity.HIGH: "ELEVATED_ATTENTION",
            DefectSeverity.MEDIUM: "STANDARD_REVIEW",
            DefectSeverity.LOW: "INFORMATIONAL",
        }
        urgency = urgency_map.get(max_severity, "STANDARD_REVIEW")

        # Generate action recommendations based on severity
        recommended_actions = []
        if max_severity == DefectSeverity.CRITICAL:
            recommended_actions.extend(
                [
                    "Halt affected production conveyor immediately.",
                    "Notify QA Lead and Maintenance Engineer.",
                    "Trigger automated safety interlock on station.",
                ]
            )
        elif max_severity == DefectSeverity.HIGH:
            recommended_actions.extend(
                [
                    "Divert flagged part to manual inspection bay.",
                    "Calibrate optical edge sensor camera.",
                ]
            )
        else:
            recommended_actions.append("Log defect occurrence in shift telemetry.")

        summary_text = (
            f"Incident {incident_id} detected in {self.facility_zone} at "
            f"{iso_timestamp}. Total defects: {len(result.defects)} with peak "
            f"severity '{max_severity.value}'. Latency: "
            f"{result.inference_time_ms} ms."
        )

        llm_context = {
            "system_instruction": (
                "You are the Vision Builder Autonomous Incident Specialist. "
                "Synthesize the visual defect telemetry, assess production risk, "
                "and draft an executive alert for the shift supervisor."
            ),
            "payload": {
                "incident_id": incident_id,
                "facility_zone": self.facility_zone,
                "timestamp": iso_timestamp,
                "urgency": urgency,
                "max_severity": max_severity.value,
                "defects": defect_entries,
                "telemetry": telemetry_dict,
                "operator_notes": operator_notes,
                "recommended_mitigation": recommended_actions,
            },
        }

        return IncidentReport(
            incident_id=incident_id,
            timestamp=iso_timestamp,
            facility_zone=self.facility_zone,
            urgency=urgency,
            defect_count=len(result.defects),
            max_severity=max_severity.value,
            detected_defects=defect_entries,
            telemetry_summary=telemetry_dict,
            llm_prompt_context=llm_context,
            summary_text=summary_text,
            recommended_actions=recommended_actions,
        )
