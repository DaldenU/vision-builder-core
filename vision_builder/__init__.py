"""Vision Builder Core Package.

Scientific edge computing, computer vision preprocessing, anomaly/defect
detection, telemetry tracking, and LLM incident reporting pipeline.
"""

from vision_builder.detector import (
    BoundingBox,
    DefectDetector,
    DefectItem,
    DetectionResult,
)
from vision_builder.preprocessor import ImagePreprocessor, PreprocessingConfig
from vision_builder.report_builder import IncidentReport, IncidentReportBuilder
from vision_builder.telemetry import TelemetryMetrics, TelemetryTracker

__version__ = "0.1.0"
__all__ = [
    "ImagePreprocessor",
    "PreprocessingConfig",
    "DefectDetector",
    "BoundingBox",
    "DefectItem",
    "DetectionResult",
    "TelemetryTracker",
    "TelemetryMetrics",
    "IncidentReportBuilder",
    "IncidentReport",
]
