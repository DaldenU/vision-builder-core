"""Lightweight HTTP inference and health-check server for CD & Kubernetes deployments.

Exposes REST endpoints (/health, /infer) using Python's standard library http.server
so containerized Edge-to-Cloud CV-Ops nodes can be verified by Jenkins CD pipelines
and Kubernetes liveness/readiness probes without heavy web framework dependencies.
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

from vision_builder import (
    DefectDetector,
    ImagePreprocessor,
    IncidentReportBuilder,
    TelemetryTracker,
    __version__,
)
from vision_builder.cli import generate_synthetic_frame
from vision_builder.detector import DetectorConfig
from vision_builder.preprocessor import PreprocessingConfig

_PREPROCESSOR = ImagePreprocessor(PreprocessingConfig())
_DETECTOR = DefectDetector(DetectorConfig())
_TELEMETRY = TelemetryTracker()
_REPORT_BUILDER = IncidentReportBuilder("Edge_CD_Test_Node")


def build_health_payload() -> Dict[str, Any]:
    """Return service health and operational telemetry summary."""
    metrics = _TELEMETRY.get_metrics()
    return {
        "status": "healthy",
        "service": "vision-builder-core",
        "version": __version__,
        "total_frames_processed": metrics.total_frames,
        "throughput_fps": metrics.throughput_fps,
    }


def run_single_inference(inject_defect: bool = True) -> Dict[str, Any]:
    """Execute a single synthetic CV-Ops frame detection and LLM report synthesis."""
    frame = generate_synthetic_frame(inject_defect=inject_defect)
    _ = _PREPROCESSOR.process(frame)
    result = _DETECTOR.detect(frame, frame_id="cd_smoke_test_frame")
    _TELEMETRY.record_result(result)
    metrics = _TELEMETRY.get_metrics()
    report = _REPORT_BUILDER.build_report(result, telemetry=metrics)
    detection_dict = result.model_dump()
    detection_dict["is_defective"] = result.is_defective
    return {
        "detection": detection_dict,
        "telemetry": metrics.model_dump(),
        "incident_report": report.model_dump(),
    }


def dispatch_get_request(path: str) -> Tuple[int, Dict[str, Any]]:
    """Route GET request paths to their corresponding JSON response payloads."""
    if path in {"/", "/health", "/livez", "/readyz"}:
        return 200, build_health_payload()
    if path.startswith("/infer"):
        return 200, run_single_inference(inject_defect=True)
    return 404, {"error": "Not Found", "path": path}


class VisionBuilderRequestHandler(BaseHTTPRequestHandler):  # pragma: no cover
    """HTTP request handler for health checks and synthetic CV-Ops inference."""

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests for /health and /infer."""
        status_code, payload = dispatch_get_request(self.path)
        self._send_json(status_code, payload)

    def log_message(self, format_str: str, *args: Any) -> None:
        """Suppress noisy stderr logging during automated checks."""
        return


def create_server(
    host: str = "127.0.0.1", port: int = 8000
) -> Tuple[HTTPServer, int]:  # pragma: no cover
    """Instantiate an HTTPServer bound to (host, port)."""
    server = HTTPServer((host, port), VisionBuilderRequestHandler)
    actual_port = int(server.server_address[1])
    return server, actual_port


def serve_main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    """Start the HTTP health and inference server from CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Vision Builder Core HTTP Health & Inference Server"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    server, port = create_server(host=args.host, port=args.port)
    print(f"Vision Builder Core HTTP Server listening on http://{args.host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(serve_main())
