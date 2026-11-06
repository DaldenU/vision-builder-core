"""Unit tests for the lightweight CD/Kubernetes HTTP health & inference server."""

from vision_builder.server import (
    build_health_payload,
    dispatch_get_request,
    run_single_inference,
)


def test_build_health_payload() -> None:
    payload = build_health_payload()
    assert payload["status"] == "healthy"
    assert payload["service"] == "vision-builder-core"
    assert "version" in payload
    assert "total_frames_processed" in payload


def test_run_single_inference() -> None:
    result = run_single_inference(inject_defect=True)
    assert "detection" in result
    assert "telemetry" in result
    assert "incident_report" in result
    assert result["detection"]["is_defective"] is True
    assert result["detection"]["status"] == "DEFECT_DETECTED"


def test_dispatch_get_request_routes() -> None:
    for health_path in ("/", "/health", "/livez", "/readyz"):
        code, data = dispatch_get_request(health_path)
        assert code == 200
        assert data["status"] == "healthy"

    code_infer, infer_data = dispatch_get_request("/infer")
    assert code_infer == 200
    assert infer_data["detection"]["is_defective"] is True

    code_404, err_data = dispatch_get_request("/unknown")
    assert code_404 == 404
    assert err_data["error"] == "Not Found"
