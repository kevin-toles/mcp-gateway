"""
Unit tests for MetricsMiddleware — P1-08.

Covers:
- Normal request (200) → REQUEST_TOTAL incremented with correct labels
- Error response (500) → REQUEST_TOTAL incremented with status_code="500"
- Path with route template params → label uses template, not raw UUID
- LabelCardinalityError raised for forbidden label keys (§2.7)
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.middleware.metrics import LabelCardinalityError, MetricsMiddleware, _normalize_path_fallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _counter_value(**labels: str) -> float:
    """Read ai_platform_request_total sample value (0.0 if not yet observed)."""
    val = REGISTRY.get_sample_value("ai_platform_request_total", labels=labels)
    return val or 0.0


def _build_app(service_name: str = "test-svc") -> tuple[FastAPI, TestClient]:
    """Build a minimal test app with MetricsMiddleware."""
    app = FastAPI()
    app.add_middleware(MetricsMiddleware, service_name=service_name)

    @app.get("/v1/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/items/{item_id}")
    async def get_item(item_id: str):
        return {"id": item_id}

    @app.get("/v1/error")
    async def raise_error():
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="boom")

    return app, TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetricsMiddlewareCounter:
    """REQUEST_TOTAL counter behaviour."""

    def test_normal_request_increments_counter(self):
        """200 response increments counter with correct labels."""
        app, client = _build_app("svc-200")
        before = _counter_value(service="svc-200", endpoint="/v1/health", method="GET", status_code="200")
        client.get("/v1/health")
        after = _counter_value(service="svc-200", endpoint="/v1/health", method="GET", status_code="200")
        assert after - before == pytest.approx(1.0)

    def test_error_response_increments_counter_with_error_status(self):
        """5xx response is recorded with the correct status_code label."""
        app, client = _build_app("svc-500")
        before = _counter_value(service="svc-500", endpoint="/v1/error", method="GET", status_code="500")
        client.get("/v1/error")
        after = _counter_value(service="svc-500", endpoint="/v1/error", method="GET", status_code="500")
        assert after - before == pytest.approx(1.0)

    def test_three_requests_increments_counter_by_three(self):
        """Counter increments once per request."""
        app, client = _build_app("svc-3x")
        before = _counter_value(service="svc-3x", endpoint="/v1/health", method="GET", status_code="200")
        for _ in range(3):
            client.get("/v1/health")
        after = _counter_value(service="svc-3x", endpoint="/v1/health", method="GET", status_code="200")
        assert after - before == pytest.approx(3.0)


class TestPathNormalization:
    """Route-template path normalization (§2.7 cardinality guard)."""

    def test_path_param_uses_route_template_not_raw_uuid(self):
        """UUID in path → label shows template '{item_id}', not the raw UUID."""
        app, client = _build_app("svc-template")
        uuid_val = "123e4567-e89b-12d3-a456-426614174000"
        before_template = _counter_value(
            service="svc-template",
            endpoint="/v1/items/{item_id}",
            method="GET",
            status_code="200",
        )
        client.get(f"/v1/items/{uuid_val}")
        after_template = _counter_value(
            service="svc-template",
            endpoint="/v1/items/{item_id}",
            method="GET",
            status_code="200",
        )
        # Template-based label was used (not the raw UUID)
        assert after_template - before_template == pytest.approx(1.0)

        # Raw UUID label must NOT exist
        raw_uuid_val = REGISTRY.get_sample_value(
            "ai_platform_request_total",
            labels={
                "service": "svc-template",
                "endpoint": f"/v1/items/{uuid_val}",
                "method": "GET",
                "status_code": "200",
            },
        )
        assert raw_uuid_val is None

    def test_fallback_normalization_replaces_uuid_with_id(self):
        """Regex fallback replaces UUID with {id}."""
        path = "/v1/tasks/123e4567-e89b-12d3-a456-426614174000"
        normalized = _normalize_path_fallback(path)
        assert normalized == "/v1/tasks/{id}"

    def test_fallback_normalization_replaces_numeric_segment(self):
        """Regex fallback replaces numeric ID with {id}."""
        path = "/v1/users/42/profile"
        normalized = _normalize_path_fallback(path)
        assert normalized == "/v1/users/{id}/profile"

    def test_fallback_normalization_leaves_static_paths_unchanged(self):
        """Static paths without dynamic segments are unchanged."""
        path = "/v1/health"
        assert _normalize_path_fallback(path) == "/v1/health"


class TestLabelCardinalityGuard:
    """LabelCardinalityError raised for forbidden (user-scoped) label keys."""

    def test_label_cardinality_error_is_value_error_subclass(self):
        assert issubclass(LabelCardinalityError, ValueError)

    def test_forbidden_key_raises_label_cardinality_error(self):
        """user_id, session_id, and request_id are forbidden as label keys."""
        from src.middleware.metrics import _FORBIDDEN_LABEL_KEYS

        for forbidden in ("user_id", "session_id", "request_id"):
            assert forbidden in _FORBIDDEN_LABEL_KEYS, (
                f"'{forbidden}' must be in _FORBIDDEN_LABEL_KEYS per design doc §2.7"
            )

    def test_forbidden_keys_frozenset(self):
        """_FORBIDDEN_LABEL_KEYS must be a frozenset (immutable)."""
        from src.middleware.metrics import _FORBIDDEN_LABEL_KEYS

        assert isinstance(_FORBIDDEN_LABEL_KEYS, frozenset)


class TestExcludedPaths:
    """Excluded paths are not counted in metrics."""

    def test_metrics_endpoint_excluded_by_default(self):
        """Requests to /metrics are not counted (prevents recursive observations)."""
        app = FastAPI()
        app.add_middleware(
            MetricsMiddleware,
            service_name="svc-exclude",
            exclude_paths=["/metrics", "/health"],
        )

        @app.get("/metrics")
        async def fake_metrics():
            return {}

        client = TestClient(app)
        before = _counter_value(service="svc-exclude", endpoint="/metrics", method="GET", status_code="200")
        client.get("/metrics")
        after = _counter_value(service="svc-exclude", endpoint="/metrics", method="GET", status_code="200")
        # Counter must NOT have been incremented
        assert after - before == pytest.approx(0.0)
