"""
RED tests for Audit MCP Tool Registration — WBS-AEI13

TDD coverage for:
  AC-AEI13.3  AUDIT_SERVICE_URL already in Settings (http://localhost:8084)
  AC-AEI13.4  audit_security_scan / audit_code_metrics / audit_corpus_search
              — registered in _INPUT_MODELS, tools.yaml, dispatch routes

Tasks: AEI13.7 (RED), AEI13.8 (GREEN)
"""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher


@pytest.fixture
def dispatcher() -> ToolDispatcher:
    return ToolDispatcher(Settings())


AUDIT_TOOL_NAMES = {
    "audit_security_scan",
    "audit_code_metrics",
    "audit_corpus_search",
}


# ===========================================================================
# AC-AEI13.3 — AUDIT_SERVICE_URL in Settings
# ===========================================================================


class TestAuditServiceURLConfig:
    """AUDIT_SERVICE_URL must default to http://localhost:8084."""

    def test_audit_service_url_default(self) -> None:
        settings = Settings()
        assert settings.AUDIT_SERVICE_URL == "http://localhost:8084"

    def test_audit_service_url_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("MCP_GATEWAY_AUDIT_SERVICE_URL", "http://remote-audit:9999")
        settings = Settings()
        assert settings.AUDIT_SERVICE_URL == "http://remote-audit:9999"

    def test_audit_service_url_used_in_dispatch_routes(self, dispatcher) -> None:
        """At least one audit tool must route to AUDIT_SERVICE_URL."""
        route = dispatcher.get_route("audit_security_scan")
        assert route is not None
        assert route.base_url == "http://localhost:8084"


# ===========================================================================
# AC-AEI13.4 — Tool Registration in _INPUT_MODELS
# ===========================================================================


class TestAuditToolRegistration:
    """All 3 audit_* tools must be registered in _INPUT_MODELS."""

    def test_all_audit_tools_in_input_models(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        for name in AUDIT_TOOL_NAMES:
            assert name in _INPUT_MODELS, f"{name} missing from _INPUT_MODELS"

    def test_audit_input_models_are_pydantic(self) -> None:
        from pydantic import BaseModel

        from src.tool_registry import _INPUT_MODELS

        for name in AUDIT_TOOL_NAMES:
            model = _INPUT_MODELS[name]
            assert issubclass(model, BaseModel), f"{name} model is not a BaseModel subclass"


# ===========================================================================
# AC-AEI13.4 — Schema field validation
# ===========================================================================


class TestAuditInputSchemas:
    """Each tool's Pydantic schema must expose the correct required fields."""

    def test_security_scan_has_code_field(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_security_scan"]
        assert "code" in model.model_fields

    def test_security_scan_has_language_field(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_security_scan"]
        assert "language" in model.model_fields

    def test_code_metrics_has_code_field(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_code_metrics"]
        assert "code" in model.model_fields

    def test_code_metrics_has_pillars_field(self) -> None:
        """audit_code_metrics must expose 'pillars' to drive per-pillar dispatch."""
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_code_metrics"]
        assert "pillars" in model.model_fields

    def test_code_metrics_pillars_default_all_three(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_code_metrics"]
        instance = model(code="x=1")
        assert set(instance.pillars) == {"structural", "architectural", "eloquence"}

    def test_corpus_search_has_query_field(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_corpus_search"]
        assert "query" in model.model_fields

    def test_corpus_search_query_is_required(self) -> None:
        from pydantic import ValidationError

        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_corpus_search"]
        with pytest.raises(ValidationError):
            model()  # missing 'query'

    def test_corpus_search_has_top_k_field(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_corpus_search"]
        assert "top_k" in model.model_fields

    def test_corpus_search_has_threshold_field(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_corpus_search"]
        assert "threshold" in model.model_fields


# ===========================================================================
# AC-AEI13.4 — Dispatch routes point to correct audit paths
# ===========================================================================


class TestAuditDispatchRoutes:
    """Each audit tool must dispatch to the correct audit-service endpoint."""

    def test_security_scan_route_path(self, dispatcher) -> None:
        route = dispatcher.get_route("audit_security_scan")
        assert route is not None
        assert route.path == "/v1/audit/security"
        assert route.base_url == "http://localhost:8084"

    def test_code_metrics_route_path(self, dispatcher) -> None:
        route = dispatcher.get_route("audit_code_metrics")
        assert route is not None
        assert route.path == "/v1/audit/metrics"
        assert route.base_url == "http://localhost:8084"

    def test_corpus_search_route_path(self, dispatcher) -> None:
        route = dispatcher.get_route("audit_corpus_search")
        assert route is not None
        assert route.path == "/v1/audit/corpus"
        assert route.base_url == "http://localhost:8084"
