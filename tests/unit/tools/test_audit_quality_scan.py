"""Tests for audit_quality_scan MCP tool — Phase 7.

Validates:
    - AuditQualityScanInput registered in _INPUT_MODELS
    - Schema has correct fields with sensible defaults
    - Dispatch route points to /v1/audit/quality on audit-service
    - Handler is registered in server TOOL_HANDLERS
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher


@pytest.fixture
def dispatcher() -> ToolDispatcher:
    return ToolDispatcher(Settings())


# ===========================================================================
# Schema registration
# ===========================================================================


class TestAuditQualityScanSchema:
    def test_registered_in_input_models(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        assert "audit_quality_scan" in _INPUT_MODELS

    def test_is_pydantic_base_model(self) -> None:
        from pydantic import BaseModel

        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_quality_scan"]
        assert issubclass(model, BaseModel)

    def test_has_code_field(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_quality_scan"]
        assert "code" in model.model_fields

    def test_code_is_required(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_quality_scan"]
        with pytest.raises(ValidationError):
            model()

    def test_has_language_field_defaulting_python(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_quality_scan"]
        assert "language" in model.model_fields
        instance = model(code="x=1")
        assert instance.language == "python"

    def test_has_severity_threshold_defaulting_info(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_quality_scan"]
        instance = model(code="x=1")
        assert instance.severity_threshold == "info"

    def test_has_include_antipatterns_defaulting_true(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_quality_scan"]
        instance = model(code="x=1")
        assert instance.include_antipatterns is True

    def test_has_rule_categories_defaulting_none(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_quality_scan"]
        instance = model(code="x=1")
        assert instance.rule_categories is None

    def test_rule_categories_accepts_list(self) -> None:
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_quality_scan"]
        instance = model(code="x=1", rule_categories=["security", "naming"])
        assert instance.rule_categories == ["security", "naming"]


# ===========================================================================
# Dispatch route
# ===========================================================================


class TestAuditQualityDispatchRoute:
    def test_route_registered(self, dispatcher) -> None:
        route = dispatcher.get_route("audit_quality_scan")
        assert route is not None

    def test_route_path_is_quality_endpoint(self, dispatcher) -> None:
        route = dispatcher.get_route("audit_quality_scan")
        assert route.path == "/v1/audit/quality"

    def test_route_base_url_is_audit_service(self, dispatcher) -> None:
        route = dispatcher.get_route("audit_quality_scan")
        assert route.base_url == "http://localhost:8084"


# ===========================================================================
# Handler registration
# ===========================================================================


class TestAuditQualityHandlerRegistration:
    def test_handler_in_tool_handlers(self) -> None:
        from src.server import _HANDLER_FACTORIES

        assert "audit_quality_scan" in _HANDLER_FACTORIES

    def test_handler_is_callable(self) -> None:
        from src.server import _HANDLER_FACTORIES

        assert callable(_HANDLER_FACTORIES["audit_quality_scan"])
