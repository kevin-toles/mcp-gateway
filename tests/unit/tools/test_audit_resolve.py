"""AEI-20 tests — audit_resolve_lookup MCP tool (RED phase).

TDD coverage for AC-AEI20.4:
  - audit_resolve_lookup registered in _INPUT_MODELS (tool_registry)
  - AuditResolveLookupInput has violation_type, pillar, include_code_examples fields
  - Dispatcher routes to /v1/audit/resolve on audit-service (:8084)
  - create_handler() returns a callable
  - Handler dispatches POST with correct payload
  - tools.yaml registration

Tasks: AEI20.7 (RED), AEI20.8 (GREEN)
"""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher


@pytest.fixture
def dispatcher():
    settings = Settings()
    return ToolDispatcher(settings)


# ---------------------------------------------------------------------------
# AC-AEI20.4a — Tool registered in _INPUT_MODELS
# ---------------------------------------------------------------------------


class TestAuditResolveLookupRegistration:
    """audit_resolve_lookup is registered in the tool registry."""

    def test_tool_in_input_models(self):
        from src.tool_registry import _INPUT_MODELS

        assert "audit_resolve_lookup" in _INPUT_MODELS, "audit_resolve_lookup is missing from _INPUT_MODELS"

    def test_input_model_is_pydantic_base_model(self):
        from pydantic import BaseModel

        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_resolve_lookup"]
        assert issubclass(model, BaseModel), "audit_resolve_lookup input model must be a Pydantic BaseModel"


# ---------------------------------------------------------------------------
# AC-AEI20.4b — AuditResolveLookupInput schema
# ---------------------------------------------------------------------------


class TestAuditResolveLookupInputSchema:
    """AuditResolveLookupInput validates all required and optional fields."""

    def test_schema_has_violation_type_field(self):
        from src.models.schemas import AuditResolveLookupInput

        assert "violation_type" in AuditResolveLookupInput.model_fields

    def test_schema_has_pillar_field(self):
        from src.models.schemas import AuditResolveLookupInput

        assert "pillar" in AuditResolveLookupInput.model_fields

    def test_schema_has_include_code_examples_field(self):
        from src.models.schemas import AuditResolveLookupInput

        assert "include_code_examples" in AuditResolveLookupInput.model_fields

    def test_pillar_defaults_to_none(self):
        from src.models.schemas import AuditResolveLookupInput

        obj = AuditResolveLookupInput(violation_type="sql_injection")
        assert obj.pillar is None

    def test_include_code_examples_defaults_to_true(self):
        from src.models.schemas import AuditResolveLookupInput

        obj = AuditResolveLookupInput(violation_type="sql_injection")
        assert obj.include_code_examples is True

    def test_violation_type_is_required(self):
        from pydantic import ValidationError

        from src.models.schemas import AuditResolveLookupInput

        with pytest.raises(ValidationError):
            AuditResolveLookupInput()  # type: ignore[call-arg]

    def test_full_construction(self):
        from src.models.schemas import AuditResolveLookupInput

        obj = AuditResolveLookupInput(
            violation_type="DEP_LOW_RATIO",
            pillar="dependency",
            include_code_examples=False,
        )
        assert obj.violation_type == "DEP_LOW_RATIO"
        assert obj.pillar == "dependency"
        assert obj.include_code_examples is False


# ---------------------------------------------------------------------------
# AC-AEI20.4c — Dispatcher routes to /v1/audit/resolve
# ---------------------------------------------------------------------------


class TestAuditResolveLookupDispatch:
    """audit_resolve_lookup dispatches to /v1/audit/resolve on audit-service."""

    def test_tool_routes_to_audit_service(self, dispatcher: ToolDispatcher):
        """Tool is mapped to 'audit-service' in the dispatch table."""
        from src.tool_dispatcher import _TOOL_SERVICE_NAMES

        assert _TOOL_SERVICE_NAMES.get("audit_resolve_lookup") == "audit-service", (
            "audit_resolve_lookup must route to audit-service"
        )

    def test_dispatch_route_has_correct_path(self, dispatcher: ToolDispatcher):
        """Dispatch route path is /v1/audit/resolve."""
        routes = dispatcher.routes  # public attribute
        route = routes.get("audit_resolve_lookup")
        assert route is not None, "No route found for audit_resolve_lookup"
        assert route.path == "/v1/audit/resolve", f"Expected path '/v1/audit/resolve', got '{route.path}'"


# ---------------------------------------------------------------------------
# AC-AEI20.4d — create_handler returns callable
# ---------------------------------------------------------------------------


class TestAuditResolveLookupHandler:
    """audit_resolve_lookup tool handler can be created and called."""

    def test_create_handler_returns_callable(self, dispatcher: ToolDispatcher):
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import audit_resolve_lookup

        sanitizer = OutputSanitizer()
        handler = audit_resolve_lookup.create_handler(dispatcher, sanitizer)
        assert callable(handler)

    def test_handler_is_async(self, dispatcher: ToolDispatcher):
        import inspect

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import audit_resolve_lookup

        sanitizer = OutputSanitizer()
        handler = audit_resolve_lookup.create_handler(dispatcher, sanitizer)
        assert inspect.iscoroutinefunction(handler)

    def test_handler_function_name_matches_tool_name(self, dispatcher: ToolDispatcher):
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import audit_resolve_lookup

        sanitizer = OutputSanitizer()
        handler = audit_resolve_lookup.create_handler(dispatcher, sanitizer)
        assert handler.__name__ == "audit_resolve_lookup"


# ---------------------------------------------------------------------------
# AC-AEI20.4e — tools.yaml registration
# ---------------------------------------------------------------------------


class TestAuditResolveLookupYamlConfig:
    """audit_resolve_lookup appears in tools.yaml configuration."""

    def test_tool_in_tools_yaml(self):
        from pathlib import Path

        import yaml

        config_path = Path(__file__).parents[3] / "config" / "tools.yaml"
        assert config_path.exists(), f"tools.yaml not found at {config_path}"

        with config_path.open() as f:
            data = yaml.safe_load(f)

        tool_names = [t["name"] for t in data.get("tools", [])]
        assert "audit_resolve_lookup" in tool_names, (
            f"audit_resolve_lookup not found in tools.yaml. Found: {tool_names}"
        )

    def test_tool_yaml_has_audit_tag(self):
        from pathlib import Path

        import yaml

        config_path = Path(__file__).parents[3] / "config" / "tools.yaml"
        with config_path.open() as f:
            data = yaml.safe_load(f)

        tool = next(
            (t for t in data.get("tools", []) if t["name"] == "audit_resolve_lookup"),
            None,
        )
        assert tool is not None
        tags = tool.get("tags", [])
        assert "audit" in tags, f"Expected 'audit' tag, got {tags}"
