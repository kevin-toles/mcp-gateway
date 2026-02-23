"""AEI-18 tests — audit_dependency_assess MCP tool (RED phase)

TDD coverage for AC-AEI18.15:
  - audit_dependency_assess registered in _INPUT_MODELS (tool_registry)
  - AuditDependencyAssessInput has source_path, manifest_path, include_transitive fields
  - Dispatcher routes to /v1/audit/dependency on audit-service (:8084)
  - create_handler() returns a callable
  - Handler dispatches POST with correct payload
  - tools.yaml registration

Tasks: AEI18.15 (RED), AEI18.16 (GREEN)
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher


@pytest.fixture
def dispatcher():
    settings = Settings()
    return ToolDispatcher(settings)


# ---------------------------------------------------------------------------
# AC-AEI18.15a — Tool registered in _INPUT_MODELS
# ---------------------------------------------------------------------------


class TestAuditDependencyAssessRegistration:
    """audit_dependency_assess is registered in the tool registry."""

    def test_tool_in_input_models(self):
        from src.tool_registry import _INPUT_MODELS

        assert "audit_dependency_assess" in _INPUT_MODELS, "audit_dependency_assess is missing from _INPUT_MODELS"

    def test_input_model_is_pydantic_base_model(self):
        from pydantic import BaseModel

        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_dependency_assess"]
        assert issubclass(model, BaseModel), "audit_dependency_assess input model must be a Pydantic BaseModel"


# ---------------------------------------------------------------------------
# AC-AEI18.15b — AuditDependencyAssessInput schema
# ---------------------------------------------------------------------------


class TestAuditDependencyAssessInputSchema:
    """AuditDependencyAssessInput validates all required and optional fields."""

    def test_schema_has_source_path_field(self):
        from src.models.schemas import AuditDependencyAssessInput

        assert "source_path" in AuditDependencyAssessInput.model_fields

    def test_schema_has_manifest_path_field(self):
        from src.models.schemas import AuditDependencyAssessInput

        assert "manifest_path" in AuditDependencyAssessInput.model_fields

    def test_schema_has_include_transitive_field(self):
        from src.models.schemas import AuditDependencyAssessInput

        assert "include_transitive" in AuditDependencyAssessInput.model_fields

    def test_include_transitive_defaults_to_true(self):
        from src.models.schemas import AuditDependencyAssessInput

        obj = AuditDependencyAssessInput(source_path="/my/src")
        assert obj.include_transitive is True

    def test_manifest_path_defaults_to_none(self):
        from src.models.schemas import AuditDependencyAssessInput

        obj = AuditDependencyAssessInput(source_path="/my/src")
        assert obj.manifest_path is None

    def test_source_path_is_required(self):
        from pydantic import ValidationError

        from src.models.schemas import AuditDependencyAssessInput

        with pytest.raises(ValidationError):
            AuditDependencyAssessInput()  # missing source_path

    def test_source_path_must_not_be_empty(self):
        from pydantic import ValidationError

        from src.models.schemas import AuditDependencyAssessInput

        with pytest.raises(ValidationError):
            AuditDependencyAssessInput(source_path="")

    def test_valid_construction_all_fields(self):
        from src.models.schemas import AuditDependencyAssessInput

        obj = AuditDependencyAssessInput(
            source_path="/project/src",
            manifest_path="/project/pyproject.toml",
            include_transitive=False,
        )
        assert obj.source_path == "/project/src"
        assert obj.manifest_path == "/project/pyproject.toml"
        assert obj.include_transitive is False

    def test_valid_construction_source_only(self):
        from src.models.schemas import AuditDependencyAssessInput

        obj = AuditDependencyAssessInput(source_path="/project/src")
        assert obj.source_path == "/project/src"
        assert obj.manifest_path is None
        assert obj.include_transitive is True


# ---------------------------------------------------------------------------
# AC-AEI18.15c — Dispatcher routes to /v1/audit/dependency on audit-service
# ---------------------------------------------------------------------------


class TestAuditDependencyAssessDispatchRoute:
    """audit_dependency_assess dispatches to audit-service /v1/audit/dependency."""

    def test_route_registered_in_dispatcher(self, dispatcher):
        route = dispatcher.get_route("audit_dependency_assess")
        assert route is not None, "audit_dependency_assess has no dispatch route"

    def test_route_path_is_dependency(self, dispatcher):
        route = dispatcher.get_route("audit_dependency_assess")
        assert route.path == "/v1/audit/dependency"

    def test_route_base_url_is_audit_service(self, dispatcher):
        route = dispatcher.get_route("audit_dependency_assess")
        assert route.base_url == "http://localhost:8084"

    @pytest.mark.asyncio
    async def test_dispatch_sends_post_to_dependency_endpoint(self, dispatcher):
        """Dispatcher sends POST to /v1/audit/dependency."""
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch(
            "audit_dependency_assess",
            {
                "source_path": "/project/src",
                "manifest_path": None,
                "include_transitive": True,
            },
        )
        assert captured["url"] == "http://localhost:8084/v1/audit/dependency"
        assert captured["method"] == "POST"

    @pytest.mark.asyncio
    async def test_dispatch_includes_source_path_in_body(self, dispatcher):
        """Dispatcher includes source_path in POST body."""
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch(
            "audit_dependency_assess",
            {
                "source_path": "/my/project/src",
                "manifest_path": None,
                "include_transitive": True,
            },
        )
        assert captured["body"]["source_path"] == "/my/project/src"

    @pytest.mark.asyncio
    async def test_dispatch_includes_manifest_path_in_body(self, dispatcher):
        """Dispatcher includes manifest_path when provided."""
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch(
            "audit_dependency_assess",
            {
                "source_path": "/src",
                "manifest_path": "/pyproject.toml",
                "include_transitive": True,
            },
        )
        assert captured["body"]["manifest_path"] == "/pyproject.toml"

    @pytest.mark.asyncio
    async def test_dispatch_includes_include_transitive_false(self, dispatcher):
        """Dispatcher forwards include_transitive=False."""
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch(
            "audit_dependency_assess",
            {
                "source_path": "/src",
                "manifest_path": None,
                "include_transitive": False,
            },
        )
        assert captured["body"]["include_transitive"] is False


# ---------------------------------------------------------------------------
# AC-AEI18.15d — create_handler factory
# ---------------------------------------------------------------------------


class TestAuditDependencyAssessHandler:
    """create_handler returns a callable that dispatches correctly."""

    def test_create_handler_exists(self):
        from src.tools.audit_dependency_assess import create_handler

        assert callable(create_handler)

    def test_create_handler_returns_callable(self, dispatcher):
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.audit_dependency_assess import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_handler_sends_source_path(self, dispatcher):
        """Handler dispatches source_path to backend."""
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.audit_dependency_assess import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_path="/project/src")
        assert captured["body"]["source_path"] == "/project/src"

    @pytest.mark.asyncio
    async def test_handler_default_include_transitive_is_true(self, dispatcher):
        """Handler defaults include_transitive=True."""
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.audit_dependency_assess import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_path="/project/src")
        assert captured["body"].get("include_transitive", True) is True

    @pytest.mark.asyncio
    async def test_handler_default_manifest_path_is_none(self, dispatcher):
        """Handler defaults manifest_path=None."""
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.audit_dependency_assess import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_path="/project/src")
        assert captured["body"].get("manifest_path") is None

    @pytest.mark.asyncio
    async def test_handler_passes_manifest_path_when_given(self, dispatcher):
        """Handler passes manifest_path when explicitly provided."""
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.audit_dependency_assess import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(
            source_path="/project/src",
            manifest_path="/project/pyproject.toml",
        )
        assert captured["body"]["manifest_path"] == "/project/pyproject.toml"

    @pytest.mark.asyncio
    async def test_handler_passes_include_transitive_false(self, dispatcher):
        """Handler passes include_transitive=False when specified."""
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.audit_dependency_assess import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(
            source_path="/project/src",
            include_transitive=False,
        )
        assert captured["body"]["include_transitive"] is False


# ---------------------------------------------------------------------------
# AC-AEI18.15e — tools.yaml registration
# ---------------------------------------------------------------------------


class TestAuditDependencyAssessYamlRegistration:
    """audit_dependency_assess is registered in config/tools.yaml."""

    def test_tool_in_yaml_registry(self):
        """audit_dependency_assess must be loaded by ToolRegistry from tools.yaml."""
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        repo_root = Path(__file__).parent.parent.parent.parent
        tools_yaml = repo_root / "config" / "tools.yaml"
        registry = ToolRegistry(config_path=tools_yaml)
        tool_names = [t.name for t in registry.list_all()]
        assert "audit_dependency_assess" in tool_names, (
            f"audit_dependency_assess not found in tools.yaml registry. Found: {tool_names}"
        )

    def test_tool_description_mentions_dependency(self):
        """Tool description should mention dependency analysis."""
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        repo_root = Path(__file__).parent.parent.parent.parent
        tools_yaml = repo_root / "config" / "tools.yaml"
        registry = ToolRegistry(config_path=tools_yaml)
        tool = next(
            (t for t in registry.list_all() if t.name == "audit_dependency_assess"),
            None,
        )
        assert tool is not None
        assert tool.description  # non-empty
        desc_lower = tool.description.lower()
        assert "depend" in desc_lower, f"Description should mention dependency, got: {tool.description}"

    def test_tool_tier_is_bronze(self):
        """audit_dependency_assess should have bronze tier."""
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        repo_root = Path(__file__).parent.parent.parent.parent
        tools_yaml = repo_root / "config" / "tools.yaml"
        registry = ToolRegistry(config_path=tools_yaml)
        tool = next(
            (t for t in registry.list_all() if t.name == "audit_dependency_assess"),
            None,
        )
        assert tool is not None
        assert tool.tier == "bronze"
