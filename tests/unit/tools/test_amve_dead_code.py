"""AEI-17 tests — amve_detect_dead_code MCP tool (RED phase)

TDD coverage for AC-AEI17.9 through AC-AEI17.11:
  - amve_detect_dead_code registered in _INPUT_MODELS (tool_registry)
  - AMVEDetectDeadCodeInput has source_path and include_unused_imports fields
  - Dispatcher routes to /v1/analysis/dead-code on AMVE service (:8088)
  - create_handler() returns a callable
  - Handler dispatches POST with correct payload

Tasks: AEI17.13 (RED), AEI17.14 (GREEN)
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
# AC-AEI17.9 — Tool registered in _INPUT_MODELS
# ---------------------------------------------------------------------------


class TestAmveDetectDeadCodeRegistration:
    """amve_detect_dead_code is registered in the tool registry."""

    def test_tool_in_input_models(self):
        from src.tool_registry import _INPUT_MODELS

        assert "amve_detect_dead_code" in _INPUT_MODELS, "amve_detect_dead_code is missing from _INPUT_MODELS"

    def test_input_model_is_pydantic_base_model(self):
        from pydantic import BaseModel

        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["amve_detect_dead_code"]
        assert issubclass(model, BaseModel), "amve_detect_dead_code input model must be a Pydantic BaseModel"


# ---------------------------------------------------------------------------
# AC-AEI17.10 — AMVEDetectDeadCodeInput schema
# ---------------------------------------------------------------------------


class TestAmveDetectDeadCodeInputSchema:
    """AMVEDetectDeadCodeInput validates source_path and include_unused_imports."""

    def test_schema_has_source_path_field(self):
        from src.models.schemas import AMVEDetectDeadCodeInput

        assert "source_path" in AMVEDetectDeadCodeInput.model_fields

    def test_schema_has_include_unused_imports_field(self):
        from src.models.schemas import AMVEDetectDeadCodeInput

        assert "include_unused_imports" in AMVEDetectDeadCodeInput.model_fields

    def test_include_unused_imports_defaults_to_true(self):
        from src.models.schemas import AMVEDetectDeadCodeInput

        instance = AMVEDetectDeadCodeInput(source_path="/src")
        assert instance.include_unused_imports is True

    def test_source_path_required(self):
        from pydantic import ValidationError

        from src.models.schemas import AMVEDetectDeadCodeInput

        with pytest.raises(ValidationError):
            AMVEDetectDeadCodeInput()  # missing source_path

    def test_source_path_must_not_be_empty(self):
        from pydantic import ValidationError

        from src.models.schemas import AMVEDetectDeadCodeInput

        with pytest.raises(ValidationError):
            AMVEDetectDeadCodeInput(source_path="")

    def test_valid_construction(self):
        from src.models.schemas import AMVEDetectDeadCodeInput

        obj = AMVEDetectDeadCodeInput(
            source_path="/my/src",
            include_unused_imports=False,
        )
        assert obj.source_path == "/my/src"
        assert obj.include_unused_imports is False


# ---------------------------------------------------------------------------
# AC-AEI17.11 — Dispatcher routes to /v1/analysis/dead-code
# ---------------------------------------------------------------------------


class TestAmveDetectDeadCodeDispatchRoute:
    """amve_detect_dead_code dispatches to AMVE /v1/analysis/dead-code."""

    def test_route_registered_in_dispatcher(self, dispatcher):
        route = dispatcher.get_route("amve_detect_dead_code")
        assert route is not None, "amve_detect_dead_code has no dispatch route"

    def test_route_path_is_dead_code(self, dispatcher):
        route = dispatcher.get_route("amve_detect_dead_code")
        assert route.path == "/v1/analysis/dead-code"

    def test_route_base_url_is_amve(self, dispatcher):
        route = dispatcher.get_route("amve_detect_dead_code")
        assert route.base_url == "http://localhost:8088"

    @pytest.mark.asyncio
    async def test_dispatch_sends_post_to_dead_code_endpoint(self, dispatcher):
        """Dispatcher sends POST to /v1/analysis/dead-code."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch(
            "amve_detect_dead_code",
            {"source_path": "/src", "include_unused_imports": True},
        )
        assert captured["url"] == "http://localhost:8088/v1/analysis/dead-code"
        assert captured["method"] == "POST"

    @pytest.mark.asyncio
    async def test_dispatch_includes_source_path_in_body(self, dispatcher):
        """Dispatcher includes source_path in POST body."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch(
            "amve_detect_dead_code",
            {"source_path": "/my/src", "include_unused_imports": True},
        )
        assert captured["body"]["source_path"] == "/my/src"

    @pytest.mark.asyncio
    async def test_dispatch_includes_include_unused_imports_in_body(self, dispatcher):
        """Dispatcher includes include_unused_imports in POST body."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch(
            "amve_detect_dead_code",
            {"source_path": "/my/src", "include_unused_imports": False},
        )
        assert captured["body"]["include_unused_imports"] is False


# ---------------------------------------------------------------------------
# AC-AEI17.12 — create_handler factory
# ---------------------------------------------------------------------------


class TestAmveDetectDeadCodeHandler:
    """create_handler returns a callable that dispatches correctly."""

    def test_create_handler_exists(self):
        from src.tools.amve_detect_dead_code import create_handler

        assert callable(create_handler)

    def test_create_handler_returns_callable(self, dispatcher):
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_dead_code import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_handler_sends_source_path(self, dispatcher):
        """Handler dispatches source_path to backend."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_dead_code import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_path="/my/source")
        assert captured["body"]["source_path"] == "/my/source"

    @pytest.mark.asyncio
    async def test_handler_sends_include_unused_imports_true_by_default(self, dispatcher):
        """Handler defaults include_unused_imports=True."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_dead_code import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_path="/my/source")
        assert captured["body"].get("include_unused_imports", True) is True

    @pytest.mark.asyncio
    async def test_handler_sends_include_unused_imports_false(self, dispatcher):
        """Handler passes include_unused_imports=False when specified."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_dead_code import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_path="/my/source", include_unused_imports=False)
        assert captured["body"]["include_unused_imports"] is False


# ---------------------------------------------------------------------------
# AC-AEI17.13 — tools.yaml registration
# ---------------------------------------------------------------------------


class TestAmveDetectDeadCodeYamlRegistration:
    """amve_detect_dead_code is registered in config/tools.yaml."""

    def test_tool_in_yaml_registry(self):
        """amve_detect_dead_code must be loaded by ToolRegistry from tools.yaml."""
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        # tools.yaml is relative to the mcp-gateway root
        repo_root = Path(__file__).parent.parent.parent.parent
        tools_yaml = repo_root / "config" / "tools.yaml"
        registry = ToolRegistry(config_path=tools_yaml)
        tool_names = [t.name for t in registry.list_all()]
        assert "amve_detect_dead_code" in tool_names, (
            f"amve_detect_dead_code not found in tools.yaml registry. Found: {tool_names}"
        )

    def test_tool_description_mentions_dead_code(self):
        """Tool description should mention dead code."""
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        repo_root = Path(__file__).parent.parent.parent.parent
        tools_yaml = repo_root / "config" / "tools.yaml"
        registry = ToolRegistry(config_path=tools_yaml)
        tool = next((t for t in registry.list_all() if t.name == "amve_detect_dead_code"), None)
        assert tool is not None
        assert tool.description  # non-empty
        desc_lower = tool.description.lower()
        assert "dead" in desc_lower or "unused" in desc_lower, (
            f"Description should mention dead code or unused code, got: {tool.description}"
        )
