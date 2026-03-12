"""G2.5 (RED) — amve_extract_architecture MCP tool.

TDD coverage for AC-2.4:
  - Tool handler exists at src/tools/amve_extract_architecture.py
  - Registered in _INPUT_MODELS with AMVEExtractArchitectureInput schema
  - Dispatcher route: GET POST /v1/architecture/extract on AMVE (:8088)
  - Handler returns {snapshot_sha, record_count, source_path, extraction_time_ms}

Tasks: G2.5 (RED), G2.6 (GREEN)
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
# AC-2.4 — Tool registered in _INPUT_MODELS
# ---------------------------------------------------------------------------


class TestAmveExtractArchitectureRegistration:
    """AC-2.4: amve_extract_architecture is registered in the tool registry."""

    def test_tool_in_input_models(self):
        from src.tool_registry import _INPUT_MODELS

        assert "amve_extract_architecture" in _INPUT_MODELS, (
            "amve_extract_architecture is missing from _INPUT_MODELS"
        )

    def test_input_model_has_source_path_field(self):
        from src.models.schemas import AMVEExtractArchitectureInput

        assert "source_path" in AMVEExtractArchitectureInput.model_fields

    def test_source_path_required(self):
        from pydantic import ValidationError

        from src.models.schemas import AMVEExtractArchitectureInput

        with pytest.raises(ValidationError):
            AMVEExtractArchitectureInput()  # missing source_path

    def test_source_path_nonempty(self):
        from pydantic import ValidationError

        from src.models.schemas import AMVEExtractArchitectureInput

        with pytest.raises(ValidationError):
            AMVEExtractArchitectureInput(source_path="")


# ---------------------------------------------------------------------------
# AC-2.4 — Dispatch route
# ---------------------------------------------------------------------------


class TestAmveExtractArchitectureDispatchRoute:
    """AC-2.4: amve_extract_architecture dispatches to AMVE /v1/architecture/extract."""

    def test_route_registered_in_dispatcher(self, dispatcher):
        route = dispatcher.get_route("amve_extract_architecture")
        assert route is not None, "amve_extract_architecture has no dispatch route"

    def test_route_path(self, dispatcher):
        route = dispatcher.get_route("amve_extract_architecture")
        assert route.path == "/v1/architecture/extract"

    def test_route_base_url_is_amve(self, dispatcher):
        route = dispatcher.get_route("amve_extract_architecture")
        assert route.base_url == "http://localhost:8088"


# ---------------------------------------------------------------------------
# AC-2.4 — create_handler returns {snapshot_sha, record_count, source_path, extraction_time_ms}
# ---------------------------------------------------------------------------


class TestAmveExtractArchitectureHandler:
    """AC-2.4: Handler returns four fields from AMVE extract response."""

    def test_handler_factory_is_importable(self):
        """Handler module can be imported."""
        from src.tools import amve_extract_architecture  # noqa: F401

    def test_create_handler_returns_callable(self, dispatcher):
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_extract_architecture

        sanitizer = OutputSanitizer()
        handler = amve_extract_architecture.create_handler(dispatcher, sanitizer)
        import asyncio
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_handler_returns_snapshot_sha(self, dispatcher):
        """Handler includes snapshot_sha from AMVE response."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_extract_architecture

        amve_response = {
            "snapshot_sha": "abc123def456" + "0" * 52,
            "record_count": 12,
            "extraction_time_ms": 42.5,
        }

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=amve_response)

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        sanitizer = OutputSanitizer()
        handler = amve_extract_architecture.create_handler(dispatcher, sanitizer)

        result = await handler(source_path="/my/src")
        assert "snapshot_sha" in result

    @pytest.mark.asyncio
    async def test_handler_returns_record_count(self, dispatcher):
        """Handler includes record_count from AMVE response."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_extract_architecture

        amve_response = {
            "snapshot_sha": "a" * 64,
            "record_count": 7,
            "extraction_time_ms": 10.0,
        }

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=amve_response)

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        sanitizer = OutputSanitizer()
        handler = amve_extract_architecture.create_handler(dispatcher, sanitizer)

        result = await handler(source_path="/my/src")
        assert result.get("record_count") == 7 or "record_count" in result

    @pytest.mark.asyncio
    async def test_handler_includes_source_path(self, dispatcher):
        """Handler dispatches with source_path forwarded to AMVE."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_extract_architecture

        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "snapshot_sha": "b" * 64,
                    "record_count": 3,
                    "extraction_time_ms": 5.0,
                },
            )

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        sanitizer = OutputSanitizer()
        handler = amve_extract_architecture.create_handler(dispatcher, sanitizer)

        await handler(source_path="/target/path")
        assert "/target/path" in json.dumps(captured.get("body", {}))

    @pytest.mark.asyncio
    async def test_dispatch_sends_post_to_extract_endpoint(self, dispatcher):
        """Dispatcher sends POST to /v1/architecture/extract."""
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            return httpx.Response(
                200,
                json={
                    "snapshot_sha": "c" * 64,
                    "record_count": 1,
                    "extraction_time_ms": 1.0,
                },
            )

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch(
            "amve_extract_architecture",
            {"source_path": "/s"},
        )
        assert captured["url"] == "http://localhost:8088/v1/architecture/extract"
        assert captured["method"] == "POST"
