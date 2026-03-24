"""Unit tests for the `search_in` facade handler — MCP-F-2 (RED phase).

TDD RED phase: all tests written before implementation.
They will FAIL until MCP-F-2.3 (GREEN) creates src/tools/search_in.py.

Acceptance Criteria covered:
- AC-MCP-F-2.1: search_in dispatches to "knowledge_refine" with correct collection/mmr_rerank/limit
- AC-MCP-F-2.2: all 4 source→collection mappings correct
- AC-MCP-F-2.3: unknown source raises ValueError before dispatch
- AC-MCP-F-2.4: handler exposes exactly 3 parameters (via inspect.signature)
"""

from __future__ import annotations

import inspect

import httpx
import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def dispatcher() -> ToolDispatcher:
    return ToolDispatcher(Settings())


# ─────────────────────────────────────────────────────────────────────────────
# AC-MCP-F-2.1  Default dispatch — textbooks → chapters, mmr_rerank=True, limit=5
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchInDefaultDispatch:
    """AC-MCP-F-2.1: search_in dispatches to knowledge_refine with correct params."""

    @pytest.mark.asyncio
    async def test_textbooks_maps_to_collection_chapters(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.search_in import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="design patterns", source="textbooks")

        assert captured.get("collection") == "chapters"

    @pytest.mark.asyncio
    async def test_mmr_rerank_is_true(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.search_in import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="design patterns", source="textbooks")

        assert captured.get("mmr_rerank") is True

    @pytest.mark.asyncio
    async def test_default_limit_is_5(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.search_in import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="design patterns", source="textbooks")

        assert captured.get("limit") == 5

    @pytest.mark.asyncio
    async def test_max_results_overrides_limit(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.search_in import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="design patterns", source="textbooks", max_results=12)

        assert captured.get("limit") == 12


# ─────────────────────────────────────────────────────────────────────────────
# AC-MCP-F-2.2  All 4 source → collection mappings
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchInSourceMapping:
    """AC-MCP-F-2.2: all 4 source values map to correct internal collections."""

    @pytest.mark.asyncio
    async def test_code_maps_to_code_chunks(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.search_in import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="singleton", source="code")

        assert captured.get("collection") == "code_chunks"

    @pytest.mark.asyncio
    async def test_patterns_maps_to_code_good_patterns(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.search_in import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="factory pattern", source="patterns")

        assert captured.get("collection") == "code_good_patterns"

    @pytest.mark.asyncio
    async def test_diagrams_maps_to_ascii_diagrams(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.search_in import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="microservices architecture", source="diagrams")

        assert captured.get("collection") == "ascii_diagrams"


# ─────────────────────────────────────────────────────────────────────────────
# AC-MCP-F-2.3  Unknown source raises ValueError before dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchInInvalidSource:
    """AC-MCP-F-2.3: Unknown source raises ValueError; dispatcher is NOT called."""

    @pytest.mark.asyncio
    async def test_unknown_source_raises_value_error(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.search_in import create_handler

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)

        with pytest.raises(ValueError, match="books"):
            await handler(query="design patterns", source="books")

    @pytest.mark.asyncio
    async def test_unknown_source_does_not_call_dispatcher(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.search_in import create_handler

        called = {"count": 0}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            called["count"] += 1
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)

        with pytest.raises(ValueError):
            await handler(query="design patterns", source="books")

        assert called["count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# AC-MCP-F-2.4  FastMCP schema surface — exactly 3 parameters (REFACTOR)
# ─────────────────────────────────────────────────────────────────────────────


class TestSearchInHandlerSignature:
    """AC-MCP-F-2.4: handler exposes exactly 3 parameters to FastMCP schema generation."""

    def test_handler_has_exactly_3_parameters(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.search_in import create_handler

        handler = create_handler(dispatcher)
        sig = inspect.signature(handler)
        param_names = list(sig.parameters.keys())

        assert len(param_names) == 3, f"Expected 3 params, got: {param_names}"

    def test_handler_parameter_names_are_query_source_max_results(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.search_in import create_handler

        handler = create_handler(dispatcher)
        sig = inspect.signature(handler)
        param_names = list(sig.parameters.keys())

        assert param_names == ["query", "source", "max_results"], f"Unexpected params: {param_names}"
