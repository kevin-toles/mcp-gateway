"""Unit tests for the `find_code_pattern` facade handler — MCP-F-3 (RED phase).

TDD RED phase: all tests written before implementation.
They will FAIL until MCP-F-3.3 (GREEN) creates src/tools/find_code_pattern.py.

Acceptance Criteria covered:
- AC-MCP-F-3.1: find_code_pattern dispatches to "pattern_search" with pattern_type="all", limit=10
- AC-MCP-F-3.2: all 3 examples→pattern_type mappings correct
- AC-MCP-F-3.3: unknown examples raises ValueError before dispatch
- AC-MCP-F-3.4: handler exposes exactly 2 parameters (via inspect.signature)
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
# AC-MCP-F-3.1  Default dispatch — examples default → pattern_type="all", limit=10
# ─────────────────────────────────────────────────────────────────────────────


class TestFindCodePatternDefaultDispatch:
    """AC-MCP-F-3.1: default call dispatches with pattern_type="all" and limit=10."""

    @pytest.mark.asyncio
    async def test_default_examples_dispatches_pattern_type_all(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.find_code_pattern import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="singleton")

        assert captured.get("pattern_type") == "all"

    @pytest.mark.asyncio
    async def test_default_limit_is_10(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.find_code_pattern import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="singleton")

        assert captured.get("limit") == 10


# ─────────────────────────────────────────────────────────────────────────────
# AC-MCP-F-3.2  All 3 examples → pattern_type mappings
# ─────────────────────────────────────────────────────────────────────────────


class TestFindCodePatternExamplesMapping:
    """AC-MCP-F-3.2: all 3 examples values map to correct pattern_type."""

    @pytest.mark.asyncio
    async def test_good_maps_to_pattern_type_good(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.find_code_pattern import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="factory", examples="good")

        assert captured.get("pattern_type") == "good"

    @pytest.mark.asyncio
    async def test_bad_maps_to_pattern_type_bad(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.find_code_pattern import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="god class", examples="bad")

        assert captured.get("pattern_type") == "bad"

    @pytest.mark.asyncio
    async def test_both_maps_to_pattern_type_all(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.find_code_pattern import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="observer", examples="both")

        assert captured.get("pattern_type") == "all"


# ─────────────────────────────────────────────────────────────────────────────
# AC-MCP-F-3.3  Unknown examples raises ValueError before dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestFindCodePatternInvalidExamples:
    """AC-MCP-F-3.3: unknown examples raises ValueError; dispatcher is NOT called."""

    @pytest.mark.asyncio
    async def test_unknown_examples_raises_value_error(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.find_code_pattern import create_handler

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)

        with pytest.raises(ValueError, match="antipattern"):
            await handler(query="singleton", examples="antipattern")

    @pytest.mark.asyncio
    async def test_unknown_examples_does_not_call_dispatcher(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.find_code_pattern import create_handler

        called = {"count": 0}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            called["count"] += 1
            return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)

        with pytest.raises(ValueError):
            await handler(query="singleton", examples="antipattern")

        assert called["count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# AC-MCP-F-3.4  FastMCP schema surface — exactly 2 parameters (REFACTOR)
# ─────────────────────────────────────────────────────────────────────────────


class TestFindCodePatternHandlerSignature:
    """AC-MCP-F-3.4: handler exposes exactly 2 parameters to FastMCP schema generation."""

    def test_handler_has_exactly_2_parameters(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.find_code_pattern import create_handler

        handler = create_handler(dispatcher)
        sig = inspect.signature(handler)
        param_names = list(sig.parameters.keys())

        assert len(param_names) == 2, f"Expected 2 params, got: {param_names}"

    def test_handler_parameter_names_are_query_and_examples(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.find_code_pattern import create_handler

        handler = create_handler(dispatcher)
        sig = inspect.signature(handler)
        param_names = list(sig.parameters.keys())

        assert param_names == ["query", "examples"], f"Unexpected params: {param_names}"
