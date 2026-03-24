"""Unit tests for the `ask` facade handler — MCP-F-1 (RED phase).

TDD RED phase: all tests in this file are written before the implementation.
They will FAIL until MCP-F-1.4 (GREEN) creates src/tools/ask.py.

Acceptance Criteria covered:
- AC-MCP-F-1.1: ask dispatches to "hybrid_search" with correct hardcoded internals
- AC-MCP-F-1.2: difficulty mapping → bloom_tier_filter
- AC-MCP-F-1.3: unknown difficulty raises ValueError before dispatch
- AC-MCP-F-1.4: FastMCP schema surface exposes exactly 3 params (via inspect.signature)
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
# AC-MCP-F-1.1  Default dispatch payload
# ─────────────────────────────────────────────────────────────────────────────


class TestAskDefaultDispatch:
    """AC-MCP-F-1.1: ask dispatches to hybrid_search with correct hardcoded internals."""

    @pytest.mark.asyncio
    async def test_ask_dispatches_correct_top_k(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="binary search", max_results=5)

        assert captured.get("limit") == 5

    @pytest.mark.asyncio
    async def test_ask_dispatches_expand_taxonomy_true(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="binary search", max_results=5)

        assert captured.get("expand_taxonomy") is True

    @pytest.mark.asyncio
    async def test_ask_dispatches_mmr_rerank_false(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="binary search", max_results=5)

        assert captured.get("mmr_rerank") is False

    @pytest.mark.asyncio
    async def test_ask_dispatches_bloom_tier_filter_none_by_default(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="binary search", max_results=5)

        assert "bloom_tier_filter" not in captured


# ─────────────────────────────────────────────────────────────────────────────
# AC-MCP-F-1.2  Difficulty → Bloom tier mapping
# ─────────────────────────────────────────────────────────────────────────────


class TestAskDifficultyMapping:
    """AC-MCP-F-1.2: difficulty parameter maps to bloom_tier_filter correctly."""

    @pytest.mark.asyncio
    async def test_beginner_maps_to_bloom_tier_0_1_2(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="recursion", difficulty="beginner")

        assert captured.get("bloom_tier_filter") == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_intermediate_maps_to_bloom_tier_3_4(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="recursion", difficulty="intermediate")

        assert captured.get("bloom_tier_filter") == [3, 4]

    @pytest.mark.asyncio
    async def test_advanced_maps_to_bloom_tier_4_5_6(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="recursion", difficulty="advanced")

        assert captured.get("bloom_tier_filter") == [4, 5, 6]

    @pytest.mark.asyncio
    async def test_none_difficulty_omits_bloom_tier_filter(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            import json

            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)
        await handler(query="recursion", difficulty=None)

        assert "bloom_tier_filter" not in captured


# ─────────────────────────────────────────────────────────────────────────────
# AC-MCP-F-1.3  Unknown difficulty raises ValueError before dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestAskInvalidDifficulty:
    """AC-MCP-F-1.3: Unknown difficulty raises ValueError; dispatcher is NOT called."""

    @pytest.mark.asyncio
    async def test_unknown_difficulty_raises_value_error(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        called = {"count": 0}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            called["count"] += 1
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)

        with pytest.raises(ValueError, match="expert"):
            await handler(query="recursion", difficulty="expert")

    @pytest.mark.asyncio
    async def test_unknown_difficulty_does_not_call_dispatcher(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        called = {"count": 0}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            called["count"] += 1
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher)

        with pytest.raises(ValueError):
            await handler(query="recursion", difficulty="expert")

        assert called["count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# AC-MCP-F-1.4  FastMCP schema surface — exactly 3 parameters (REFACTOR)
# ─────────────────────────────────────────────────────────────────────────────


class TestAskHandlerSignature:
    """AC-MCP-F-1.4: handler exposes exactly 3 parameters to FastMCP schema generation."""

    def test_handler_has_exactly_3_parameters(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        handler = create_handler(dispatcher)
        sig = inspect.signature(handler)
        param_names = list(sig.parameters.keys())

        assert len(param_names) == 3, f"Expected 3 params, got: {param_names}"

    def test_handler_parameter_names_are_query_max_results_difficulty(self, dispatcher: ToolDispatcher) -> None:
        from src.tools.ask import create_handler

        handler = create_handler(dispatcher)
        sig = inspect.signature(handler)
        param_names = list(sig.parameters.keys())

        assert param_names == ["query", "max_results", "difficulty"], f"Unexpected params: {param_names}"
