"""Tests for ToolDispatcher — WBS-MCP2 (RED).

Covers all 5 Acceptance Criteria:
- AC-2.1: Dispatches HTTP POST to correct backend for each of 9 tools
- AC-2.2: Configurable timeout per tool (default 30s)
- AC-2.3: Structured DispatchResult with status_code, headers, body, elapsed_ms
- AC-2.4: Connection errors raise BackendUnavailableError with service name
- AC-2.5: Timeout errors raise ToolTimeoutError with tool name and timeout

Technical debt: httpx responses are mocked; resolved in WBS-MCP9.
"""

import httpx
import pytest

from src.core.errors import BackendUnavailableError, ToolTimeoutError

# ── Route table expectations ────────────────────────────────────────────
EXPECTED_ROUTES = {
    "semantic_search": {"base_url": "http://localhost:8081", "path": "/v1/search"},
    "hybrid_search": {"base_url": "http://localhost:8081", "path": "/v1/search/hybrid"},
    "code_analyze": {"base_url": "http://localhost:8084", "path": "/v1/patterns/detect"},
    "code_pattern_audit": {"base_url": "http://localhost:8084", "path": "/v1/patterns/detect"},
    "graph_query": {"base_url": "http://localhost:8081", "path": "/v1/graph/query"},
    "llm_complete": {"base_url": "http://localhost:8080", "path": "/v1/chat/completions"},
    "a2a_send_message": {"base_url": "http://localhost:8082", "path": "/a2a/v1/message:send"},
    "a2a_get_task": {"base_url": "http://localhost:8082", "path": "/a2a/v1/tasks"},
    "a2a_cancel_task": {"base_url": "http://localhost:8082", "path": "/a2a/v1/tasks"},
}


@pytest.fixture
def dispatcher():
    """Create a ToolDispatcher with default settings."""
    from src.core.config import Settings
    from src.tool_dispatcher import ToolDispatcher

    settings = Settings()
    return ToolDispatcher(settings)


# ═══════════════════════════════════════════════════════════════════════
# AC-2.1: ToolDispatcher dispatches HTTP POST to correct backend
# ═══════════════════════════════════════════════════════════════════════


class TestRouteTable:
    """AC-2.1: ToolDispatcher has correct routes for all 9 tools."""

    def test_has_route_for_all_nine_tools(self, dispatcher):
        for tool_name in EXPECTED_ROUTES:
            route = dispatcher.get_route(tool_name)
            assert route is not None, f"Missing route for {tool_name}"

    def test_unknown_tool_returns_none(self, dispatcher):
        assert dispatcher.get_route("nonexistent_tool") is None

    @pytest.mark.parametrize("tool_name,expected", list(EXPECTED_ROUTES.items()))
    def test_route_base_url(self, dispatcher, tool_name, expected):
        route = dispatcher.get_route(tool_name)
        assert route.base_url == expected["base_url"], (
            f"{tool_name}: expected base_url={expected['base_url']}, got {route.base_url}"
        )

    @pytest.mark.parametrize("tool_name,expected", list(EXPECTED_ROUTES.items()))
    def test_route_path(self, dispatcher, tool_name, expected):
        route = dispatcher.get_route(tool_name)
        assert route.path == expected["path"], f"{tool_name}: expected path={expected['path']}, got {route.path}"

    def test_total_route_count_is_15(self, dispatcher):
        assert len(dispatcher.routes) == 15


class TestDispatchRouting:
    """AC-2.1: Dispatch sends HTTP POST to the correct URL."""

    @pytest.mark.asyncio
    async def test_semantic_search_dispatches_to_correct_url(self, dispatcher):
        """First tool smoke-test: semantic_search → :8081/v1/search."""
        captured_request = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_request["url"] = str(request.url)
            captured_request["method"] = request.method
            return httpx.Response(200, json={"results": []})

        transport = httpx.MockTransport(mock_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        await dispatcher.dispatch("semantic_search", {"query": "test"})
        assert captured_request["url"] == "http://localhost:8081/v1/search"
        assert captured_request["method"] == "POST"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_name,expected", list(EXPECTED_ROUTES.items()))
    async def test_all_tools_dispatch_to_correct_url(self, dispatcher, tool_name, expected):
        """AC-2.1: Verify every tool dispatches to the right backend+path."""
        captured_request = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_request["url"] = str(request.url)
            captured_request["method"] = request.method
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(mock_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        await dispatcher.dispatch(tool_name, {"input": "data"})
        expected_url = f"{expected['base_url']}{expected['path']}"
        assert captured_request["url"] == expected_url
        assert captured_request["method"] == "POST"

    @pytest.mark.asyncio
    async def test_dispatch_sends_payload_as_json_body(self, dispatcher):
        """Verify the input dict is sent as JSON POST body."""
        captured_body = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(mock_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        payload = {"query": "hello", "top_k": 5}
        await dispatcher.dispatch("semantic_search", payload)
        assert captured_body == payload

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool_raises_value_error(self, dispatcher):
        with pytest.raises(ValueError, match="Unknown tool"):
            await dispatcher.dispatch("nonexistent_tool", {})


# ═══════════════════════════════════════════════════════════════════════
# AC-2.2 / AC-2.5: Configurable timeout, ToolTimeoutError
# ═══════════════════════════════════════════════════════════════════════


class TestTimeout:
    """AC-2.2: Per-tool timeout; AC-2.5: ToolTimeoutError on timeout."""

    def test_default_timeout_is_30s(self, dispatcher):
        route = dispatcher.get_route("semantic_search")
        assert route.timeout == 30.0

    def test_custom_timeout_on_route(self):
        from src.tool_dispatcher import DispatchRoute

        route = DispatchRoute(base_url="http://localhost:8080", path="/v1/test", timeout=60.0)
        assert route.timeout == 60.0

    @pytest.mark.asyncio
    async def test_timeout_raises_tool_timeout_error(self, dispatcher):
        """AC-2.5: Timeout → ToolTimeoutError with tool name and timeout value."""

        async def slow_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

        transport = httpx.MockTransport(slow_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        with pytest.raises(ToolTimeoutError) as exc_info:
            await dispatcher.dispatch("semantic_search", {"query": "slow"})

        assert exc_info.value.tool_name == "semantic_search"
        assert exc_info.value.timeout_seconds == 30.0

    @pytest.mark.asyncio
    async def test_timeout_error_message_includes_tool_and_seconds(self, dispatcher):
        async def slow_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

        transport = httpx.MockTransport(slow_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        with pytest.raises(ToolTimeoutError, match="semantic_search.*30"):
            await dispatcher.dispatch("semantic_search", {"query": "slow"})


# ═══════════════════════════════════════════════════════════════════════
# AC-2.4: Connection errors → BackendUnavailableError
# ═══════════════════════════════════════════════════════════════════════


class TestConnectionErrors:
    """AC-2.4: Connection errors raise BackendUnavailableError."""

    @pytest.mark.asyncio
    async def test_connect_error_raises_backend_unavailable(self, dispatcher):
        async def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(fail_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        with pytest.raises(BackendUnavailableError) as exc_info:
            await dispatcher.dispatch("semantic_search", {"query": "test"})

        assert "semantic-search" in exc_info.value.service_name.lower().replace("_", "-")

    @pytest.mark.asyncio
    async def test_connect_timeout_raises_backend_unavailable(self, dispatcher):
        async def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("Connect timeout")

        transport = httpx.MockTransport(fail_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        with pytest.raises(BackendUnavailableError):
            await dispatcher.dispatch("llm_complete", {"prompt": "test"})

    @pytest.mark.asyncio
    async def test_backend_unavailable_error_contains_service_name(self, dispatcher):
        async def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(fail_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        with pytest.raises(BackendUnavailableError) as exc_info:
            await dispatcher.dispatch("graph_query", {"cypher": "MATCH (n) RETURN n"})

        assert exc_info.value.service_name != ""


# ═══════════════════════════════════════════════════════════════════════
# AC-2.3: DispatchResult structure
# ═══════════════════════════════════════════════════════════════════════


class TestDispatchResult:
    """AC-2.3: Structured response with status_code, headers, body, elapsed_ms."""

    @pytest.mark.asyncio
    async def test_result_has_status_code(self, dispatcher):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": []})

        transport = httpx.MockTransport(handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        result = await dispatcher.dispatch("semantic_search", {"query": "test"})
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_result_has_body_dict(self, dispatcher):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [1, 2, 3]})

        transport = httpx.MockTransport(handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        result = await dispatcher.dispatch("semantic_search", {"query": "test"})
        assert isinstance(result.body, dict)
        assert result.body == {"results": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_result_has_headers_dict(self, dispatcher):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"ok": True},
                headers={"x-custom": "value"},
            )

        transport = httpx.MockTransport(handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        result = await dispatcher.dispatch("semantic_search", {"query": "test"})
        assert isinstance(result.headers, dict)

    @pytest.mark.asyncio
    async def test_result_has_elapsed_ms(self, dispatcher):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        result = await dispatcher.dispatch("semantic_search", {"query": "test"})
        assert isinstance(result.elapsed_ms, float)
        assert result.elapsed_ms >= 0

    @pytest.mark.asyncio
    async def test_result_preserves_non_200_status(self, dispatcher):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"error": "validation failed"})

        transport = httpx.MockTransport(handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        result = await dispatcher.dispatch("semantic_search", {"query": "test"})
        assert result.status_code == 422
        assert result.body == {"error": "validation failed"}

    @pytest.mark.asyncio
    async def test_result_handles_empty_body(self, dispatcher):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(204)

        transport = httpx.MockTransport(handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        result = await dispatcher.dispatch("semantic_search", {"query": "test"})
        assert result.status_code == 204
        assert result.body == {}


# ═══════════════════════════════════════════════════════════════════════
# DispatchRoute / DispatchResult dataclass tests
# ═══════════════════════════════════════════════════════════════════════


class TestDataclasses:
    """Verify DispatchRoute and DispatchResult are well-formed dataclasses."""

    def test_dispatch_route_fields(self):
        from src.tool_dispatcher import DispatchRoute

        route = DispatchRoute(base_url="http://localhost:8081", path="/v1/search", timeout=10.0)
        assert route.base_url == "http://localhost:8081"
        assert route.path == "/v1/search"
        assert route.timeout == 10.0

    def test_dispatch_result_fields(self):
        from src.tool_dispatcher import DispatchResult

        result = DispatchResult(
            status_code=200,
            body={"key": "value"},
            headers={"content-type": "application/json"},
            elapsed_ms=42.5,
        )
        assert result.status_code == 200
        assert result.body == {"key": "value"}
        assert result.headers == {"content-type": "application/json"}
        assert result.elapsed_ms == 42.5


# ═══════════════════════════════════════════════════════════════════════
# Error class tests
# ═══════════════════════════════════════════════════════════════════════


class TestErrorClasses:
    """Verify custom error classes from src/core/errors.py."""

    def test_backend_unavailable_error_attrs(self):
        err = BackendUnavailableError("semantic-search", "Connection refused")
        assert err.service_name == "semantic-search"
        assert err.detail == "Connection refused"
        assert "semantic-search" in str(err)

    def test_backend_unavailable_error_no_detail(self):
        err = BackendUnavailableError("ai-agents")
        assert err.service_name == "ai-agents"
        assert err.detail == ""

    def test_tool_timeout_error_attrs(self):
        err = ToolTimeoutError("llm_complete", 30.0)
        assert err.tool_name == "llm_complete"
        assert err.timeout_seconds == 30.0
        assert "llm_complete" in str(err)
        assert "30" in str(err)


# ═══════════════════════════════════════════════════════════════════════
# MCP2.11 REFACTOR: Connection pooling
# ═══════════════════════════════════════════════════════════════════════


class TestConnectionPooling:
    """MCP2.11: Single httpx.AsyncClient per backend for connection reuse."""

    def test_no_clients_created_at_init(self, dispatcher):
        """Clients are lazily created, not at __init__ time."""
        assert dispatcher._client is None
        assert len(dispatcher._clients) == 0

    @pytest.mark.asyncio
    async def test_get_client_creates_pooled_client(self, dispatcher):
        """_get_client creates a new client when base_url is not in pool."""
        client = dispatcher._get_client("http://localhost:8081")
        assert isinstance(client, httpx.AsyncClient)
        assert "http://localhost:8081" in dispatcher._clients
        await dispatcher.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_pooled_client(self, dispatcher):
        """_get_client returns the same client for the same base_url."""
        first = dispatcher._get_client("http://localhost:8081")
        second = dispatcher._get_client("http://localhost:8081")
        assert first is second
        assert len(dispatcher._clients) == 1
        await dispatcher.close()

    @pytest.mark.asyncio
    async def test_get_client_different_urls_different_clients(self, dispatcher):
        """_get_client creates separate clients for different base_urls."""
        c1 = dispatcher._get_client("http://localhost:8081")
        c2 = dispatcher._get_client("http://localhost:8082")
        assert c1 is not c2
        assert len(dispatcher._clients) == 2
        await dispatcher.close()

    @pytest.mark.asyncio
    async def test_get_client_prefers_injected_client(self, dispatcher):
        """When _client is set (test injection), _get_client returns it."""
        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        dispatcher._client = mock_client
        result = dispatcher._get_client("http://localhost:9999")
        assert result is mock_client
        assert len(dispatcher._clients) == 0  # pool not used
        await dispatcher.close()

    @pytest.mark.asyncio
    async def test_close_clears_injected_client(self, dispatcher):
        """close() should close and clear the injected _client."""
        transport = httpx.MockTransport(lambda r: httpx.Response(200))
        dispatcher._client = httpx.AsyncClient(transport=transport)

        await dispatcher.close()

        assert dispatcher._client is None

    @pytest.mark.asyncio
    async def test_close_clears_pooled_clients(self, dispatcher):
        """close() should close all pooled clients and clear the pool."""
        # Create real pooled clients
        dispatcher._get_client("http://localhost:8081")
        dispatcher._get_client("http://localhost:8082")
        assert len(dispatcher._clients) == 2

        await dispatcher.close()

        assert len(dispatcher._clients) == 0

    @pytest.mark.asyncio
    async def test_close_with_no_clients_does_not_error(self, dispatcher):
        """close() on a fresh dispatcher should not raise."""
        await dispatcher.close()
