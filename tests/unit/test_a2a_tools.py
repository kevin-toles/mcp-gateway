"""Tests for A2A Protocol MCP Tools — MCP ↔ A2A ↔ Temporal bridge.

Verifies the 3 A2A tools that bridge MCP protocol to ai-agents A2A endpoints:
- a2a_send_message: POST /a2a/v1/message:send → TaskExecutor → Temporal
- a2a_get_task: GET /a2a/v1/tasks/{id}
- a2a_cancel_task: POST /a2a/v1/tasks/{id}:cancel

Integration Chain: MCP → mcp-gateway → ai-agents A2A → TaskExecutor → Temporal

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → A2A ↔ MCP ↔ Temporal
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from src.core.config import Settings
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import DispatchResult, ToolDispatcher

# ── Fixtures ────────────────────────────────────────────────────────────


def _make_result(body: dict, status_code: int = 200) -> DispatchResult:
    return DispatchResult(
        status_code=status_code,
        body=body,
        headers={"content-type": "application/json"},
        elapsed_ms=15.0,
    )


@pytest.fixture
def dispatcher():
    settings = Settings()
    return ToolDispatcher(settings)


@pytest.fixture
def mock_dispatcher():
    d = AsyncMock(spec=ToolDispatcher)
    d.dispatch = AsyncMock(return_value=_make_result({"taskId": "task-abc-123"}))
    return d


@pytest.fixture
def sanitizer():
    return OutputSanitizer()


# ═══════════════════════════════════════════════════════════════════════
# Route Table Tests
# ═══════════════════════════════════════════════════════════════════════


class TestA2ARouteTable:
    """Verify A2A tools have correct dispatch routes."""

    def test_a2a_send_message_route_exists(self, dispatcher):
        route = dispatcher.get_route("a2a_send_message")
        assert route is not None
        assert route.base_url == "http://localhost:8082"
        assert route.path == "/a2a/v1/message:send"

    def test_a2a_get_task_route_exists(self, dispatcher):
        route = dispatcher.get_route("a2a_get_task")
        assert route is not None
        assert route.base_url == "http://localhost:8082"
        assert route.path == "/a2a/v1/tasks"

    def test_a2a_cancel_task_route_exists(self, dispatcher):
        route = dispatcher.get_route("a2a_cancel_task")
        assert route is not None
        assert route.base_url == "http://localhost:8082"
        assert route.path == "/a2a/v1/tasks"

    def test_all_a2a_routes_point_to_ai_agents(self, dispatcher):
        """All A2A tools route to ai-agents on :8082."""
        for tool in ("a2a_send_message", "a2a_get_task", "a2a_cancel_task"):
            route = dispatcher.get_route(tool)
            assert route is not None
            assert "8082" in route.base_url, f"{tool} should route to ai-agents :8082"


# ═══════════════════════════════════════════════════════════════════════
# Handler Tests — a2a_send_message
# ═══════════════════════════════════════════════════════════════════════


class TestA2ASendMessageHandler:
    """Test the a2a_send_message tool handler."""

    @pytest.mark.asyncio
    async def test_sends_correct_payload(self, mock_dispatcher, sanitizer):
        from src.tools.a2a_send_message import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        result = await handler(content="Analyze this code", skill_id="cross-reference")

        mock_dispatcher.dispatch.assert_called_once()
        call_args = mock_dispatcher.dispatch.call_args
        assert call_args[0][0] == "a2a_send_message"
        payload = call_args[0][1]
        assert payload["message"]["parts"][0]["text"] == "Analyze this code"
        assert payload["message"]["skillId"] == "cross-reference"

    @pytest.mark.asyncio
    async def test_returns_task_id(self, mock_dispatcher, sanitizer):
        from src.tools.a2a_send_message import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        result = await handler(content="Test message")
        assert result["taskId"] == "task-abc-123"

    @pytest.mark.asyncio
    async def test_optional_skill_id_defaults_empty(self, mock_dispatcher, sanitizer):
        from src.tools.a2a_send_message import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(content="Test")

        payload = mock_dispatcher.dispatch.call_args[0][1]
        assert payload["message"]["skillId"] is None

    @pytest.mark.asyncio
    async def test_optional_context_id(self, mock_dispatcher, sanitizer):
        from src.tools.a2a_send_message import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(content="Test", context_id="ctx-456")

        payload = mock_dispatcher.dispatch.call_args[0][1]
        assert payload["contextId"] == "ctx-456"

    @pytest.mark.asyncio
    async def test_validates_empty_content_rejected(self, mock_dispatcher, sanitizer):
        from pydantic import ValidationError

        from src.tools.a2a_send_message import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(content="")


# ═══════════════════════════════════════════════════════════════════════
# Handler Tests — a2a_get_task
# ═══════════════════════════════════════════════════════════════════════


class TestA2AGetTaskHandler:
    """Test the a2a_get_task tool handler."""

    @pytest.mark.asyncio
    async def test_uses_get_method_with_path_override(self, sanitizer):
        mock_disp = AsyncMock(spec=ToolDispatcher)
        mock_disp.dispatch = AsyncMock(
            return_value=_make_result(
                {
                    "id": "task-123",
                    "status": "COMPLETED",
                    "artifacts": [{"parts": [{"type": "text", "text": "result"}]}],
                }
            )
        )

        from src.tools.a2a_get_task import create_handler

        handler = create_handler(mock_disp, sanitizer)
        result = await handler(task_id="task-123")

        mock_disp.dispatch.assert_called_once()
        call_kwargs = mock_disp.dispatch.call_args
        # Check method="GET" and path_override
        assert call_kwargs.kwargs["method"] == "GET"
        assert call_kwargs.kwargs["path_override"] == "/a2a/v1/tasks/task-123"

    @pytest.mark.asyncio
    async def test_returns_task_details(self, sanitizer):
        mock_disp = AsyncMock(spec=ToolDispatcher)
        mock_disp.dispatch = AsyncMock(
            return_value=_make_result(
                {
                    "id": "task-456",
                    "status": "WORKING",
                }
            )
        )

        from src.tools.a2a_get_task import create_handler

        handler = create_handler(mock_disp, sanitizer)
        result = await handler(task_id="task-456")
        assert result["id"] == "task-456"
        assert result["status"] == "WORKING"

    @pytest.mark.asyncio
    async def test_validates_empty_task_id_rejected(self, mock_dispatcher, sanitizer):
        from pydantic import ValidationError

        from src.tools.a2a_get_task import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(task_id="")


# ═══════════════════════════════════════════════════════════════════════
# Handler Tests — a2a_cancel_task
# ═══════════════════════════════════════════════════════════════════════


class TestA2ACancelTaskHandler:
    """Test the a2a_cancel_task tool handler."""

    @pytest.mark.asyncio
    async def test_sends_post_with_path_override(self, sanitizer):
        mock_disp = AsyncMock(spec=ToolDispatcher)
        mock_disp.dispatch = AsyncMock(
            return_value=_make_result(
                {
                    "id": "task-789",
                    "status": "CANCELED",
                }
            )
        )

        from src.tools.a2a_cancel_task import create_handler

        handler = create_handler(mock_disp, sanitizer)
        result = await handler(task_id="task-789")

        mock_disp.dispatch.assert_called_once()
        call_kwargs = mock_disp.dispatch.call_args
        assert call_kwargs.kwargs["path_override"] == "/a2a/v1/tasks/task-789:cancel"
        # Default method is POST (no explicit method kwarg needed)

    @pytest.mark.asyncio
    async def test_returns_canceled_task(self, sanitizer):
        mock_disp = AsyncMock(spec=ToolDispatcher)
        mock_disp.dispatch = AsyncMock(
            return_value=_make_result(
                {
                    "id": "task-789",
                    "status": "CANCELED",
                }
            )
        )

        from src.tools.a2a_cancel_task import create_handler

        handler = create_handler(mock_disp, sanitizer)
        result = await handler(task_id="task-789")
        assert result["status"] == "CANCELED"

    @pytest.mark.asyncio
    async def test_validates_empty_task_id_rejected(self, mock_dispatcher, sanitizer):
        from pydantic import ValidationError

        from src.tools.a2a_cancel_task import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(task_id="")


# ═══════════════════════════════════════════════════════════════════════
# Dispatcher GET method support
# ═══════════════════════════════════════════════════════════════════════


class TestDispatcherGETSupport:
    """Verify dispatcher handles GET requests for a2a_get_task."""

    @pytest.mark.asyncio
    async def test_get_method_sends_get_request(self, dispatcher):
        """Dispatcher sends GET (not POST) when method='GET'."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"id": "task-1", "status": "COMPLETED"})

        transport = httpx.MockTransport(mock_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        result = await dispatcher.dispatch(
            "a2a_get_task",
            {},
            method="GET",
            path_override="/a2a/v1/tasks/task-1",
        )
        assert captured["method"] == "GET"
        assert "task-1" in captured["url"]
        assert result.body["status"] == "COMPLETED"

    @pytest.mark.asyncio
    async def test_post_with_path_override(self, dispatcher):
        """Dispatcher sends POST to custom path for a2a_cancel_task."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"id": "task-2", "status": "CANCELED"})

        transport = httpx.MockTransport(mock_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        result = await dispatcher.dispatch(
            "a2a_cancel_task",
            {},
            path_override="/a2a/v1/tasks/task-2:cancel",
        )
        assert captured["method"] == "POST"
        assert "task-2:cancel" in captured["url"]


# ═══════════════════════════════════════════════════════════════════════
# Integration Chain Verification
# ═══════════════════════════════════════════════════════════════════════


class TestMCPA2ATemporalChain:
    """Verify the MCP → A2A → Temporal integration chain architecture.

    These are architectural verification tests that confirm the tool wiring
    correctly routes through ai-agents A2A endpoints, which in turn use
    TaskExecutor to route to Temporal workflows.

    Full chain: MCP tool → dispatcher → ai-agents :8082 A2A endpoint
                → TaskExecutor → Temporal EnrichmentWorkflow (when available)
    """

    def test_a2a_send_message_routes_to_a2a_endpoint(self, dispatcher):
        """a2a_send_message dispatches to the A2A message:send endpoint."""
        route = dispatcher.get_route("a2a_send_message")
        assert route.path == "/a2a/v1/message:send"

    def test_a2a_tools_all_target_ai_agents(self, dispatcher):
        """All A2A tools target ai-agents service on port 8082."""
        for tool_name in ("a2a_send_message", "a2a_get_task", "a2a_cancel_task"):
            route = dispatcher.get_route(tool_name)
            assert route.base_url == "http://localhost:8082"

    def test_settings_has_ai_agents_url(self):
        """AI_AGENTS_URL is configurable in Settings."""
        settings = Settings()
        assert settings.AI_AGENTS_URL == "http://localhost:8082"

    def test_dispatcher_has_16_routes(self, dispatcher):
        """Dispatcher maintains 16 total routes (6 original + 3 A2A + 7 workflow)."""
        assert len(dispatcher.routes) == 16

    def test_a2a_tools_registered_in_yaml(self):
        """All 3 A2A tools are in the tool registry YAML."""
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        config_path = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"
        registry = ToolRegistry(config_path)

        for tool_name in ("a2a_send_message", "a2a_get_task", "a2a_cancel_task"):
            tool = registry.get(tool_name)
            assert tool is not None, f"{tool_name} not in registry"
            assert "a2a" in tool.tags, f"{tool_name} missing 'a2a' tag"
