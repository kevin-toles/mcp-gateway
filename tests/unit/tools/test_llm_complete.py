"""TWR5: llm_complete handler tests — dispatch path & payload format.

TDD RED Phase: Verify D5 fixes.

AC-TWR5.1: llm_complete dispatch route uses path /v1/chat/completions
AC-TWR5.2: Request payload uses OpenAI messages format
           {"model": "...", "messages": [{"role": "user", "content": "..."}]}

Reference: WBS_TOOL_WIRING_REMEDIATION.md §TWR5
Defect: D5 — Dispatches to /v1/completions instead of /v1/chat/completions
"""

import json

import httpx
import pytest

# ═══════════════════════════════════════════════════════════════════════
# AC-TWR5.1: Dispatch route uses /v1/chat/completions
# ═══════════════════════════════════════════════════════════════════════


class TestTWR5DispatchPath:
    """AC-TWR5.1: llm_complete dispatches to /v1/chat/completions."""

    @pytest.fixture
    def dispatcher(self):
        from src.core.config import Settings
        from src.tool_dispatcher import ToolDispatcher

        return ToolDispatcher(Settings())

    def test_llm_complete_route_path_is_chat_completions(self, dispatcher):
        """
        AC-TWR5.1: The route table entry for llm_complete must use
        /v1/chat/completions, not /v1/completions.
        """
        route = dispatcher.get_route("llm_complete")
        assert route is not None
        assert route.path == "/v1/chat/completions", (
            f"D5: llm_complete dispatches to '{route.path}' — should be '/v1/chat/completions'"
        )

    def test_llm_complete_route_base_url_is_llm_gateway(self, dispatcher):
        """AC-TWR5.1: llm_complete routes to llm-gateway on :8080."""
        route = dispatcher.get_route("llm_complete")
        assert route.base_url == "http://localhost:8080"

    @pytest.mark.asyncio
    async def test_llm_complete_dispatch_hits_chat_completions_url(self, dispatcher):
        """
        AC-TWR5.1: Full dispatch sends to http://localhost:8080/v1/chat/completions.
        """
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await dispatcher.dispatch("llm_complete", {"model": "qwen3-8b", "messages": []})

        assert captured["url"] == "http://localhost:8080/v1/chat/completions"
        assert captured["method"] == "POST"


# ═══════════════════════════════════════════════════════════════════════
# AC-TWR5.2: Payload uses OpenAI messages format
# ═══════════════════════════════════════════════════════════════════════


class TestTWR5PayloadFormat:
    """AC-TWR5.2: llm_complete handler transforms input to OpenAI messages format."""

    @pytest.fixture
    def dispatcher(self):
        from src.core.config import Settings
        from src.tool_dispatcher import ToolDispatcher

        return ToolDispatcher(Settings())

    @pytest.mark.asyncio
    async def test_handler_sends_messages_array(self, dispatcher):
        """
        AC-TWR5.2: The payload dispatched must contain a 'messages' key
        with an array of message objects (not a flat 'prompt' string).
        """
        captured_body = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.llm_complete import create_handler

        handler_fn = create_handler(dispatcher, OutputSanitizer())
        await handler_fn(prompt="hello world")

        assert "messages" in captured_body, "Payload must contain 'messages' key in OpenAI format"
        assert isinstance(captured_body["messages"], list)
        assert len(captured_body["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_handler_sends_model_field(self, dispatcher):
        """
        AC-TWR5.2: The payload must contain a 'model' field.
        """
        captured_body = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.llm_complete import create_handler

        handler_fn = create_handler(dispatcher, OutputSanitizer())
        await handler_fn(prompt="hello world")

        assert "model" in captured_body, "Payload must contain 'model' field"
        assert isinstance(captured_body["model"], str)
        assert len(captured_body["model"]) > 0

    @pytest.mark.asyncio
    async def test_handler_prompt_becomes_user_message(self, dispatcher):
        """
        AC-TWR5.2: The 'prompt' input must become a user message in the
        messages array: {"role": "user", "content": "<prompt>"}.
        """
        captured_body = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.llm_complete import create_handler

        handler_fn = create_handler(dispatcher, OutputSanitizer())
        await handler_fn(prompt="explain recursion")

        messages = captured_body["messages"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1, "Must have exactly one user message"
        assert user_msgs[0]["content"] == "explain recursion"

    @pytest.mark.asyncio
    async def test_handler_system_prompt_becomes_system_message(self, dispatcher):
        """
        AC-TWR5.2: When system_prompt is provided, it must appear as a
        system message: {"role": "system", "content": "<system_prompt>"}.
        """
        captured_body = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.llm_complete import create_handler

        handler_fn = create_handler(dispatcher, OutputSanitizer())
        await handler_fn(prompt="hello", system_prompt="You are a helpful assistant")

        messages = captured_body["messages"]
        sys_msgs = [m for m in messages if m["role"] == "system"]
        assert len(sys_msgs) == 1, "Must have system message when system_prompt is provided"
        assert sys_msgs[0]["content"] == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_handler_empty_system_prompt_omits_system_message(self, dispatcher):
        """
        AC-TWR5.2: When system_prompt is empty, no system message should
        be included in the messages array.
        """
        captured_body = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.llm_complete import create_handler

        handler_fn = create_handler(dispatcher, OutputSanitizer())
        await handler_fn(prompt="hello")

        messages = captured_body["messages"]
        sys_msgs = [m for m in messages if m["role"] == "system"]
        assert len(sys_msgs) == 0, "No system message when system_prompt is empty"

    @pytest.mark.asyncio
    async def test_handler_includes_temperature_and_max_tokens(self, dispatcher):
        """
        AC-TWR5.2: temperature and max_tokens must be passed through.
        """
        captured_body = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.llm_complete import create_handler

        handler_fn = create_handler(dispatcher, OutputSanitizer())
        await handler_fn(prompt="hello", temperature=0.5, max_tokens=2048)

        assert captured_body["temperature"] == 0.5
        assert captured_body["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_handler_does_not_send_flat_prompt(self, dispatcher):
        """
        AC-TWR5.2: The payload must NOT contain a top-level 'prompt' key.
        That was the D5 bug — sending flat prompt instead of messages.
        """
        captured_body = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.llm_complete import create_handler

        handler_fn = create_handler(dispatcher, OutputSanitizer())
        await handler_fn(prompt="hello")

        assert "prompt" not in captured_body, "D5: payload must NOT contain flat 'prompt' key — use messages format"
