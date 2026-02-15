"""TWR6 tests — code_analyze + code_pattern_audit handler payload format.

Verifies AC-TWR6.1, AC-TWR6.2, AC-TWR6.3:
- Both tools dispatch to audit-service :8084 at /v1/patterns/detect
- Handlers transform MCP input into audit-service request format
"""

import json

import httpx
import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher


@pytest.fixture
def dispatcher():
    settings = Settings()
    return ToolDispatcher(settings)


# ═══════════════════════════════════════════════════════════════════════
# AC-TWR6.1 / AC-TWR6.2: Dispatch routes
# ═══════════════════════════════════════════════════════════════════════


class TestTWR6DispatchRoutes:
    """AC-TWR6.1/6.2: Both tools dispatch to audit-service :8084."""

    def test_code_analyze_route_base_url(self, dispatcher):
        route = dispatcher.get_route("code_analyze")
        assert route.base_url == "http://localhost:8084"

    def test_code_analyze_route_path(self, dispatcher):
        route = dispatcher.get_route("code_analyze")
        assert route.path == "/v1/patterns/detect"

    def test_code_pattern_audit_route_base_url(self, dispatcher):
        route = dispatcher.get_route("code_pattern_audit")
        assert route.base_url == "http://localhost:8084"

    def test_code_pattern_audit_route_path(self, dispatcher):
        route = dispatcher.get_route("code_pattern_audit")
        assert route.path == "/v1/patterns/detect"

    @pytest.mark.asyncio
    async def test_code_analyze_full_dispatch_url(self, dispatcher):
        """AC-TWR6.1: code_analyze dispatches to localhost:8084/v1/patterns/detect."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"matches": [], "total": 0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch("code_analyze", {"code": "x=1", "language": "python"})
        assert captured["url"] == "http://localhost:8084/v1/patterns/detect"

    @pytest.mark.asyncio
    async def test_code_pattern_audit_full_dispatch_url(self, dispatcher):
        """AC-TWR6.2: code_pattern_audit dispatches to localhost:8084/v1/patterns/detect."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"matches": [], "total": 0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch("code_pattern_audit", {"code": "x=1", "language": "python"})
        assert captured["url"] == "http://localhost:8084/v1/patterns/detect"


# ═══════════════════════════════════════════════════════════════════════
# AC-TWR6.3: Handler payload transformation
# ═══════════════════════════════════════════════════════════════════════


class TestTWR6CodeAnalyzePayload:
    """AC-TWR6.3: code_analyze handler transforms input to audit-service format."""

    @pytest.mark.asyncio
    async def test_payload_contains_code_field(self, dispatcher):
        captured_body = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"matches": [], "total": 0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.code_analyze import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(code="def foo(): pass", language="python")

        assert "code" in captured_body
        assert captured_body["code"] == "def foo(): pass"

    @pytest.mark.asyncio
    async def test_payload_contains_language_field(self, dispatcher):
        captured_body = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"matches": [], "total": 0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.code_analyze import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(code="x = 1", language="python")

        assert captured_body["language"] == "python"

    @pytest.mark.asyncio
    async def test_payload_contains_file_path_field(self, dispatcher):
        """audit-service requires file_path in the request body."""
        captured_body = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"matches": [], "total": 0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.code_analyze import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(code="x = 1", language="python")

        assert "file_path" in captured_body

    @pytest.mark.asyncio
    async def test_payload_excludes_analysis_type(self, dispatcher):
        """analysis_type is a MCP-only field, not sent to audit-service."""
        captured_body = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"matches": [], "total": 0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.code_analyze import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(code="x = 1", language="python", analysis_type="all")

        assert "analysis_type" not in captured_body


class TestTWR6CodePatternAuditPayload:
    """AC-TWR6.3: code_pattern_audit handler transforms input to audit-service format."""

    @pytest.mark.asyncio
    async def test_payload_contains_code_and_language(self, dispatcher):
        captured_body = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"matches": [], "total": 0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.code_pattern_audit import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(code="class Foo: pass", language="python")

        assert captured_body["code"] == "class Foo: pass"
        assert captured_body["language"] == "python"

    @pytest.mark.asyncio
    async def test_payload_contains_file_path(self, dispatcher):
        """audit-service requires file_path in the request body."""
        captured_body = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"matches": [], "total": 0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.code_pattern_audit import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(code="class Foo: pass", language="python")

        assert "file_path" in captured_body

    @pytest.mark.asyncio
    async def test_payload_excludes_confidence_threshold(self, dispatcher):
        """confidence_threshold is a MCP-only field, not sent to audit-service."""
        captured_body = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json={"matches": [], "total": 0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.code_pattern_audit import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(code="class Foo: pass", language="python", confidence_threshold=0.5)

        assert "confidence_threshold" not in captured_body
