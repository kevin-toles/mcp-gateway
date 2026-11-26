"""sa_build_call_graph forwards min_confidence to struct-analyzer :8088.

struct-analyzer's POST /v1/analysis/call-graph drops edges below min_confidence
(high | medium | low | unknown); the MCP tool must accept and forward it.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher


class TestSACallGraphInputMinConfidence:
    def test_accepts_min_confidence(self):
        from src.models.schemas import SACallGraphInput

        model = SACallGraphInput(source_paths=["/src"], min_confidence="high")
        assert model.min_confidence == "high"

    def test_min_confidence_defaults_to_empty(self):
        from src.models.schemas import SACallGraphInput

        assert SACallGraphInput(source_paths=["/src"]).min_confidence == ""

    def test_rejects_unknown_min_confidence(self):
        from src.models.schemas import SACallGraphInput

        with pytest.raises(ValidationError):
            SACallGraphInput(source_paths=["/src"], min_confidence="very")


class TestSABuildCallGraphHandlerMinConfidence:
    @pytest.mark.asyncio
    async def test_handler_forwards_min_confidence(self):
        from src.tools.sa_build_call_graph import create_handler

        captured = {}

        def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"call_graph": {"edges": [], "nodes": []}})

        dispatcher = ToolDispatcher(Settings())
        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        handler = create_handler(dispatcher, OutputSanitizer())

        await handler(source_paths=["/src"], include_external=True, min_confidence="medium")

        assert captured["body"]["min_confidence"] == "medium"
        assert captured["body"]["include_external"] is True

    @pytest.mark.asyncio
    async def test_handler_omits_min_confidence_when_empty(self):
        from src.tools.sa_build_call_graph import create_handler

        captured = {}

        def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"call_graph": {"edges": [], "nodes": []}})

        dispatcher = ToolDispatcher(Settings())
        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        handler = create_handler(dispatcher, OutputSanitizer())

        await handler(source_paths=["/src"])

        assert "min_confidence" not in captured["body"]
