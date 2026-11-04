"""sa_build_call_graph forwards reference_paths to struct-analyzer :8088.

reference_paths are read-only dependency sources (an Elixir project's deps,
for instance) read for what their `use` macros inject, never graphed; without
them such calls stay unresolved and the response reports use_source_unread.
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.core.config import Settings
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher


class TestSACallGraphInputReferencePaths:
    def test_accepts_reference_paths(self):
        from src.models.schemas import SACallGraphInput

        model = SACallGraphInput(source_paths=["/src"], reference_paths=["/deps/phoenix"])
        assert model.reference_paths == ["/deps/phoenix"]

    def test_reference_paths_default_to_empty(self):
        from src.models.schemas import SACallGraphInput

        assert SACallGraphInput(source_paths=["/src"]).reference_paths == []


def _capturing_handler(captured: dict):
    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"call_graph": {"edges": [], "nodes": []}})

    return mock_handler


class TestSABuildCallGraphHandlerReferencePaths:
    @pytest.mark.asyncio
    async def test_handler_forwards_reference_paths(self):
        from src.tools.sa_build_call_graph import create_handler

        captured: dict = {}
        dispatcher = ToolDispatcher(Settings())
        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(_capturing_handler(captured)))
        handler = create_handler(dispatcher, OutputSanitizer())

        await handler(source_paths=["/src"], reference_paths=["/deps/ecto", "/deps/phoenix"])

        assert captured["body"]["reference_paths"] == ["/deps/ecto", "/deps/phoenix"]

    @pytest.mark.asyncio
    async def test_handler_omits_reference_paths_when_empty(self):
        from src.tools.sa_build_call_graph import create_handler

        captured: dict = {}
        dispatcher = ToolDispatcher(Settings())
        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(_capturing_handler(captured)))
        handler = create_handler(dispatcher, OutputSanitizer())

        await handler(source_paths=["/src"])

        assert "reference_paths" not in captured["body"]
