"""uss_ring_search forwards scope to unified-search-rs GET /v1/search/rings (ETDI-1.0 v0.22 BEH-11.08).

`scope` is se (default) | theory | all: theory searches the theory collection, all searches both and keeps the better
hit. The tool description tells LLM callers to pass theory for mathematics, statistics or other theory questions.
"""

from __future__ import annotations

import httpx
import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from src.core.config import Settings
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher


class TestUSSRingSearchInputScope:
    def test_accepts_each_scope(self):
        from src.models.schemas import USSRingSearchInput

        for scope in ("se", "theory", "all"):
            assert USSRingSearchInput(q="kalman filter", scope=scope).scope == scope

    def test_scope_defaults_to_empty(self):
        from src.models.schemas import USSRingSearchInput

        assert USSRingSearchInput(q="kalman filter").scope == ""

    def test_rejects_unknown_scope(self):
        from src.models.schemas import USSRingSearchInput

        with pytest.raises(ValidationError):
            USSRingSearchInput(q="kalman filter", scope="foundations")


class TestUSSRingSearchHandlerScope:
    @staticmethod
    def _handler(captured: dict):
        from src.tools.uss_ring_search import create_handler

        def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["params"] = dict(request.url.params)
            return httpx.Response(200, json={"ring_0": {}, "rings": {}})

        dispatcher = ToolDispatcher(Settings())
        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        return create_handler(dispatcher, OutputSanitizer())

    @pytest.mark.asyncio
    async def test_handler_forwards_scope(self):
        captured: dict = {}
        await self._handler(captured)(q="kalman filter covariance", scope="theory")
        assert captured["params"]["scope"] == "theory"

    @pytest.mark.asyncio
    async def test_handler_omits_scope_when_empty(self):
        captured: dict = {}
        await self._handler(captured)(q="circuit breaker")
        assert "scope" not in captured["params"]


def test_the_tool_description_tells_callers_when_to_use_theory():
    tools = yaml.safe_load((Path(__file__).resolve().parents[3] / "config" / "tools.yaml").read_text())["tools"]
    [ring] = [t for t in tools if t["name"] == "uss_ring_search"]
    text = (ring["description"] + " " + ring["when_to_use"]).lower()
    assert "scope" in text and "theory" in text and "underlying theory" in text
