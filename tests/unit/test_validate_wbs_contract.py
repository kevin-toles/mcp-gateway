"""ASCP.INT.1 — Contract tests: validate_wbs plumbing, no live service required.

These tests verify the MCP tool call chain
  (FastMCP Client → server → ToolDispatcher → HTTP → response parsing)
using httpx.MockTransport to simulate the validation-service.  They do NOT
verify actual WBS compliance logic — for that, see:
  - Live integration tests:  test_validate_wbs_mcp_roundtrip.py  (INTEGRATION=1)
  - Rust GHK tests:          validation-service/rust/tests/bdd_ghk.rs

These run in normal CI without INTEGRATION=1.

Runtime gate note: "POST /validate/wbs returns score=100 for the ASCP WBS
document" is a runtime property verifiable only with live services (ASCP.INT.1).
These contract tests verify the plumbing is correct so that when the live
service IS present and returns score=100, the result reaches the MCP client
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.core.config import Settings
from src.security.output_sanitizer import OutputSanitizer
from src.server import create_mcp_server
from src.tool_dispatcher import ToolDispatcher
from src.tool_registry import ToolRegistry
from tests.integration.conftest import _extract_body

try:
    from fastmcp import Client
except ImportError:
    pytest.skip("fastmcp not installed", allow_module_level=True)

_VALIDATION_SERVICE_BASE_URL = "http://localhost:8091"
_VALIDATE_WBS_PATH = "/validate/wbs"
_VALIDATE_WBS_URL = f"{_VALIDATION_SERVICE_BASE_URL}{_VALIDATE_WBS_PATH}"

_PASS_BODY: dict = {
    "verdict": "PASS",
    "score": 100,
    "failures": [],
    "phase": None,
    "human_review_items": [],
    "ready_for_approval": False,
    "stub": False,
}

_FAIL_BODY: dict = {
    "verdict": "FAIL",
    "score": 60,
    "failures": [
        {
            "check_id": "S-1",
            "severity": "ERROR",
            "section": "§1",
            "description": "Missing required section header",
            "detail": None,
        }
    ],
    "phase": None,
    "human_review_items": [],
    "ready_for_approval": False,
    "stub": False,
}


def _make_server(handler) -> object:
    """Create an in-process MCP server wired to a mock HTTP transport."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"
    if not config_path.exists():
        pytest.skip("config/tools.yaml not found")
    registry = ToolRegistry(config_path)
    dispatcher = ToolDispatcher(Settings())
    dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sanitizer = OutputSanitizer()
    return create_mcp_server(registry, dispatcher, sanitizer)


# ── AC-ASCP.1: dispatch routing ───────────────────────────────────────────────


class TestValidateWBSDispatch:
    """validate_wbs tool routes to the correct endpoint."""

    async def test_dispatches_post_to_validate_wbs_endpoint(self):
        """Tool sends POST to {VALIDATION_SERVICE_URL}/validate/wbs."""
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            return httpx.Response(200, json=_PASS_BODY)

        server = _make_server(handler)
        async with Client(server) as client:
            await client.call_tool("validate_wbs", {"content": "## WBS-X: Test\n"})

        assert captured["method"] == "POST", "validate_wbs must use POST"
        assert captured["url"] == _VALIDATE_WBS_URL, (
            f"Expected route {_VALIDATE_WBS_URL!r}, got {captured.get('url')!r}"
        )

    async def test_wbs_content_forwarded_in_request_body(self):
        """WBS content is sent as JSON {'content': ...} in the POST body."""
        captured: dict = {}
        wbs_content = "## WBS-CONTRACT1: Routing check\n**Dependencies:** none\n"

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_PASS_BODY)

        server = _make_server(handler)
        async with Client(server) as client:
            await client.call_tool("validate_wbs", {"content": wbs_content})

        assert "content" in captured.get("body", {}), (
            f"Expected 'content' key in request body, got: {captured.get('body')}"
        )
        assert captured["body"]["content"] == wbs_content


# ── AC-ASCP.1: response parsing — PASS path ───────────────────────────────────


class TestValidateWBSPassResponse:
    """validate_wbs tool correctly propagates PASS verdict and score=100.

    These cover the static half of the ASCP.INT.1 exit milestone:
    the plumbing carries score=100 from the service to the MCP client.
    """

    async def test_pass_verdict_propagated(self):
        """verdict=PASS from the service reaches the MCP client unchanged."""
        server = _make_server(lambda r: httpx.Response(200, json=_PASS_BODY))
        async with Client(server) as client:
            result = await client.call_tool("validate_wbs", {"content": "## WBS-X\n"})
        body = _extract_body(result)
        assert body.get("verdict") == "PASS"

    async def test_score_100_propagated(self):
        """score=100 from the service reaches the MCP client unchanged.

        This is the static half of ASCP.INT.1 — the runtime half requires
        the live validation-service (see test_validate_wbs_mcp_roundtrip.py).
        """
        server = _make_server(lambda r: httpx.Response(200, json=_PASS_BODY))
        async with Client(server) as client:
            result = await client.call_tool("validate_wbs", {"content": "## WBS-X\n"})
        body = _extract_body(result)
        assert body.get("score") == 100

    async def test_empty_failures_list_propagated(self):
        """failures=[] from the service reaches the MCP client unchanged."""
        server = _make_server(lambda r: httpx.Response(200, json=_PASS_BODY))
        async with Client(server) as client:
            result = await client.call_tool("validate_wbs", {"content": "## WBS-X\n"})
        body = _extract_body(result)
        assert body.get("failures") == []


# ── AC-ASCP.1: response parsing — FAIL path ───────────────────────────────────


class TestValidateWBSFailResponse:
    """validate_wbs tool correctly propagates FAIL verdict and structured failures."""

    async def test_fail_verdict_propagated(self):
        """verdict=FAIL from the service reaches the MCP client."""
        server = _make_server(lambda r: httpx.Response(200, json=_FAIL_BODY))
        async with Client(server) as client:
            result = await client.call_tool("validate_wbs", {"content": "## WBS-X\n"})
        body = _extract_body(result)
        assert body.get("verdict") == "FAIL"

    async def test_partial_score_propagated(self):
        """score < 100 is forwarded to the MCP client when checks fail."""
        server = _make_server(lambda r: httpx.Response(200, json=_FAIL_BODY))
        async with Client(server) as client:
            result = await client.call_tool("validate_wbs", {"content": "## WBS-X\n"})
        body = _extract_body(result)
        assert body.get("score") == 60

    async def test_structured_failures_propagated(self):
        """CheckFailure list is forwarded intact — check_id, severity, section preserved."""
        server = _make_server(lambda r: httpx.Response(200, json=_FAIL_BODY))
        async with Client(server) as client:
            result = await client.call_tool("validate_wbs", {"content": "## WBS-X\n"})
        body = _extract_body(result)
        failures = body.get("failures", [])
        assert len(failures) == 1
        assert failures[0]["check_id"] == "S-1"
        assert failures[0]["severity"] == "ERROR"
