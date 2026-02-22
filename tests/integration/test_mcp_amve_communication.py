"""E2E: amve_detect_communication via MCP SSE — WBS-AEI8.

AC-AEI8.3: amve_detect_communication returns events + messaging results
            for scope='all' on a real codebase.

Scans /Users/kevintoles/POC/ai-agents/src via the full MCP stack:
  FastMCP Client → mcp-gateway → ToolDispatcher → AMVE :8088/v1/analysis/communication

The communication endpoint with scope='all' returns:
  {"scope": "all", "events": {success, result, error}, "messaging": {success, result, error}}
"""

import pytest
from fastmcp import Client

from tests.integration.conftest import AI_AGENTS_SRC, _check_backend, _extract_body

pytestmark = pytest.mark.integration


# ── AC-AEI8.3: Communication Detection E2E ─────────────────────────────────


class TestAMVEDetectCommunicationE2E:
    """amve_detect_communication via MCP SSE against real ai-agents codebase."""

    async def test_scope_all_returns_both_sections(self, mcp_server):
        """scope='all' returns both events and messaging sections."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_communication",
                {
                    "source_path": AI_AGENTS_SRC,
                    "scope": "all",
                    "include_confidence": False,
                },
            )
        body = _extract_body(result)
        assert body["scope"] == "all", f"Expected scope='all', got {body.get('scope')}"
        assert "events" in body, "Missing 'events' section in scope='all' response"
        assert "messaging" in body, "Missing 'messaging' section in scope='all' response"

    async def test_events_section_has_result(self, mcp_server):
        """The events section contains a success/result/error structure."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_communication",
                {
                    "source_path": AI_AGENTS_SRC,
                    "scope": "all",
                    "include_confidence": False,
                },
            )
        body = _extract_body(result)
        events = body["events"]
        # Either {success, result, error} or {feature_disabled: true}
        if "feature_disabled" not in events:
            assert "success" in events, "events section missing 'success' field"
            assert events["success"] is True, f"Events detection failed: {events.get('error')}"

    async def test_messaging_section_has_result(self, mcp_server):
        """The messaging section contains a success/result/error structure."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_communication",
                {
                    "source_path": AI_AGENTS_SRC,
                    "scope": "all",
                    "include_confidence": False,
                },
            )
        body = _extract_body(result)
        messaging = body["messaging"]
        if "feature_disabled" not in messaging:
            assert "success" in messaging, "messaging section missing 'success' field"
            assert messaging["success"] is True, f"Messaging detection failed: {messaging.get('error')}"

    async def test_scope_events_only(self, mcp_server):
        """scope='events' returns only events section."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_communication",
                {
                    "source_path": AI_AGENTS_SRC,
                    "scope": "events",
                    "include_confidence": False,
                },
            )
        body = _extract_body(result)
        assert body["scope"] == "events"
        assert "events" in body

    async def test_scope_messaging_only(self, mcp_server):
        """scope='messaging' returns only messaging section."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_communication",
                {
                    "source_path": AI_AGENTS_SRC,
                    "scope": "messaging",
                    "include_confidence": False,
                },
            )
        body = _extract_body(result)
        assert body["scope"] == "messaging"
        assert "messaging" in body
