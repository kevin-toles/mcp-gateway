"""E2E: amve_detect_patterns via MCP SSE — WBS-AEI8.

AC-AEI8.1: amve_detect_patterns returns pattern instances with confidence
            scores for a real codebase (inference-service).

Scans /Users/kevintoles/POC/inference-service/src for Repository, UseCase,
and other architecture patterns via the full MCP dispatch stack:
  FastMCP Client → mcp-gateway → ToolDispatcher → AMVE :8088/v1/analysis/patterns
"""

import pytest
from fastmcp import Client

from tests.integration.conftest import INFERENCE_SERVICE_SRC, _check_backend, _extract_body

pytestmark = pytest.mark.integration


# ── AC-AEI8.1: Pattern Detection E2E ───────────────────────────────────────


class TestAMVEDetectPatternsE2E:
    """amve_detect_patterns via MCP SSE against real inference-service."""

    async def test_returns_successful_result(self, mcp_server):
        """Pattern detection returns success=True for a real codebase."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_patterns",
                {
                    "source_path": INFERENCE_SERVICE_SRC,
                    "include_confidence": True,
                },
            )
        assert result is not None
        # FastMCP wraps tool results in a list of content blocks
        body = _extract_body(result)
        assert body["success"] is True, f"Detection failed: {body.get('error')}"

    async def test_returns_pattern_instances(self, mcp_server):
        """Result contains at least 1 pattern instance."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_patterns",
                {
                    "source_path": INFERENCE_SERVICE_SRC,
                    "include_confidence": True,
                },
            )
        body = _extract_body(result)
        patterns = body["result"]["patterns"]
        assert len(patterns) >= 1, "Expected at least 1 detected pattern"

    async def test_pattern_has_confidence_score(self, mcp_server):
        """Each pattern has confidence_score > 0.0 when include_confidence=True."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_patterns",
                {
                    "source_path": INFERENCE_SERVICE_SRC,
                    "include_confidence": True,
                },
            )
        body = _extract_body(result)
        for p in body["result"]["patterns"]:
            assert "confidence_score" in p, f"Missing confidence_score in {p}"
            assert p["confidence_score"] > 0.0, f"Expected confidence > 0.0, got {p['confidence_score']}"

    async def test_pattern_has_required_fields(self, mcp_server):
        """Each pattern instance has pattern_type, source_file, confidence."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_patterns",
                {
                    "source_path": INFERENCE_SERVICE_SRC,
                    "include_confidence": True,
                },
            )
        body = _extract_body(result)
        required_keys = {"pattern_type", "source_file", "confidence"}
        for p in body["result"]["patterns"]:
            missing = required_keys - set(p.keys())
            assert not missing, f"Pattern missing keys: {missing}"

    async def test_pattern_types_include_known_types(self, mcp_server):
        """Detected patterns include at least one known architecture pattern type."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_patterns",
                {
                    "source_path": INFERENCE_SERVICE_SRC,
                    "include_confidence": True,
                },
            )
        body = _extract_body(result)
        known_types = {
            "repository",
            "use_case",
            "service",
            "adapter",
            "entity",
            "value_object",
            "factory",
            "event_handler",
            "domain_event",
            "controller",
        }
        detected_types = {p["pattern_type"] for p in body["result"]["patterns"]}
        overlap = detected_types & known_types
        assert len(overlap) >= 1, f"No known pattern types found. Detected: {detected_types}"
