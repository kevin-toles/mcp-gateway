"""E2E: amve_detect_boundaries via MCP SSE — WBS-AEI8.

AC-AEI8.5: amve_detect_boundaries returns boundary candidates with composite
            heuristic scores for real platform service directories.

Scans /Users/kevintoles/POC/ai-agents/src via the full MCP stack:
  FastMCP Client → mcp-gateway → ToolDispatcher → AMVE :8088/v1/analysis/boundaries

The boundaries endpoint returns:
  {"success": true, "result": {boundaries: [...], violations: [...], ...}, "error": null}

Each boundary has:
  - confidence_score (float 0.0-1.0) — composite heuristic score
  - heuristic_matches[] — breakdown per heuristic type
  - heuristic types: package_structure, entry_points, http_clients,
                     database_access, messaging
"""

import pytest
from fastmcp import Client

from tests.integration.conftest import AI_AGENTS_SRC, _check_backend, _extract_body

pytestmark = pytest.mark.integration


# ── AC-AEI8.5: Boundary Detection E2E ──────────────────────────────────────


class TestAMVEDetectBoundariesE2E:
    """amve_detect_boundaries via MCP SSE against real ai-agents codebase."""

    async def test_returns_successful_result(self, mcp_server):
        """Boundary detection returns success=True for a real codebase."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_boundaries",
                {
                    "source_path": AI_AGENTS_SRC,
                    "include_confidence": True,
                },
            )
        assert result is not None
        body = _extract_body(result)
        assert body["success"] is True, f"Boundary detection failed: {body.get('error')}"

    async def test_contains_boundary_candidates(self, mcp_server):
        """Result contains at least 1 boundary candidate."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_boundaries",
                {
                    "source_path": AI_AGENTS_SRC,
                    "include_confidence": True,
                },
            )
        body = _extract_body(result)
        boundaries = body["result"]["boundaries"]
        assert len(boundaries) >= 1, "Expected at least 1 boundary candidate"

    async def test_boundary_has_confidence_score(self, mcp_server):
        """Each boundary has confidence_score > 0.0."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_boundaries",
                {
                    "source_path": AI_AGENTS_SRC,
                    "include_confidence": True,
                },
            )
        body = _extract_body(result)
        for b in body["result"]["boundaries"]:
            assert "confidence_score" in b, f"Missing confidence_score in boundary {b.get('boundary_id')}"
            assert b["confidence_score"] > 0.0, f"Expected confidence_score > 0.0, got {b['confidence_score']}"

    async def test_boundary_has_heuristic_matches(self, mcp_server):
        """Each boundary has heuristic_matches with known heuristic types."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        known_heuristic_types = {
            "package_structure",
            "entry_points",
            "http_clients",
            "database_access",
            "messaging",
        }

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_boundaries",
                {
                    "source_path": AI_AGENTS_SRC,
                    "include_confidence": True,
                },
            )
        body = _extract_body(result)
        for b in body["result"]["boundaries"]:
            if "heuristic_matches" in b and b["heuristic_matches"]:
                for hm in b["heuristic_matches"]:
                    assert "heuristic_type" in hm, "Missing heuristic_type in heuristic_match"
                    assert hm["heuristic_type"] in known_heuristic_types, (
                        f"Unknown heuristic_type: {hm['heuristic_type']}"
                    )

    async def test_boundary_has_required_fields(self, mcp_server):
        """Each boundary has boundary_id, confidence, packages."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_boundaries",
                {
                    "source_path": AI_AGENTS_SRC,
                    "include_confidence": True,
                },
            )
        body = _extract_body(result)
        required_keys = {"boundary_id", "confidence", "packages"}
        for b in body["result"]["boundaries"]:
            missing = required_keys - set(b.keys())
            assert not missing, f"Boundary missing keys: {missing}"

    async def test_result_includes_analysis_counts(self, mcp_server):
        """Result includes total_packages_analyzed and total_files_analyzed."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_detect_boundaries",
                {
                    "source_path": AI_AGENTS_SRC,
                    "include_confidence": True,
                },
            )
        body = _extract_body(result)
        result_data = body["result"]
        assert "total_packages_analyzed" in result_data or "total_files_analyzed" in result_data, (
            "Expected analysis count fields in result"
        )
