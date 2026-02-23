"""E2E: amve_evaluate_fitness via MCP SSE — WBS-AEI8 / WBS-AEIBF1.

AC-AEI8.4:   amve_evaluate_fitness returns per-function scores and overall
             fitness score for a real architecture snapshot.
AC-AEIBF1.3: POST /v1/fitness/evaluate builds EvaluateFitnessRequest from
             snapshot_id (source path) + FitnessFunctionRegistry.
AC-AEIBF1.4: All 5 previously-skipped tests pass against a real AMVE instance.

Calls AMVE :8088/v1/fitness/evaluate via the full MCP stack:
  FastMCP Client → mcp-gateway → ToolDispatcher → AMVE :8088/v1/fitness/evaluate

The fitness endpoint returns:
  {"overall_passed": bool, "overall_score": float, "results": [{function_id,
   function_name, passed, score, details}]}

Built-in fitness functions (from FitnessFunctionRegistry):
  FF-001: Circular Dependencies (structural)
  FF-002: Component Coupling (relationship)
  FF-003: Code Similarity (semantic)
  FF-004: Service Health (operational)
"""

import pytest
from fastmcp import Client

from tests.integration.conftest import _check_backend, _extract_body

pytestmark = pytest.mark.integration


def _is_valid_fitness_response(body: dict) -> bool:
    """Check whether the response looks like a proper EvaluateResponse."""
    return "overall_passed" in body and "overall_score" in body and "results" in body


# ── AC-AEI8.4: Fitness Evaluation E2E ──────────────────────────────────────────


class TestAMVEEvaluateFitnessE2E:
    """amve_evaluate_fitness via MCP SSE with real AMVE fitness functions."""

    async def test_dispatch_completes_without_exception(self, mcp_server):
        """MCP dispatch path to AMVE fitness endpoint completes (no crash)."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_evaluate_fitness",
                {
                    "snapshot_id": "integration-test-snapshot",
                },
            )
        assert result is not None
        body = _extract_body(result)
        # Gateway always returns a dict (possibly empty if AMVE 500s)
        assert isinstance(body, dict)

    async def test_returns_evaluation_result(self, mcp_server):
        """Fitness evaluation returns overall_passed, overall_score, results."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_evaluate_fitness",
                {
                    "snapshot_id": "integration-test-snapshot",
                },
            )
        body = _extract_body(result)
        assert _is_valid_fitness_response(body), (
            f"Expected valid fitness response with overall_passed/overall_score/results, got: {body!r}"
        )
        assert body["overall_passed"] is not None
        assert body["overall_score"] is not None
        assert isinstance(body["results"], list)

    async def test_overall_score_in_valid_range(self, mcp_server):
        """overall_score is between 0.0 and 1.0."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_evaluate_fitness",
                {
                    "snapshot_id": "integration-test-snapshot",
                },
            )
        body = _extract_body(result)
        assert _is_valid_fitness_response(body), f"Expected valid fitness response, got: {body!r}"
        score = body["overall_score"]
        assert 0.0 <= score <= 1.0, f"overall_score {score} out of range [0.0, 1.0]"

    async def test_results_contain_per_function_scores(self, mcp_server):
        """Each result entry has function_id, passed, score, details."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_evaluate_fitness",
                {
                    "snapshot_id": "integration-test-snapshot",
                },
            )
        body = _extract_body(result)
        assert _is_valid_fitness_response(body), f"Expected valid fitness response, got: {body!r}"
        results = body["results"]
        assert len(results) >= 1, "Expected at least 1 fitness function result"
        required_keys = {"function_id", "passed", "score", "details"}
        for r in results:
            missing = required_keys - set(r.keys())
            assert not missing, f"Result missing keys: {missing} in {r}"

    async def test_specific_function_ids_filter(self, mcp_server):
        """Passing fitness_function_ids filters the evaluation to those IDs."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_evaluate_fitness",
                {
                    "snapshot_id": "integration-test-snapshot",
                    "fitness_function_ids": ["FF-001", "FF-002"],
                },
            )
        body = _extract_body(result)
        assert _is_valid_fitness_response(body), f"Expected valid fitness response, got: {body!r}"
        result_ids = {r["function_id"] for r in body["results"]}
        assert len(result_ids) >= 1, "Expected at least 1 result"
        # All returned ids must be from the requested set
        assert result_ids <= {"FF-001", "FF-002"}, (
            f"Unexpected function ids returned: {result_ids - {'FF-001', 'FF-002'}}"
        )

    async def test_overall_passed_is_boolean(self, mcp_server):
        """overall_passed must be a boolean."""
        if not await _check_backend("http://localhost:8088/v1/health"):
            pytest.skip("AMVE service not running on :8088")

        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "amve_evaluate_fitness",
                {
                    "snapshot_id": "integration-test-snapshot",
                },
            )
        body = _extract_body(result)
        assert _is_valid_fitness_response(body), f"Expected valid fitness response, got: {body!r}"
        assert isinstance(body["overall_passed"], bool), (
            f"overall_passed should be bool, got {type(body['overall_passed'])}"
        )
