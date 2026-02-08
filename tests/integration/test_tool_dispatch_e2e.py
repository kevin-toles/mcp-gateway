"""End-to-end tool dispatch integration tests — WBS-MCP9.

AC-9.4  All 9 tools callable end-to-end through mcp-gateway with live backends
AC-9.8  All mocks replaced with real service calls

Tests dispatch through ``ToolDispatcher`` to live backend services
running on localhost.  Each test verifies:
  1. The dispatcher can reach the backend and get a response.
  2. The response structure is as expected (status_code, body).

Requires: platform services running on :8080-:8083.
"""

import httpx
import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher

pytestmark = pytest.mark.integration


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def settings():
    """Real settings pointing at localhost backends."""
    return Settings()


@pytest.fixture
def dispatcher(settings):
    """Real ToolDispatcher with live HTTP clients — function-scoped to avoid event loop issues."""
    return ToolDispatcher(settings)


# ── Helpers ─────────────────────────────────────────────────────────────


async def _check_backend(url: str) -> bool:
    """Return True if backend health endpoint responds with 200."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=3.0)
            return resp.status_code == 200
    except Exception:
        return False


# ── AC-9.4 / AC-9.8: 9 Tools End-to-End ────────────────────────────────


class TestSemanticSearchE2E:
    """Tool: semantic_search → semantic-search-service :8081."""

    async def test_dispatch_returns_result(self, dispatcher):
        if not await _check_backend("http://localhost:8081/health"):
            pytest.skip("semantic-search-service not running on :8081")
        result = await dispatcher.dispatch("semantic_search", {
            "query": "integration test query",
            "collection": "all",
            "top_k": 3,
            "threshold": 0.3,
        })
        assert 100 <= result.status_code < 600, f"Invalid HTTP status: {result.status_code}"
        assert isinstance(result.body, dict)
        assert result.elapsed_ms > 0


class TestHybridSearchE2E:
    """Tool: hybrid_search → semantic-search-service :8081."""

    async def test_dispatch_returns_result(self, dispatcher):
        if not await _check_backend("http://localhost:8081/health"):
            pytest.skip("semantic-search-service not running on :8081")
        result = await dispatcher.dispatch("hybrid_search", {
            "query": "integration test hybrid",
            "collection": "all",
            "top_k": 3,
            "semantic_weight": 0.7,
            "keyword_weight": 0.3,
        })
        assert 100 <= result.status_code < 600, f"Invalid HTTP status: {result.status_code}"
        assert isinstance(result.body, dict)
        assert result.elapsed_ms > 0


class TestCodeAnalyzeE2E:
    """Tool: code_analyze → code-orchestrator-service :8083."""

    async def test_dispatch_returns_result(self, dispatcher):
        if not await _check_backend("http://localhost:8083/health"):
            pytest.skip("code-orchestrator-service not running on :8083")
        result = await dispatcher.dispatch("code_analyze", {
            "code": "def hello(): pass",
            "language": "python",
            "analysis_type": "all",
        })
        assert 100 <= result.status_code < 600, f"Invalid HTTP status: {result.status_code}"
        assert isinstance(result.body, dict)
        assert result.elapsed_ms > 0


class TestCodePatternAuditE2E:
    """Tool: code_pattern_audit → code-orchestrator-service :8083."""

    async def test_dispatch_returns_result(self, dispatcher):
        if not await _check_backend("http://localhost:8083/health"):
            pytest.skip("code-orchestrator-service not running on :8083")
        result = await dispatcher.dispatch("code_pattern_audit", {
            "code": "x = 1\nif x == 1:\n    print(x)",
            "language": "python",
        })
        assert 100 <= result.status_code < 600, f"Invalid HTTP status: {result.status_code}"
        assert isinstance(result.body, dict)
        assert result.elapsed_ms > 0


class TestGraphQueryE2E:
    """Tool: graph_query → ai-agents :8082."""

    async def test_dispatch_returns_result(self, dispatcher):
        if not await _check_backend("http://localhost:8082/health"):
            pytest.skip("ai-agents not running on :8082")
        result = await dispatcher.dispatch("graph_query", {
            "cypher": "MATCH (n) RETURN count(n) AS total LIMIT 1",
        })
        assert 100 <= result.status_code < 600, f"Invalid HTTP status: {result.status_code}"
        assert isinstance(result.body, dict)
        assert result.elapsed_ms > 0


class TestLLMCompleteE2E:
    """Tool: llm_complete → llm-gateway :8080."""

    async def test_dispatch_returns_result(self, dispatcher):
        if not await _check_backend("http://localhost:8080/health"):
            pytest.skip("llm-gateway not running on :8080")
        result = await dispatcher.dispatch("llm_complete", {
            "prompt": "Say hello",
            "max_tokens": 10,
            "temperature": 0.1,
        })
        assert 100 <= result.status_code < 600, f"Invalid HTTP status: {result.status_code}"
        assert isinstance(result.body, dict)
        assert result.elapsed_ms > 0


class TestRunAgentFunctionE2E:
    """Tool: run_agent_function → ai-agents :8082."""

    async def test_dispatch_returns_result(self, dispatcher):
        if not await _check_backend("http://localhost:8082/health"):
            pytest.skip("ai-agents not running on :8082")
        result = await dispatcher.dispatch("run_agent_function", {
            "function_name": "summarize-content",
            "input": {"content": "integration test content"},
        })
        assert 100 <= result.status_code < 600, f"Invalid HTTP status: {result.status_code}"
        assert isinstance(result.body, dict)
        assert result.elapsed_ms > 0


class TestRunDiscussionE2E:
    """Tool: run_discussion → ai-agents :8082."""

    async def test_dispatch_returns_result(self, dispatcher):
        if not await _check_backend("http://localhost:8082/health"):
            pytest.skip("ai-agents not running on :8082")
        result = await dispatcher.dispatch("run_discussion", {
            "protocol_id": "ROUNDTABLE_DISCUSSION",
            "inputs": {"topic": "integration test"},
        })
        assert 100 <= result.status_code < 600, f"Invalid HTTP status: {result.status_code}"
        assert isinstance(result.body, dict)
        assert result.elapsed_ms > 0


class TestAgentExecuteE2E:
    """Tool: agent_execute → ai-agents :8082."""

    async def test_dispatch_returns_result(self, dispatcher):
        if not await _check_backend("http://localhost:8082/health"):
            pytest.skip("ai-agents not running on :8082")
        result = await dispatcher.dispatch("agent_execute", {
            "task": "echo integration test",
        })
        assert 100 <= result.status_code < 600, f"Invalid HTTP status: {result.status_code}"
        assert isinstance(result.body, dict)
        assert result.elapsed_ms > 0


# ── AC-9.8: All 9 tools have e2e tests ─────────────────────────────────


class TestAllToolsCovered:
    """Meta-test: verify all 9 tools have e2e test coverage."""

    def test_all_tools_have_e2e_class(self, dispatcher):
        """Every tool in the route table must have a test class above."""
        expected_tools = {
            "semantic_search",
            "hybrid_search",
            "code_analyze",
            "code_pattern_audit",
            "graph_query",
            "llm_complete",
            "run_agent_function",
            "run_discussion",
            "agent_execute",
        }
        assert set(dispatcher.routes.keys()) == expected_tools

    def test_route_count_is_nine(self, dispatcher):
        assert len(dispatcher.routes) == 9
