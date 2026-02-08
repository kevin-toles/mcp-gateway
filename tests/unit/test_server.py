"""Tests for MCP server — WBS-MCP8 (RED).

AC-8.1 (SSE transport), AC-8.2 (tools/list 9 tools), AC-8.3 (tools/call pipeline).
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client, FastMCP

from src.core.errors import BackendUnavailableError, ToolTimeoutError
from src.security.output_sanitizer import OutputSanitizer
from src.server import create_mcp_server
from src.tool_dispatcher import DispatchResult
from src.tool_registry import ToolRegistry

# ── Constants & Fixtures ────────────────────────────────────────────────

EXPECTED_TOOL_NAMES = {
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

VALID_TOOLS_YAML = """\
tools:
  - name: semantic_search
    description: "Search using semantic similarity"
    tier: bronze
    tags: [search, rag]
  - name: hybrid_search
    description: "Hybrid semantic + keyword search"
    tier: bronze
    tags: [search, rag]
  - name: code_analyze
    description: "Analyze code quality"
    tier: silver
    tags: [code, analysis]
  - name: code_pattern_audit
    description: "Audit code anti-patterns"
    tier: silver
    tags: [code, audit]
  - name: graph_query
    description: "Query Neo4j knowledge graph"
    tier: gold
    tags: [graph, query]
  - name: llm_complete
    description: "LLM completion with fallback"
    tier: gold
    tags: [llm, completion]
  - name: run_agent_function
    description: "Run single agent function"
    tier: gold
    tags: [agent, function]
  - name: run_discussion
    description: "Multi-LLM discussion"
    tier: enterprise
    tags: [agent, discussion]
  - name: agent_execute
    description: "Autonomous agent execution"
    tier: enterprise
    tags: [agent, execution]
"""


def _make_dispatch_result(body=None, status_code=200, elapsed_ms=42.0):
    return DispatchResult(
        status_code=status_code,
        body=body or {"results": [{"text": "hello"}]},
        headers={"content-type": "application/json"},
        elapsed_ms=elapsed_ms,
    )


@pytest.fixture()
def registry(tmp_path: Path) -> ToolRegistry:
    p = tmp_path / "tools.yaml"
    p.write_text(VALID_TOOLS_YAML)
    return ToolRegistry(p)


@pytest.fixture()
def mock_dispatcher() -> AsyncMock:
    dispatcher = AsyncMock()
    dispatcher.dispatch = AsyncMock(return_value=_make_dispatch_result())
    return dispatcher


@pytest.fixture()
def sanitizer() -> OutputSanitizer:
    return OutputSanitizer()


@pytest.fixture()
def mcp_server(registry, mock_dispatcher, sanitizer) -> FastMCP:
    return create_mcp_server(registry, mock_dispatcher, sanitizer)


# ── Server Creation ─────────────────────────────────────────────────────


class TestCreateMCPServer:
    def test_returns_fastmcp(self, mcp_server):
        assert isinstance(mcp_server, FastMCP)

    def test_server_name(self, mcp_server):
        assert mcp_server.name == "mcp-gateway"


# ── tools/list (AC-8.2) ────────────────────────────────────────────────


class TestToolsList:
    async def test_returns_9_tools(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        assert len(tools) == 9

    async def test_tool_names(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == EXPECTED_TOOL_NAMES

    async def test_each_tool_has_description(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        for tool in tools:
            assert tool.description, f"{tool.name} missing description"

    async def test_each_tool_has_input_schema(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        for tool in tools:
            schema = tool.inputSchema
            assert schema, f"{tool.name} missing inputSchema"
            assert "properties" in schema, f"{tool.name} schema has no properties"

    async def test_semantic_search_schema_fields(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        ss = next(t for t in tools if t.name == "semantic_search")
        props = ss.inputSchema["properties"]
        assert "query" in props
        assert "collection" in props
        assert "top_k" in props
        assert "threshold" in props

    async def test_hybrid_search_schema_fields(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        hs = next(t for t in tools if t.name == "hybrid_search")
        props = hs.inputSchema["properties"]
        assert "query" in props
        assert "semantic_weight" in props
        assert "keyword_weight" in props

    async def test_graph_query_schema_fields(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        gq = next(t for t in tools if t.name == "graph_query")
        props = gq.inputSchema["properties"]
        assert "cypher" in props
        assert "parameters" in props

    async def test_llm_complete_schema_fields(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        lc = next(t for t in tools if t.name == "llm_complete")
        props = lc.inputSchema["properties"]
        assert "prompt" in props
        assert "temperature" in props
        assert "max_tokens" in props
        assert "model_preference" in props

    async def test_agent_execute_schema_fields(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        ae = next(t for t in tools if t.name == "agent_execute")
        props = ae.inputSchema["properties"]
        assert "task" in props
        assert "max_steps" in props

    async def test_run_discussion_schema_fields(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        rd = next(t for t in tools if t.name == "run_discussion")
        props = rd.inputSchema["properties"]
        assert "protocol_id" in props
        assert "topic" in props
        assert "context" in props

    async def test_semantic_search_query_is_required(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        ss = next(t for t in tools if t.name == "semantic_search")
        assert "query" in ss.inputSchema.get("required", [])


# ── tools/call — valid inputs (AC-8.3) ─────────────────────────────────


class TestToolsCallValid:
    async def test_semantic_search(self, mcp_server, mock_dispatcher):
        async with Client(mcp_server) as client:
            result = await client.call_tool("semantic_search", {"query": "test search"})
        assert not result.is_error
        mock_dispatcher.dispatch.assert_called_once()
        args = mock_dispatcher.dispatch.call_args
        assert args[0][0] == "semantic_search"
        assert args[0][1]["query"] == "test search"

    async def test_hybrid_search(self, mcp_server, mock_dispatcher):
        async with Client(mcp_server) as client:
            result = await client.call_tool("hybrid_search", {"query": "test"})
        assert not result.is_error
        mock_dispatcher.dispatch.assert_called_once()
        assert mock_dispatcher.dispatch.call_args[0][0] == "hybrid_search"

    async def test_code_analyze(self, mcp_server, mock_dispatcher):
        async with Client(mcp_server) as client:
            result = await client.call_tool("code_analyze", {"code": "print('hi')"})
        assert not result.is_error

    async def test_code_pattern_audit(self, mcp_server, mock_dispatcher):
        async with Client(mcp_server) as client:
            result = await client.call_tool("code_pattern_audit", {"code": "x = 1"})
        assert not result.is_error

    async def test_graph_query(self, mcp_server, mock_dispatcher):
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "graph_query", {"cypher": "MATCH (n) RETURN n LIMIT 5"},
            )
        assert not result.is_error

    async def test_llm_complete(self, mcp_server, mock_dispatcher):
        async with Client(mcp_server) as client:
            result = await client.call_tool("llm_complete", {"prompt": "Hello world"})
        assert not result.is_error

    async def test_run_discussion(self, mcp_server, mock_dispatcher):
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "run_discussion",
                {"protocol_id": "ROUNDTABLE_DISCUSSION", "topic": "Testing"},
            )
        assert not result.is_error

    async def test_run_agent_function(self, mcp_server, mock_dispatcher):
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "run_agent_function", {"function_name": "summarize-content"},
            )
        assert not result.is_error

    async def test_agent_execute(self, mcp_server, mock_dispatcher):
        async with Client(mcp_server) as client:
            result = await client.call_tool("agent_execute", {"task": "Do something"})
        assert not result.is_error


# ── tools/call — response handling (AC-8.3) ────────────────────────────


class TestToolsCallResponse:
    async def test_returns_backend_body(self, mcp_server, mock_dispatcher):
        mock_dispatcher.dispatch.return_value = _make_dispatch_result(
            body={"results": [{"score": 0.95, "text": "match"}]},
        )
        async with Client(mcp_server) as client:
            result = await client.call_tool("semantic_search", {"query": "test"})
        content_text = result.content[0].text
        body = json.loads(content_text)
        assert body["results"][0]["score"] == 0.95

    async def test_output_passes_through_sanitizer(self, mcp_server, mock_dispatcher):
        """Phase 1: sanitizer is passthrough, data unchanged."""
        original = {"data": "sensitive", "nested": {"key": "value"}}
        mock_dispatcher.dispatch.return_value = _make_dispatch_result(body=original)
        async with Client(mcp_server) as client:
            result = await client.call_tool("semantic_search", {"query": "test"})
        body = json.loads(result.content[0].text)
        assert body == original

    async def test_dispatches_validated_payload(self, mcp_server, mock_dispatcher):
        """Handler validates input via Pydantic before dispatch."""
        async with Client(mcp_server) as client:
            await client.call_tool(
                "semantic_search",
                {"query": "test", "collection": "code", "top_k": 5, "threshold": 0.8},
            )
        payload = mock_dispatcher.dispatch.call_args[0][1]
        assert payload["query"] == "test"
        assert payload["collection"] == "code"
        assert payload["top_k"] == 5
        assert payload["threshold"] == 0.8


# ── tools/call — error handling (AC-8.3) ───────────────────────────────


class TestToolsCallErrors:
    async def test_invalid_input_returns_error(self, mcp_server):
        """Pydantic validation error (min_length=1) surfaces as MCP tool error."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "semantic_search", {"query": ""}, raise_on_error=False,
            )
        assert result.is_error

    async def test_missing_required_field_returns_error(self, mcp_server):
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "semantic_search", {}, raise_on_error=False,
            )
        assert result.is_error

    async def test_cypher_injection_blocked(self, mcp_server):
        """Cypher write operations rejected by input validation."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "graph_query", {"cypher": "CREATE (n:Test)"}, raise_on_error=False,
            )
        assert result.is_error

    async def test_backend_unavailable_returns_error(self, mcp_server, mock_dispatcher):
        mock_dispatcher.dispatch.side_effect = BackendUnavailableError(
            "semantic-search", "Connection refused",
        )
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "semantic_search", {"query": "test"}, raise_on_error=False,
            )
        assert result.is_error

    async def test_timeout_returns_error(self, mcp_server, mock_dispatcher):
        mock_dispatcher.dispatch.side_effect = ToolTimeoutError("semantic_search", 30.0)
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "semantic_search", {"query": "test"}, raise_on_error=False,
            )
        assert result.is_error


# ── SSE Transport (AC-8.1) ──────────────────────────────────────────────


class TestSSETransport:
    def test_http_app_creates_starlette(self, mcp_server):
        app = mcp_server.http_app(transport="sse")
        assert app is not None

    def test_sse_route_exists(self, mcp_server):
        app = mcp_server.http_app(transport="sse")
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/sse" in paths

    def test_messages_route_exists(self, mcp_server):
        app = mcp_server.http_app(transport="sse")
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/messages" in paths
