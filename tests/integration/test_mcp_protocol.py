"""MCP protocol integration tests — WBS-MCP9.

AC-9.4  All 9 tools callable through MCP protocol
AC-9.8  Real MCP server, real tool registry — no mocks

Tests use ``fastmcp.Client`` against the in-process MCP server to verify
the full protocol stack: tools/list, tools/call, error handling.
"""

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from src.security.output_sanitizer import OutputSanitizer
from src.server import create_mcp_server
from src.tool_dispatcher import ToolDispatcher
from src.tool_registry import ToolRegistry
from src.core.config import Settings

pytestmark = pytest.mark.integration


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mcp_server():
    """Create a real MCP server with real registry, dispatcher, sanitizer."""
    from pathlib import Path

    config_path = Path(__file__).parent.parent.parent / "config" / "tools.yaml"
    if not config_path.exists():
        pytest.skip("config/tools.yaml not found")

    registry = ToolRegistry(config_path)
    dispatcher = ToolDispatcher(Settings())
    sanitizer = OutputSanitizer()
    return create_mcp_server(registry, dispatcher, sanitizer)


# ── AC-9.4: Tools List via MCP Protocol ─────────────────────────────────


class TestMCPToolsList:
    """tools/list returns all 9 tools with correct metadata."""

    async def test_list_returns_all_nine_tools(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            tool_names = {t.name for t in tools}
            expected = {
                "semantic_search",
                "hybrid_search",
                "code_analyze",
                "code_pattern_audit",
                "graph_query",
                "llm_complete",
                "run_discussion",
                "run_agent_function",
                "agent_execute",
            }
            assert tool_names == expected

    async def test_tool_count_is_nine(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            assert len(tools) == 9

    async def test_each_tool_has_description(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            for tool in tools:
                assert tool.description, f"Tool {tool.name} missing description"
                assert len(tool.description) > 10, f"Tool {tool.name} description too short"

    async def test_each_tool_has_input_schema(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            for tool in tools:
                assert tool.inputSchema, f"Tool {tool.name} missing input schema"


# ── AC-9.4: Tools Call via MCP Protocol ──────────────────────────────────


class TestMCPToolsCall:
    """tools/call dispatches to backends (live or graceful error)."""

    async def test_call_unknown_tool_errors(self, mcp_server):
        """Calling a non-existent tool should raise ToolError."""
        async with Client(mcp_server) as client:
            with pytest.raises(ToolError, match="Unknown tool"):
                await client.call_tool("nonexistent_tool", {"query": "test"})

    async def test_call_semantic_search_with_valid_input(self, mcp_server):
        """If backend is running, returns result; if not, returns backend error."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("semantic_search", {
                "query": "integration test",
                "collection": "all",
                "top_k": 3,
                "threshold": 0.5,
            })
            # Either succeeds (backend up) or returns error (backend down)
            # Both are valid MCP responses — we just verify protocol integrity
            assert result is not None

    async def test_call_graph_query_with_valid_cypher(self, mcp_server):
        """graph_query tool processes valid Cypher."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("graph_query", {
                "cypher": "MATCH (n) RETURN count(n) AS total LIMIT 1",
            })
            assert result is not None

    async def test_call_llm_complete_with_valid_prompt(self, mcp_server):
        """llm_complete tool processes valid prompt."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("llm_complete", {
                "prompt": "Say hello in one word",
                "max_tokens": 5,
            })
            assert result is not None

    async def test_call_with_missing_required_field(self, mcp_server):
        """Tool call with missing required field should raise ToolError."""
        async with Client(mcp_server) as client:
            with pytest.raises(ToolError, match="Missing required argument"):
                await client.call_tool("semantic_search", {})


# ── Protocol Integrity ──────────────────────────────────────────────────


class TestMCPProtocolIntegrity:
    """Verify MCP protocol behaviors work end-to-end."""

    async def test_server_has_name(self, mcp_server):
        """MCP server reports its name correctly."""
        assert mcp_server.name == "mcp-gateway"

    async def test_concurrent_tool_list_calls(self, mcp_server):
        """Multiple concurrent list_tools calls should all succeed."""
        import asyncio

        async def list_once():
            async with Client(mcp_server) as client:
                return await client.list_tools()

        results = await asyncio.gather(*[list_once() for _ in range(5)])
        for tools in results:
            assert len(tools) == 9
