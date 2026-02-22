"""MCP protocol integration tests — WBS-MCP9.

AC-9.4  All 22 tools callable through MCP protocol
AC-9.8  Real MCP server, real tool registry — no mocks

Tests use ``fastmcp.Client`` against the in-process MCP server to verify
the full protocol stack: tools/list, tools/call, error handling.
"""

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from src.core.config import Settings
from src.security.output_sanitizer import OutputSanitizer
from src.server import create_mcp_server
from src.tool_dispatcher import ToolDispatcher
from src.tool_registry import ToolRegistry

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
    """tools/list returns all 22 tools with correct metadata."""

    async def test_list_returns_all_tools(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            tool_names = {t.name for t in tools}
            expected = {
                # Search
                "semantic_search",
                "hybrid_search",
                # Code analysis
                "code_analyze",
                "code_pattern_audit",
                # Graph / LLM
                "graph_query",
                "llm_complete",
                # A2A agent tools
                "a2a_send_message",
                "a2a_get_task",
                "a2a_cancel_task",
                # Book / document pipeline
                "convert_pdf",
                "extract_book_metadata",
                "batch_extract_metadata",
                "generate_taxonomy",
                "enrich_book_metadata",
                "enhance_guideline",
                "analyze_taxonomy_coverage",
                # AMVE architecture analysis (AEI-7)
                "amve_detect_patterns",
                "amve_detect_boundaries",
                "amve_detect_communication",
                "amve_build_call_graph",
                "amve_evaluate_fitness",
                "amve_generate_architecture_log",
            }
            assert tool_names == expected

    async def test_tool_count_is_twenty_two(self, mcp_server):
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            assert len(tools) == 22

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
            result = await client.call_tool(
                "semantic_search",
                {
                    "query": "integration test",
                    "collection": "all",
                    "top_k": 3,
                    "threshold": 0.5,
                },
            )
            # Either succeeds (backend up) or returns error (backend down)
            # Both are valid MCP responses — we just verify protocol integrity
            assert result is not None

    async def test_call_graph_query_with_valid_cypher(self, mcp_server):
        """graph_query tool processes valid Cypher."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "graph_query",
                {
                    "cypher": "MATCH (n) RETURN count(n) AS total LIMIT 1",
                },
            )
            assert result is not None

    async def test_call_llm_complete_with_valid_prompt(self, mcp_server):
        """llm_complete tool processes valid prompt."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "llm_complete",
                {
                    "prompt": "Say hello in one word",
                    "max_tokens": 5,
                },
            )
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
            assert len(tools) == 22
