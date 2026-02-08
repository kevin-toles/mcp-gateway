"""FastMCP server — WBS-MCP8 (GREEN).

Creates a FastMCP server instance with all 9 platform tools registered
from the ``ToolRegistry``.  Each tool handler validates input via its
Pydantic model, dispatches to the appropriate backend service, and
sanitizes the output before returning.

Reference: AC-8.1 (SSE/HTTP transport), AC-8.2 (tools/list),
           AC-8.3 (tools/call pipeline), AC-8.4 (registry-driven)
"""

from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP

from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher
from src.tool_registry import ToolRegistry
from src.tools import (
    agent_execute,
    code_analyze,
    code_pattern_audit,
    graph_query,
    hybrid_search,
    llm_complete,
    run_agent_function,
    run_discussion,
    semantic_search,
)

# ── Handler factory mapping ────────────────────────────────────────────

_HANDLER_FACTORIES: dict[str, Callable[..., Any]] = {
    "semantic_search": semantic_search.create_handler,
    "hybrid_search": hybrid_search.create_handler,
    "code_analyze": code_analyze.create_handler,
    "code_pattern_audit": code_pattern_audit.create_handler,
    "graph_query": graph_query.create_handler,
    "llm_complete": llm_complete.create_handler,
    "run_discussion": run_discussion.create_handler,
    "run_agent_function": run_agent_function.create_handler,
    "agent_execute": agent_execute.create_handler,
}


# ── Server factory ─────────────────────────────────────────────────────


def create_mcp_server(
    registry: ToolRegistry,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
) -> FastMCP:
    """Create a FastMCP server with all tools from the registry.

    Each tool is registered with its YAML-defined description and a
    handler that performs: validate → dispatch → sanitize → return.

    Args:
        registry:   Loaded ``ToolRegistry`` (from YAML config).
        dispatcher: ``ToolDispatcher`` for backend HTTP calls.
        sanitizer:  ``OutputSanitizer`` (passthrough in Phase 1).

    Returns:
        A configured ``FastMCP`` instance ready for SSE/HTTP transport.
    """
    mcp = FastMCP(
        name="mcp-gateway",
    )

    for tool_def in registry.list_all():
        factory = _HANDLER_FACTORIES.get(tool_def.name)
        if factory is None:
            raise ValueError(
                f"No handler factory for tool '{tool_def.name}'. "
                f"Available: {sorted(_HANDLER_FACTORIES.keys())}"
            )
        handler = factory(dispatcher, sanitizer)
        mcp.tool(
            name=tool_def.name,
            description=tool_def.description,
        )(handler)

    return mcp
