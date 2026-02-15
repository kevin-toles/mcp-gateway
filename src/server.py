"""FastMCP server — WBS-MCP8 (GREEN).

Creates a FastMCP server instance with all 6 platform tools registered
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
    a2a_cancel_task,
    a2a_get_task,
    a2a_send_message,
    analyze_taxonomy_coverage,
    batch_extract_metadata,
    code_analyze,
    code_pattern_audit,
    convert_pdf,
    enhance_guideline,
    enrich_book_metadata,
    extract_book_metadata,
    generate_taxonomy,
    graph_query,
    hybrid_search,
    llm_complete,
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
    "a2a_send_message": a2a_send_message.create_handler,
    "a2a_get_task": a2a_get_task.create_handler,
    "a2a_cancel_task": a2a_cancel_task.create_handler,
    # Workflow tools (WBS-WF6)
    "convert_pdf": convert_pdf.create_handler,
    "extract_book_metadata": extract_book_metadata.create_handler,
    "batch_extract_metadata": batch_extract_metadata.create_handler,
    "generate_taxonomy": generate_taxonomy.create_handler,
    "enrich_book_metadata": enrich_book_metadata.create_handler,
    "enhance_guideline": enhance_guideline.create_handler,
    # Taxonomy Analysis (WBS-TAP9)
    "analyze_taxonomy_coverage": analyze_taxonomy_coverage.create_handler,
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
                f"No handler factory for tool '{tool_def.name}'. Available: {sorted(_HANDLER_FACTORIES.keys())}"
            )
        handler = factory(dispatcher, sanitizer)
        mcp.tool(
            name=tool_def.name,
            description=tool_def.description,
        )(handler)

    return mcp
