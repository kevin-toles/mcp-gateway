"""WBS-F7: `foundation_search` tool handler.

Routes to USS ``/v1/search/foundation`` for mathematical, statistical,
or theoretical underpinnings of software concepts.
"""

from __future__ import annotations

from src.models.schemas import FoundationSearchInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "foundation_search"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def foundation_search(
        query: str,
        domains: list[str] | None = None,
        include_graph_neighbors: bool = False,
        limit: int = 5,
    ) -> dict:
        """Search the scientific foundation layer for mathematical, statistical, or
        theoretical underpinnings of software concepts.
        """
        validated = FoundationSearchInput(
            query=query,
            domains=domains,
            include_graph_neighbors=include_graph_neighbors,
            limit=limit,
        )
        payload = validated.model_dump(exclude_none=False)
        # Drop the top_k alias field — backend only accepts limit
        payload.pop("top_k", None)
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return foundation_search
