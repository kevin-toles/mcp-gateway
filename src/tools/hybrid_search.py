"""hybrid_search tool handler — WBS-MCP8."""

from src.models.schemas import HybridSearchInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "hybrid_search"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def hybrid_search(
        query: str,
        collection: str = "all",
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> dict:
        """Combines semantic search with keyword matching for better precision."""
        validated = HybridSearchInput(
            query=query,
            collection=collection,
            top_k=top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return hybrid_search
