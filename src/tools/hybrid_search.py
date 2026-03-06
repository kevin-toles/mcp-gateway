"""hybrid_search tool handler — WBS-MCP8 / WBS-TXS5."""

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
        bloom_tier_filter: list[int] | None = None,
        quality_tier_filter: list[int] | None = None,
        bloom_tier_boost: bool = True,
    ) -> dict:
        """Combines semantic search with keyword matching for better precision."""
        validated = HybridSearchInput(
            query=query,
            collection=collection,
            top_k=top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            bloom_tier_filter=bloom_tier_filter,
            quality_tier_filter=quality_tier_filter,
            bloom_tier_boost=bloom_tier_boost,
        )
        # Map MCP param names → semantic-search API param names
        payload: dict = {
            "query": validated.query,
            "collection": validated.collection,
            "limit": validated.top_k,
            "alpha": validated.semantic_weight,
            # TXS5: always include tier_boost (bool) so downstream honours it
            "tier_boost": validated.bloom_tier_boost,
        }
        # Conditionally include tier filters (omit when None for backward compat)
        if validated.bloom_tier_filter is not None:
            payload["bloom_tier_filter"] = validated.bloom_tier_filter
        if validated.quality_tier_filter is not None:
            payload["quality_tier_filter"] = validated.quality_tier_filter
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return hybrid_search
