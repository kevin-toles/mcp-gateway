"""hybrid_search tool handler — WBS-MCP8 / WBS-TXS5.

Accepts a human-readable ``source`` parameter and resolves it to the
internal backend collection name via ``resolve_search_collection()``.
Callers never see or need to know internal collection routing values.
"""

from src.models.schemas import HybridSearchInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher
from src.tools._resolvers import resolve_search_collection

TOOL_NAME = "hybrid_search"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def hybrid_search(
        query: str,
        source: str = "all",
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        bloom_tier_filter: list[int] | None = None,
        quality_tier_filter: list[int] | None = None,
        bloom_tier_boost: bool = True,
        # Graph control
        include_graph: bool = True,
        # MMR reranking (score/MMR traversal)
        mmr_rerank: bool = False,
        mmr_lambda: float = 0.5,
        # Taxonomy query expansion
        expand_taxonomy: bool = False,
        # Custom traversal / domain focus
        focus_areas: list[str] | None = None,
        focus_keywords: list[str] | None = None,
    ) -> dict:
        """Combines semantic search with keyword matching for better precision.

        Search modes via parameters:
        - Basic/eager: default settings, pure vector+graph fusion
        - MMR traversal: mmr_rerank=True — diversity-aware reranking via MMR
        - Score traversal: mmr_rerank=False, semantic_weight controls score mix
        - Custom traversal: focus_areas + focus_keywords + include_graph for domain-specific scoring
        - Taxonomy-expanded: expand_taxonomy=True — expands query via Neo4j SIMILAR_TO edges

        Args:
            source: Knowledge collection to search. One of:
                - "all"       — search across all collections (default)
                - "code"      — CRE code implementation examples
                - "docs"      — documentation
                - "textbooks" — textbook chapter prose
        """
        collection = resolve_search_collection(source)
        validated = HybridSearchInput(
            query=query,
            collection=collection,
            top_k=top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            bloom_tier_filter=bloom_tier_filter,
            quality_tier_filter=quality_tier_filter,
            bloom_tier_boost=bloom_tier_boost,
            include_graph=include_graph,
            mmr_rerank=mmr_rerank,
            mmr_lambda=mmr_lambda,
            expand_taxonomy=expand_taxonomy,
            focus_areas=focus_areas,
            focus_keywords=focus_keywords,
        )
        # Map MCP param names → semantic-search API param names
        payload: dict = {
            "query": validated.query,
            "collection": validated.collection,
            "limit": validated.top_k,
            "alpha": validated.semantic_weight,
            # TXS5: always include tier_boost (bool) so downstream honours it
            "tier_boost": validated.bloom_tier_boost,
            # Graph scoring
            "include_graph": validated.include_graph,
            # MMR reranking
            "mmr_rerank": validated.mmr_rerank,
            "mmr_lambda": validated.mmr_lambda,
            # Taxonomy expansion
            "expand_taxonomy": validated.expand_taxonomy,
        }
        # Conditionally include tier filters (omit when None for backward compat)
        if validated.bloom_tier_filter is not None:
            payload["bloom_tier_filter"] = validated.bloom_tier_filter
        if validated.quality_tier_filter is not None:
            payload["quality_tier_filter"] = validated.quality_tier_filter
        # Conditionally include custom focus fields (omit when None)
        if validated.focus_areas is not None:
            payload["focus_areas"] = validated.focus_areas
        if validated.focus_keywords is not None:
            payload["focus_keywords"] = validated.focus_keywords
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return hybrid_search
