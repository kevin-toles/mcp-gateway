"""hybrid_search tool handler — WBS-MCP8 / WBS-TXS5.

Accepts a human-readable ``source`` parameter and resolves it to the
internal backend collection name via ``resolve_search_collection()``.
Callers never see or need to know internal collection routing values.
"""

from src.models.schemas import HybridSearchInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher
from src.tools._resolvers import SEARCH_GRANULARITY, granularity_for, resolve_search_collection

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
        # Code facets (ETDI BEH-06.04 step 5) — applied to code_chunks
        structural_types: list[str] | None = None,
        tested_structural_types: list[str] | None = None,
        is_test_unit: bool | None = None,
        subject_subdomains: list[str] | None = None,
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
                - "textbooks" — book chapters
            structural_types: code units whose structural types include any of these (e.g. ["http_endpoint"]).
            tested_structural_types: test units linked to code of any of these types — "how to test an HTTP
                endpoint" is is_test_unit=True plus tested_structural_types=["http_endpoint"].
            is_test_unit: True for test code only, False for non-test code only.
            subject_subdomains: test units whose subject — the code they test — is in any of these subdomains
                (e.g. ["filesystems-implementation"]).
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
            structural_types=structural_types,
            tested_structural_types=tested_structural_types,
            is_test_unit=is_test_unit,
            subject_subdomains=subject_subdomains,
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
        granularity = granularity_for(validated.collection, SEARCH_GRANULARITY)
        if granularity:
            payload["granularity"] = granularity
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
        # Code facets (ETDI BEH-06.04 step 5): omitted when None so absent fields change nothing
        for facet in ("structural_types", "tested_structural_types", "is_test_unit", "subject_subdomains"):
            value = getattr(validated, facet)
            if value is not None:
                payload[facet] = value
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return hybrid_search
