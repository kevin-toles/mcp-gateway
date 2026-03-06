"""knowledge_search tool handler — Issue #6: consolidated KB search.

Batteries-included search across the full knowledge base (all 6 primary
collections: chapters, code_chunks, textbook_code, pattern_instances,
code_good_patterns, repo_concepts) with taxonomy query expansion ON by default.

Use this tool when you want broad knowledge retrieval from the textbook
library. Prefer `knowledge_refine` when you know the target collection.
"""

from src.models.schemas import KnowledgeSearchInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "knowledge_search"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def knowledge_search(
        query: str,
        limit: int = 10,
        expand_taxonomy: bool = True,
        mmr_rerank: bool = False,
        mmr_lambda: float = 0.5,
        bloom_tier_filter: list[int] | None = None,
    ) -> dict:
        """Search all KB collections (textbooks, code, patterns) with taxonomy expansion.

        Fans out across chapters, code_chunks, textbook_code, pattern_instances,
        code_good_patterns, and repo_concepts in a single call. Taxonomy expansion
        is ON by default — the router selects the domain taxonomy that best matches
        the query (AI/ML, DevSecOps, C++, Code Defects, Integration, FP, or uber).

        Args:
            query: Natural language question or concept to retrieve knowledge for.
            limit: Maximum results to return (1-50, default 10).
            expand_taxonomy: Expand query via Neo4j SIMILAR_TO edges for related
                concepts. Default True (disable only for exact-match queries).
            mmr_rerank: Apply Maximal Marginal Relevance reranking for diversity.
                Useful when results from multiple collections are too similar.
            mmr_lambda: MMR tuning — 0.0=maximum diversity, 1.0=maximum relevance.
                Only used when mmr_rerank=True.
            bloom_tier_filter: Restrict chapter results to specific Bloom cognitive
                tiers. E.g., [5, 6] for Evaluation/Innovation-tier only.
        """
        validated = KnowledgeSearchInput(
            query=query,
            limit=limit,
            expand_taxonomy=expand_taxonomy,
            mmr_rerank=mmr_rerank,
            mmr_lambda=mmr_lambda,
            bloom_tier_filter=bloom_tier_filter,
        )
        payload: dict = {
            "query": validated.query,
            "collection": "all",
            "limit": validated.limit,
            "include_graph": True,
            "tier_boost": True,
            "expand_taxonomy": validated.expand_taxonomy,
            "mmr_rerank": validated.mmr_rerank,
            "mmr_lambda": validated.mmr_lambda,
        }
        if validated.bloom_tier_filter is not None:
            payload["bloom_tier_filter"] = validated.bloom_tier_filter
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return knowledge_search
