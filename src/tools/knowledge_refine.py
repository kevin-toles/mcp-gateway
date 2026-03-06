"""knowledge_refine tool handler — Issue #6: targeted single-collection KB search.

Surgical, precision search within a single named collection. Unlike
knowledge_search (broad fan-out), this tool stays within the specified
collection and applies MMR reranking by default for within-collection diversity.

Use this tool when:
- You already know which collection holds the relevant content
- You want to narrow down results within chapters or code_chunks
- You need filtered results (Bloom tier, CRE quality tier)
"""

from src.models.schemas import KnowledgeRefineInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "knowledge_refine"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def knowledge_refine(
        query: str,
        collection: str = "chapters",
        limit: int = 5,
        bloom_tier_filter: list[int] | None = None,
        quality_tier_filter: list[int] | None = None,
        mmr_rerank: bool = True,
    ) -> dict:
        """Targeted search within a single KB collection.

        Stays within the specified collection rather than fanning out across all
        collections. MMR reranking is ON by default to maximise within-collection
        diversity (turn off for ranked-list retrieval).

        Args:
            query: Natural language question or concept to search for.
            collection: Collection to search. Valid values:
                - 'chapters' / 'textbooks' — textbook chapter prose
                - 'code_chunks' / 'code'   — CRE code examples
                - 'pattern_instances' / 'patterns' — architectural pattern instances
                - 'code_good_patterns'     — canonical good-code examples
                - 'repo_concepts' / 'concepts' — repository concept extractions
            limit: Maximum results (1-20, default 5).
            bloom_tier_filter: Filter chapters by Bloom cognitive tier (0-6).
                E.g., [0, 1, 2] for Foundational/Comprehension tiers.
            quality_tier_filter: Filter code_chunks by CRE tier (1=flagship,
                2=standard, 3=supplemental).
            mmr_rerank: Apply MMR diversity reranking (default True).
        """
        validated = KnowledgeRefineInput(
            query=query,
            collection=collection,
            limit=limit,
            bloom_tier_filter=bloom_tier_filter,
            quality_tier_filter=quality_tier_filter,
            mmr_rerank=mmr_rerank,
        )
        payload: dict = {
            "query": validated.query,
            "collection": validated.collection,
            "limit": validated.limit,
            "include_graph": True,
            "tier_boost": True,
            "expand_taxonomy": False,
            "mmr_rerank": validated.mmr_rerank,
        }
        if validated.bloom_tier_filter is not None:
            payload["bloom_tier_filter"] = validated.bloom_tier_filter
        if validated.quality_tier_filter is not None:
            payload["quality_tier_filter"] = validated.quality_tier_filter
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return knowledge_refine
