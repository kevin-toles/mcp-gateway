"""ask facade handler — MCP-F-1 (GREEN).

Intent-level entry point for general software-engineering questions.
Encapsulates all hybrid_search routing parameters; exposes only three
parameters to FastMCP schema generation: query, max_results, difficulty.

Reference: Shim layer decoupling pattern (CRE-3); enum-bounded routing
signals (CRE-2); MCP JSON-RPC tool handler pattern (KB-1).
"""

from __future__ import annotations

from src.models.schemas import HybridSearchInput
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "ask"

# ── Difficulty → Bloom tier mapping ──────────────────────────────────────────

_DIFFICULTY_MAP: dict[str, list[int]] = {
    "beginner": [0, 1, 2],
    "intermediate": [3, 4],
    "advanced": [4, 5, 6],
}


def _resolve_difficulty(difficulty: str | None) -> list[int] | None:
    """Map a human-readable difficulty label to a Bloom tier filter.

    Args:
        difficulty: One of "beginner", "intermediate", "advanced", or None.

    Returns:
        A list of Bloom tier integers, or None when no difficulty specified.

    Raises:
        ValueError: If the difficulty value is not recognised.
    """
    if difficulty is None:
        return None
    resolved = _DIFFICULTY_MAP.get(difficulty)
    if resolved is None:
        valid = ", ".join(f'"{k}"' for k in _DIFFICULTY_MAP)
        msg = f"Unknown difficulty {difficulty!r}. Valid values: {valid}"
        raise ValueError(msg)
    return resolved


def create_handler(dispatcher: ToolDispatcher, sanitizer=None):  # noqa: ANN001
    """Return an async handler with a 3-param signature for FastMCP schema generation.

    The returned handler exposes exactly: query, max_results, difficulty.
    All other hybrid_search parameters are hardcoded and never appear in
    the FastMCP tools/list schema.

    Args:
        dispatcher: ToolDispatcher for backend HTTP calls.
        sanitizer:  Accepted for compatibility with server.py factory calls;
                    not used — facade layer returns raw dispatcher result.
    """

    async def ask(
        query: str,
        max_results: int = 10,
        difficulty: str | None = None,
    ) -> dict:
        """Answer a general software-engineering question from the knowledge base.

        Uses hybrid search (semantic + keyword fusion) with taxonomy expansion
        enabled by default. Optionally filter results by Bloom cognitive tier.

        Args:
            query: The software-engineering question or concept to search for.
            max_results: Maximum number of results to return (default 10).
            difficulty: Optional difficulty filter — "beginner", "intermediate",
                        or "advanced". Constrains results by Bloom taxonomy tier.
        """
        bloom_tier_filter = _resolve_difficulty(difficulty)

        validated = HybridSearchInput(
            query=query,
            top_k=max_results,
            expand_taxonomy=True,
            mmr_rerank=False,
            bloom_tier_boost=True,
            include_graph=True,
            bloom_tier_filter=bloom_tier_filter,
        )

        payload: dict = {
            "query": validated.query,
            "collection": validated.collection,
            "limit": validated.top_k,
            "expand_taxonomy": validated.expand_taxonomy,
            "mmr_rerank": validated.mmr_rerank,
            "tier_boost": validated.bloom_tier_boost,
            "include_graph": validated.include_graph,
        }
        if validated.bloom_tier_filter is not None:
            payload["bloom_tier_filter"] = validated.bloom_tier_filter

        result = await dispatcher.dispatch("hybrid_search", payload)
        return result.body

    return ask
