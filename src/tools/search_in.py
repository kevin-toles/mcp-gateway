"""search_in facade handler — MCP-F-2 (GREEN).

Intent-level entry point for targeted knowledge shelf search.
Accepts a human-readable source name and maps it to the correct internal
collection. Exposes only three parameters to FastMCP schema generation:
query, source, max_results.

Reference: Shim layer abstraction pattern (CRE-3); trust boundary
enforcement (KB-6); enum-bounded routing signals (CRE-2).
"""

from __future__ import annotations

from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "search_in"

# ── Source → collection mapping ───────────────────────────────────────────────

_COLLECTION_MAP: dict[str, str] = {
    "textbooks": "chapters",
    "code": "code_chunks",
    "patterns": "code_good_patterns",
    "diagrams": "ascii_diagrams",
}


def _resolve_source(source: str) -> str:
    """Map a human-readable shelf name to an internal collection identifier.

    Args:
        source: One of "textbooks", "code", "patterns", "diagrams".

    Returns:
        The internal collection name string.

    Raises:
        ValueError: If the source value is not recognised.
    """
    resolved = _COLLECTION_MAP.get(source)
    if resolved is None:
        valid = ", ".join(f'"{k}"' for k in _COLLECTION_MAP)
        msg = f"Unknown source {source!r}. Valid values: {valid}"
        raise ValueError(msg)
    return resolved


def create_handler(dispatcher: ToolDispatcher, sanitizer=None):  # noqa: ANN001
    """Return an async handler with a 3-param signature for FastMCP schema generation.

    The returned handler exposes exactly: query, source, max_results.
    mmr_rerank and mmr_lambda are hardcoded and never appear in the schema.

    Args:
        dispatcher: ToolDispatcher for backend HTTP calls.
        sanitizer:  Accepted for compatibility with server.py factory calls;
                    not used — facade layer returns raw dispatcher result.
    """

    async def search_in(
        query: str,
        source: str = "textbooks",
        max_results: int = 5,
    ) -> dict:
        """Search within a specific knowledge shelf.

        More targeted than `ask` — use when you know the type of content you need.

        Args:
            query: The question or concept to search for.
            source: Knowledge shelf to search. One of:
                - "textbooks" — textbook chapter prose (default)
                - "code" — CRE code implementation examples
                - "patterns" — design pattern / anti-pattern instances
                - "diagrams" — architecture ASCII diagrams
            max_results: Maximum number of results to return (default 5).
        """
        collection = _resolve_source(source)

        payload: dict = {
            "query": query,
            "collection": collection,
            "limit": max_results,
            "mmr_rerank": True,
            "include_graph": True,
            "tier_boost": True,
            "expand_taxonomy": False,
        }

        result = await dispatcher.dispatch("knowledge_refine", payload)
        return result.body

    return search_in
