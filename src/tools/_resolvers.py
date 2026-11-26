"""Shared collection resolution helpers for MCP tool handlers.

All MCP tools that route to a vector search backend accept a human-readable
``source`` parameter and resolve it to the internal collection name here.
Callers never need to know internal collection or routing names.

Two resolution paths exist:

- ``resolve_search_collection`` — for ``hybrid_search`` / ``semantic_search``
  which dispatch to the hybrid_search / semantic_search backend routes.
  Valid source values: ``"all"``, ``"code"``, ``"docs"``, ``"textbooks"``.

- ``resolve_shelf_collection`` — for ``search_in`` which dispatches to the
  ``knowledge_refine`` backend route.
  Valid source values: ``"textbooks"``, ``"code"``, ``"patterns"``, ``"diagrams"``.
"""

from __future__ import annotations

# ── hybrid_search / semantic_search backend routing names ────────────────────

_SEARCH_COLLECTION_MAP: dict[str, str] = {
    "all": "all",  # search across all collections (default)
    "code": "code",  # CRE code implementation examples
    "docs": "docs",  # documentation
    "textbooks": "chapters",  # book text: every level lives in `chapters`, reached by granularity (ETDI §5.1.8)
}

# ── knowledge_refine backend Qdrant collection names ─────────────────────────

_SHELF_COLLECTION_MAP: dict[str, str] = {
    # ETDI-1.0 (v0.48, S3/S4): the `chunks` collection is retired; book text at every level (chapter, passage,
    # paragraph group) lives in `chapters` with its text in `content`, reached by a `granularity` filter.
    "textbooks": "chapters",
    "code": "code_chunks",
    "patterns": "code_good_patterns",
    "diagrams": "ascii_diagrams",
}


#: Level each kind of call asks `chapters` for (ETDI §5.1.8): broad search returns chapters, targeted refining passages.
SEARCH_GRANULARITY = "chapter"
SHELF_GRANULARITY = "passage"


def granularity_for(collection: str, level: str) -> str | None:
    """The `granularity` to send with `collection` — only `chapters` is multi-level; other collections take none."""
    return level if collection == "chapters" else None


def resolve_search_collection(source: str) -> str:
    """Map a human-readable source name to a hybrid/semantic backend collection value.

    Args:
        source: One of ``"all"``, ``"code"``, ``"docs"``, ``"textbooks"``.

    Returns:
        The internal collection routing string accepted by the backend.

    Raises:
        ValueError: If ``source`` is not a recognised value.
    """
    resolved = _SEARCH_COLLECTION_MAP.get(source)
    if resolved is None:
        valid = ", ".join(f'"{k}"' for k in _SEARCH_COLLECTION_MAP)
        msg = f"Unknown source {source!r}. Valid values: {valid}"
        raise ValueError(msg)
    return resolved


def resolve_shelf_collection(source: str) -> str:
    """Map a human-readable shelf name to a knowledge_refine Qdrant collection name.

    Args:
        source: One of ``"textbooks"``, ``"code"``, ``"patterns"``, ``"diagrams"``.

    Returns:
        The internal Qdrant collection name string.

    Raises:
        ValueError: If ``source`` is not a recognised value.
    """
    resolved = _SHELF_COLLECTION_MAP.get(source)
    if resolved is None:
        valid = ", ".join(f'"{k}"' for k in _SHELF_COLLECTION_MAP)
        msg = f"Unknown source {source!r}. Valid values: {valid}"
        raise ValueError(msg)
    return resolved
