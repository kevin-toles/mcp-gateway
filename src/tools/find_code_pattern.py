"""MCP-F-3: `find_code_pattern` facade handler.

Intent-level tool that surfaces code pattern search to LLM clients using
user-friendly `examples` vocabulary ("good", "bad", "both") instead of the
internal `pattern_type` values accepted by the `pattern_search` backend route.

The facade resolves `examples` → `collection` and builds a payload that USS
``/v1/search/hybrid`` accepts directly, avoiding the ``pattern_type`` field
that USS rejects as an extra/forbidden body parameter.
"""

from __future__ import annotations

from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "find_code_pattern"

# Maps user-facing `examples` values to USS collection names.
# Must stay in sync with pattern_search._PATTERN_COLLECTION_MAP.
_EXAMPLES_TO_COLLECTION: dict[str, str] = {
    "good": "code_good_patterns",
    "bad": "code_bad_patterns",
    "both": "all",
}


def _resolve_collection(examples: str) -> str:
    collection = _EXAMPLES_TO_COLLECTION.get(examples)
    if collection is None:
        valid = ", ".join(f'"{k}"' for k in _EXAMPLES_TO_COLLECTION)
        raise ValueError(f"Unknown examples {examples!r}. Valid values: {valid}")
    return collection


def create_handler(dispatcher: ToolDispatcher, sanitizer=None):
    async def find_code_pattern(query: str, examples: str = "both") -> dict:
        collection = _resolve_collection(examples)
        payload = {
            "query": query,
            "collection": collection,
            "limit": 10,
            "include_graph": True,
            "tier_boost": True,
            "expand_taxonomy": False,
        }
        result = await dispatcher.dispatch("pattern_search", payload)
        return result.body

    return find_code_pattern
