"""MCP-F-3: `find_code_pattern` facade handler.

Intent-level tool that surfaces code pattern search to LLM clients using
user-friendly `examples` vocabulary ("good", "bad", "both") instead of the
internal `pattern_type` values accepted by the `pattern_search` backend route.

Translates `examples` to a Qdrant collection name that the unified-search-service
HybridSearchRequest model accepts (avoids sending unknown fields like `pattern_type`
which would cause a 422 Pydantic validation error).
"""

from __future__ import annotations

from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "find_code_pattern"

_EXAMPLES_MAP: dict[str, str] = {
    "good": "good",
    "bad": "bad",
    "both": "all",
}

# Maps resolved pattern_type → USS collection value
# Must match _PATTERN_COLLECTION_MAP in pattern_search.py
_PATTERN_COLLECTION_MAP: dict[str, str] = {
    "good": "code_good_patterns",
    "bad": "code_bad_patterns",
    "all": "all",
}


def _resolve_examples(examples: str) -> str:
    resolved = _EXAMPLES_MAP.get(examples)
    if resolved is None:
        valid = ", ".join(f'"{k}"' for k in _EXAMPLES_MAP)
        raise ValueError(f"Unknown examples {examples!r}. Valid values: {valid}")
    return resolved


def create_handler(dispatcher: ToolDispatcher, sanitizer=None):
    async def find_code_pattern(query: str, examples: str = "both") -> dict:
        pattern_type = _resolve_examples(examples)
        collection = _PATTERN_COLLECTION_MAP[pattern_type]
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
