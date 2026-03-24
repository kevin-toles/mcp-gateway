"""MCP-F-3: `find_code_pattern` facade handler.

Intent-level tool that surfaces code pattern search to LLM clients using
user-friendly `examples` vocabulary ("good", "bad", "both") instead of the
internal `pattern_type` values accepted by the `pattern_search` backend route.
"""

from __future__ import annotations

from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "find_code_pattern"

_EXAMPLES_MAP: dict[str, str] = {
    "good": "good",
    "bad": "bad",
    "both": "all",
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
        payload = {"query": query, "pattern_type": pattern_type, "limit": 10}
        result = await dispatcher.dispatch("pattern_search", payload)
        return result.body

    return find_code_pattern
