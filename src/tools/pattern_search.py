"""pattern_search tool handler — Issue #6: code pattern and anti-pattern retrieval.

Specialized search within code pattern collections. Routes to:
  - 'good'  → code_good_patterns (canonical positive examples)
  - 'bad'   → pattern_instances (architectural/code pattern instances,
               includes both anti-patterns and pattern violations)
  - 'all'   → fan-out across all primary collections (includes both
               pattern collections + chapters, code_chunks, etc.)

Use this tool when you are looking for code patterns, design patterns,
anti-patterns, or "right way / wrong way" code examples.
"""

from src.models.schemas import PatternSearchInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "pattern_search"

# Maps pattern_type → USS collection value
_PATTERN_COLLECTION_MAP: dict[str, str] = {
    "good": "code_good_patterns",
    "bad": "pattern_instances",
    "all": "all",
}


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def pattern_search(
        query: str,
        pattern_type: str = "all",
        limit: int = 10,
    ) -> dict:
        """Search code pattern and anti-pattern collections.

        Returns code examples demonstrating correct patterns, anti-patterns,
        or both. When pattern_type='all', results are drawn from all primary
        collections (not just pattern collections) which can surface textbook
        discussions of the pattern alongside code examples.

        Args:
            query: Pattern description or code smell to search for.
                E.g., 'singleton pattern', 'god class', 'dependency injection'.
            pattern_type: Which pattern collection(s) to search:
                - 'good' — code_good_patterns: canonical positive code examples
                - 'bad'  — pattern_instances: anti-pattern and violation instances
                - 'all'  — all primary collections (broadest coverage)
            limit: Maximum results (1-30, default 10).
        """
        validated = PatternSearchInput(
            query=query,
            pattern_type=pattern_type,
            limit=limit,
        )
        collection = _PATTERN_COLLECTION_MAP[validated.pattern_type]
        payload: dict = {
            "query": validated.query,
            "collection": collection,
            "limit": validated.limit,
            "include_graph": True,
            "tier_boost": True,
            "expand_taxonomy": False,
        }
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return pattern_search
