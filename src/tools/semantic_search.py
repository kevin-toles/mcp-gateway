"""semantic_search tool handler — WBS-MCP8.

Factory creates a closure that validates input via Pydantic,
dispatches to the semantic-search backend, and sanitizes output.
"""

from src.models.schemas import SemanticSearchInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "semantic_search"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def semantic_search(
        query: str,
        collection: str = "all",
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> dict:
        """Search across code, documentation, and textbooks using semantic similarity."""
        validated = SemanticSearchInput(
            query=query,
            collection=collection,
            top_k=top_k,
            threshold=threshold,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return semantic_search
