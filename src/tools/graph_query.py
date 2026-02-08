"""graph_query tool handler — WBS-MCP8."""

from src.models.schemas import GraphQueryInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "graph_query"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def graph_query(
        cypher: str,
        parameters: dict | None = None,
    ) -> dict:
        """Query the Neo4j knowledge graph using Cypher (read-only)."""
        validated = GraphQueryInput(
            cypher=cypher,
            parameters=parameters if parameters is not None else {},
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return graph_query
