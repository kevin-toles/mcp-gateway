"""uss_graph_query — unified-search-rs :8089 [PENDING: graph API endpoint needed]."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "uss_graph_query"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def uss_graph_query(cypher: str, params: dict | None = None) -> dict:
        """[PENDING] Read-only Cypher query on unified-search-rs Neo4j graph."""
        payload: dict = {"cypher": cypher}
        if params:
            payload["params"] = params
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)
    return uss_graph_query
