"""uss_graph_traverse — unified-search-rs :8089 [PENDING: graph API endpoint needed]."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "uss_graph_traverse"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def uss_graph_traverse(
        start_node: str,
        strategy: str = "bfs",
        max_depth: int = 3,
        top_k: int = 10,
    ) -> dict:
        """[PENDING] BFS/MMR graph traversal on unified-search-rs Neo4j."""
        result = await dispatcher.dispatch(TOOL_NAME, {
            "start_node": start_node,
            "strategy": strategy,
            "max_depth": max_depth,
            "top_k": top_k,
        })
        return sanitizer.sanitize(result.body)
    return uss_graph_traverse
