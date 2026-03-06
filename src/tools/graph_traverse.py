"""graph_traverse tool handler — BFS + MMR graph traversal via USS /v1/graph/traverse.

Supports two traversal styles:
- Basic/eager BFS (mmr_enabled=False): Full breadth-first graph traversal returning
  all reachable nodes and edges up to max_depth. Use for exploration and relationship
  mapping.
- MMR traversal (mmr_enabled=True): Diversity-aware traversal using Maximal Marginal
  Relevance to select diverse, non-redundant nodes from the BFS frontier. Use when
  you want a representative sample rather than exhaustive coverage.
"""

from src.models.schemas import GraphTraverseInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "graph_traverse"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def graph_traverse(
        start_nodes: list[str],
        relationship_types: list[str] | None = None,
        max_depth: int = 3,
        limit: int = 50,
        mmr_enabled: bool = False,
        mmr_lambda: float = 0.5,
    ) -> dict:
        """Traverse the Neo4j knowledge graph from one or more starting nodes.

        Two traversal styles:
        - Basic/eager BFS (mmr_enabled=False): breadth-first, returns all reachable
          nodes and edges. Best for full relationship exploration.
        - MMR traversal (mmr_enabled=True): diversity-aware selection using Maximal
          Marginal Relevance. Returns a representative, non-redundant subset of nodes.
          mmr_lambda controls relevance vs diversity (0.0=max diversity, 1.0=max relevance).

        Args:
            start_nodes: One or more Neo4j node IDs to start BFS from.
            relationship_types: Relationship types to follow (None = all types).
                Common types: SIMILAR_TO, REQUIRES, RELATED_TO, IMPLEMENTS.
            max_depth: Maximum BFS hops from start nodes (1-10, default 3).
            limit: Maximum nodes returned (1-500, default 50).
            mmr_enabled: True = MMR diversity traversal, False = basic BFS.
            mmr_lambda: MMR tuning (0.0=pure diversity, 1.0=pure relevance).
                Only used when mmr_enabled=True.
        """
        validated = GraphTraverseInput(
            start_nodes=start_nodes,
            relationship_types=relationship_types,
            max_depth=max_depth,
            limit=limit,
            mmr_enabled=mmr_enabled,
            mmr_lambda=mmr_lambda,
        )
        payload: dict = {
            "start_nodes": validated.start_nodes,
            "max_depth": validated.max_depth,
            "limit": validated.limit,
            "mmr_enabled": validated.mmr_enabled,
            "mmr_lambda": validated.mmr_lambda,
        }
        if validated.relationship_types is not None:
            payload["relationship_types"] = validated.relationship_types
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return graph_traverse
