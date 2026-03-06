"""diagram_search tool handler — semantic search over KB ASCII / sequence diagrams.

Searches the ``ascii_diagrams`` Qdrant collection (10,035 diagrams extracted from
platform textbooks) using CLIP text→image cross-modal retrieval.  The USS hybrid
search handler detects ``collection=ascii_diagrams`` and routes embedding through
``CLIPEncoder.encode_text()`` (512-dim) instead of the normal MiniLM path (384-dim)
so query semantics align with the stored CLIP image vectors.

When ``diagram_type`` is provided the tool fetches 3× the requested limit from USS
and post-filters on the ``diagram_type`` payload field before truncating to ``limit``.

Diagram type distribution in the collection (from 10,035 points):
  - ascii     — plain ASCII art boxes / trees  (~75%)
  - sequence  — UML-style sequence / interaction diagrams (~25%)
  - box_flow  — box-and-arrow flow diagrams (<1%)
"""

from src.models.schemas import DiagramSearchInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "diagram_search"

# Over-fetch multiplier when diagram_type filter is active.
# Fetching 3× improves recall for post-filtering without significant latency cost.
_FILTER_OVERFETCH = 3


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def diagram_search(
        query: str,
        diagram_type: str | None = None,
        limit: int = 10,
    ) -> dict:
        """Search KB architecture diagrams using natural-language descriptions.

        Retrieves diagrams extracted from platform textbooks (chapters, code
        walkthroughs, architecture guides) by semantic similarity.  The search
        uses CLIP cross-modal encoding so descriptions like "microservices
        call flow" or "hexagonal architecture layers" match stored diagram
        images even when the exact words don't appear in the ASCII text.

        Args:
            query: Natural-language description of the diagram you're looking for.
                E.g., 'layered architecture', 'event-driven message flow',
                'dependency injection container', 'circuit breaker state machine'.
            diagram_type: Optional filter restricting results to one type:
                - 'ascii'    — plain ASCII art (boxes, trees, borders)
                - 'sequence' — UML sequence / interaction diagrams
                - 'box_flow' — box-and-arrow flow / decision diagrams
                Omit (or pass null) to search all diagram types.
            limit: Maximum results to return (1-30, default 10).
        """
        validated = DiagramSearchInput(
            query=query,
            diagram_type=diagram_type,
            limit=limit,
        )

        # When filtering by diagram_type, over-fetch so post-filtering has
        # enough candidates to fill the requested limit.
        fetch_limit = validated.limit * _FILTER_OVERFETCH if validated.diagram_type else validated.limit

        payload: dict = {
            "query": validated.query,
            "collection": "ascii_diagrams",
            "limit": fetch_limit,
            "include_graph": False,  # ascii_diagrams has no Neo4j graph nodes
            "tier_boost": False,  # no bloom/quality tier metadata on diagrams
            "expand_taxonomy": False,  # CLIP already captures semantic similarity
        }

        result = await dispatcher.dispatch(TOOL_NAME, payload)
        sanitized = sanitizer.sanitize(result.body)

        # Post-filter on diagram_type if requested, then truncate to validated.limit.
        if validated.diagram_type and isinstance(sanitized, dict):
            results_key = next(
                (k for k in ("results", "hits", "items") if k in sanitized),
                None,
            )
            if results_key and isinstance(sanitized.get(results_key), list):
                filtered = [
                    r
                    for r in sanitized[results_key]
                    if (isinstance(r, dict) and r.get("payload", {}).get("diagram_type") == validated.diagram_type)
                ]
                sanitized = {
                    **sanitized,
                    results_key: filtered[: validated.limit],
                    "total": len(filtered[: validated.limit]),
                    "diagram_type_filter": validated.diagram_type,
                }

        return sanitized

    return diagram_search
