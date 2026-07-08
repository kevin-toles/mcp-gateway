"""generate_taxonomy tool handler — WBS-WF6.

Dispatches to Code-Orchestrator POST /api/v1/workflows/generate-taxonomy.
Builds full taxonomy from a tiered book corpus.
"""

from src.models.schemas import GenerateTaxonomyInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "generate_taxonomy"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def generate_taxonomy(
        tier_books: dict,
        output_path: str | None = None,
        concepts: list[str] | None = None,
        domain: str = "auto",
    ) -> dict:
        """Build full taxonomy from a tiered book corpus.

        Aggregates keywords and concepts per book, applies vocabulary quality
        gates, infers Bloom tiers, and writes the full uber_taxonomy format
        (metadata + concept_categories + tiers).

        Args:
            tier_books: Mapping of tier name to list of enriched book JSON paths.
            output_path: Output taxonomy JSON path (auto-generated if omitted).
            concepts: Seed concept list (optional).
            domain: Knowledge domain — python, architecture, data_science, or auto.
        """
        validated = GenerateTaxonomyInput(
            tier_books=tier_books,
            output_path=output_path,
            concepts=concepts,
            domain=domain,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return generate_taxonomy
