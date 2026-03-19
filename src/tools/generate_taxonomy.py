"""generate_taxonomy tool handler — WBS-WF6.

Routes to CO /generate-taxonomy-from-enriched, which reads *_enriched.json
files and builds the full uber_taxonomy format (metadata + concept_categories
+ tiers).  This is the correct taxonomy generation path.
"""

from src.models.schemas import GenerateTaxonomyInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "generate_taxonomy"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def generate_taxonomy(
        enriched_dir: str = "/Users/kevintoles/POC/ai-platform-data/books/enriched",
        output_path: str | None = None,
    ) -> dict:
        """Build full taxonomy from enriched corpus.

        Reads all *_enriched.json files in enriched_dir, aggregates keywords
        and concepts per book, applies vocabulary quality gates, infers Bloom
        tiers, and writes the full uber_taxonomy JSON (metadata +
        concept_categories + tiers).
        """
        validated = GenerateTaxonomyInput(
            enriched_dir=enriched_dir,
            output_path=output_path,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return generate_taxonomy
