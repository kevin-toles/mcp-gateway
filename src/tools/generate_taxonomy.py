"""generate_taxonomy tool handler — WBS-WF6."""

from src.models.schemas import GenerateTaxonomyInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "generate_taxonomy"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def generate_taxonomy(
        tier_books: dict[str, list[str]],
        output_path: str | None = None,
        concepts: list[str] | None = None,
        domain: str = "auto",
    ) -> dict:
        """Generate a concept taxonomy from tier-organized book JSON files."""
        validated = GenerateTaxonomyInput(
            tier_books=tier_books,
            output_path=output_path,
            concepts=concepts,
            domain=domain,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return generate_taxonomy
