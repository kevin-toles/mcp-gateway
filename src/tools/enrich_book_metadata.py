"""enrich_book_metadata tool handler — WBS-WF6."""

from src.models.schemas import EnrichBookMetadataInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "enrich_book_metadata"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def enrich_book_metadata(
        input_path: str,
        output_path: str | None = None,
        taxonomy_path: str | None = None,
        mode: str = "msep",
    ) -> dict:
        """Enrich book metadata with MSEP (multi-source enrichment pipeline)."""
        validated = EnrichBookMetadataInput(
            input_path=input_path,
            output_path=output_path,
            taxonomy_path=taxonomy_path,
            mode=mode,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return enrich_book_metadata
