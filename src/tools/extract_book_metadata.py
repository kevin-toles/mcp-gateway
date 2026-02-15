"""extract_book_metadata tool handler — WBS-WF6."""

from src.models.schemas import ExtractBookMetadataInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "extract_book_metadata"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def extract_book_metadata(
        input_path: str,
        output_path: str | None = None,
        chapters: list[dict] | None = None,
        options: dict | None = None,
    ) -> dict:
        """Extract metadata, keywords, concepts, and code blocks from a book JSON file."""
        validated = ExtractBookMetadataInput(
            input_path=input_path,
            output_path=output_path,
            chapters=chapters,
            options=options,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return extract_book_metadata
