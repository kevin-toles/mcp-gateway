"""batch_enrich_metadata tool handler — WF-ENRICH batch variant.

Enriches all *_metadata.json files in a directory via CO /api/v1/workflows/batch-enrich.
No ai-agents dependency. Produces DMA-§2.3 compliant output for every book.
"""

from src.models.schemas import BatchEnrichMetadataInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "batch_enrich_metadata"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def batch_enrich_metadata(
        metadata_dir: str = "/Users/kevintoles/POC/ai-platform-data/books/metadata",
        output_dir: str = "/Users/kevintoles/POC/ai-platform-data/books/enriched",
        taxonomy_path: str | None = None,
        resume: bool = True,
        limit: int = 0,
        book: str = "",
    ) -> dict:
        """Batch-enrich all *_metadata.json files in a directory to DMA-§2.3 format.

        Calls code-orchestrator (:8083) directly — no ai-agents dependency.
        Each book goes through: keywords → topics → similarity → classify → concepts → §2.3 transform.

        Args:
            metadata_dir: Directory containing *_metadata.json files.
            output_dir: Where to write *_enriched.json files.
            taxonomy_path: Path to taxonomy JSON (defaults to uber_taxonomy_v8.json).
            resume: If True, skip books that already have an enriched output file.
            limit: Cap at N books (0 = process all).
            book: Filter — process only books whose filename contains this string.
        """
        validated = BatchEnrichMetadataInput(
            metadata_dir=metadata_dir,
            output_dir=output_dir,
            taxonomy_path=taxonomy_path,
            resume=resume,
            limit=limit,
            book=book,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return batch_enrich_metadata
