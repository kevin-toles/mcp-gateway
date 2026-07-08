"""convert_pdf tool handler — WBS-WF6.

Dispatches to Code-Orchestrator POST /api/v1/workflows/convert-pdf.
"""

from src.models.schemas import ConvertPDFInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "convert_pdf"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def convert_pdf(
        input_path: str,
        output_path: str | None = None,
        enable_ocr: bool = True,
    ) -> dict:
        """Convert a PDF file to structured JSON.

        Extracts text per page with optional OCR fallback and detects chapters.
        Dispatches to Code-Orchestrator POST /api/v1/workflows/convert-pdf.

        Args:
            input_path: Path to a PDF file or a directory of PDFs.
            output_path: Output JSON file path (auto-generated if omitted).
            enable_ocr: Enable OCR fallback for image-only pages.
        """
        validated = ConvertPDFInput(
            input_path=input_path,
            output_path=output_path,
            enable_ocr=enable_ocr,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return convert_pdf
