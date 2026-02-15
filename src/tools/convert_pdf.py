"""convert_pdf tool handler — WBS-WF6."""

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
        """Convert a PDF file to structured JSON with optional OCR fallback."""
        validated = ConvertPDFInput(
            input_path=input_path,
            output_path=output_path,
            enable_ocr=enable_ocr,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return convert_pdf
