"""enhance_guideline tool handler — WBS-WF6."""

from src.models.schemas import EnhanceGuidelineInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "enhance_guideline"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def enhance_guideline(
        aggregate_path: str,
        guideline_path: str,
        output_dir: str = "output",
        provider: str = "gateway",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict:
        """Enhance a guideline's chapters via LLM with cross-book context."""
        validated = EnhanceGuidelineInput(
            aggregate_path=aggregate_path,
            guideline_path=guideline_path,
            output_dir=output_dir,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return enhance_guideline
