"""code_analyze tool handler — WBS-MCP8."""

from src.models.schemas import CodeAnalyzeInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "code_analyze"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def code_analyze(
        code: str,
        language: str = "",
        analysis_type: str = "all",
    ) -> dict:
        """Analyze code for patterns, complexity, dependencies, and quality metrics."""
        validated = CodeAnalyzeInput(
            code=code,
            language=language,
            analysis_type=analysis_type,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return code_analyze
