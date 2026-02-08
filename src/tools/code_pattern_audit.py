"""code_pattern_audit tool handler — WBS-MCP8."""

from src.models.schemas import CodePatternAuditInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "code_pattern_audit"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def code_pattern_audit(
        code: str,
        language: str = "",
        confidence_threshold: float = 0.3,
    ) -> dict:
        """Detect code anti-patterns using dual-net detection with confidence scoring."""
        validated = CodePatternAuditInput(
            code=code,
            language=language,
            confidence_threshold=confidence_threshold,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return code_pattern_audit
