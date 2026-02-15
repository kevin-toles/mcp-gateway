"""code_pattern_audit tool handler — WBS-MCP8, TWR6 (GREEN).

Dispatches to audit-service :8084 POST /v1/patterns/detect.
Transforms MCP input {code, language, confidence_threshold} into
audit-service format {code, language, file_path}.
"""

from src.models.schemas import CodePatternAuditInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "code_pattern_audit"


def _build_audit_payload(validated: CodePatternAuditInput) -> dict:
    """Transform MCP input into audit-service /v1/patterns/detect request."""
    return {
        "code": validated.code,
        "language": validated.language,
        "file_path": "<mcp-input>",
    }


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
        payload = _build_audit_payload(validated)
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return code_pattern_audit
