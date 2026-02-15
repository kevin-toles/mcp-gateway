"""code_analyze tool handler — WBS-MCP8, TWR6 (GREEN).

Dispatches to audit-service :8084 POST /v1/patterns/detect.
Transforms MCP input {code, language, analysis_type} into
audit-service format {code, language, file_path}.
"""

from src.models.schemas import CodeAnalyzeInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "code_analyze"


def _build_audit_payload(validated: CodeAnalyzeInput) -> dict:
    """Transform MCP input into audit-service /v1/patterns/detect request."""
    return {
        "code": validated.code,
        "language": validated.language,
        "file_path": "<mcp-input>",
    }


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
        payload = _build_audit_payload(validated)
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return code_analyze
