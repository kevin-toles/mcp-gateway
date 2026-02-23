"""amve_detect_dead_code tool handler — AEI-17.

Dispatches to AMVE :8088 POST /v1/analysis/dead-code.
Detects dead code (unused functions + unused imports) in a source tree.

AC-AEI17.9:  Tool registered in _INPUT_MODELS.
AC-AEI17.11: Route dispatches to /v1/analysis/dead-code on AMVE (:8088).
AC-AEI17.12: create_handler() returns async callable with typed signature.
"""

from src.models.schemas import AMVEDetectDeadCodeInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "amve_detect_dead_code"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def handler(
        source_path: str,
        include_unused_imports: bool = True,
    ) -> dict:
        """Detect dead code in source_path via AMVE.

        :param source_path:            Path to the source directory or file.
        :param include_unused_imports: Whether to include unused-import analysis.
        :returns:                      Dead-code report dict.
        """
        validated = AMVEDetectDeadCodeInput(
            source_path=source_path,
            include_unused_imports=include_unused_imports,
        )
        payload = validated.model_dump()
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return handler
