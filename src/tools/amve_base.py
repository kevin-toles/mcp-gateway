"""AMVE tool base — AEI-7 REFACTOR.

Shared base for all 6 amve_* tool handlers. Reduces dispatch
boilerplate by providing a reusable create_handler factory pattern.
"""

from src.models.schemas import AMVEAnalysisInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher


def create_analysis_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
):
    """Return an async handler for an AMVE analysis endpoint.

    All 4 analysis tools (patterns, boundaries, call-graph, and the
    resolve endpoint) share the same {source_path, include_confidence}
    input schema. This factory eliminates per-tool boilerplate.
    """

    async def handler(
        source_path: str,
        include_confidence: bool = False,
    ) -> dict:
        validated = AMVEAnalysisInput(
            source_path=source_path,
            include_confidence=include_confidence,
        )
        payload = validated.model_dump()
        result = await dispatcher.dispatch(tool_name, payload)
        return sanitizer.sanitize(result.body)

    return handler
