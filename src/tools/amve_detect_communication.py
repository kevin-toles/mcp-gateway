"""amve_detect_communication tool handler — AEI-7.

Dispatches to AMVE :8088 POST /v1/analysis/communication.
Consolidated events + messaging detection with scope routing.
"""

from src.models.schemas import AMVECommunicationInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "amve_detect_communication"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def amve_detect_communication(
        source_path: str,
        scope: str = "all",
        include_confidence: bool = False,
    ) -> dict:
        """Detect event-driven and message bus patterns with scope routing."""
        validated = AMVECommunicationInput(
            source_path=source_path,
            scope=scope,
            include_confidence=include_confidence,
        )
        payload = validated.model_dump()
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return amve_detect_communication
