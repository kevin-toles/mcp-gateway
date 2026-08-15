"""ASCP.VS7: `validate_wbs` tool handler.

Routes to validation-service ``/validate/wbs`` for WBS document validation.
Returns a ValidationResult with verdict, score, and structured failures.
"""

from __future__ import annotations

from src.models.schemas import ValidateWBSInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "validate_wbs"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def validate_wbs(content: str) -> dict:
        """Validate a WBS document against the ASCP-1.0 rule set.

        Returns a ValidationResult with verdict (PASS/FAIL), score, and failures list.
        """
        validated = ValidateWBSInput(content=content)
        payload = validated.model_dump()
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return validate_wbs
