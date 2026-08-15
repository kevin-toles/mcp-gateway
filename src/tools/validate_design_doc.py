"""ASCP.VS7: `validate_design_doc` tool handler.

Routes to validation-service ``/validate/design_document`` for design
document validation. Returns a ValidationResult with verdict, score, and failures.
"""

from __future__ import annotations

from src.models.schemas import ValidateDesignDocInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "validate_design_doc"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def validate_design_doc(content: str, phase: str = "") -> dict:
        """Validate a design document against the ASCP-1.0 rule set.

        Returns a ValidationResult with verdict (PASS/FAIL), score, and failures list.
        Accepts an optional phase hint (e.g. ARCHITECT) to enable phase-specific checks.
        """
        validated = ValidateDesignDocInput(content=content, phase=phase)
        payload = validated.model_dump()
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return validate_design_doc
