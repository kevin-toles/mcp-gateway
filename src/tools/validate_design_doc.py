"""ASCP.VS7: `validate_design_doc` tool handler.

Routes to validation-service ``/validate/design_document`` for design
document validation. Returns a ValidationResult with verdict, score, and failures.
"""

from __future__ import annotations

from src.models.schemas import ValidateDesignDocInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "validate_design_doc"

# The Rust service deserializes `phase` into a strict enum — anything else
# is a 422 ("unknown variant"). A MISSING key is accepted (phase optional).
VALID_PHASES = ("DRAFT", "REVIEW", "APPROVED")


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def validate_design_doc(content: str, phase: str = "") -> dict:
        """Validate a design document against the ASCP-1.0 rule set.

        Returns a ValidationResult with verdict (PASS/FAIL), score, and failures list.
        Accepts an optional phase hint (DRAFT | REVIEW | APPROVED) to enable
        phase-specific checks; omit for the default (DRAFT-equivalent) behavior.
        """
        validated = ValidateDesignDocInput(content=content, phase=phase)
        payload = validated.model_dump()

        # Contract with the Rust service: omit `phase` entirely when empty
        # (an empty string is a 422), and fail fast on values it would reject.
        phase_value = str(payload.get("phase") or "").strip().upper()
        if not phase_value:
            payload.pop("phase", None)
        elif phase_value in VALID_PHASES:
            payload["phase"] = phase_value
        else:
            raise ValueError(
                f"invalid phase {phase!r} — validation-service accepts: "
                f"{', '.join(VALID_PHASES)} (or omit for default)"
            )

        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return validate_design_doc
