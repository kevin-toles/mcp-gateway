"""Audit tool base — WBS-AEI13.

Shared factory for all audit_* tool handlers.
"""

from __future__ import annotations

from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher


def create_audit_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
    input_model,
):
    """Return an async handler that validates input, dispatches to audit-service,
    and sanitizes output.

    ``input_model`` is the Pydantic model class for this tool's inputs.
    The handler's keyword arguments are inferred by FastMCP from the model fields.
    """

    async def _handler(**kwargs) -> dict:
        validated = input_model(**kwargs)
        payload = validated.model_dump()
        result = await dispatcher.dispatch(tool_name, payload)
        return sanitizer.sanitize(result.body)

    return _handler
