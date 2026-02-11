"""a2a_send_message tool handler — MCP ↔ A2A bridge.

Factory creates a closure that validates input via Pydantic,
dispatches to the ai-agents A2A endpoint, and sanitizes output.

This tool bridges the MCP protocol to A2A task submission,
which in turn routes through TaskExecutor to Temporal workflows
when available: MCP → A2A → TaskExecutor → Temporal.

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → A2A ↔ MCP Integration
"""

from src.models.schemas import A2ASendMessageInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "a2a_send_message"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def a2a_send_message(
        content: str,
        skill_id: str = "",
        context_id: str = "",
    ) -> dict:
        """Send a message to an AI agent via A2A protocol. Returns a task ID for tracking."""
        validated = A2ASendMessageInput(
            content=content,
            skill_id=skill_id,
            context_id=context_id,
        )
        # Build A2A SendMessageRequest payload
        payload = {
            "message": {
                "parts": [{"text": validated.content}],
                "skillId": validated.skill_id or None,
            },
            "contextId": validated.context_id or None,
        }
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return a2a_send_message
