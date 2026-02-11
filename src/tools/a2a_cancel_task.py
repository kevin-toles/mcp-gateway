"""a2a_cancel_task tool handler — MCP ↔ A2A bridge.

Factory creates a closure that validates input via Pydantic,
dispatches a POST request to the ai-agents A2A cancel endpoint,
and sanitizes output.

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → A2A ↔ MCP Integration
"""

from src.models.schemas import A2ACancelTaskInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "a2a_cancel_task"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def a2a_cancel_task(
        task_id: str,
    ) -> dict:
        """Cancel a running A2A agent task by its ID."""
        validated = A2ACancelTaskInput(task_id=task_id)
        # POST /a2a/v1/tasks/{task_id}:cancel — path parameter in URL, empty body
        path = f"/a2a/v1/tasks/{validated.task_id}:cancel"
        result = await dispatcher.dispatch(
            TOOL_NAME,
            {},
            path_override=path,
        )
        return sanitizer.sanitize(result.body)

    return a2a_cancel_task
