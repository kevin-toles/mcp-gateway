"""a2a_get_task tool handler — MCP ↔ A2A bridge.

Factory creates a closure that validates input via Pydantic,
dispatches a GET request to the ai-agents A2A task endpoint,
and sanitizes output.

Reference: PROTOCOL_INTEGRATION_ARCHITECTURE.md → A2A ↔ MCP Integration
"""

from src.models.schemas import A2AGetTaskInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "a2a_get_task"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def a2a_get_task(
        task_id: str,
    ) -> dict:
        """Get the status and results of an A2A agent task by its ID."""
        validated = A2AGetTaskInput(task_id=task_id)
        # GET /a2a/v1/tasks/{task_id} — path parameter in URL, no body
        path = f"/a2a/v1/tasks/{validated.task_id}"
        result = await dispatcher.dispatch(
            TOOL_NAME,
            {},
            method="GET",
            path_override=path,
        )
        return sanitizer.sanitize(result.body)

    return a2a_get_task
