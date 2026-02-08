"""agent_execute tool handler — WBS-MCP8."""

from src.models.schemas import AgentExecuteInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "agent_execute"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def agent_execute(
        task: str,
        max_steps: int = 15,
    ) -> dict:
        """Execute an autonomous multi-step agent task."""
        validated = AgentExecuteInput(
            task=task,
            max_steps=max_steps,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return agent_execute
