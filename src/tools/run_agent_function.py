"""run_agent_function tool handler — WBS-MCP8."""

from src.models.schemas import RunAgentFunctionInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "run_agent_function"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def run_agent_function(  # noqa: A002
        function_name: str,
        input: dict | None = None,  # noqa: A002
    ) -> dict:
        """Execute a single-purpose agent function by name."""
        validated = RunAgentFunctionInput(
            function_name=function_name,
            input=input if input is not None else {},  # noqa: A002
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return run_agent_function
