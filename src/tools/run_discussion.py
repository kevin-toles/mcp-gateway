"""run_discussion tool handler — WBS-MCP8."""

from src.models.schemas import RunDiscussionInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "run_discussion"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def run_discussion(
        protocol_id: str,
        topic: str,
        context: str = "",
    ) -> dict:
        """Run multi-LLM discussion using a Kitchen Brigade protocol."""
        validated = RunDiscussionInput(
            protocol_id=protocol_id,
            topic=topic,
            context=context,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return run_discussion
