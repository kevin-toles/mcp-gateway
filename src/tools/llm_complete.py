"""llm_complete tool handler — WBS-MCP8."""

from src.models.schemas import LLMCompleteInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "llm_complete"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def llm_complete(
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model_preference: str = "auto",
    ) -> dict:
        """Generate LLM completion with tiered fallback (local then cloud)."""
        validated = LLMCompleteInput(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model_preference=model_preference,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return llm_complete
