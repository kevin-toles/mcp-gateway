"""llm_complete tool handler — WBS-MCP8, TWR5 (D5 fix).

Transforms flat MCP input (prompt, system_prompt, model_preference)
into OpenAI chat completions format (model, messages, temperature,
max_tokens) before dispatching to llm-gateway /v1/chat/completions.

Reference: WBS_TOOL_WIRING_REMEDIATION.md §TWR5
"""

from src.models.schemas import LLMCompleteInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "llm_complete"

# TWR5: Map model_preference hint → default model name for llm-gateway
_MODEL_PREFERENCE_MAP: dict[str, str] = {
    "auto": "qwen3-8b",
    "local": "qwen3-8b",
    "cloud": "claude-sonnet-4-20250514",
}


def _build_openai_payload(validated: LLMCompleteInput) -> dict:
    """Transform validated MCP input into OpenAI chat completions format.

    Args:
        validated: Pydantic-validated input with prompt, system_prompt, etc.

    Returns:
        Dict with model, messages, temperature, max_tokens — ready for
        POST /v1/chat/completions.
    """
    messages: list[dict[str, str]] = []
    if validated.system_prompt:
        messages.append({"role": "system", "content": validated.system_prompt})
    messages.append({"role": "user", "content": validated.prompt})

    model = _MODEL_PREFERENCE_MAP.get(validated.model_preference, "qwen3-8b")

    return {
        "model": model,
        "messages": messages,
        "temperature": validated.temperature,
        "max_tokens": validated.max_tokens,
    }


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
        # TWR5 (D5): Transform to OpenAI messages format
        payload = _build_openai_payload(validated)
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return llm_complete
