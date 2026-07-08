"""inference — inference-service-cpp :8085 POST /v1/models."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "inference"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def inference(
        model: str = "",
        prompt: str = "",
        messages: list[dict] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict:
        """Run inference on locally hosted llama.cpp models."""
        payload: dict = {"max_tokens": max_tokens, "temperature": temperature}
        if model:
            payload["model"] = model
        if messages:
            payload["messages"] = messages
        elif prompt:
            payload["prompt"] = prompt
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)
    return inference
