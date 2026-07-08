"""uss_embed — unified-search-rs :8089 POST /v1/embed [PENDING]."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "uss_embed"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def uss_embed(text: str) -> dict:
        """[PENDING] Embed text into 384-dim MiniLM vector on unified-search-rs."""
        result = await dispatcher.dispatch(TOOL_NAME, {"text": text})
        return sanitizer.sanitize(result.body)
    return uss_embed
