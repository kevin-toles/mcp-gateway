"""cms_warmup — context-management-service :8086 POST /v1/workflows/warmup."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "cms_warmup"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def cms_warmup() -> dict:
        """Pre-warm CMS and verify struct-analyzer skeletonize connectivity."""
        result = await dispatcher.dispatch(TOOL_NAME, {})
        return sanitizer.sanitize(result.body)
    return cms_warmup
