"""cms_chunk — context-management-service :8086 POST /v1/context/chunk."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "cms_chunk"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def cms_chunk(content: str, conversation_id: str = "") -> dict:
        """Bifurcate input into TextSegment and CodeBlock typed segments."""
        payload: dict = {"content": content}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)
    return cms_chunk
