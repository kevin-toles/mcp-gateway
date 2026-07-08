"""cms_validate — context-management-service :8086 POST /v1/context/validate."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "cms_validate"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def cms_validate(content: str, conversation_id: str = "") -> dict:
        """Validate a context payload against CMS schema."""
        payload = {"content": content}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)
    return cms_validate
