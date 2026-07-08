"""cms_get_glossary — context-management-service :8086 GET /v1/context/glossary/{id}."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "cms_get_glossary"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def cms_get_glossary(conversation_id: str) -> dict:
        """Retrieve the accumulated glossary for a conversation_id."""
        result = await dispatcher.dispatch(
            TOOL_NAME, {"conversation_id": conversation_id}
        )
        return sanitizer.sanitize(result.body)
    return cms_get_glossary
