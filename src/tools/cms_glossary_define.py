"""cms_glossary_define — context-management-service :8086 POST /v1/context/glossary/define."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "cms_glossary_define"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def cms_glossary_define(
        term: str,
        definition: str,
        conversation_id: str = "",
    ) -> dict:
        """Add a term-definition pair to the active conversation glossary."""
        payload: dict = {"term": term, "definition": definition}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)
    return cms_glossary_define
