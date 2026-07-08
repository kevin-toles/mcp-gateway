"""cms_process — context-management-service :8086 POST /v1/context/process."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "cms_process"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def cms_process(
        content: str,
        conversation_id: str = "",
        token_budget: int = 0,
    ) -> dict:
        """Full bifurcated processing pipeline: validate→chunk→optimize→glossary."""
        payload: dict = {"content": content}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if token_budget:
            payload["token_budget"] = token_budget
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)
    return cms_process
