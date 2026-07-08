"""cms_optimize — context-management-service :8086 POST /v1/context/optimize."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "cms_optimize"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def cms_optimize(
        content: str,
        conversation_id: str = "",
        token_budget: int = 0,
    ) -> dict:
        """Optimize a context window via token budget analysis and segment routing."""
        payload: dict = {"content": content}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if token_budget:
            payload["token_budget"] = token_budget
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)
    return cms_optimize
