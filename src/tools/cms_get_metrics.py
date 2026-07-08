"""cms_get_metrics — context-management-service :8086 POST /v1/context/metrics."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "cms_get_metrics"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def cms_get_metrics(conversation_id: str = "") -> dict:
        """Retrieve context window metrics and token utilization statistics."""
        payload: dict = {}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)
    return cms_get_metrics
