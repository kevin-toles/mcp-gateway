"""uss_fitness_evaluate — unified-search-rs :8089 POST /v1/fitness/evaluate [PENDING]."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "uss_fitness_evaluate"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def uss_fitness_evaluate(
        function_id: str = "",
        function: dict | None = None,
        snapshot: dict | None = None,
    ) -> dict:
        """[PENDING] Evaluate architecture fitness on unified-search-rs."""
        payload: dict = {}
        if function_id:
            payload["function_id"] = function_id
        if function:
            payload["function"] = function
        if snapshot:
            payload["snapshot"] = snapshot
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)
    return uss_fitness_evaluate
