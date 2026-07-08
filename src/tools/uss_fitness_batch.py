"""uss_fitness_batch — unified-search-rs :8089 POST /v1/fitness/batch [PENDING]."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "uss_fitness_batch"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def uss_fitness_batch(requests: list[dict]) -> dict:
        """[PENDING] Batch evaluate multiple fitness functions on unified-search-rs."""
        result = await dispatcher.dispatch(TOOL_NAME, {"requests": requests})
        return sanitizer.sanitize(result.body)
    return uss_fitness_batch
