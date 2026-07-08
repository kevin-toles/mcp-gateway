"""uss_hydrate — unified-search-rs :8089 POST /v1/hydrate [PENDING]."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "uss_hydrate"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def uss_hydrate(ids: list[str], collection: str = "") -> dict:
        """[PENDING] Retrieve full chunk content by vector IDs on unified-search-rs."""
        payload: dict = {"ids": ids}
        if collection:
            payload["collection"] = collection
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)
    return uss_hydrate
