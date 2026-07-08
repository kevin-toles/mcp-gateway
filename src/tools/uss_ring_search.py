"""uss_ring_search — unified-search-rs :8089 GET /v1/search/rings [PENDING]."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "uss_ring_search"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def uss_ring_search(
        q: str,
        collection: str = "",
        mmr_strategy: str = "",
    ) -> dict:
        """[PENDING] 6-ring concentric search pipeline on unified-search-rs."""
        payload: dict = {"q": q}
        if collection:
            payload["collection"] = collection
        if mmr_strategy:
            payload["mmr_strategy"] = mmr_strategy
        result = await dispatcher.dispatch(TOOL_NAME, payload, method="GET")
        return sanitizer.sanitize(result.body)
    return uss_ring_search
