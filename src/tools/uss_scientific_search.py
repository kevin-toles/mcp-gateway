"""uss_scientific_search — unified-search-rs :8089 POST /v1/search/scientific [PENDING]."""
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "uss_scientific_search"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    async def uss_scientific_search(query: str, top_k: int = 10) -> dict:
        """[PENDING] Scientific/theoretical grounding search on unified-search-rs."""
        result = await dispatcher.dispatch(TOOL_NAME, {"query": query, "top_k": top_k})
        return sanitizer.sanitize(result.body)
    return uss_scientific_search
