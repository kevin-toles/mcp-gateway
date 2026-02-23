"""audit_corpus_search tool handler — WBS-AEI13.

Dispatches to audit-service :8084 POST /v1/audit/corpus.
Searches the code corpus and textbook chapters via Qdrant.
"""

from __future__ import annotations

from src.models.schemas import AuditCorpusSearchInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "audit_corpus_search"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def audit_corpus_search(
        query: str,
        collections: list[str] | None = None,
        top_k: int = 10,
        threshold: float = 0.7,
    ) -> dict:
        """Search the code corpus and textbook chapters using Qdrant vector similarity."""
        if collections is None:
            collections = ["code_chunks", "chapters"]
        validated = AuditCorpusSearchInput(
            query=query,
            collections=collections,
            top_k=top_k,
            threshold=threshold,
        )
        payload = validated.model_dump()
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return audit_corpus_search
