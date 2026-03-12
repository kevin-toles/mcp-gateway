"""amve_extract_architecture tool handler — WBS Phase 2.

Dispatches to AMVE :8088  POST /v1/architecture/extract.
Returns {snapshot_sha, record_count, source_path, extraction_time_ms,  ...} as
returned by AMVE; the snapshot_sha is the SHA-256 fingerprint written to the
Redis stream when SNAPSHOT_STORE_ENABLED=true on the AMVE side.

AC-2.4 (G2.5 RED → G2.6 GREEN)
"""

from __future__ import annotations

from src.models.schemas import AMVEExtractArchitectureInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "amve_extract_architecture"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async MCP handler for amve_extract_architecture."""

    async def handler(source_path: str) -> dict:
        validated = AMVEExtractArchitectureInput(source_path=source_path)
        payload = {"source_paths": [validated.source_path]}
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return handler
