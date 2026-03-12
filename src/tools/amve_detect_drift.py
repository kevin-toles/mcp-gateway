"""amve_detect_drift tool handler — WBS Phase 2.

Two operating modes:
  SHA-pair mode  (AC-2.5): resolver SHAs from Redis XRANGE → POST /v1/architecture/drift
  Passthrough    (AC-2.6): snapshot dicts provided inline → POST /v1/architecture/drift (no Redis)

Not-found handling (AC-2.7): returns {"error": "snapshot_not_found", "sha": ..., "stream": ...}
G2.11 REFACTOR: _resolve_snapshot_from_stream() helper extracted.

Tasks: G2.7 (RED) → G2.8 (GREEN), G2.9 (RED) → G2.10 (GREEN), G2.11 (REFACTOR)
"""

from __future__ import annotations

import json

import redis.asyncio as redis_async  # imported at module level so tests can patch it

from src.models.schemas import AMVEDetectDriftInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "amve_detect_drift"
_STREAM_KEY = "amve:findings:anonymous"


# ---------------------------------------------------------------------------
# G3.13 (REFACTOR) — no hardcoded stream key strings
# ---------------------------------------------------------------------------


def _stream_key(tenant_id: str) -> str:
    """Return the Redis stream key scoped to *tenant_id*.

    G3.13 (REFACTOR): centralises construction of ``amve:findings:{tenant_id}``
    so callers never embed the pattern inline.
    """
    return f"amve:findings:{tenant_id}"


# ---------------------------------------------------------------------------
# G2.11 REFACTOR — extracted helper
# ---------------------------------------------------------------------------


async def _resolve_snapshot_from_stream(
    sha: str,
    stream_key: str,
    redis_client,
) -> dict | None:
    """Scan *stream_key* for an entry whose ``snapshot_sha`` field matches *sha*.

    Returns the deserialised snapshot dict, or ``None`` if not found.
    """
    entries = await redis_client.xrange(stream_key, "-", "+")
    for _msg_id, fields in entries:
        if fields.get("snapshot_sha") == sha:
            return json.loads(fields["snapshot"])
    return None


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async MCP handler for amve_detect_drift."""

    async def handler(
        snapshot_a_sha: str | None = None,
        snapshot_b_sha: str | None = None,
        snapshot_a: dict | None = None,
        snapshot_b: dict | None = None,
        tenant_id: str | None = None,
    ) -> dict:
        # ------------------------------------------------------------------
        # Passthrough mode — AC-2.6
        # ------------------------------------------------------------------
        if snapshot_a is not None and snapshot_b is not None:
            result = await dispatcher.dispatch(
                TOOL_NAME,
                {"snapshot_a": snapshot_a, "snapshot_b": snapshot_b},
            )
            return sanitizer.sanitize(result.body)

        # ------------------------------------------------------------------
        # SHA-pair mode — AC-2.5
        # ------------------------------------------------------------------
        AMVEDetectDriftInput(snapshot_a_sha=snapshot_a_sha, snapshot_b_sha=snapshot_b_sha)

        # G3.10 (GREEN) — Phase 3: use tenant-scoped stream key when flag enabled
        effective_tenant = tenant_id or "anonymous"
        stream_key = (
            _stream_key("anonymous")
            if not dispatcher._settings.IDENTITY_PROPAGATION
            else _stream_key(effective_tenant)
        )

        redis_url = dispatcher._settings.REDIS_URL
        redis_client = redis_async.from_url(redis_url, decode_responses=True)
        try:
            snap_a = await _resolve_snapshot_from_stream(snapshot_a_sha, stream_key, redis_client)
            if snap_a is None:
                return {
                    "error": "snapshot_not_found",
                    "sha": snapshot_a_sha,
                    "stream": stream_key,
                }

            snap_b = await _resolve_snapshot_from_stream(snapshot_b_sha, stream_key, redis_client)
            if snap_b is None:
                return {
                    "error": "snapshot_not_found",
                    "sha": snapshot_b_sha,
                    "stream": stream_key,
                }
        finally:
            await redis_client.aclose()

        result = await dispatcher.dispatch(
            TOOL_NAME,
            {"snapshot_a": snap_a, "snapshot_b": snap_b},
        )
        return sanitizer.sanitize(result.body)

    return handler
