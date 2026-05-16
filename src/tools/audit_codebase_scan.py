"""audit_codebase_scan MCP tool handler — VRE-SCAN.

Dispatches to audit-service :8084 POST /v1/audit/scan.

Walks a local source directory, runs the full 4-layer detection pipeline
(AST → Enrichment → Scoring → Reporting) on every scannable file
(.py .js .ts .tsx .jsx .go .java), deduplicates findings by
(pattern_id, file), and enriches any SEC* findings with VRE exploit evidence
and advisory context from Qdrant :6336.

Note:
    ``confidence_threshold`` (0.80) and ``max_findings`` (uncapped) are
    hardcoded on the backend for accuracy — not exposed to the LLM.

Usage (via FastMCP server):
    result = await audit_codebase_scan(
        source_path="/Users/me/POC/my-service",
        priority_filter=["CRITICAL", "HIGH"],
    )
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time

from src.models.schemas import AuditCodebaseScanInput

TOOL_NAME = "audit_codebase_scan"

_DEDUP_TTL_SECONDS = 300.0
_INFLIGHT_LOCK = asyncio.Lock()
_INFLIGHT: dict[str, asyncio.Task[dict]] = {}
_RECENT_RESULTS: dict[str, tuple[float, dict]] = {}


def _normalize_payload(payload: dict) -> dict:
    """Return a canonical payload representation for dedupe keys."""
    normalized = dict(payload)
    pf = normalized.get("priority_filter")
    if pf:
        normalized["priority_filter"] = sorted({str(p).upper() for p in pf})
    return normalized


def _payload_fingerprint(payload: dict) -> str:
    """Create a stable fingerprint for request-level single-flight dedupe."""
    canonical = json.dumps(_normalize_payload(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evict_expired(now: float) -> None:
    """Drop stale cached results to keep memory bounded."""
    stale = [key for key, (ts, _) in _RECENT_RESULTS.items() if now - ts > _DEDUP_TTL_SECONDS]
    for key in stale:
        _RECENT_RESULTS.pop(key, None)


def create_handler(dispatcher, sanitizer):  # type: ignore[type-arg]
    """Return a FastMCP-compatible async handler for *audit_codebase_scan*.

    Parameters
    ----------
    dispatcher:
        ``ToolDispatcher`` instance — used to forward requests to audit-service.
    sanitizer:
        ``OutputSanitizer`` instance — cleans the service response before
        returning to the MCP client.

    Returns
    -------
    async callable
        Bound handler function ready to be registered with FastMCP.
    """

    async def audit_codebase_scan(
        source_path: str,
        priority_filter: list[str] | None = None,
    ) -> dict:
        """Scan an entire local codebase for security and quality findings.

        Walks *source_path* recursively, runs the full 4-layer detection
        pipeline on each scannable file, deduplicates by (pattern_id, file),
        assigns a priority level (CRITICAL/HIGH/MEDIUM/LOW/NEGLIGIBLE) to each
        finding, and enriches all SEC* findings with real-world exploit evidence
        and CVE advisory context from the Vulnerability Reference Engine (459k
        records across vuln_exploits and vuln_advisories).

        Note:
            ``confidence_threshold`` and ``max_findings`` are deliberately
            unavailable — these are hardcoded on the backend for accuracy.
            Every finding returned has confidence >= 0.80 (the accuracy
            threshold). All findings above this threshold are returned
            (no cap).

        Args:
            source_path: Absolute path to the local directory (or file) to
                scan. Must exist on the audit-service host.
            priority_filter: Optional list of priority levels to include.
                Choices: CRITICAL, HIGH, MEDIUM. Omit for all.
                Example: ["CRITICAL", "HIGH"] returns only security and
                high-confidence anti-pattern findings.

        Returns:
            dict with keys:
                - ``source_path``: echoed input path
                - ``findings``: list of ScanFinding dicts, each containing
                  ``pattern_id``, ``pattern_name``, ``confidence``,
                  ``classification``, ``priority``, ``vre_max_severity``
                  (highest NVD/GHSA severity from matched advisories, or null),
                  ``has_exploit_evidence`` (True when Exploit-DB match found),
                  ``file``, ``line_start``, ``line_end``, ``code_snippet``,
                  ``exploit_evidence`` (SEC*), ``advisory_context`` (SEC*),
                  ``citations``
                - ``stats``: ``files_scanned``, ``files_with_findings``,
                  ``total_findings``, ``findings_by_priority``, ``scan_time_ms``
        """
        validated = AuditCodebaseScanInput(
            source_path=source_path,
            priority_filter=priority_filter,
        )
        payload = validated.model_dump()
        fingerprint = _payload_fingerprint(payload)

        async with _INFLIGHT_LOCK:
            now = time.monotonic()
            _evict_expired(now)

            cached = _RECENT_RESULTS.get(fingerprint)
            if cached is not None:
                return copy.deepcopy(cached[1])

            task = _INFLIGHT.get(fingerprint)
            if task is None:
                task = asyncio.create_task(dispatcher.dispatch(TOOL_NAME, payload))
                _INFLIGHT[fingerprint] = task

        try:
            result = await task
            sanitized = sanitizer.sanitize(result.body)
        except Exception:
            async with _INFLIGHT_LOCK:
                if _INFLIGHT.get(fingerprint) is task:
                    _INFLIGHT.pop(fingerprint, None)
            raise

        async with _INFLIGHT_LOCK:
            if _INFLIGHT.get(fingerprint) is task:
                _INFLIGHT.pop(fingerprint, None)
            _RECENT_RESULTS[fingerprint] = (time.monotonic(), copy.deepcopy(sanitized))

        return sanitized

    return audit_codebase_scan
