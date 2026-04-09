"""audit_codebase_scan MCP tool handler — VRE-SCAN.

Dispatches to audit-service :8084 POST /v1/audit/scan.

Walks a local source directory, runs the full 4-layer detection pipeline
(AST → Enrichment → Scoring → Reporting) on every scannable file
(.py .js .ts .tsx .jsx .go .java), deduplicates findings by
(pattern_id, file), and enriches any SEC* findings with VRE exploit evidence
and advisory context from Qdrant :6336.

Usage (via FastMCP server):
    result = await audit_codebase_scan(
        source_path="/Users/me/POC/my-service",
        confidence_threshold=0.3,
        max_findings=200,
    )
"""

from __future__ import annotations

from src.models.schemas import AuditCodebaseScanInput

TOOL_NAME = "audit_codebase_scan"


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
        confidence_threshold: float = 0.3,
        max_findings: int | None = None,
        priority_filter: list[str] | None = None,
    ) -> dict:
        """Scan an entire local codebase for security and quality findings.

        Walks *source_path* recursively, runs the full 4-layer detection
        pipeline on each scannable file, deduplicates by (pattern_id, file),
        assigns a priority level (CRITICAL/HIGH/MEDIUM/LOW/NEGLIGIBLE) to each
        finding, and enriches all SEC* findings with real-world exploit evidence
        and CVE advisory context from the Vulnerability Reference Engine (459k
        records across vuln_exploits and vuln_advisories).

        Args:
            source_path: Absolute path to the local directory (or file) to
                scan. Must exist on the audit-service host.
            confidence_threshold: Minimum confidence (0.0–1.0) to include a
                finding. Default 0.3.
            max_findings: Maximum deduplicated findings to return. None
                (default) = unlimited full scan. Use with priority_filter to
                scope large codebases.
            priority_filter: List of priority levels to include. Options:
                CRITICAL, HIGH, MEDIUM, LOW, NEGLIGIBLE. None = all.
                Example: ["CRITICAL", "HIGH"] returns only SEC* and
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
            confidence_threshold=confidence_threshold,
            max_findings=max_findings,
            priority_filter=priority_filter,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return audit_codebase_scan
