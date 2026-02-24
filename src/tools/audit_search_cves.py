"""audit_search_cves MCP tool handler — WBS-AEI23.

AC-AEI23.5: Registers audit_search_cves in the MCP tool registry and
            dispatches to audit-service POST /v1/audit/cves.

Usage (via FastMCP server):
    result = await audit_search_cves(
        cwe_id="CWE-89",
        severity="critical",
        ecosystem="python",
        limit=50,
    )
"""

from __future__ import annotations

from src.models.schemas import AuditSearchCVEsInput

TOOL_NAME = "audit_search_cves"


def create_handler(dispatcher, sanitizer):  # type: ignore[type-arg]
    """Return a FastMCP-compatible async handler for *audit_search_cves*.

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

    async def audit_search_cves(
        cwe_id: str | None = None,
        severity: str | None = None,
        ecosystem: str | None = None,
        limit: int = 50,
    ) -> dict:
        """Retrieve CVE records from PostgreSQL filtered by CWE, severity, or ecosystem.

        Args:
            cwe_id:    Filter by CWE identifier (e.g. "CWE-89"). Optional.
            severity:  Filter by severity level ("critical", "high", "medium",
                "low"). Optional.
            ecosystem: Filter by affected ecosystem (e.g. "python", "npm",
                "java"). Optional.
            limit:     Maximum number of records to return (default 50).

        Returns:
            dict with keys ``records``, ``total``, ``filters_applied``.
        """
        validated = AuditSearchCVEsInput(
            cwe_id=cwe_id,
            severity=severity,
            ecosystem=ecosystem,
            limit=limit,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return audit_search_cves
