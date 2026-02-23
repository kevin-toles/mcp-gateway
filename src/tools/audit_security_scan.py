"""audit_security_scan tool handler — WBS-AEI13.

Dispatches to audit-service :8084 POST /v1/audit/security.
Scans source code for security vulnerabilities.
"""

from __future__ import annotations

from src.models.schemas import AuditSecurityScanInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "audit_security_scan"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def audit_security_scan(
        code: str,
        language: str = "python",
        mode: str = "quick",
        domains: list[str] | None = None,
        severity_threshold: str = "low",
    ) -> dict:
        """Scan source code for security vulnerabilities."""
        validated = AuditSecurityScanInput(
            code=code,
            language=language,
            mode=mode,
            domains=domains,
            severity_threshold=severity_threshold,
        )
        payload = validated.model_dump()
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return audit_security_scan
