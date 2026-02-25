"""audit_quality_scan tool handler — Phase 7.

Dispatches to audit-service :8084 POST /v1/audit/quality.
Runs coding pattern rules (CP001-CP011), static analysis rules (SA001-SA008),
security rules (SEC001-SEC008), and structural anti-pattern detection against
submitted source code.
"""

from __future__ import annotations

from src.models.schemas import AuditQualityScanInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "audit_quality_scan"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def audit_quality_scan(
        code: str,
        language: str = "python",
        rule_categories: list[str] | None = None,
        severity_threshold: str = "info",
        include_antipatterns: bool = True,
    ) -> dict:
        """Audit source code for coding pattern violations and structural anti-patterns.

        Runs three rule sets (CP001-CP011 coding patterns, SA001-SA008 static
        analysis, SEC001-SEC008 security) plus AntiPatternDetector (Blob/God
        Class, Lava Flow, Boat Anchor, Redundant Wrapper, Premature Abstraction).

        Args:
            code: Source code string to audit.
            language: Language hint (Python AST only currently).
            rule_categories: Restrict to specific categories (null = all).
            severity_threshold: Minimum severity to include in output.
            include_antipatterns: Whether to run structural anti-pattern detection.

        Returns:
            Dict with pattern_findings, antipattern_findings, summary, and timing.
        """
        validated = AuditQualityScanInput(
            code=code,
            language=language,
            rule_categories=rule_categories,
            severity_threshold=severity_threshold,
            include_antipatterns=include_antipatterns,
        )
        payload = validated.model_dump()
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return audit_quality_scan
