"""audit_code_metrics tool handler — WBS-AEI13.

Dispatches to audit-service :8084 POST /v1/audit/metrics.
Computes structural, architectural, and eloquence pillar scores.
"""

from __future__ import annotations

from src.models.schemas import AuditCodeMetricsInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "audit_code_metrics"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def audit_code_metrics(
        code: str,
        language: str = "python",
        pillars: list[str] | None = None,
    ) -> dict:
        """Compute per-pillar code metrics and composite score."""
        if pillars is None:
            pillars = ["structural", "architectural", "eloquence"]
        validated = AuditCodeMetricsInput(
            code=code,
            language=language,
            pillars=pillars,
        )
        payload = validated.model_dump()
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return audit_code_metrics
