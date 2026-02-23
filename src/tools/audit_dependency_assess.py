"""audit_dependency_assess MCP tool handler — WBS-AEI18.

AC-AEI18.15: Registers audit_dependency_assess in the MCP tool registry and
             dispatches to audit-service POST /v1/audit/dependency.

Usage (via FastMCP server):
    result = await audit_dependency_assess(
        source_path="/path/to/project/src",
        manifest_path="/path/to/pyproject.toml",
        include_transitive=True,
    )
"""

from __future__ import annotations

from src.models.schemas import AuditDependencyAssessInput

TOOL_NAME = "audit_dependency_assess"


def create_handler(dispatcher, sanitizer):  # type: ignore[type-arg]
    """Return a FastMCP-compatible async handler for *audit_dependency_assess*.

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

    async def audit_dependency_assess(
        source_path: str,
        manifest_path: str | None = None,
        include_transitive: bool = True,
    ) -> dict:
        """Assess dependency health for a Python project.

        Args:
            source_path: Absolute path to the Python source directory to scan.
            manifest_path: Optional path to pyproject.toml or requirements.txt.
                If omitted, auto-detected from source_path.
            include_transitive: When True (default) computes transitive dep
                counts via BFS over ``pip show --dependencies``.

        Returns:
            dict with keys ``dependency_health_score``, ``violations``,
            ``zone_classification``, ``package_metrics``, ``scan_time_ms``.
        """
        validated = AuditDependencyAssessInput(
            source_path=source_path,
            manifest_path=manifest_path,
            include_transitive=include_transitive,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return audit_dependency_assess
