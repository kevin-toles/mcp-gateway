"""audit_resolve_lookup MCP tool handler — WBS-AEI20.

AC-AEI20.4: Registers audit_resolve_lookup in the MCP tool registry and
             dispatches to audit-service POST /v1/audit/resolve.

Usage (via FastMCP server):
    result = await audit_resolve_lookup(
        violation_type="sql_injection",
        pillar="security",
        include_code_examples=True,
    )
"""

from __future__ import annotations

from src.models.schemas import AuditResolveLookupInput

TOOL_NAME = "audit_resolve_lookup"


def create_handler(dispatcher, sanitizer):  # type: ignore[type-arg]
    """Return a FastMCP-compatible async handler for *audit_resolve_lookup*.

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

    async def audit_resolve_lookup(
        violation_type: str,
        pillar: str | None = None,
        include_code_examples: bool = True,
    ) -> dict:
        """Look up the full resolution evidence chain for a violation type.

        Searches the resolution knowledge base (Qdrant :6335) to return:
          - The matched violation type (what went wrong)
          - The governing principle (why it matters)
          - The resolution pattern (how to fix it)
          - Documentation references (source book, chapter)
          - Code examples from the corpus (optional)
          - A human-readable evidence chain list

        Args:
            violation_type: Violation identifier or description to look up.
                E.g. 'sql_injection', 'DEP_LOW_RATIO', 'circular_dependency'.
            pillar: Optional pillar hint — structural, architectural, eloquence,
                security, or dependency.
            include_code_examples: When True (default), fetches matching code
                examples from the code_chunks corpus via semantic-search.

        Returns:
            dict with keys ``violation``, ``principle``, ``resolution``,
            ``documentation``, ``code_examples``, ``evidence_chain``,
            ``resolve_time_ms``.
        """
        validated = AuditResolveLookupInput(
            violation_type=violation_type,
            pillar=pillar,
            include_code_examples=include_code_examples,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return audit_resolve_lookup
