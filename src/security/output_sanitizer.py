"""Output sanitization pipeline — WBS-MCP8 (GREEN).

Phase 1: passthrough (no-op).  Returns tool output unchanged.
Phase 2 (AGT2): will add active filtering for injection patterns,
PII stripping, and response size enforcement.

Reference: AC-8.3 (step 7 in data flow), Strategy §8
"""

from typing import Any


class OutputSanitizer:
    """Sanitizes tool output before returning to the MCP client.

    In Phase 1 this is a no-op passthrough.  Set ``active=True`` to
    enable filtering in Phase 2 (implementation deferred to AGT2).

    Args:
        active: When ``True``, output filtering is engaged.
                Defaults to ``False`` (passthrough).
    """

    def __init__(self, active: bool = False) -> None:
        self.active = active

    def sanitize(self, data: Any) -> Any:
        """Return *data* unchanged (Phase 1 passthrough).

        Phase 2 will inspect and filter the data when ``self.active``
        is ``True``.
        """
        return data
