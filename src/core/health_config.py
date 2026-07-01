"""
Health Configuration — HWC-PY-1 COLD/WARM tier definitions
===========================================================

Defines ``SERVICE_TIERS``, a global dict tracking which backend services
are considered COLD (unpromoted) or WARM (promoted).  Used by
``ColdWarmPromoter`` in ``idle_timeout.py``.

Each key is a hyphen-form service key (e.g. ``"semantic-search"``)
matching ``_TOOL_SERVICE_NAMES`` values in ``tool_dispatcher.py``.
"""

from __future__ import annotations

# ── Global service tier registry ────────────────────────────────────────

SERVICE_TIERS: dict[str, str] = {
    "semantic-search": "COLD",
    "code-orchestrator": "COLD",
    "llm-gateway": "COLD",
    "ai-agents": "COLD",
    "amve": "COLD",
    "audit-service": "COLD",
    "unified-search-service": "COLD",
}


def reset_service_tiers() -> None:
    """Reset all tiers back to COLD (useful for testing)."""
    for key in SERVICE_TIERS:
        SERVICE_TIERS[key] = "COLD"
