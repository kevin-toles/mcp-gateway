"""
Health Configuration — HWC-PY-1 COLD/WARM tier definitions
===========================================================

Defines ``SERVICE_TIERS``, a global dict tracking which backend services
are considered COLD (unpromoted) or WARM (promoted).  Uses the
new deque-based ``ColdWarmPromoter`` in ``idle_timeout.py``.

Each key is a hyphen-form service key (e.g. ``"semantic-search"``)
matching ``_TOOL_SERVICE_NAMES`` values in ``tool_dispatcher.py``.
"""

from __future__ import annotations

import os

# ── Lazy import inside auto_warm_service() body to avoid circular deps ──
# from src.core.idle_timeout import record_dispatch_for_promotion

# ── HWC-PY-1: Promotion window constants ───────────────────────────────

# Number of successful dispatches within the sliding window to trigger COLD→WARM
COLD_PROMOTION_REQUESTS = int(os.getenv("COLD_PROMOTION_REQUESTS", "5"))

# Sliding window duration (seconds) for the promotion check
COLD_PROMOTION_WINDOW_SECS = int(os.getenv("COLD_PROMOTION_WINDOW_SECS", "600"))


# ── HWC-PY-2: Warm→Cold demotion constants ─────────────────────────────

WARM_IDLE_TIMEOUT_SECS = int(os.getenv("WARM_IDLE_TIMEOUT_SECS", "600"))
SIGTERM_DRAIN_SECS = int(os.getenv("SIGTERM_DRAIN_SECS", "30"))


# ── Global service tier registry ────────────────────────────────────────

SERVICE_TIERS: dict[str, str] = {
    "semantic-search": "COLD",
    "code-orchestrator": "COLD",
    "llm-gateway": "COLD",
    "ai-agents": "COLD",
    "amve": "COLD",
    "audit-service": "COLD",
    "unified-search-service": "COLD",
    "context-management-service": "COLD",
    "struct-analyzer-service": "COLD",
}


def reset_service_tiers() -> None:
    """Reset all tiers back to COLD (useful for testing)."""
    for key in SERVICE_TIERS:
        SERVICE_TIERS[key] = "COLD"


def auto_warm_service(service_key: str) -> bool:
    """Directly promote *service_key* to WARM.

    Unlike the old HTTP health-check promoter, the new spec-based
    ``ColdWarmPromoter`` uses dispatch-driven promotion via
    ``record_dispatch_for_promotion()``.  This function performs an
    explicit warm-up by setting the tier directly and recording a
    dispatch record.

    Returns ``True`` on success, ``False`` on error.
    """
    try:
        SERVICE_TIERS[service_key] = "WARM"
        # Lazy import to avoid circular dependency: health_config → idle_timeout → health_config
        from src.core.idle_timeout import record_dispatch_for_promotion  # noqa: PLC0415

        record_dispatch_for_promotion(service_key)
        return True
    except Exception:
        return False


def health_timeout_for(service_key: str) -> float:
    """Return the health-check timeout (seconds) for *service_key* by tier.

    Tier timeouts:
    - HOT:  2 s   (already running, fast)
    - WARM: 15 s  (process alive, may be warming up)
    - COLD: 60 s  (needs to be launched)
    - BOOT: 120 s (first-time cold boot)
    """
    tier = SERVICE_TIERS.get(service_key, "COLD")
    if tier == "COLD":
        return 60.0
    if tier == "BOOT":
        return 120.0
    if tier == "WARM":
        return 15.0
    return 2.0  # HOT
