"""Tier-aware health check timeout configuration — HWC F5.

Maps each backend service to a hot/warm/cold/boot tier and provides
a ``health_timeout_for()`` function used by ``preflight.py`` to set
per-service HTTP timeouts during health checks.

Tiers
-----
HOT   (2.0s)  — Services that are always expected to be running.
WARM  (15.0s) — Services that may need a moment to respond.
COLD  (60.0s) — Services that must be cold-started on first call.
BOOT (120.0s) — Services with unusually slow boot or cold-start penalty.
"""

from __future__ import annotations

import os

from src.core.config import ServiceKey

# ── Tier timeout definitions ────────────────────────────────────────────
# Each value can be overridden via environment variable.
# Base values (GREEN spec):
#   HOT=2.0, WARM=15.0, COLD=60.0, BOOT=120.0

TIER_HEALTH_TIMEOUTS: dict[str, float] = {
    "hot": float(os.environ.get("HWC_HOT_TIMEOUT", "2.0")),
    "warm": float(os.environ.get("HWC_WARM_TIMEOUT", "15.0")),
    "cold": float(os.environ.get("HWC_COLD_TIMEOUT", "60.0")),
    "boot": float(os.environ.get("HWC_BOOT_TIMEOUT", "120.0")),
}

# ── Service tier assignments ────────────────────────────────────────────
# Maps canonical service keys (hyphen-form, matching ServiceKey.__str__)
# to their tier classification.

SERVICE_TIERS: dict[str, str] = {
    # HOT — always-running core services
    "llm-gateway": "hot",
    "semantic-search": "hot",
    "unified-search-service": "hot",
    "mcp-gateway": "hot",
    # WARM — frequently used but may need a moment
    "ai-agents": "warm",
    "audit-service": "warm",
    "code-orchestrator": "warm",
    "amve": "warm",
    # COLD — on-demand / infrequently used
    "inference-service-cpp": "cold",
    "unified-search-rs": "hot",
    "context-management-service": "cold",
}


def health_timeout_for(service_key: str | ServiceKey) -> float:
    """Return the health-check timeout for a service based on its tier.

    Parameters
    ----------
    service_key:
        Canonical service name (hyphen-form) or ``ServiceKey`` instance.

    Returns
    -------
    float
        Timeout in seconds.  If the service is unknown, returns the
        **warm** timeout (15.0s) as a safe default.
    """
    key = str(service_key) if isinstance(service_key, ServiceKey) else service_key
    tier = SERVICE_TIERS.get(key, "warm")
    return TIER_HEALTH_TIMEOUTS.get(tier, 15.0)
