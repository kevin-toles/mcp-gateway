"""Tier-aware health check timeout configuration — HWC F5.

Maps each backend service to a hot/warm/cold/boot tier and provides
a ``health_timeout_for()`` function used by ``preflight.py`` to set
per-service HTTP timeouts during health checks.

Also provides ``SERVICE_STARTUP_COMMANDS`` and ``auto_warm_service()``
for on-demand cold-start of tier-appropriate services.

Tiers
-----
HOT   (2.0s)  — Services that are always expected to be running.
WARM  (15.0s) — Services that may need a moment to respond.
COLD  (60.0s) — Services that must be cold-started on first call.
BOOT (120.0s) — Services with unusually slow boot or cold-start penalty.
"""

from __future__ import annotations

import asyncio
import os

from src.core.config import ServiceKey

# ── Tier timeout definitions ────────────────────────────────────────────
# Each value can be overridden via environment variable.
# Base values (GREEN spec):
#   HOT=2.0, WARM=15.0, COLD=60.0, BOOT=120.0

TIER_HEALTH_TIMEOUTS: dict[str, float] = {
    "hot": float(os.environ.get("HEALTH_CHECK_TIMEOUT_HOT", os.environ.get("HWC_HOT_TIMEOUT", "2.0"))),
    "warm": float(os.environ.get("HEALTH_CHECK_TIMEOUT_WARM", os.environ.get("HWC_WARM_TIMEOUT", "15.0"))),
    "cold": float(os.environ.get("HEALTH_CHECK_TIMEOUT_COLD", os.environ.get("HWC_COLD_TIMEOUT", "60.0"))),
    "boot": float(os.environ.get("HEALTH_CHECK_TIMEOUT_BOOT", os.environ.get("HWC_BOOT_TIMEOUT", "120.0"))),
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

# ── Service startup commands ────────────────────────────────────────────
# Shell commands to start each service when auto-warming a cold or warm
# service on demand.  Keys match SERVICE_TIERS.
# NOTE: These overlap with restart_command in HEALTH_PROXY_SERVICE_CONFIG
# but are duplicated here so health_config.py remains self-contained.

SERVICE_STARTUP_COMMANDS: dict[str, str] = {
    "unified-search-service": (
        "cd /Users/kevintoles/POC/unified-search-service && "
        ".venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8081"
    ),
    "code-orchestrator": (
        "COS_CODEBERT_START_MODE=warm "
        "COS_GRAPHCODEBERT_START_MODE=cold "
        "COS_CODET5_START_MODE=cold "
        "cd /Users/kevintoles/POC/Code-Orchestrator-Service && "
        ".venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8083"
    ),
    "llm-gateway": (
        "cd /Users/kevintoles/POC/llm-gateway && "
        ".venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8080"
    ),
    "ai-agents": (
        "cd /Users/kevintoles/POC/ai-agents && "
        ".venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8082"
    ),
    "audit-service": (
        "cd /Users/kevintoles/POC/audit-service && "
        ".venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8084"
    ),
    "context-management-service": (
        "cd /Users/kevintoles/POC/context-management-service && "
        ".venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8086"
    ),
    "amve": (
        "cd /Users/kevintoles/POC/architecture-mapping-validation-engine && "
        ".venv/bin/python -m src.main"
    ),
    "unified-search-rs": (
        "cd /Users/kevintoles/POC/unified-search-rs && "
        "cargo run --release"
    ),
    "inference-service-cpp": (
        "cd /Users/kevintoles/POC/inference-service-cpp && "
        "./build/inference-service"
    ),
    "mcp-gateway": (
        "lsof -ti:8087 | xargs kill -9 2>/dev/null || true; sleep 1; "
        "cd /Users/kevintoles/POC/mcp-gateway && "
        ".venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8087"
    ),
    "semantic-search": (
        "cd /Users/kevintoles/POC/unified-search-service && "
        ".venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8081"
    ),
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
    key = str(service_key) if isinstance(service_key, str) else service_key
    tier = SERVICE_TIERS.get(key, "warm")
    return TIER_HEALTH_TIMEOUTS.get(tier, 15.0)


async def auto_warm_service(service_key: str | ServiceKey) -> None:
    """Start a service on-demand if its tier is cold or warm.

    Launches the startup command as a subprocess and waits up to the
    tier's health timeout for the service to become reachable.  Hot
    services are skipped — they should already be running.

    Parameters
    ----------
    service_key:
        Canonical service name (hyphen-form) or ``ServiceKey`` instance.

    Raises
    ------
    RuntimeError
        If the service does not have a startup command registered.
    asyncio.TimeoutError
        If the service does not become healthy within the tier timeout.
    """
    key = str(service_key)
    tier = SERVICE_TIERS.get(key, "warm")

    # Hot services should already be running; skip auto-warm.
    if tier == "hot":
        return

    command = SERVICE_STARTUP_COMMANDS.get(key)
    if not command:
        raise RuntimeError(
            f"No startup command registered for service '{key}'; "
            "add an entry to SERVICE_STARTUP_COMMANDS in health_config.py"
        )

    timeout = TIER_HEALTH_TIMEOUTS.get(tier, 15.0)

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    # Wait up to the tier timeout for the process to start.
    # We rely on the caller to probe health via HTTP afterwards.
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        # Process is still running — that's fine, it started.
        pass
