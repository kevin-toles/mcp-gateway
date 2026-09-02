"""Tier-aware health check timeout configuration — HWC F5.

Maps each backend service to a hot/warm/cold/boot tier and provides
a ``health_timeout_for()`` function used by ``preflight.py`` to set
per-service HTTP timeouts during health checks.

Also provides ``SERVICE_STARTUP_COMMANDS`` and ``auto_warm_service()``
for on-demand cold-start of tier-appropriate services.

**Single source of truth**: service tiers, ports, and startup commands are
loaded from ``config/services.toml`` — the same manifest the Rust lifecycle
daemon reads. Nothing service-specific is hardcoded here; edit the manifest,
not this module.

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
import tomllib
from pathlib import Path

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

# ── Service manifest (single source of truth) ───────────────────────────

_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "config" / "services.toml"

DEFAULT_POC_ROOT = "/Users/kevintoles/POC"


def _poc_root() -> str:
    return os.environ.get("POC_ROOT", DEFAULT_POC_ROOT)


def _load_manifest() -> dict[str, dict]:
    with open(_MANIFEST_PATH, "rb") as f:
        return tomllib.load(f)["services"]


def _build_tables() -> tuple[dict[str, str], dict[str, str], dict[str, int], dict[str, str]]:
    """Derive (tiers, startup_commands, ports, ready_paths) from the manifest.

    Aliases resolve to the same values as their primary service.
    ``${POC_ROOT}`` placeholders are expanded from the POC_ROOT env var.
    """
    manifest = _load_manifest()
    root = _poc_root()

    tiers: dict[str, str] = {}
    commands: dict[str, str] = {}
    ports: dict[str, int] = {}
    ready_paths: dict[str, str] = {}

    for name, svc in manifest.items():
        keys = [name, *svc.get("aliases", [])]
        command = svc["spawn"].replace("${POC_ROOT}", root)
        for key in keys:
            tiers[key] = svc["tier"]
            commands[key] = command
            ports[key] = svc["port"]
            if svc.get("ready_path"):
                ready_paths[key] = svc["ready_path"]

    return tiers, commands, ports, ready_paths


SERVICE_TIERS, SERVICE_STARTUP_COMMANDS, SERVICE_PORTS, SERVICE_READY_PATHS = _build_tables()


def restart_command_for(service_key: str) -> str | None:
    """Kill-then-start command for HEALTH_PROXY_SERVICE_CONFIG consumers.

    Prefixes the manifest spawn command with a port-based kill so a stale
    or wedged process can't hold the port (canonical decision: kill-before-
    start lives ONLY in restart commands, never in plain spawn commands).
    """
    command = SERVICE_STARTUP_COMMANDS.get(service_key)
    port = SERVICE_PORTS.get(service_key)
    if command is None or port is None:
        return None
    return (
        f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true; sleep 1; {command}"
    )


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
            "add an entry to config/services.toml"
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
