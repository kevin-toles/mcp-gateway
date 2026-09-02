"""H/W/C E2E Startup Test Suite — Lifecycle Validation.

Validates the full HWC lifecycle works, not just that services exist:
  1. Core infrastructure is up (lifecycle daemon, gateway)
  2. KeepAlive services (launchd-managed) are healthy
  3. On-demand services auto-start when triggered
  4. Tier transitions work (Cold → Hot on activity, stability check)

This is a startup validation suite. Every test either passes or fails.
A Cold service that can't auto-start is a FAILURE, not a skip.

Usage:
    INTEGRATION=1 pytest tests/integration/test_hwc_e2e_startup.py -v
    ./scripts/hwc_e2e_test_runner.sh
"""

import asyncio
import os
import time
from datetime import datetime

import httpx
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

LIFECYCLE_DAEMON_PORT = 8079
# 150s: heavy Python services (context-management measured ~100s cold under
# load; boot storms at login make it worse). The Rust spawn path polls its
# own BOOT-tier budget of 120s — the test window must exceed it.
AUTO_START_TIMEOUT_SECS = 150
HEALTH_POLL_INTERVAL_SECS = 2

# KeepAlive services: launchd manages these, they should always be up.
KEEPALIVE_SERVICES = {
    "llm-gateway": (8080, "/health"),
    "mcp-gateway": (8087, "/health"),
    "unified-search-rs": (8081, "/health"),
}

# On-demand services: Rust shim spawns these via spawn_cmd.
# The test triggers auto-start and verifies they come up.
ONDEMAND_SERVICES = {
    "ai-agents": (8082, "/health"),
    "code-orchestrator": (8083, "/health"),
    "audit-service": (8084, "/health"),
    "inference-service-cpp": (8085, "/health"),
    "context-management-service": (8086, "/health"),
    "struct-analyzer": (8088, "/health"),
    "validation-service": (8091, "/health"),
    "amve": (8092, "/v1/health"),
}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


async def poll_health(
    port: int,
    health_path: str,
    timeout_secs: float = AUTO_START_TIMEOUT_SECS,
) -> float:
    """Poll a service's health endpoint until it returns 200.

    Returns the elapsed time in seconds.
    Raises AssertionError if the service doesn't come up within timeout.
    """
    start = time.time()
    deadline = start + timeout_secs
    async with httpx.AsyncClient(timeout=5) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(f"http://localhost:{port}{health_path}")
                if resp.status_code == 200:
                    return time.time() - start
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            await asyncio.sleep(HEALTH_POLL_INTERVAL_SECS)

    elapsed = time.time() - start
    raise AssertionError(
        f"Service on :{port} did not become healthy within {timeout_secs}s "
        f"(polled for {elapsed:.1f}s)"
    )


async def trigger_auto_start(service: str) -> None:
    """Report activity to the lifecycle daemon to trigger tier promotion.

    POST /activity/{service} on :8079 records a request and promotes
    Cold/Warm → Hot. For Native-runtime services, health_failure_monitor
    will detect the unhealthy state and respawn via spawn_cmd.
    """
    async with httpx.AsyncClient(timeout=5) as client:
        await client.post(f"http://localhost:{LIFECYCLE_DAEMON_PORT}/activity/{service}")


async def is_service_up(port: int, health_path: str) -> bool:
    """Quick non-blocking health check."""
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            resp = await client.get(f"http://localhost:{port}{health_path}")
            return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 0: Config Integrity — manifest vs LaunchAgent plists
# ═════════════════════════════════════════════════════════════════════════════
#
# The manifest (config/services.toml) is the single source of truth for
# service config. LaunchAgent plists are hand-maintained, so they CAN drift
# (this is exactly how validation-service broke: rewritten Python→Rust,
# plist kept the deleted Python path, exit code 78 on every launchd start).
# These tests catch that drift on every boot instead of letting it fester.

import plistlib
import tomllib
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "config" / "services.toml"
LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"

# manifest service name -> plist basename (where they differ)
PLIST_NAME_OVERRIDES = {
    "semantic-search": "unified-search-rs",
}


def _load_manifest() -> dict:
    with open(MANIFEST_PATH, "rb") as f:
        return tomllib.load(f)["services"]


def _plist_for(service: str) -> Path:
    basename = PLIST_NAME_OVERRIDES.get(service, service)
    return LAUNCH_AGENTS / f"com.kevintoles.{basename}.plist"


@pytest.mark.integration
def test_hwc_00_manifest_exists_and_parses():
    """The service manifest must exist — everything derives from it."""
    assert MANIFEST_PATH.is_file(), f"MANIFEST MISSING: {MANIFEST_PATH}"
    services = _load_manifest()
    assert len(services) >= 12, f"manifest has only {len(services)} services"
    print(f"  ✓ Manifest OK: {len(services)} services")


@pytest.mark.integration
@pytest.mark.parametrize("service", sorted(_load_manifest()))
def test_hwc_01_plist_matches_manifest(service: str):
    """LaunchAgent plist working dirs/commands must exist and agree with
    the manifest — a plist pointing at a deleted directory is exactly the
    validation-service failure mode."""
    plist_path = _plist_for(service)
    if not plist_path.is_file():
        pytest.skip(f"{service}: no LaunchAgent plist (spawned on demand only)")
        return

    with open(plist_path, "rb") as f:
        plist = plistlib.load(f)

    # 1. WorkingDirectory must exist on disk
    wd = plist.get("WorkingDirectory")
    if wd:
        assert Path(wd).is_dir(), (
            f"{service}: plist WorkingDirectory does not exist: {wd} "
            f"— update {plist_path}"
        )

    # 2. Any absolute path mentioned in ProgramArguments must exist
    args = " ".join(plist.get("ProgramArguments", []))
    for token in args.replace("&&", " ").split():
        if token.startswith("/Users/") and ".venv" not in token:
            path = Path(token)
            base = path.parent if not path.exists() and path.suffix else path
            assert base.exists(), (
                f"{service}: plist references nonexistent path {token} "
                f"— update {plist_path}"
            )

    # 3. Port in plist env (if declared) must match the manifest
    manifest_port = _load_manifest()[service]["port"]
    env = plist.get("EnvironmentVariables", {})
    if "PORT" in env:
        assert int(env["PORT"]) == manifest_port, (
            f"{service}: plist PORT={env['PORT']} but manifest says "
            f"{manifest_port} — update {plist_path} or config/services.toml"
        )
    print(f"  ✓ {service}: plist agrees with manifest")


@pytest.mark.integration
def test_hwc_00b_mcp_json_matches_manifest():
    """~/.claude/mcp.json must point to the mcp-gateway port declared in
    services.toml.  This is the drift that broke every Claude Code session
    after the 2026-08-16 proxy removal: port 8090 was intentionally removed
    from the shim, but mcp.json was never updated to point directly to 8087.
    """
    import json

    mcp_json_path = Path.home() / ".claude" / "mcp.json"
    assert mcp_json_path.is_file(), f"~/.claude/mcp.json not found at {mcp_json_path}"

    services = _load_manifest()
    assert "mcp-gateway" in services, "mcp-gateway not in services.toml — cannot validate"
    canonical_port = services["mcp-gateway"]["port"]
    expected_url = f"http://localhost:{canonical_port}/mcp/sse"

    with open(mcp_json_path) as f:
        mcp_config = json.load(f)

    servers = mcp_config.get("mcpServers", {})
    assert "ai-platform" in servers, (
        "~/.claude/mcp.json has no 'ai-platform' server entry — "
        "add: {\"mcpServers\": {\"ai-platform\": {\"type\": \"sse\", \"url\": \"" + expected_url + "\"}}}"
    )

    actual_url = servers["ai-platform"].get("url", "")
    assert actual_url == expected_url, (
        f"~/.claude/mcp.json ai-platform URL is '{actual_url}' "
        f"but services.toml declares mcp-gateway on port {canonical_port} — "
        f"expected '{expected_url}'. "
        f"Run: scripts/validate_mcp_json.py --fix"
    )
    print(f"  ✓ mcp.json ai-platform URL matches manifest: {actual_url}")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: Core Infrastructure
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_01_lifecycle_daemon_listening():
    """Platform lifecycle daemon (:8079) is up and serving health checks."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"http://localhost:{LIFECYCLE_DAEMON_PORT}/health")
            assert response.status_code == 200, \
                f"Lifecycle daemon returned {response.status_code}, expected 200"
            print(f"  ✓ Lifecycle daemon healthy on :{LIFECYCLE_DAEMON_PORT}")
    except httpx.ConnectError:
        pytest.fail(
            f"Lifecycle daemon :{LIFECYCLE_DAEMON_PORT} is not listening. "
            "Check LaunchAgent com.kevintoles.mcp-gateway-shim"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_02_gateway_healthy():
    """MCP Gateway (:8087) responds with healthy status."""
    async with httpx.AsyncClient(timeout=10) as client:
        start = time.time()
        try:
            response = await client.get("http://localhost:8087/health")
            elapsed_ms = (time.time() - start) * 1000
            assert response.status_code == 200, \
                f"Gateway returned {response.status_code}, expected 200"
            data = response.json()
            assert data.get("status") == "healthy", \
                f"Gateway status is '{data.get('status')}', expected 'healthy'"
            print(f"  ✓ Gateway healthy in {elapsed_ms:.0f}ms")
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.fail("Gateway :8087 is not reachable")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: KeepAlive Services (must be running — launchd manages them)
# ═════════════════════════════════════════════════════════════════════════════


# Generous: at login every service boots simultaneously (boot storm), so a
# service that takes ~38s on an idle system can take far longer under load.
KEEPALIVE_BOOT_TIMEOUT_SECS = 90


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(KEEPALIVE_BOOT_TIMEOUT_SECS + 10)
@pytest.mark.parametrize("service,endpoint", KEEPALIVE_SERVICES.items())
async def test_hwc_10_keepalive_service_healthy(service: str, endpoint: tuple):
    """KeepAlive service is running and healthy.

    These services have KeepAlive=true in their LaunchAgent plists.
    Polls briefly to handle startup race (launchd may still be booting them).
    """
    port, health_path = endpoint
    try:
        elapsed = await poll_health(port, health_path, timeout_secs=KEEPALIVE_BOOT_TIMEOUT_SECS)
        print(f"  ✓ {service} (:{port}): healthy (KeepAlive, {elapsed:.1f}s)")
    except AssertionError:
        pytest.fail(
            f"{service} (:{port}): NOT RUNNING after {KEEPALIVE_BOOT_TIMEOUT_SECS}s. "
            f"KeepAlive=true service should be up. "
            f"Check: launchctl list | grep {service}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: On-Demand Auto-Start (the whole point of HWC)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(AUTO_START_TIMEOUT_SECS + 10)
@pytest.mark.parametrize("service,endpoint", ONDEMAND_SERVICES.items())
async def test_hwc_20_ondemand_service_auto_starts(service: str, endpoint: tuple):
    """On-demand service auto-starts and becomes healthy.

    This is the core HWC validation:
    1. If the service is already up → pass (fast path)
    2. If the service is down → trigger auto-start via activity report
    3. Poll health until it comes up or timeout
    4. FAIL if it doesn't start — that means HWC is broken

    A Cold service that refuses to start is a FAILURE, not a skip.
    """
    port, health_path = endpoint

    # Fast path: already running
    if await is_service_up(port, health_path):
        print(f"  ✓ {service} (:{port}): already healthy")
        return

    # Trigger auto-start via activity report to lifecycle daemon
    print(f"  → {service} (:{port}): not running, triggering auto-start...")
    await trigger_auto_start(service)

    # Poll until healthy
    try:
        elapsed = await poll_health(port, health_path)
        print(f"  ✓ {service} (:{port}): auto-started in {elapsed:.1f}s")
    except AssertionError:
        pytest.fail(
            f"{service} (:{port}): FAILED TO AUTO-START within {AUTO_START_TIMEOUT_SECS}s. "
            f"HWC lifecycle is broken for this service. "
            f"Check: spawn_cmd in platform_services.rs, "
            f"SERVICE_STARTUP_COMMANDS in health_config.py"
        )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4: Tier State After Auto-Start
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_30_slo_endpoint_available():
    """SLO endpoint returns per-service health metrics."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(f"http://localhost:{LIFECYCLE_DAEMON_PORT}/slo")
            assert response.status_code == 200, \
                f"SLO endpoint returned {response.status_code}"
            data = response.json()
            assert isinstance(data, list), "SLO response should be a JSON array"
            assert len(data) > 0, "SLO response should contain service entries"

            service_names = {entry["service"] for entry in data}
            assert "mcp-gateway" in service_names, "mcp-gateway missing from SLO data"
            assert "llm-gateway" in service_names, "llm-gateway missing from SLO data"

            print(f"  ✓ SLO endpoint: {len(data)} services reporting")
            for entry in sorted(data, key=lambda e: e["service"]):
                slo_status = "✓" if entry.get("within_slo") else "✗"
                print(
                    f"    {slo_status} {entry['service']}: "
                    f"uptime={entry.get('uptime_ratio', 0):.4f} "
                    f"target={entry.get('slo_target', 0)} "
                    f"tier={entry.get('tier', '?')}"
                )
        except httpx.ConnectError:
            pytest.fail("Lifecycle daemon not reachable for SLO check")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_32_status_endpoint_full_diagnostics():
    """GET /status returns the full per-service diagnostic snapshot:
    tier, runtime, failure counts, circuit state, last transition."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(f"http://localhost:{LIFECYCLE_DAEMON_PORT}/status")
        except httpx.ConnectError:
            pytest.fail("Lifecycle daemon not reachable for /status check")
    assert response.status_code == 200, f"/status returned {response.status_code}"
    data = response.json()
    services = data.get("services")
    assert isinstance(services, list) and services, "/status must list services"
    required = {
        "service", "port", "tier", "runtime", "failure_count",
        "circuit_open", "last_transition", "health_checks_total",
    }
    for entry in services:
        missing = required - set(entry)
        assert not missing, f"{entry.get('service')}: /status missing {missing}"
    transitions = sum(1 for e in services if e.get("last_transition"))
    print(f"  ✓ /status: {len(services)} services, {transitions} with recorded transitions")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_31_tier_state_after_startup():
    """After auto-start, all services should be in a non-Boot tier."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(f"http://localhost:{LIFECYCLE_DAEMON_PORT}/slo")
            if response.status_code != 200:
                pytest.skip("SLO endpoint not available")
                return
            data = response.json()

            boot_services = [
                entry["service"] for entry in data
                if entry.get("tier") == "Boot"
            ]
            assert not boot_services, (
                f"Services still in Boot tier after startup: {boot_services}. "
                "startup_scan or boot_to_cold should have promoted them."
            )
            print("  ✓ No services stuck in Boot tier")
        except httpx.ConnectError:
            pytest.skip("Lifecycle daemon not reachable")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5: Stability
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_hwc_40_gateway_uptime_stability():
    """Gateway remains stable (no unexpected restarts during test window)."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            resp1 = await client.get("http://localhost:8087/health")
            uptime_start = resp1.json().get("uptime_seconds")
        except Exception:
            pytest.fail("Gateway not responding for stability check")

        if uptime_start is None:
            pytest.fail("Gateway health response missing uptime_seconds")

        await asyncio.sleep(5)

        try:
            resp2 = await client.get("http://localhost:8087/health")
            uptime_end = resp2.json().get("uptime_seconds")
        except Exception:
            pytest.fail("Gateway stopped responding during stability window")

        delta = uptime_end - uptime_start
        assert delta > 3, (
            f"Uptime delta was {delta:.1f}s over 5s window — "
            "gateway likely restarted during the test"
        )
        print(f"  ✓ Gateway stable: uptime {uptime_start:.0f}s → {uptime_end:.0f}s")


# ═════════════════════════════════════════════════════════════════════════════
# Final Report
# ═════════════════════════════════════════════════════════════════════════════


def test_hwc_99_final_report():
    """Summary report."""
    print("\n" + "=" * 70)
    print("H/W/C E2E STARTUP VALIDATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"\nValidated:")
    print(f"  • Core infrastructure (lifecycle daemon + gateway)")
    print(f"  • {len(KEEPALIVE_SERVICES)} KeepAlive services (launchd-managed)")
    print(f"  • {len(ONDEMAND_SERVICES)} on-demand services (auto-start lifecycle)")
    print(f"  • SLO metrics endpoint")
    print(f"  • Tier state consistency")
    print(f"  • Gateway stability")
    print("=" * 70 + "\n")
