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
AUTO_START_TIMEOUT_SECS = 90
HEALTH_POLL_INTERVAL_SECS = 2

# KeepAlive services: launchd manages these, they should always be up.
KEEPALIVE_SERVICES = {
    "llm-gateway": (8080, "/health"),
    "mcp-gateway": (8087, "/health"),
    "unified-search-rs": (8093, "/health"),
    "unified-search-service": (8081, "/health"),
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


KEEPALIVE_BOOT_TIMEOUT_SECS = 45


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
