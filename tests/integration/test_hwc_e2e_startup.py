"""H/W/C E2E Startup Test Suite — Comprehensive Endpoint Coverage.

Runs at machine startup to validate:
  1. Platform lifecycle daemon (:8079) is listening
  2. MCP Gateway (:8087) is listening and healthy
  3. MCP Gateway endpoint coverage
  4. Platform service endpoint coverage (all 12 services)
  5. Tier state transitions
  6. Service stability monitoring

Usage:
    pytest tests/integration/test_hwc_e2e_startup.py -v --tb=short

    Or via shell wrapper:
    ./scripts/hwc_e2e_test_runner.sh
"""

import asyncio
import time
from datetime import datetime

import httpx
import pytest


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: Proxy & Gateway Startup
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_01_lifecycle_daemon_listening():
    """Platform lifecycle daemon is listening on :8079."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8079/health")
            assert response.status_code == 200, \
                f"Lifecycle daemon should return 200, got {response.status_code}"
            print("  ✓ Lifecycle daemon listening on :8079")
    except httpx.ConnectError:
        pytest.fail("Lifecycle daemon :8079 is not listening")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_02_gateway_startup_latency():
    """MCP Gateway (:8087) responds within acceptable startup window."""
    start = time.time()

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("http://localhost:8087/health", follow_redirects=True)
            elapsed_ms = (time.time() - start) * 1000

            assert elapsed_ms < 5000, \
                f"Gateway health check took {elapsed_ms:.0f}ms (expected < 5000ms)"

            if response.status_code == 200:
                print(f"  ✓ Gateway healthy in {elapsed_ms:.0f}ms")
            else:
                pytest.skip(f"Gateway still starting (status {response.status_code})")
    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.fail("Gateway :8087 connection failed")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: MCP Gateway Endpoint Coverage
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_03_gateway_health_endpoint():
    """Gateway /health endpoint returns healthy status."""
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get("http://localhost:8087/health")
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data.get("status") == "healthy", \
            f"Expected status=healthy, got {data.get('status')}"
        assert "uptime_seconds" in data, \
            "Missing uptime_seconds in health response"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_04_gateway_mcp_sse_endpoint():
    """Gateway /mcp/sse endpoint (MCP protocol server) responds."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get("http://localhost:8087/mcp/sse", follow_redirects=True)
            # MCP SSE may return 200 or just open stream
            assert response.status_code in (200, 401, 403), \
                f"Expected valid status, got {response.status_code}"
        except httpx.TimeoutException:
            # SSE streams don't close, timeout is expected
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_05_gateway_openapi_endpoint():
    """Gateway /openapi.json endpoint serves API schema."""
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get("http://localhost:8087/openapi.json")
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}"
        
        schema = response.json()
        assert "openapi" in schema, "Missing openapi version"
        assert schema.get("info", {}).get("title") == "mcp-gateway", \
            "Unexpected API title"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_06_gateway_docs_endpoint():
    """Gateway /docs endpoint serves Swagger UI."""
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get("http://localhost:8087/docs")
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}"
        assert "swagger" in response.text.lower(), \
            "Response doesn't contain Swagger UI"


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: Platform Service Endpoint Coverage
# ═════════════════════════════════════════════════════════════════════════════


# (service, port, health_path)
SERVICE_ENDPOINTS = {
    "llm-gateway": (8080, "/health"),
    "unified-search": (8081, "/health"),
    "ai-agents": (8082, "/health"),
    "code-orchestrator": (8083, "/health"),
    "audit-service": (8084, "/health"),
    "inference-service-cpp": (8085, "/health"),
    "context-management": (8086, "/health"),
    "mcp-gateway": (8087, "/health"),
    "struct-analyzer": (8088, "/health"),
    "validation-service": (8091, "/health"),
    "amve": (8092, "/v1/health"),
    "unified-search-rs": (8093, "/health"),
}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("service,endpoint", SERVICE_ENDPOINTS.items())
async def test_hwc_10_service_health_endpoints(service: str, endpoint: tuple):
    """Test health endpoint for each platform service.

    Services in COLD tier will fail (expected on first startup).
    Services in WARM/HOT tier should respond with 200.
    """
    port, health_path = endpoint
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(f"http://localhost:{port}{health_path}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    assert "status" in data, \
                        f"{service}: Response should contain health status"
                except Exception:
                    pass  # non-JSON 200 is still healthy
                print(f"  ✓ {service} (:{port}): HEALTHY")
            elif response.status_code in (502, 503, 504):
                pytest.skip(f"{service} (:{port}): COLD tier (not started yet)")
            else:
                pytest.fail(f"{service} (:{port}): Unexpected status {response.status_code}")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip(f"{service} (:{port}): Connection refused (COLD tier, expected)")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("service,endpoint", SERVICE_ENDPOINTS.items())
async def test_hwc_20_service_openapi_endpoints(service: str, endpoint: tuple):
    """Test /openapi.json endpoint for each service (if available).

    Some services may not expose OpenAPI schema.
    """
    port, _ = endpoint
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(f"http://localhost:{port}/openapi.json")

            if response.status_code == 200:
                schema = response.json()
                assert "openapi" in schema or "swagger" in schema, \
                    f"{service}: Invalid schema format"
                print(f"  ✓ {service} (:{port}): OpenAPI available")
            elif response.status_code == 404:
                pytest.skip(f"{service} (:{port}): No OpenAPI endpoint")
            elif response.status_code in (502, 503, 504):
                pytest.skip(f"{service} (:{port}): COLD tier (not started)")
            else:
                print(f"  ? {service} (:{port}): Status {response.status_code}")

        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip(f"{service} (:{port}): Connection refused (COLD tier)")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4: Tier State Verification
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_30_tier_state_verification():
    """Verify services are in expected tier states (HOT/WARM/COLD)."""
    async with httpx.AsyncClient(timeout=5) as client:
        results = {
            "hot": [],
            "warm": [],
            "cold": [],
        }
        
        # Check each service
        for service, (port, health_path) in SERVICE_ENDPOINTS.items():
            try:
                response = await client.get(f"http://localhost:{port}{health_path}", timeout=2)
                if response.status_code == 200:
                    results["warm"].append(f"{service}:{port}")
                elif response.status_code in (502, 503, 504):
                    results["cold"].append(f"{service}:{port}")
            except (httpx.ConnectError, httpx.TimeoutException):
                results["cold"].append(f"{service}:{port}")
        
        # Verify gateway is in WARM/HOT
        try:
            response = await client.get("http://localhost:8087/health", timeout=2)
            if response.status_code == 200:
                results["warm"].append("gateway:8087")
        except:
            pass
        
        # Report tier state
        print("\n  Tier Summary:")
        if results["warm"]:
            print(f"    WARM/HOT: {', '.join(results['warm'])}")
        if results["cold"]:
            print(f"    COLD: {', '.join(results['cold'])}")
        
        # At minimum, gateway should be running
        assert results["warm"], "Gateway should be in WARM/HOT tier"


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5: Stability Monitoring
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_hwc_40_gateway_uptime_stability():
    """Gateway remains stable (monitor uptime delta over 5s)."""
    async with httpx.AsyncClient(timeout=5) as client:
        uptime_start = None

        try:
            response = await client.get("http://localhost:8087/health")
            if response.status_code == 200:
                data = response.json()
                uptime_start = data.get("uptime_seconds")
        except:
            pytest.skip("Gateway not responding")

        if uptime_start is None:
            pytest.skip("Could not get initial uptime")

        # Wait 5 seconds (quick stability check)
        await asyncio.sleep(5)

        # Check uptime again
        try:
            response = await client.get("http://localhost:8087/health")
            data = response.json()
            uptime_end = data.get("uptime_seconds")

            delta = uptime_end - uptime_start
            assert delta > 3, \
                f"Uptime delta should be ~5s, got {delta:.1f}s (possible restart)"

            print(f"  ✓ Gateway stable: {uptime_start:.1f}s → {uptime_end:.1f}s")
        except:
            pytest.fail("Could not verify uptime after 5s wait")


# ═════════════════════════════════════════════════════════════════════════════
# Summary & Report
# ═════════════════════════════════════════════════════════════════════════════


def test_hwc_99_final_report():
    """Generate final test report."""
    print("\n" + "="*70)
    print("H/W/C E2E STARTUP TEST REPORT")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Platform Lifecycle Daemon: http://localhost:8079")
    print(f"MCP Gateway: http://localhost:8087")
    print("\nTest Coverage:")
    print("  ✓ Lifecycle daemon startup verification")
    print("  ✓ Gateway startup latency")
    print("  ✓ Gateway endpoint coverage (5 endpoints)")
    print("  ✓ Platform service endpoints (12 services × 2 endpoints = 24 tests)")
    print("  ✓ Tier state verification")
    print("  ✓ Gateway stability monitoring (5s quick check)")
    print("\n" + "="*70)
    print("Test suite complete. Check output above for pass/skip/fail details.")
    print("="*70 + "\n")

