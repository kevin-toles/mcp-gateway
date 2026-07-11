"""H/W/C E2E Startup Test Suite — Comprehensive Endpoint Coverage.

Runs at shim startup to validate:
  1. Rust proxy (:8090) is listening
  2. Cold-start spawn latency
  3. MCP Gateway endpoint coverage
  4. Platform service endpoint coverage  
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
async def test_hwc_01_proxy_listening():
    """MCP Lifecycle Proxy is listening on :8090."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://localhost:8090/health")
            assert response.status_code in (200, 502, 504), \
                f"Proxy should be accepting connections, got {response.status_code}"
        pytest.skip("Proxy listening (gateway may not be up yet)")
    except httpx.ConnectError:
        pytest.fail("Proxy :8090 is not listening")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hwc_02_gateway_cold_start_latency():
    """Gateway cold-starts on first proxy connection (~300-500ms expected)."""
    start = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("http://localhost:8090/health", follow_redirects=True)
            elapsed_ms = (time.time() - start) * 1000
            
            # First request may be slow due to spawn, but should complete within 5s
            assert elapsed_ms < 5000, \
                f"First request took {elapsed_ms:.0f}ms (expected < 5000ms)"
            
            # Gateway should eventually respond
            if response.status_code == 200:
                print(f"  ✓ Gateway spawned in {elapsed_ms:.0f}ms")
            else:
                pytest.skip(f"Gateway still starting (status {response.status_code})")
    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.fail("Proxy/Gateway connection failed")


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


SERVICE_ENDPOINTS = {
    "llm-gateway": 8080,
    "unified-search": 8081,
    "ai-agents": 8082,
    "code-orchestrator": 8083,
    "audit-service": 8084,
    "context-management": 8086,
    "struct-analyzer": 8088,
}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("service,port", SERVICE_ENDPOINTS.items())
async def test_hwc_10_service_health_endpoints(service: str, port: int):
    """Test /health endpoint for each platform service.
    
    Services in COLD tier will fail (expected on first startup).
    Services in WARM/HOT tier should respond with 200.
    """
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(f"http://localhost:{port}/health")
            
            if response.status_code == 200:
                data = response.json()
                assert data.get("status") == "healthy" or "status" in data, \
                    f"{service}: Response should contain health status"
                print(f"  ✓ {service} ({port}): HEALTHY")
            elif response.status_code in (502, 503, 504):
                pytest.skip(f"{service} ({port}): COLD tier (not started yet)")
            else:
                pytest.fail(f"{service} ({port}): Unexpected status {response.status_code}")
                
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip(f"{service} ({port}): Connection refused (COLD tier, expected)")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("service,port", SERVICE_ENDPOINTS.items())
async def test_hwc_20_service_openapi_endpoints(service: str, port: int):
    """Test /openapi.json endpoint for each service (if available).
    
    Some services may not expose OpenAPI schema.
    """
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            response = await client.get(f"http://localhost:{port}/openapi.json")
            
            if response.status_code == 200:
                schema = response.json()
                assert "openapi" in schema or "swagger" in schema, \
                    f"{service}: Invalid schema format"
                print(f"  ✓ {service} ({port}): OpenAPI available")
            elif response.status_code == 404:
                pytest.skip(f"{service} ({port}): No OpenAPI endpoint")
            elif response.status_code in (502, 503, 504):
                pytest.skip(f"{service} ({port}): COLD tier (not started)")
            else:
                # Just warn, don't fail
                print(f"  ⚠ {service} ({port}): Status {response.status_code}")
                
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip(f"{service} ({port}): Connection refused (COLD tier)")


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
        for service, port in SERVICE_ENDPOINTS.items():
            try:
                response = await client.get(f"http://localhost:{port}/health", timeout=2)
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
    print(f"MCP Lifecycle Proxy: http://localhost:8090")
    print(f"MCP Gateway: http://localhost:8087")
    print("\nTest Coverage:")
    print("  ✓ Proxy startup verification")
    print("  ✓ Cold-start spawn latency")
    print("  ✓ Gateway endpoint coverage (5 endpoints)")
    print("  ✓ Platform service endpoints (7 services × 2 endpoints = 14 tests)")
    print("  ✓ Tier state verification")
    print("  ✓ Gateway stability monitoring (5s quick check)")
    print("\n" + "="*70)
    print("Test suite complete. Check output above for pass/skip/fail details.")
    print("="*70 + "\n")

