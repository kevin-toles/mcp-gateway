"""Tests for FastAPI app and /health endpoint — WBS-MCP1.4 (RED).

Verifies that:
- FastAPI app exists and is importable
- GET /health returns 200
- Response includes service name, version, status, and uptime
- Response content-type is application/json
- No imports from ai-agents or other platform packages

Reference: AC-1.1, AC-1.2, AC-1.5
"""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    """Import and return the FastAPI app."""
    from src.main import app

    return app


@pytest.fixture
async def client(app):
    """Create an async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


class TestHealthEndpoint:
    """AC-1.1: FastAPI application starts; AC-1.2: /health returns metadata."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_json(self, client):
        response = await client.get("/health")
        assert response.headers["content-type"] == "application/json"

    @pytest.mark.asyncio
    async def test_health_contains_service_name(self, client):
        response = await client.get("/health")
        data = response.json()
        assert data["service"] == "mcp-gateway"

    @pytest.mark.asyncio
    async def test_health_contains_version(self, client):
        response = await client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_health_contains_status(self, client):
        response = await client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_contains_uptime(self, client):
        response = await client.get("/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0


class TestAppImport:
    """AC-1.5: Service runs independently — no platform imports."""

    def test_app_is_fastapi_instance(self):
        from fastapi import FastAPI

        from src.main import app

        assert isinstance(app, FastAPI)

    def test_no_platform_imports(self):
        """Verify src/ has no imports from ai-agents or other platform services."""
        import importlib
        import sys

        # Import the main module
        import src.main  # noqa: F401

        # Check that no platform-service modules are loaded
        platform_packages = [
            "ai_agents",
            "llm_gateway",
            "semantic_search",
            "code_orchestrator",
            "inference_service",
        ]
        loaded_platform = [
            mod
            for mod in sys.modules
            if any(mod.startswith(pkg) for pkg in platform_packages)
        ]
        assert loaded_platform == [], (
            f"Platform packages imported: {loaded_platform}"
        )


class TestRequestID:
    """AC-7.6: Every response includes X-Request-ID."""

    @pytest.mark.asyncio
    async def test_health_has_request_id(self, client):
        response = await client.get("/health")
        assert "x-request-id" in response.headers
        rid = response.headers["x-request-id"]
        assert len(rid) == 36  # UUID v4 format

    @pytest.mark.asyncio
    async def test_request_ids_are_unique(self, client):
        r1 = await client.get("/health")
        r2 = await client.get("/health")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]

    @pytest.mark.asyncio
    async def test_client_request_id_preserved(self, client):
        """If client sends X-Request-ID, server should use it."""
        response = await client.get(
            "/health", headers={"X-Request-ID": "client-req-999"}
        )
        assert response.headers["x-request-id"] == "client-req-999"
