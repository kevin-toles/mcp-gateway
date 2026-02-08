"""Redis-backed rate limiter tests — WBS-MCP6 (RED).

Covers AC-6.1 (per-tenant RPM via Redis), AC-6.2 (rate limit headers),
AC-6.3 (429 + Retry-After), AC-6.6 (graceful Redis failure).

Uses fakeredis for Redis simulation.
"""

import time
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis
import pytest
from starlette.testclient import TestClient

from src.security.rate_limiter import RateLimitMiddleware


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_app(
    rpm: int = 5,
    redis_client=None,
):
    """Build a minimal FastAPI app with RateLimitMiddleware."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        rpm=rpm,
        redis_client=redis_client,
    )

    @app.get("/test")
    async def _test(request: Request):
        return JSONResponse({"ok": True})

    @app.get("/health")
    async def _health(request: Request):
        return JSONResponse({"status": "ok"})

    return app


@pytest.fixture()
def fake_redis():
    """Provide a fresh fakeredis async client."""
    return fakeredis.aioredis.FakeRedis()


# ── AC-6.1: Per-tenant rate limiting via Redis ──────────────────────────


class TestPerTenantRateLimiting:
    """Requests counted per tenant."""

    def test_first_request_succeeds(self, fake_redis) -> None:
        app = _make_app(rpm=5, redis_client=fake_redis)
        client = TestClient(app)
        resp = client.get("/test", headers={"X-Tenant-ID": "tenant-a"})
        assert resp.status_code == 200

    def test_requests_within_limit_succeed(self, fake_redis) -> None:
        app = _make_app(rpm=5, redis_client=fake_redis)
        client = TestClient(app)
        for _ in range(5):
            resp = client.get("/test", headers={"X-Tenant-ID": "tenant-a"})
            assert resp.status_code == 200

    def test_different_tenants_tracked_separately(self, fake_redis) -> None:
        app = _make_app(rpm=2, redis_client=fake_redis)
        client = TestClient(app)
        # Fill tenant-a
        for _ in range(2):
            client.get("/test", headers={"X-Tenant-ID": "tenant-a"})
        # tenant-a exhausted
        resp_a = client.get("/test", headers={"X-Tenant-ID": "tenant-a"})
        assert resp_a.status_code == 429
        # tenant-b still ok
        resp_b = client.get("/test", headers={"X-Tenant-ID": "tenant-b"})
        assert resp_b.status_code == 200

    def test_anonymous_uses_ip_fallback(self, fake_redis) -> None:
        app = _make_app(rpm=3, redis_client=fake_redis)
        client = TestClient(app)
        for _ in range(3):
            resp = client.get("/test")
            assert resp.status_code == 200
        resp = client.get("/test")
        assert resp.status_code == 429


# ── AC-6.2: Rate limit headers ──────────────────────────────────────────


class TestRateLimitHeaders:
    """X-RateLimit-* headers on every response."""

    def test_limit_header_present(self, fake_redis) -> None:
        app = _make_app(rpm=10, redis_client=fake_redis)
        client = TestClient(app)
        resp = client.get("/test", headers={"X-Tenant-ID": "t1"})
        assert resp.headers["X-RateLimit-Limit"] == "10"

    def test_remaining_decrements(self, fake_redis) -> None:
        app = _make_app(rpm=5, redis_client=fake_redis)
        client = TestClient(app)
        resp1 = client.get("/test", headers={"X-Tenant-ID": "t1"})
        assert resp1.headers["X-RateLimit-Remaining"] == "4"
        resp2 = client.get("/test", headers={"X-Tenant-ID": "t1"})
        assert resp2.headers["X-RateLimit-Remaining"] == "3"

    def test_reset_header_is_epoch_int(self, fake_redis) -> None:
        app = _make_app(rpm=10, redis_client=fake_redis)
        client = TestClient(app)
        resp = client.get("/test", headers={"X-Tenant-ID": "t1"})
        reset = int(resp.headers["X-RateLimit-Reset"])
        assert reset > int(time.time()) - 5  # within last 5s tolerance

    def test_headers_on_429_too(self, fake_redis) -> None:
        app = _make_app(rpm=1, redis_client=fake_redis)
        client = TestClient(app)
        client.get("/test", headers={"X-Tenant-ID": "t1"})  # exhaust
        resp = client.get("/test", headers={"X-Tenant-ID": "t1"})
        assert resp.status_code == 429
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers


# ── AC-6.3: 429 with Retry-After ────────────────────────────────────────


class TestRateLimitExceeded:
    """Exceeding RPM returns 429 with Retry-After."""

    def test_429_when_exceeded(self, fake_redis) -> None:
        app = _make_app(rpm=2, redis_client=fake_redis)
        client = TestClient(app)
        client.get("/test", headers={"X-Tenant-ID": "t1"})
        client.get("/test", headers={"X-Tenant-ID": "t1"})
        resp = client.get("/test", headers={"X-Tenant-ID": "t1"})
        assert resp.status_code == 429

    def test_retry_after_header_present(self, fake_redis) -> None:
        app = _make_app(rpm=1, redis_client=fake_redis)
        client = TestClient(app)
        client.get("/test", headers={"X-Tenant-ID": "t1"})
        resp = client.get("/test", headers={"X-Tenant-ID": "t1"})
        assert resp.status_code == 429
        retry = int(resp.headers["Retry-After"])
        assert 1 <= retry <= 60

    def test_429_body_has_error_message(self, fake_redis) -> None:
        app = _make_app(rpm=1, redis_client=fake_redis)
        client = TestClient(app)
        client.get("/test", headers={"X-Tenant-ID": "t1"})
        resp = client.get("/test", headers={"X-Tenant-ID": "t1"})
        body = resp.json()
        assert "rate limit" in body.get("detail", "").lower() or "rate limit" in body.get("error", "").lower()


# ── AC-6.6: Graceful Redis failure ──────────────────────────────────────


class TestRedisGracefulDegradation:
    """Redis unavailable → allow traffic, log warning."""

    def test_allows_traffic_when_redis_unavailable(self) -> None:
        """With no Redis client (None), middleware degrades to allow."""
        app = _make_app(rpm=5, redis_client=None)
        client = TestClient(app)
        resp = client.get("/test", headers={"X-Tenant-ID": "t1"})
        assert resp.status_code == 200

    def test_allows_traffic_when_redis_raises(self, fake_redis) -> None:
        """When Redis raises ConnectionError, traffic is still allowed."""
        app = _make_app(rpm=5, redis_client=fake_redis)
        client = TestClient(app)
        # Monkey-patch the pipeline to raise
        with patch.object(fake_redis, "pipeline", side_effect=ConnectionError("gone")):
            resp = client.get("/test", headers={"X-Tenant-ID": "t1"})
        assert resp.status_code == 200

    def test_logs_warning_when_redis_fails(self, fake_redis, caplog) -> None:
        import logging
        app = _make_app(rpm=5, redis_client=fake_redis)
        client = TestClient(app)
        with caplog.at_level(logging.WARNING, logger="mcp_gateway.security"):
            with patch.object(fake_redis, "pipeline", side_effect=ConnectionError("gone")):
                client.get("/test", headers={"X-Tenant-ID": "t1"})
        assert any("redis" in r.message.lower() for r in caplog.records)


# ── Health path bypass ──────────────────────────────────────────────────


class TestHealthPathBypass:
    """Health endpoint should not be rate-limited."""

    def test_health_not_counted(self, fake_redis) -> None:
        app = _make_app(rpm=1, redis_client=fake_redis)
        client = TestClient(app)
        # Many health checks should not count
        for _ in range(10):
            resp = client.get("/health")
            assert resp.status_code == 200
        # Actual endpoint still has full budget
        resp = client.get("/test", headers={"X-Tenant-ID": "t1"})
        assert resp.status_code == 200
