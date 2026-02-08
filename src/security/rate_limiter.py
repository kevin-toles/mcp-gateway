"""Redis-backed rate limiter — WBS-MCP6 (GREEN).

Starlette middleware that enforces per-tenant requests-per-minute via a
Redis INCR + EXPIRE pipeline (token bucket).  Degrades gracefully when
Redis is unavailable.

Reference: Strategy §5.3 (Resource Exhaustion — P1), §7.1 Controls #4, #13
"""

from __future__ import annotations

import logging
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_logger = logging.getLogger("mcp_gateway.security")

# Paths excluded from rate limiting
_EXCLUDED_PATHS: set[str] = {"/health", "/health/"}

# Window duration in seconds (1 minute)
_WINDOW_SECONDS: int = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant rate limiting with Redis token bucket."""

    def __init__(self, app: Any, rpm: int = 100, redis_client: Any = None) -> None:
        super().__init__(app)
        self.rpm = rpm
        self.redis_client = redis_client

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Skip rate limiting for excluded paths
        if request.url.path in _EXCLUDED_PATHS:
            return await call_next(request)

        tenant_id = self._extract_tenant(request)
        now = int(time.time())
        window_reset = now - (now % _WINDOW_SECONDS) + _WINDOW_SECONDS

        count = await self._increment(tenant_id, window_reset)

        # Build rate-limit headers
        remaining = max(0, self.rpm - count)
        headers = {
            "X-RateLimit-Limit": str(self.rpm),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(window_reset),
        }

        if count > self.rpm:
            retry_after = max(1, window_reset - int(time.time()))
            headers["Retry-After"] = str(retry_after)
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "detail": "Rate limit exceeded. Try again later."},
                headers=headers,
            )

        response = await call_next(request)

        # Inject headers into the successful response
        for key, value in headers.items():
            response.headers[key] = value

        return response

    async def _increment(self, tenant_id: str, window_reset: int) -> int:
        """Atomically increment the request count for *tenant_id*.

        Returns the current count.  On Redis failure, returns 0 (allow).
        """
        if self.redis_client is None:
            _logger.warning("Redis unavailable — rate limiting disabled")
            return 0

        key = f"ratelimit:{tenant_id}:{window_reset}"
        ttl = _WINDOW_SECONDS + 1  # slight buffer

        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, ttl)
            results = await pipe.execute()
            return int(results[0])
        except Exception:
            _logger.warning("Redis error during rate limiting — allowing request")
            return 0

    @staticmethod
    def _extract_tenant(request: Request) -> str:
        """Extract tenant identifier from request.

        Priority: X-Tenant-ID header > authenticated user > client IP.
        """
        tenant = request.headers.get("X-Tenant-ID")
        if tenant:
            return tenant

        # Fall back to auth context if available
        auth = getattr(request.state, "auth", None)
        if auth and hasattr(auth, "tenant_id") and auth.tenant_id:
            return auth.tenant_id

        # Last resort: client IP
        client = request.client
        return client.host if client else "unknown"
