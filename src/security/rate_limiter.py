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

from starlette.responses import JSONResponse

_logger = logging.getLogger("mcp_gateway.security")

# Paths excluded from rate limiting
_EXCLUDED_PATHS: set[str] = {"/health", "/health/"}

# SSE/streaming paths incompatible with BaseHTTPMiddleware (breaks streaming)
_SSE_PREFIX: str = "/mcp"

# Window duration in seconds (1 minute)
_WINDOW_SECONDS: int = 60


class RateLimitMiddleware:
    """Per-tenant rate limiting with Redis token bucket.

    Raw ASGI implementation avoids BaseHTTPMiddleware anyio buffer wrapping
    that breaks SSE streams under Starlette 0.52+.
    """

    def __init__(self, app: Any, rpm: int = 100, redis_client: Any = None) -> None:
        self.app = app
        self.rpm = rpm
        self.redis_client = redis_client

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _EXCLUDED_PATHS or path.startswith(_SSE_PREFIX):
            await self.app(scope, receive, send)
            return

        tenant_id = self._extract_tenant(scope)
        now = int(time.time())
        window_reset = now - (now % _WINDOW_SECONDS) + _WINDOW_SECONDS

        count = await self._increment(tenant_id, window_reset)

        remaining = max(0, self.rpm - count)
        rl_headers = {
            "X-RateLimit-Limit": str(self.rpm),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(window_reset),
        }

        if count > self.rpm:
            retry_after = max(1, window_reset - int(time.time()))
            rl_headers["Retry-After"] = str(retry_after)
            response = JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "detail": "Rate limit exceeded. Try again later."},
                headers=rl_headers,
            )
            await response(scope, receive, send)
            return

        async def send_wrapper(message: Any) -> None:
            if message["type"] == "http.response.start":
                extra = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in rl_headers.items()]
                await send({**message, "headers": list(message.get("headers", [])) + extra})
            else:
                await send(message)

        await self.app(scope, receive, send_wrapper)

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
    def _extract_tenant(scope: dict) -> str:
        """Extract tenant identifier from scope.

        Priority: X-Tenant-ID header > authenticated user > client IP.
        """
        header_dict = {name.lower(): value for name, value in scope.get("headers", [])}

        tenant = header_dict.get(b"x-tenant-id")
        if tenant:
            return tenant.decode()

        auth = getattr(scope.get("state"), "auth", None)
        if auth and hasattr(auth, "tenant_id") and auth.tenant_id:
            return auth.tenant_id

        client = scope.get("client")
        return client[0] if client else "unknown"
