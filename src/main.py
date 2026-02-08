"""FastAPI application entrypoint — WBS-MCP1, MCP7, MCP8.

Provides the main FastAPI app with a ``/health`` endpoint,
request-ID middleware, full middleware chain (AC-8.5), and
the FastMCP SSE/HTTP server mounted at ``/mcp``.

Reference: AC-1.1, AC-1.2, AC-1.5, AC-7.6, AC-8.1, AC-8.5
"""

import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, Response

from src.core.config import Settings
from src.models.schemas import HealthResponse
from src.security.audit import AuditMiddleware
from src.security.authn import OIDCAuthMiddleware
from src.security.rate_limiter import RateLimitMiddleware

logger = logging.getLogger(__name__)

settings = Settings()

_start_time = time.monotonic()

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
)


# ── Middleware chain (AC-8.5) ───────────────────────────────────────────
# Order: RequestID → Auth → RateLimit → [handler] → Audit
# Starlette add_middleware prepends, so LAST added = OUTERMOST.
# We add innermost first:

app.add_middleware(AuditMiddleware, log_path=settings.AUDIT_LOG_PATH)

try:
    import redis.asyncio as aioredis

    _redis_client = aioredis.from_url(settings.REDIS_URL)
except Exception:
    _redis_client = None

app.add_middleware(
    RateLimitMiddleware,
    rpm=settings.RATE_LIMIT_RPM,
    redis_client=_redis_client,
)

app.add_middleware(OIDCAuthMiddleware, settings=settings)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    """Assign or preserve a unique request ID on every request (AC-7.6)."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health with name, version, status, and uptime."""
    return HealthResponse(
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        status="healthy",
        uptime_seconds=round(time.monotonic() - _start_time, 2),
    )


# ── MCP Protocol Server (AC-8.1, AC-8.5) ───────────────────────────────

_config_path = Path(__file__).parent.parent / "config" / "tools.yaml"

if _config_path.exists():
    from src.security.output_sanitizer import OutputSanitizer
    from src.server import create_mcp_server
    from src.tool_dispatcher import ToolDispatcher
    from src.tool_registry import ToolRegistry

    _registry = ToolRegistry(_config_path)
    _dispatcher = ToolDispatcher(settings)
    _sanitizer = OutputSanitizer()
    mcp_server = create_mcp_server(_registry, _dispatcher, _sanitizer)
    app.mount("/mcp", mcp_server.http_app(transport="sse"))
    logger.info("MCP server mounted at /mcp with %d tools", _registry.tool_count)
else:
    logger.warning("config/tools.yaml not found — MCP server not mounted")
