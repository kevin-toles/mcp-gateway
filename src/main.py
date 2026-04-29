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

# P1-02 / P1-04: Platform metrics and middleware
from src.ai_platform_metrics import mount_metrics
from src.core.config import Settings
from src.middleware.metrics import MetricsMiddleware
from src.middleware.session_recovery import SessionRecoveryMiddleware

# P4-07: configure_otel() — idempotent OTel init with FastAPIInstrumentor
from src.middleware.tracing import configure_otel
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

# P1-04: Platform-standard MetricsMiddleware (ai_platform_* names for fleet dashboards)
# Added innermost so it runs after auth/rate-limit and records only admitted requests.
# /metrics excluded so scraping does not produce recursive observations.
app.add_middleware(
    MetricsMiddleware,
    service_name="mcp-gateway",
    exclude_paths=["/metrics", "/health"],
)

# P4-07: configure_otel() — env-var-driven, idempotent; no-op if OTEL_EXPORTER_OTLP_ENDPOINT unset
configure_otel(service_name="mcp-gateway", app=app)

# Session recovery: convert 404s on stale sessions to 410 with recovery instructions
# This prevents the "tool hangs forever" UX when server restarts.
# Added early in chain so it catches 404s from downstream handlers.
app.add_middleware(SessionRecoveryMiddleware, service_version=settings.SERVICE_VERSION)


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


@app.get("/mcp/sessions/{session_id}/health")
async def check_session_health(session_id: str):
    """Check if a specific session ID is valid.

    This endpoint allows MCP clients to verify session validity without
    making a full tool call. Useful for implementing client-side heartbeats
    or reconnection logic.

    Returns:
        {
            "session_id": "abc123...",
            "valid": false,
            "message": "Session not found - server may have restarted",
            "server_version": "1.0.0",
            "server_uptime_seconds": 123.45,
            "recovery_endpoint": "/mcp/sse"
        }
    """
    # FastMCP doesn't expose session_exists() publicly, so we infer from
    # whether we can find the session in the active sessions.
    # For now, return a helpful message - clients should use this to detect
    # stale sessions and reconnect.

    # TODO: Once FastMCP exposes session introspection, actually check validity
    # For now, always return guidance to reconnect on 404s

    return {
        "session_id": session_id,
        "valid": "unknown",
        "message": (
            "Session validity check not fully implemented. "
            "If you receive 410 errors on tool calls, reconnect via /mcp/sse"
        ),
        "server_version": settings.SERVICE_VERSION,
        "server_uptime_seconds": round(time.monotonic() - _start_time, 2),
        "recovery_endpoint": "/mcp/sse",
    }


# ── Admin / Diagnostics ────────────────────────────────────────────────


@app.get("/admin/circuits")
async def get_circuit_breakers():
    """Return snapshot of all circuit breakers (state, counters)."""
    if "_dispatcher" not in globals():
        return {"error": "dispatcher not initialized"}
    return {"breakers": _dispatcher.circuit_breakers.all_snapshots()}


@app.post("/admin/circuits/reset")
async def reset_circuit_breakers():
    """Force-reset ALL circuit breakers to CLOSED."""
    if "_dispatcher" not in globals():
        return {"error": "dispatcher not initialized"}
    await _dispatcher.circuit_breakers.reset_all()
    return {"status": "all circuit breakers reset to CLOSED"}


@app.post("/admin/circuits/{backend_name}/reset")
async def reset_circuit_breaker(backend_name: str):
    """Force-reset a single backend's circuit breaker to CLOSED."""
    if "_dispatcher" not in globals():
        return {"error": "dispatcher not initialized"}
    cb = _dispatcher.circuit_breakers.get(backend_name)
    await cb.reset()
    return {"status": f"circuit breaker '{backend_name}' reset to CLOSED"}


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

# P1-04: Mount /metrics endpoint for Prometheus scraping (ai_platform_* metrics)
mount_metrics(app, service_name="mcp-gateway")
