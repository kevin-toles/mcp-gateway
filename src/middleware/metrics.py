"""
Platform MetricsMiddleware — P1-03, P1-08 (Phase 1: Prometheus Metrics Pipeline).

Records ai_platform_request_total and ai_platform_request_duration_seconds
on every HTTP response using FastAPI route-template path normalization to
prevent label cardinality explosion (design doc §2.7).

Usage:
    app.add_middleware(PlatformMetricsMiddleware, service_name="llm-gateway")
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.ai_platform_metrics import REQUEST_DURATION, REQUEST_TOTAL

# ---------------------------------------------------------------------------
# Label cardinality guard  (P1-08, design doc §2.7)
# ---------------------------------------------------------------------------

_FORBIDDEN_LABEL_KEYS: frozenset[str] = frozenset({"user_id", "session_id", "request_id"})


class LabelCardinalityError(ValueError):
    """Raised when a user-scoped identifier is passed as a Prometheus label.

    Per design doc §2.7 no user_id, session_id, or request_id may appear
    as metric label values — they cause unbounded cardinality that exhausts
    Prometheus memory.
    """


# ---------------------------------------------------------------------------
# Path normalization fallback (when route-template lookup fails)
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
_DIGIT_SEGMENT_RE = re.compile(r"/\d+(?=/|$)")


def _normalize_path_fallback(path: str) -> str:
    """Regex-based path normalization (prevents UUID/numeric cardinality)."""
    path = _UUID_RE.sub("{id}", path)
    path = _DIGIT_SEGMENT_RE.sub("/{id}", path)
    return path


def _get_endpoint_label(request: Request) -> str:
    """Return the route-template path for cardinality-safe metric labelling.

    Primary strategy: FastAPI stores the matched route in
    request.scope['route'] after the route handler runs (i.e. after
    call_next returns).  Its .path attribute contains the template string,
    e.g. '/v1/a2a/tasks/{task_id}'.

    Fallback: regex-based normalization of the raw URL path.
    """
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return route.path  # type: ignore[no-any-return]
    return _normalize_path_fallback(request.url.path)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record ai_platform_request_total and ai_platform_request_duration_seconds.

    Mount with:
        app.add_middleware(MetricsMiddleware, service_name="<svc>")

    Uses BaseHTTPMiddleware so the route template is available in
    request.scope['route'] after call_next() resolves.
    """

    def __init__(
        self,
        app,
        service_name: str,
        exclude_paths: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._service = service_name
        self._exclude: frozenset[str] = frozenset(
            exclude_paths if exclude_paths is not None else ["/metrics", "/health", "/health/ready", "/health/live"]
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self._exclude:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        endpoint = _get_endpoint_label(request)

        REQUEST_TOTAL.labels(
            service=self._service,
            endpoint=endpoint,
            method=request.method,
            status_code=str(response.status_code),
        ).inc()

        REQUEST_DURATION.labels(
            service=self._service,
            endpoint=endpoint,
        ).observe(duration)

        return response
