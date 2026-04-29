"""
Platform-standard Prometheus metrics for mcp-gateway — P1-02.

All ai_platform_* metrics follow design doc §2.3.
Never add user_id, session_id, or request_id as label values (§2.7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

if TYPE_CHECKING:
    from fastapi import FastAPI

# ---------------------------------------------------------------------------
# Platform-standard metrics (all services)  design doc §2.3
# ---------------------------------------------------------------------------

REQUEST_TOTAL = Counter(
    "ai_platform_request_total",
    "Total HTTP requests to this service",
    ["service", "endpoint", "method", "status_code"],
)

REQUEST_DURATION = Histogram(
    "ai_platform_request_duration_seconds",
    "HTTP request duration in seconds",
    ["service", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0],
)

CIRCUIT_BREAKER_STATE = Gauge(
    "ai_platform_circuit_breaker_state",
    "Circuit breaker state: 0=CLOSED, 1=OPEN",
    ["service", "circuit_name"],
)

# ---------------------------------------------------------------------------
# mcp-gateway domain metric  design doc §2.3
# ---------------------------------------------------------------------------

TOOL_CALLS_TOTAL = Counter(
    "ai_platform_tool_calls_total",
    "MCP tool dispatch calls",
    ["tool_name", "status"],  # status: "success" | "error" | "filtered"
)


def mount_metrics(app: FastAPI, service_name: str) -> None:
    """Mount the Prometheus /metrics ASGI sub-application.

    Args:
        app: FastAPI application instance.
        service_name: Service label value for all ai_platform_* metrics.
    """
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
