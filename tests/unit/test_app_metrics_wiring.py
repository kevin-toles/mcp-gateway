"""The gateway app records platform request metrics and serves them on /metrics.

src/middleware/metrics.py was never added to src.main.app and nothing mounted /metrics (GET
/metrics was 404), so the gateway reported no ai_platform_* series, including the
ai_platform_tool_calls_total counter the dispatcher already increments.
Label format: service="mcp-gateway", endpoint=<route template>, method, status_code.
The long-lived MCP SSE stream (GET /mcp/sse) must still stream and is not counted.
"""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY

SERVICE = "mcp-gateway"


def _total(**labels: str) -> float:
    return REGISTRY.get_sample_value("ai_platform_request_total", labels=labels) or 0.0


@pytest.fixture
def app():
    from src.main import app

    return app


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def test_a_routed_request_is_counted_under_its_route_template(client):
    template = "/api/v1/tools/{tool_name:path}"
    labels = {"service": SERVICE, "endpoint": template, "method": "POST", "status_code": "404"}
    before = _total(**labels)

    resp = await client.post("/api/v1/tools/no_such_tool_for_metrics_test", json={})

    assert resp.status_code == 404
    assert _total(**labels) - before == 1.0
    exposition = (await client.get("/metrics/")).text
    assert f'endpoint="{template}"' in exposition
    assert "no_such_tool_for_metrics_test" not in exposition


async def test_metrics_serves_the_platform_series_in_prometheus_format(client):
    await client.get("/api/idle-timeout/status")

    resp = await client.get("/metrics/")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "ai_platform_request_duration_seconds_bucket{" in resp.text
    assert f'service="{SERVICE}"' in resp.text
    assert "# TYPE ai_platform_tool_calls_total counter" in resp.text


async def test_health_and_metrics_scrapes_are_not_counted(client):
    def snapshot():
        return [
            _total(service=SERVICE, endpoint=e, method="GET", status_code="200")
            for e in ("/health", "/metrics", "/metrics/")
        ]

    before = snapshot()
    await client.get("/health")
    await client.get("/metrics/")
    assert snapshot() == before


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_mcp_sse_stream_still_streams_and_is_not_counted(app):
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off",
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        before = _total(service=SERVICE, endpoint="/mcp/sse", method="GET", status_code="200")

        with httpx.stream("GET", f"http://127.0.0.1:{port}/mcp/sse", timeout=10) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            first_event = next(line for line in resp.iter_lines() if line.startswith("event:"))

        assert first_event == "event: endpoint"
        assert _total(service=SERVICE, endpoint="/mcp/sse", method="GET", status_code="200") == before
    finally:
        server.should_exit = True
        thread.join(timeout=10)
