"""Option A — the shim is the single spawn authority (mcp-gateway GAP-MCP-005).

The gateway must not spawn services itself (`SERVICE_STARTUP_COMMANDS` subprocess):
when a target is not healthy it asks the lifecycle daemon to make it ready via
``POST :8079/activity/{service}?wait=true&timeout_secs=N`` and proceeds only on 200.

RED state: `_try_auto_start` spawns via `asyncio.create_subprocess_shell` and polls
readiness itself; a bound-but-unhealthy (still starting) service returns False.
GREEN state: one shim call per attempt; no local spawn; crash-loop cooldown kept.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.core.config import Settings
from src.tool_dispatcher import AUTO_START_MAX_CONSECUTIVE_FAILURES, ToolDispatcher


class _Resp:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}

    def json(self) -> dict:
        return self._body


@pytest.fixture()
def dispatcher() -> ToolDispatcher:
    return ToolDispatcher(Settings())


def _down(*_a, **_k):
    raise httpx.ConnectError("down")


async def test_dead_service_is_started_by_the_shim_not_the_gateway(dispatcher):
    post = AsyncMock(return_value=_Resp(200, {"ready": True, "spawned": True}))
    with (
        patch("httpx.AsyncClient.get", side_effect=_down),
        patch("httpx.AsyncClient.post", new=post),
        patch("asyncio.create_subprocess_shell") as local_spawn,
    ):
        assert await dispatcher._try_auto_start("code-orchestrator", "http://localhost:8083") is True
    local_spawn.assert_not_called()
    url = str(post.call_args.args[0] if post.call_args.args else post.call_args.kwargs["url"])
    assert url.startswith("http://127.0.0.1:8079/activity/code-orchestrator?wait=true")


async def test_shim_not_ready_is_a_failure_and_counts_toward_cooldown(dispatcher):
    post = AsyncMock(return_value=_Resp(503, {"ready": False, "reason": "timeout"}))
    with (
        patch("httpx.AsyncClient.get", side_effect=_down),
        patch("httpx.AsyncClient.post", new=post),
    ):
        for _ in range(AUTO_START_MAX_CONSECUTIVE_FAILURES):
            assert await dispatcher._try_auto_start("audit-service", "http://localhost:8084") is False
        calls = post.await_count
        # Circuit open: no further shim calls until the cooldown expires.
        assert await dispatcher._try_auto_start("audit-service", "http://localhost:8084") is False
    assert post.await_count == calls


async def test_bound_but_still_starting_service_is_waited_on_via_shim(dispatcher):
    post = AsyncMock(return_value=_Resp(200, {"ready": True, "spawned": False}))
    with (
        patch("httpx.AsyncClient.get", return_value=_Resp(503)),
        patch("httpx.AsyncClient.post", new=post),
    ):
        assert await dispatcher._try_auto_start("context-management-service", "http://localhost:8086") is True
    post.assert_awaited_once()


async def test_shim_unreachable_fails_without_local_spawn_fallback(dispatcher):
    with (
        patch("httpx.AsyncClient.get", side_effect=_down),
        patch("httpx.AsyncClient.post", side_effect=_down),
        patch("asyncio.create_subprocess_shell") as local_spawn,
    ):
        assert await dispatcher._try_auto_start("ai-agents", "http://localhost:8082") is False
    local_spawn.assert_not_called()


async def test_healthy_service_skips_the_shim(dispatcher):
    post = AsyncMock()
    with (
        patch("httpx.AsyncClient.get", return_value=_Resp(200)),
        patch("httpx.AsyncClient.post", new=post),
    ):
        assert await dispatcher._try_auto_start("ai-agents", "http://localhost:8082") is True
    post.assert_not_awaited()


def test_gateway_does_not_run_its_own_idle_shutdown_actuator():
    """Option A: the shim is the only lifecycle authority (start AND stop).

    The gateway lifespan previously called ``get_checker().start()`` — an
    un-awaited coroutine, so the WarmColdActuator (``kill -9`` via
    SERVICE_SHUTDOWN_COMMANDS) never actually ran; awaiting it would have
    created a second stop authority racing the shim's Warm→Cold stop.
    """
    import inspect

    import src.main as gateway_main

    source = inspect.getsource(gateway_main)
    assert "idle_timeout_checker" not in source
    assert "get_checker" not in source
