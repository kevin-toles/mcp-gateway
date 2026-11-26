"""Auto-start backoff + readiness polling — assessment findings #3/#5.

_try_auto_start must:
  1. Stop hammering a crash-looping service: after N consecutive failed
     auto-starts, skip attempts for a cooldown window (spawn backoff).
  2. Delegate start + readiness to the lifecycle daemon (Option A, 2026-10-10):
     the shim polls the manifest ready_path; the gateway passes the tier wait
     budget and never spawns services itself.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from src.core.config import Settings
from src.tool_dispatcher import (
    AUTO_START_COOLDOWN_SECS,
    AUTO_START_MAX_CONSECUTIVE_FAILURES,
    ToolDispatcher,
)


@pytest.fixture()
def dispatcher() -> ToolDispatcher:
    return ToolDispatcher(Settings())


class _Shim:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


async def _fail_start(dispatcher: ToolDispatcher, service: str) -> bool:
    """Drive one _try_auto_start attempt that the lifecycle daemon reports as not ready."""
    with (
        patch("httpx.AsyncClient.get", side_effect=__import__("httpx").ConnectError("down")),
        patch("httpx.AsyncClient.post", new=AsyncMock(return_value=_Shim(503))),
    ):
        return await dispatcher._try_auto_start(service, "http://localhost:9")


@pytest.mark.asyncio
async def test_backoff_opens_after_max_consecutive_failures(dispatcher):
    for _ in range(AUTO_START_MAX_CONSECUTIVE_FAILURES):
        assert await _fail_start(dispatcher, "audit-service") is False

    # Circuit is now open: the next attempt must be skipped WITHOUT
    # probing or asking the lifecycle daemon at all.
    with (
        patch("httpx.AsyncClient.get") as probe,
        patch("httpx.AsyncClient.post", new=AsyncMock()) as start,
    ):
        result = await dispatcher._try_auto_start("audit-service", "http://localhost:9")
    assert result is False
    probe.assert_not_called()
    start.assert_not_awaited()


@pytest.mark.asyncio
async def test_backoff_expires_after_cooldown(dispatcher):
    for _ in range(AUTO_START_MAX_CONSECUTIVE_FAILURES):
        await _fail_start(dispatcher, "audit-service")

    # Simulate the cooldown having elapsed.
    count, _blocked_until = dispatcher._auto_start_failures["audit-service"]
    dispatcher._auto_start_failures["audit-service"] = (
        count,
        time.monotonic() - 1.0,
    )

    # Attempt proceeds again (and is refused by the daemon — that's fine, it TRIED).
    with (
        patch("httpx.AsyncClient.get", side_effect=__import__("httpx").ConnectError("down")),
        patch("httpx.AsyncClient.post", new=AsyncMock(return_value=_Shim(503))) as start,
    ):
        await dispatcher._try_auto_start("audit-service", "http://localhost:9")
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_success_resets_failure_streak(dispatcher):
    await _fail_start(dispatcher, "audit-service")
    assert dispatcher._auto_start_failures["audit-service"][0] == 1

    # Service is already healthy — fast path returns True and resets.
    class FakeResp:
        status_code = 200

    with patch("httpx.AsyncClient.get", return_value=FakeResp()):
        assert await dispatcher._try_auto_start("audit-service", "http://localhost:9") is True
    assert "audit-service" not in dispatcher._auto_start_failures


@pytest.mark.asyncio
async def test_backoff_is_per_service(dispatcher):
    for _ in range(AUTO_START_MAX_CONSECUTIVE_FAILURES):
        await _fail_start(dispatcher, "audit-service")

    # A different service is unaffected by audit-service's open circuit.
    with (
        patch("httpx.AsyncClient.get", side_effect=__import__("httpx").ConnectError("down")),
        patch("httpx.AsyncClient.post", new=AsyncMock(return_value=_Shim(503))) as start,
    ):
        await dispatcher._try_auto_start("ai-agents", "http://localhost:9")
    start.assert_awaited_once()


def test_cooldown_constant_is_minutes_scale():
    assert AUTO_START_COOLDOWN_SECS >= 60.0


@pytest.mark.asyncio
async def test_start_and_readiness_are_delegated_to_the_lifecycle_daemon(dispatcher):
    """Readiness (manifest ready_path) is polled by the shim's ensure_ready; the
    gateway asks it to wait and never spawns or polls readiness itself."""
    start = AsyncMock(return_value=_Shim(200))
    with (
        patch("httpx.AsyncClient.get", side_effect=__import__("httpx").ConnectError("down")),
        patch("httpx.AsyncClient.post", new=start),
        patch("asyncio.create_subprocess_shell") as local_spawn,
    ):
        assert await dispatcher._try_auto_start("code-orchestrator", "http://localhost:9") is True
    local_spawn.assert_not_called()
    assert "/activity/code-orchestrator?wait=true" in str(start.call_args.args[0])
