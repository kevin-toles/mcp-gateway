"""Auto-start backoff + readiness polling — assessment findings #3/#5.

_try_auto_start must:
  1. Stop hammering a crash-looping service: after N consecutive failed
     auto-starts, skip attempts for a cooldown window (spawn backoff).
  2. Poll the service's READINESS endpoint (manifest ready_path) after
     spawning, not just liveness — an alive-but-loading service must not
     be reported as started.
"""

from __future__ import annotations

import time
from unittest.mock import patch

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


async def _fail_spawn(dispatcher: ToolDispatcher, service: str) -> bool:
    """Drive one _try_auto_start attempt that fails at the spawn step."""
    with (
        # Probe raises ConnectError -> port not bound -> spawn path
        patch("httpx.AsyncClient.get", side_effect=__import__("httpx").ConnectError("down")),
        patch(
            "asyncio.create_subprocess_shell",
            side_effect=OSError("spawn failed"),
        ),
    ):
        return await dispatcher._try_auto_start(service, "http://localhost:9")


@pytest.mark.asyncio
async def test_backoff_opens_after_max_consecutive_failures(dispatcher):
    for _ in range(AUTO_START_MAX_CONSECUTIVE_FAILURES):
        assert await _fail_spawn(dispatcher, "audit-service") is False

    # Circuit is now open: the next attempt must be skipped WITHOUT
    # probing or spawning at all.
    with (
        patch("httpx.AsyncClient.get") as probe,
        patch("asyncio.create_subprocess_shell") as spawn,
    ):
        result = await dispatcher._try_auto_start("audit-service", "http://localhost:9")
    assert result is False
    probe.assert_not_called()
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_backoff_expires_after_cooldown(dispatcher):
    for _ in range(AUTO_START_MAX_CONSECUTIVE_FAILURES):
        await _fail_spawn(dispatcher, "audit-service")

    # Simulate the cooldown having elapsed.
    count, _blocked_until = dispatcher._auto_start_failures["audit-service"]
    dispatcher._auto_start_failures["audit-service"] = (
        count,
        time.monotonic() - 1.0,
    )

    # Attempt proceeds again (and fails at spawn — that's fine, it TRIED).
    with (
        patch("httpx.AsyncClient.get", side_effect=__import__("httpx").ConnectError("down")),
        patch("asyncio.create_subprocess_shell", side_effect=OSError("nope")) as spawn,
    ):
        await dispatcher._try_auto_start("audit-service", "http://localhost:9")
    spawn.assert_called_once()


@pytest.mark.asyncio
async def test_success_resets_failure_streak(dispatcher):
    await _fail_spawn(dispatcher, "audit-service")
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
        await _fail_spawn(dispatcher, "audit-service")

    # A different service is unaffected by audit-service's open circuit.
    with (
        patch("httpx.AsyncClient.get", side_effect=__import__("httpx").ConnectError("down")),
        patch("asyncio.create_subprocess_shell", side_effect=OSError("nope")) as spawn,
    ):
        await dispatcher._try_auto_start("ai-agents", "http://localhost:9")
    spawn.assert_called_once()


def test_cooldown_constant_is_minutes_scale():
    assert AUTO_START_COOLDOWN_SECS >= 60.0


@pytest.mark.asyncio
async def test_post_spawn_poll_uses_ready_endpoint(dispatcher):
    """After spawning code-orchestrator (manifest ready_path=/ready), the
    readiness poll must hit /ready, not /health."""
    polled_urls: list[str] = []

    class FakeResp:
        status_code = 503

    async def fake_get(self, url, *a, **kw):
        polled_urls.append(str(url))
        if not polled_urls[:-1]:  # first call = pre-spawn probe -> down
            raise __import__("httpx").ConnectError("down")
        return FakeResp()

    class FakeProc:
        async def wait(self):
            return 0

    with (
        patch("httpx.AsyncClient.get", new=fake_get),
        patch("asyncio.create_subprocess_shell", return_value=FakeProc()),
        # health_timeout_for is imported at call time inside the method,
        # so patching the source module is sufficient.
        patch("src.config.health_config.health_timeout_for", return_value=0.3),
    ):
        await dispatcher._try_auto_start("code-orchestrator", "http://localhost:9")

    ready_polls = [u for u in polled_urls[1:]]
    assert ready_polls, "no readiness polls happened"
    assert all(u.endswith("/ready") for u in ready_polls), (
        f"readiness poll must use manifest ready_path: {ready_polls}"
    )
