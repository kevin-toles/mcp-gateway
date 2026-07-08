#!/usr/bin/env python3
"""
Idle Timeout Checker Tests
===========================

Tests for the WarmColdActuator from src.core.idle_timeout_checker.

Covers:
  - Singleton via get_checker() / reset_checker()
  - start() / stop() lifecycle (idempotent)
  - is_running property
  - _shutdown_service (SIGTERM → drain → SIGKILL)
  - _resolve_pid from env vars
  - Integration with IdleTimeoutTracker
"""

import asyncio
import os
import signal
from typing import Optional
from unittest.mock import patch

import pytest

from src.core.idle_timeout_checker import (
    WarmColdActuator,
    get_checker,
    reset_checker,
    _POLL_INTERVAL,
    _DRAIN_SECONDS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test to avoid cross-test pollution."""
    reset_checker()
    yield
    # Ensure the checker is stopped if it was started
    instance = get_checker()
    if instance.is_running:
        # Can't await in a fixture finalizer easily — mark as stopped
        instance._task = None
        instance._stopped.set()
    reset_checker()


# =============================================================================
# Default Constants
# =============================================================================

class TestDefaults:
    """Default configuration values."""

    def test_poll_interval_default(self):
        """Default poll interval should be 60s."""
        assert _POLL_INTERVAL == 60.0

    def test_drain_seconds_default(self):
        """Default drain seconds should be 30s."""
        assert _DRAIN_SECONDS == 30.0


# =============================================================================
# WarmColdActuator — Init
# =============================================================================

class TestInit:
    """WarmColdActuator initialization."""

    def test_default_constructor(self):
        """Default constructor should use standard poll/drain values."""
        actuator = WarmColdActuator()
        assert actuator.poll_interval == 60.0
        assert actuator.drain_seconds == 30.0
        assert actuator._task is None
        assert actuator.is_running is False

    def test_custom_poll_and_drain(self):
        """Constructor should accept custom poll interval and drain seconds."""
        actuator = WarmColdActuator(poll_interval=10.0, drain_seconds=5.0)
        assert actuator.poll_interval == 10.0
        assert actuator.drain_seconds == 5.0


# =============================================================================
# WarmColdActuator — Lifecycle
# =============================================================================

class TestLifecycle:
    """start() / stop() / is_running."""

    @pytest.mark.asyncio
    async def test_start_sets_is_running(self):
        """After start(), is_running should be True."""
        actuator = WarmColdActuator(poll_interval=3600)  # 1h = won't fire
        assert actuator.is_running is False
        await actuator.start()
        assert actuator.is_running is True
        await actuator.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_is_running(self):
        """After stop(), is_running should be False."""
        actuator = WarmColdActuator(poll_interval=3600)
        await actuator.start()
        assert actuator.is_running is True
        await actuator.stop()
        assert actuator.is_running is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        """Calling start() twice should not create duplicate tasks."""
        actuator = WarmColdActuator(poll_interval=3600)
        await actuator.start()
        task = actuator._task
        await actuator.start()  # second call — no-op
        assert actuator._task is task  # same task object
        await actuator.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self):
        """Calling stop() when not running should not raise."""
        actuator = WarmColdActuator()
        await actuator.stop()  # should not raise
        assert actuator.is_running is False

    @pytest.mark.asyncio
    async def test_stop_after_stop_is_safe(self):
        """Calling stop() twice should not raise."""
        actuator = WarmColdActuator(poll_interval=3600)
        await actuator.start()
        await actuator.stop()
        await actuator.stop()  # second stop — no-op
        assert actuator.is_running is False


# =============================================================================
# WarmColdActuator — _resolve_pid
# =============================================================================

class TestResolvePid:
    """_resolve_pid static method."""

    def test_known_service_id(self):
        """Resolve a PID from SERVICE_PID_* env."""
        with patch.dict(os.environ, {"SERVICE_PID_CODE_ORCHESTRATOR": "12345"}):
            pid = WarmColdActuator._resolve_pid("code-orchestrator")
        assert pid == 12345

    def test_unknown_service_id(self):
        """Resolve a service ID with no matching env var -> None."""
        pid = WarmColdActuator._resolve_pid("nonexistent-service")
        assert pid is None

    def test_invalid_pid_str_returns_none(self):
        """When env var holds non-numeric value, return None."""
        with patch.dict(os.environ, {"SERVICE_PID_BOGUS": "not-a-number"}):
            pid = WarmColdActuator._resolve_pid("bogus")
        assert pid is None


# =============================================================================
# WarmColdActuator — _shutdown_service
# =============================================================================

class TestShutdownService:
    """_shutdown_service internal method."""

    @pytest.mark.asyncio
    async def test_unknown_service_skips_kill(self):
        """Shutdown with unresolvable PID should skip kill calls."""
        actuator = WarmColdActuator(drain_seconds=0.01)
        with patch.object(actuator, "_resolve_pid", return_value=None) as mock_resolve:
            with patch("os.kill") as mock_kill:
                await actuator._shutdown_service("unknown-service")
        mock_resolve.assert_called_once_with("unknown-service")
        mock_kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_sigterm_on_known_service(self):
        """Shutdown should send SIGTERM to the resolved PID."""
        actuator = WarmColdActuator(drain_seconds=0.01)
        with patch.object(actuator, "_resolve_pid", return_value=99999):
            with patch("os.kill") as mock_kill:
                await actuator._shutdown_service("test-service")
        # First kill call should be SIGTERM
        mock_kill.assert_any_call(99999, signal.SIGTERM)

    @pytest.mark.asyncio
    async def test_sigterm_sigkill_sequence(self):
        """Verify SIGTERM → drain → SIGKILL sequence."""
        actuator = WarmColdActuator(drain_seconds=0.01)
        calls: list[tuple[int, int]] = []

        def track_kill(pid: int, sig: int) -> None:
            calls.append((pid, sig))

        with patch.object(actuator, "_resolve_pid", return_value=99999):
            with patch("os.kill", side_effect=track_kill):
                await actuator._shutdown_service("test-service")

        assert len(calls) == 2
        assert calls[0] == (99999, signal.SIGTERM)
        assert calls[1] == (99999, signal.SIGKILL)

    @pytest.mark.asyncio
    async def test_process_lookup_error_on_sigterm(self):
        """ProcessLookupError on SIGTERM should be silently handled."""
        actuator = WarmColdActuator(drain_seconds=0.01)
        with patch.object(actuator, "_resolve_pid", return_value=99999):
            with patch("os.kill", side_effect=ProcessLookupError):
                # Should not raise
                await actuator._shutdown_service("test-service")

    @pytest.mark.asyncio
    async def test_process_lookup_error_on_sigkill(self):
        """ProcessLookupError on SIGKILL should be silently handled (graceful exit)."""
        actuator = WarmColdActuator(drain_seconds=0.01)

        def kill_side_effect(pid: int, sig: int) -> None:
            if sig == signal.SIGKILL:
                raise ProcessLookupError
            # SIGTERM succeeds

        with patch.object(actuator, "_resolve_pid", return_value=99999):
            with patch("os.kill", side_effect=kill_side_effect):
                await actuator._shutdown_service("test-service")

    @pytest.mark.asyncio
    async def test_permission_error_skips_sigkill(self):
        """PermissionError should be logged and not crash."""
        actuator = WarmColdActuator(drain_seconds=0.01)

        def kill_side_effect(pid: int, sig: int) -> None:
            if sig == signal.SIGKILL:
                raise PermissionError
            # SIGTERM succeeds

        with patch.object(actuator, "_resolve_pid", return_value=99999):
            with patch("os.kill", side_effect=kill_side_effect):
                await actuator._shutdown_service("test-service")


# =============================================================================
# Singleton
# =============================================================================

class TestSingleton:
    """get_checker() / reset_checker()."""

    def test_get_checker_returns_same_instance(self):
        """Calling get_checker() twice should return the same object."""
        c1 = get_checker()
        c2 = get_checker()
        assert c1 is c2

    def test_reset_checker_creates_new_instance(self):
        """After reset_checker(), get_checker() returns a new object."""
        c1 = get_checker()
        reset_checker()
        c2 = get_checker()
        assert c2 is not c1

    def test_get_checker_returns_warm_cold_actuator(self):
        """get_checker() should return a WarmColdActuator instance."""
        checker = get_checker()
        assert isinstance(checker, WarmColdActuator)


# =============================================================================
# Integration — Wiring with IdleTimeoutTracker
# =============================================================================

class TestIntegration:
    """End-to-end wiring with the IdleTimeoutTracker."""

    @pytest.mark.asyncio
    async def test_run_loop_queries_tracker(self):
        """The _run loop should call get_tracker().get_services_needing_shutdown()."""
        from src.core.idle_timeout import get_tracker, reset_tracker

        reset_tracker()
        tracker = get_tracker()
        actuator = WarmColdActuator(poll_interval=0.05, drain_seconds=0.01)

        # Record a request, then make it look idle by setting the dict value
        tracker.record_request("code-orchestrator")
        import datetime
        tracker._service_states["code-orchestrator"]["last_request"] -= datetime.timedelta(hours=24)

        with patch.object(actuator, "_shutdown_service") as mock_shutdown:
            await actuator.start()
            await asyncio.sleep(0.15)  # Allow a couple of poll cycles
            await actuator.stop()

        mock_shutdown.assert_any_call("code-orchestrator")
