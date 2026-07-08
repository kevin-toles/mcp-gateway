"""
Warm→Cold Actuator Tests — HWC-PY-2
=====================================

RED tests for the ``WarmColdActuator`` in ``src.core.idle_timeout``.

Tests assert WARM→COLD demotion via idle timeout thresholds,
covering SIGTERM/SIGKILL signalling, tier registry updates,
HOT-skip logic, inference-cpp binary shutdown, and structured
logging on transition.

Spec source: TDD_AMENDMENT_PLAN_CONSOLIDATED (3).md §HWC-PY-2
"""

from __future__ import annotations

import logging
import os
import signal
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.core.idle_timeout import (
    get_warm_cold_actuator,
    WarmColdActuator,
    reset_warm_cold_actuator,
    on_service_idle,
)
from src.core.health_config import (
    SERVICE_TIERS,
    WARM_IDLE_TIMEOUT_SECS,
    SIGTERM_DRAIN_SECS,
    reset_service_tiers,
)


# ── Helpers ─────────────────────────────────────────────────────────────

_FROZEN = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _dt(offset_minutes: int = 0) -> datetime:
    """Return *offset_minutes* past the frozen base time."""
    return _FROZEN.replace(tzinfo=timezone.utc) + __import__(
        "datetime"
    ).timedelta(minutes=offset_minutes)


def _idle_secs(offset_minutes: int) -> float:
    """Return the idle seconds *offset_minutes* past the frozen base."""
    return offset_minutes * 60.0


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_tiers() -> None:
    """Reset every test to a clean tier slate."""
    reset_service_tiers()


@pytest.fixture(autouse=True)
def _reset_actuator() -> None:
    """Reset the global actuator singleton."""
    reset_warm_cold_actuator()


@pytest.fixture
def actuator() -> WarmColdActuator:
    """Fresh actuator per test — never share the global singleton."""
    return WarmColdActuator()


# ── Tests ───────────────────────────────────────────────────────────────


class TestWarmToColdDemotion:
    """HWC-PY-2: WARM→COLD demotion via idle timeout."""

    def test_warm_to_cold_fires_on_idle_timeout(
        self, actuator: WarmColdActuator
    ) -> None:
        """Service in WARM tier idle beyond threshold → demotion fires.

        Given a WARM service that has been idle for > WARM_IDLE_TIMEOUT_SECS,
        ``check_and_demote()`` returns ``True`` (demotion occurred).
        """
        service = "semantic-search"
        SERVICE_TIERS[service] = "WARM"

        # Idle well beyond the threshold
        idle = WARM_IDLE_TIMEOUT_SECS + 60.0

        # Mock _stop_service so we don't actually kill anything
        with patch.object(actuator, "_stop_service") as mock_stop:
            result = actuator.check_and_demote(service, idle)

        assert result is True, "Demotion should fire when WARM + idle beyond threshold"
        mock_stop.assert_called_once_with(service)

    def test_warm_to_cold_updates_tier_registry(
        self, actuator: WarmColdActuator
    ) -> None:
        """Demotion fires → SERVICE_TIERS updated to ``"COLD"``."""
        service = "code-orchestrator"
        SERVICE_TIERS[service] = "WARM"
        idle = WARM_IDLE_TIMEOUT_SECS + 60.0

        with patch.object(actuator, "_stop_service"):
            actuator.check_and_demote(service, idle)

        assert SERVICE_TIERS[service] == "COLD", (
            "SERVICE_TIERS must be updated to COLD after demotion"
        )

    def test_warm_to_cold_sends_sigterm_to_native(
        self, actuator: WarmColdActuator
    ) -> None:
        """Native service (PID-based) → SIGTERM sent to the correct PID.

        When a service has a ``SERVICE_PID_<NAME>`` env var set,
        ``_stop_service()`` sends SIGTERM to that PID.
        """
        service = "semantic-search"
        SERVICE_TIERS[service] = "WARM"
        idle = WARM_IDLE_TIMEOUT_SECS + 60.0

        fake_pid = 12345
        env_key = f"SERVICE_PID_{service.upper().replace('-', '_')}"

        with (
            patch.dict(os.environ, {env_key: str(fake_pid)}),
            patch("src.core.idle_timeout._os.kill") as mock_kill,
            patch("src.core.idle_timeout._time.sleep"),
        ):
            actuator.check_and_demote(service, idle)

        # Should have sent SIGTERM first
        sigterm_call = mock_kill.call_args_list[0]
        assert sigterm_call == ((fake_pid, signal.SIGTERM),), (
            f"Expected SIGTERM to PID {fake_pid}, got {sigterm_call}"
        )

    def test_warm_to_cold_sigkill_fallback(
        self, actuator: WarmColdActuator
    ) -> None:
        """Process still alive after SIGTERM drain → SIGKILL sent.

        After ``SIGTERM_DRAIN_SECS`` seconds, if the process is still
        alive, ``_stop_service()`` sends SIGKILL.
        """
        service = "semantic-search"
        SERVICE_TIERS[service] = "WARM"
        idle = WARM_IDLE_TIMEOUT_SECS + 60.0

        fake_pid = 12345
        env_key = f"SERVICE_PID_{service.upper().replace('-', '_')}"

        with (
            patch.dict(os.environ, {env_key: str(fake_pid)}),
            patch("src.core.idle_timeout._os.kill") as mock_kill,
            patch("src.core.idle_timeout._time.sleep"),
        ):
            # First call triggers SIGTERM; after sleep, process still alive so SIGKILL sent
            actuator.check_and_demote(service, idle)

        # Should have called kill twice: SIGTERM then SIGKILL
        assert len(mock_kill.call_args_list) == 2, (
            f"Expected 2 kill calls (SIGTERM + SIGKILL), got {len(mock_kill.call_args_list)}"
        )

        sigterm_call = mock_kill.call_args_list[0]
        sigkill_call = mock_kill.call_args_list[1]

        assert sigterm_call == ((fake_pid, signal.SIGTERM),), (
            f"First call should be SIGTERM to PID {fake_pid}"
        )
        assert sigkill_call == ((fake_pid, signal.SIGKILL),), (
            f"Second call should be SIGKILL to PID {fake_pid}"
        )

    def test_warm_to_cold_skips_hot_services(
        self, actuator: WarmColdActuator
    ) -> None:
        """HOT-tier service idle beyond threshold → skipped (returns False).

        Only WARM services are eligible for demotion. HOT services are
        production-live and must never be stopped by the actuator.
        """
        service = "llm-gateway"
        SERVICE_TIERS[service] = "HOT"
        idle = WARM_IDLE_TIMEOUT_SECS + 60.0

        with patch.object(actuator, "_stop_service") as mock_stop:
            result = actuator.check_and_demote(service, idle)

        assert result is False, "HOT services must never be demoted"
        mock_stop.assert_not_called()

    def test_warm_to_cold_inference_cpp_binary(
        self, actuator: WarmColdActuator
    ) -> None:
        """inference-service-cpp demotion uses binary shutdown command.

        When a service is registered in SERVICE_SHUTDOWN_COMMANDS,
        the shutdown command is used instead of PID-based kill.
        """
        service = "inference-service-cpp"
        SERVICE_TIERS[service] = "WARM"
        idle = WARM_IDLE_TIMEOUT_SECS + 60.0

        with (
            patch("src.core.idle_timeout.subprocess.run") as mock_run,
        ):
            actuator.check_and_demote(service, idle)

        # Should have called subprocess.run with the shutdown command
        assert mock_run.called, (
            "Expected subprocess.run for inference-cpp shutdown command"
        )
        call_args, call_kwargs = mock_run.call_args
        assert call_args[0] is not None, "subprocess.run must receive a command"
        assert "pkill -f inference-service" in call_args[0], (
            f"Expected pkill command for inference-cpp, got: {call_args[0]}"
        )
        assert call_kwargs.get("shell") is True, (
            "Shutdown command must run with shell=True"
        )

    def test_warm_to_cold_logs_transition(
        self, actuator: WarmColdActuator, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Demotion fires → structured log entry with service name and idle duration."""
        service = "struct-analyzer-service"
        SERVICE_TIERS[service] = "WARM"
        idle = WARM_IDLE_TIMEOUT_SECS + 60.0

        caplog.set_level(logging.INFO)

        with patch.object(actuator, "_stop_service"):
            actuator.check_and_demote(service, idle)

        # Check for the warm_to_cold_demotion_start log message
        matching = [
            r for r in caplog.records
            if r.msg == "warm_to_cold_demotion_start"
        ]
        assert len(matching) == 1, (
            f"Expected exactly 1 'warm_to_cold_demotion_start' log, "
            f"got {len(matching)}"
        )
        record = matching[0]
        assert record.service == service, (
            f"Expected service={service}, got {record.service}"
        )
        # The idle_secs extra field should be present
        assert hasattr(record, "idle_secs"), (
            "Log record must include idle_secs extra field"
        )
        # And also the warm_to_cold_demotion_complete log
        complete_matching = [
            r for r in caplog.records
            if r.msg == "warm_to_cold_demotion_complete"
        ]
        assert len(complete_matching) >= 1, (
            "Expected at least 1 'warm_to_cold_demotion_complete' log"
        )
