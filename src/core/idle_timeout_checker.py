"""Idle Timeout Checker — WBS-MCP1.4 (GREEN), Warm/Cold Lifecycle.

Periodically checks ``IdleTimeoutTracker.get_services_needing_shutdown()``
and issues SIGTERM (graceful) → 30 s drain → SIGKILL (force) for any
service that has exceeded its idle timeout.

The checker is wired into the FastAPI lifespan in ``main.py``:
``await checker.start()`` on startup, ``await checker.stop()`` on shutdown.

Reference: AC-7.1 (Warm/Cold lifecycle), Strategy §6.2 (idle timeout),
           config/hot_warm_cold.yaml
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Optional

from src.core.idle_timeout import (
    ColdWarmPromoter,
    get_tracker,
    record_dispatch_for_promotion,
)

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────

_POLL_INTERVAL = float(os.getenv("MCP_GATEWAY_IDLE_POLL_INTERVAL", "60"))
"""Seconds between idle-timeout checks (default 60)."""

_DRAIN_SECONDS = float(os.getenv("MCP_GATEWAY_IDLE_DRAIN_SECONDS", "30"))
"""Seconds to wait between SIGTERM and SIGKILL (default 30)."""


# =============================================================================
# WarmColdActuator
# =============================================================================

class WarmColdActuator:
    """Asynchronously monitors and shuts down idle backend services.

    The actuator polls ``IdleTimeoutTracker.get_services_needing_shutdown()``
    on a fixed interval.  For each idle service it:
      1. Sends **SIGTERM** (allows graceful teardown).
      2. Waits the configured drain window (default 30 s).
      3. Sends **SIGKILL** if the process is still alive.

    Attributes:
        poll_interval: Seconds between idle-timeout checks.
        drain_seconds: Seconds between SIGTERM and SIGKILL.
    """

    def __init__(
        self,
        poll_interval: float = _POLL_INTERVAL,
        drain_seconds: float = _DRAIN_SECONDS,
    ) -> None:
        self.poll_interval = poll_interval
        self.drain_seconds = drain_seconds
        self._task: Optional[asyncio.Task[None]] = None
        self._stopped = asyncio.Event()

    # ── Public API ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background polling loop.

        Called from the FastAPI lifespan startup.  Idempotent — safe to
        call multiple times.
        """
        if self._task is not None and not self._task.done():
            logger.warning("WarmColdActuator already running")
            return
        self._stopped.clear()
        self._task = asyncio.create_task(self._run())
        logger.info(
            "WarmColdActuator started (poll=%ss, drain=%ss)",
            self.poll_interval,
            self.drain_seconds,
        )

    async def stop(self) -> None:
        """Stop the background polling loop.

        Called from the FastAPI lifespan shutdown.  Idempotent — safe to
        call even if the actuator was never started.
        """
        if self._task is None or self._task.done():
            return
        self._stopped.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("WarmColdActuator stopped")

    @property
    def is_running(self) -> bool:
        """True while the polling loop is active."""
        return self._task is not None and not self._task.done()

    # ── Internal ─────────────────────────────────────────────────────

    async def _run(self) -> None:
        """Background loop: poll tracker → shutdown idle services."""
        while not self._stopped.is_set():
            try:
                idle_services = get_tracker().get_services_needing_shutdown()
                for service_id in idle_services:
                    await self._shutdown_service(service_id)
            except Exception:
                logger.exception("Error in idle timeout checker loop")
            await asyncio.sleep(self.poll_interval)

    async def _shutdown_service(self, service_id: str) -> None:
        """Issue SIGTERM → drain → SIGKILL for a single service.

        Checks ``SERVICE_SHUTDOWN_COMMANDS`` from config first
        (covers binaries like ``inference-service-cpp``), then falls
        back to PID-based kill via the ``SERVICE_PID_<NAME>`` env var.

        After the service is killed (or already gone), updates the
        service tier to ``"COLD"`` so the idle checker stops reporting it.

        Args:
            service_id: The service identifier (e.g. ``"code-orchestrator"``).
        """
        # ── Try SERVICE_SHUTDOWN_COMMANDS first ──────────────────────────
        from src.core.config import settings  # noqa: PLC0415

        shutdown_cmd = settings.SERVICE_SHUTDOWN_COMMANDS.get(service_id)
        if shutdown_cmd:
            try:
                proc = await asyncio.create_subprocess_shell(
                    shutdown_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(
                    proc.wait(), timeout=self.drain_seconds + 5
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "service_shutdown_command_timeout",
                    extra={"service": service_id, "command": shutdown_cmd},
                )
            except Exception:
                logger.exception(
                    "service_shutdown_command_failed",
                    extra={"service": service_id, "command": shutdown_cmd},
                )
            self._mark_cold(service_id)
            return

        # ── Fallback: PID-based kill ────────────────────────────────────
        pid = self._resolve_pid(service_id)
        if pid is None:
            logger.debug("No PID found for service %s — marking COLD", service_id)
            self._mark_cold(service_id)
            return

        logger.info("Shutting down idle service %s (PID %d)", service_id, pid)

        # Phase 1: SIGTERM (graceful)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            logger.debug("PID %d already gone (service %s)", pid, service_id)
            self._mark_cold(service_id)
            return
        except PermissionError:
            logger.warning("No permission to kill PID %d (service %s)", pid, service_id)
            return

        # Phase 2: Drain window
        await asyncio.sleep(self.drain_seconds)

        # Phase 3: SIGKILL if still alive
        try:
            os.kill(pid, signal.SIGKILL)
            logger.info("Force-killed idle service %s (PID %d)", service_id, pid)
        except ProcessLookupError:
            logger.debug(
                "PID %d exited gracefully during drain (service %s)",
                pid,
                service_id,
            )
        except PermissionError:
            logger.warning(
                "No permission to SIGKILL PID %d (service %s) — may still be running",
                pid,
                service_id,
            )

        self._mark_cold(service_id)

    @staticmethod
    def _mark_cold(service_id: str) -> None:
        """Set the service tier to COLD after shutdown."""
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS[service_id] = "COLD"
        logger.debug("Service %s marked cold after idle shutdown", service_id)

    @staticmethod
    def _resolve_pid(service_id: str) -> Optional[int]:
        """Resolve a service identifier to a PID.

        Uses the ``SERVICE_PID_<NAME>`` environment variable convention
        (e.g. ``SERVICE_PID_CODE_ORCHESTRATOR=12345``).  Returns ``None``
        when no PID is configured.

        Args:
            service_id: Hyphenated service identifier (e.g. ``"code-orchestrator"``).

        Returns:
            The PID, or ``None`` if not found.
        """
        env_key = f"SERVICE_PID_{service_id.upper().replace('-', '_')}"
        pid_str = os.getenv(env_key)
        if pid_str is None:
            return None
        try:
            return int(pid_str)
        except ValueError:
            logger.warning("Invalid PID in %s: %r", env_key, pid_str)
            return None


# =============================================================================
# Idle-callback registration for RED test compliance
# =============================================================================

_IDLE_CHECKER_CALLBACK_REGISTERED: bool = False
"""Flag to ensure the idle callback is registered exactly once."""


def start_idle_timeout_checker() -> None:
    """Register the idle callback on ``IdleTimeoutTracker``.

    This is the startup entry point expected by GAP-1 RED tests.  It
    registers ``on_service_idle`` (from ``idle_timeout.py``) as a callback
    so that idle→WARM re-promotion triggers the promoter.

    Idempotent — the callback is only registered on the first call.
    """
    global _IDLE_CHECKER_CALLBACK_REGISTERED
    if _IDLE_CHECKER_CALLBACK_REGISTERED:
        return

    from src.core.idle_timeout import on_service_idle

    tracker = get_tracker()
    tracker.register_idle_callback(on_service_idle)

    # AC-4: Expose the WarmColdActuator as tracker._actuator for
    # acceptance-criteria compliance.  The actuator's on_service_idle
    # method is already registered via register_idle_callback above;
    # this attribute gives direct access for introspection and testing.
    from src.core.idle_timeout import get_warm_cold_actuator
    tracker._actuator = get_warm_cold_actuator()

    _IDLE_CHECKER_CALLBACK_REGISTERED = True
    logger.debug("Idle timeout checker callback registered")


# =============================================================================
# Singleton Pattern
# =============================================================================

_checker_instance: Optional[WarmColdActuator] = None


def get_checker() -> WarmColdActuator:
    """Get the global ``WarmColdActuator`` instance (singleton).

    Returns:
        Global checker instance.
    """
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = WarmColdActuator()
    return _checker_instance


def reset_checker() -> None:
    """Reset the global checker instance (useful for testing).

    If the current instance is running, ``reset_checker()`` does **not**
    stop it — call ``await instance.stop()`` first if needed.
    """
    global _checker_instance
    _checker_instance = None
