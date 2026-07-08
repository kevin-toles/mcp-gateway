"""
Idle Timeout Service Tracker
=============================

Tracks service last-request timestamps and enforces idle timeouts.

Default timeout: 10 minutes (600 seconds)
Configurable per-service via environment variables.

Usage:
    from src.core.idle_timeout import IdleTimeoutTracker, get_tracker
    
    tracker = get_tracker()
    tracker.record_request("semantic-search")
    
    if tracker.is_idle("semantic-search"):
        print("Service should be shut down")
"""

from __future__ import annotations

import logging
import os
import os as _os  # noqa: PLC0415  -- aliased for _stop_service patching
import signal  # noqa: PLC0415  -- module-level for _stop_service
import subprocess
import threading
import time as _time  # noqa: PLC0415  -- aliased for _stop_service patching
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Optional
from functools import lru_cache

from src.core.keys import normalize_service_key

logger = logging.getLogger(__name__)

# ── HWC-PY-1: Deque-based ColdWarmPromoter ────────────────────────────


class ColdWarmPromoter:
    """Promotes a service from COLD→WARM based on recent dispatch success.

    Tracks a sliding window of successful dispatch timestamps per service
    key using a ``deque[datetime]``.  When ``record_dispatch()`` is called
    and the deque reaches ``COLD_PROMOTION_REQUESTS`` entries within the
    window, the service tier is flipped to ``"WARM"`` in ``SERVICE_TIERS``
    and the deque is cleared.

    This is the spec-backed (success-triggered) promoter, distinct from the
    old ``cold_warm_promoter.py`` implementation which used failure-triggered
    HTTP health checks.
    """

    def __init__(self) -> None:
        self._request_windows: dict[str, deque[datetime]] = {}
        self._lock = threading.Lock()

    def record_dispatch(self, service_key: str, _now: datetime | None = None) -> bool:
        """Record a successful dispatch and check promotion.

        Args:
            service_key: Hyphen-form service key (e.g. ``"semantic-search"``).
            _now:       Optional injected timestamp (for test time injection).
                        Defaults to ``datetime.now(timezone.utc)``.

        Returns:
            ``True`` if the service was promoted from COLD→WARM, ``False``
            if it is already WARM or has not yet met the threshold.
        """
        # ── Early short-circuit: only track COLD services ────────────
        # Lazy imports to avoid circular dependency:
        #   health_config → idle_timeout → health_config
        from src.core.health_config import COLD_PROMOTION_REQUESTS, COLD_PROMOTION_WINDOW_SECS  # noqa: PLC0415
        from src.core.health_config import SERVICE_TIERS  # noqa: PLC0415

        tier = SERVICE_TIERS.get(service_key, "WARM")
        if tier != "COLD":
            return False

        now = _now if _now is not None else datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=COLD_PROMOTION_WINDOW_SECS)

        with self._lock:
            if service_key not in self._request_windows:
                self._request_windows[service_key] = deque()

            win = self._request_windows[service_key]

            # Prune entries outside the sliding window
            while win and win[0] < cutoff:
                win.popleft()

            win.append(now)

            # Check threshold
            if len(win) < COLD_PROMOTION_REQUESTS:
                return False

            # ── Promote ──────────────────────────────────────────────
            SERVICE_TIERS[service_key] = "WARM"

            # Clear the window so only ONE promotion per burst
            win.clear()

            logger.info(
                "cold_to_warm_promotion",
                extra={"service": service_key, "request_count": COLD_PROMOTION_REQUESTS, "threshold": COLD_PROMOTION_REQUESTS},
            )
            return True


# Singleton
_cold_warm_promoter: Optional[ColdWarmPromoter] = None


def record_dispatch_for_promotion(service_key: str, _now: datetime | None = None) -> bool:
    """Record a successful dispatch for COLD→WARM promotion (singleton).

    Convenience function that delegates to
    ``ColdWarmPromoter.record_dispatch()`` via the global singleton.

    Args:
        service_key: Hyphen-form service key.
        _now:        Optional injected timestamp (for test time injection).

    Returns:
        ``True`` if the service was promoted.
    """
    global _cold_warm_promoter
    if _cold_warm_promoter is None:
        _cold_warm_promoter = ColdWarmPromoter()
    return _cold_warm_promoter.record_dispatch(service_key, _now=_now)


def reset_cold_warm_promoter() -> None:
    """Reset the global ColdWarmPromoter singleton (useful for testing)."""
    global _cold_warm_promoter
    _cold_warm_promoter = None


# ── HWC-PY-2: Warm→Cold demotion constants ────────────────────────────

# Imported from src.core.health_config:
#   WARM_IDLE_TIMEOUT_SECS, SIGTERM_DRAIN_SECS


class WarmColdActuator:
    """Demotes a WARM service back to COLD when it has been idle too long.

    Sends SIGTERM → waits ``SIGTERM_DRAIN_SECS`` → SIGKILL if still alive.
    Flips ``SERVICE_TIERS[service_key]`` to ``"COLD"``.
    """

    def check_and_demote(self, service_key: str, idle_secs: float) -> bool:
        """Demote *service_key* from WARM to COLD if idle threshold exceeded.

        Only acts on WARM-tier services — HOT and COLD services are skipped.

        Args:
            service_key: Hyphen-form service key.
            idle_secs: Seconds since last request.

        Returns:
            ``True`` if the service was demoted, ``False`` if not eligible.
        """
        # ── Only demote WARM services ────────────────────────────────
        from src.core.health_config import SERVICE_TIERS, WARM_IDLE_TIMEOUT_SECS  # noqa: PLC0415

        tier = SERVICE_TIERS.get(service_key)
        if tier != "WARM":
            return False

        if idle_secs < WARM_IDLE_TIMEOUT_SECS:
            return False

        logger.info(
            "warm_to_cold_demotion_start",
            extra={"service": service_key, "idle_secs": idle_secs},
        )
        self._stop_service(service_key)

        # Update tier registry (belt-and-suspenders with _stop_service)
        SERVICE_TIERS[service_key] = "COLD"

        logger.info(
            "warm_to_cold_demotion_complete",
            extra={"service": service_key},
        )

        return True

    def _stop_service(self, service_key: str) -> None:
        """Send SIGTERM → drain → SIGKILL to the process.

        Resolves the PID from the ``SERVICE_PID_<NAME>`` environment
        variable (upper-cased, hyphens→underscores), OR uses
        ``SERVICE_SHUTDOWN_COMMANDS`` from config if registered.
        """
        # ── Try SERVICE_SHUTDOWN_COMMANDS first (covers inference-cpp et al.) ──
        from src.core.config import settings  # noqa: PLC0415
        from src.core.health_config import SIGTERM_DRAIN_SECS  # noqa: PLC0415

        shutdown_cmd = settings.SERVICE_SHUTDOWN_COMMANDS.get(service_key)
        if shutdown_cmd:
            try:
                subprocess.run(
                    shutdown_cmd,
                    shell=True,
                    timeout=SIGTERM_DRAIN_SECS + 5,
                    capture_output=True,
                    text=True,
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "service_shutdown_command_timeout",
                    extra={"service": service_key, "command": shutdown_cmd},
                )
            except Exception:
                logger.exception(
                    "service_shutdown_command_failed",
                    extra={"service": service_key, "command": shutdown_cmd},
                )

            # Mark COLD
            from src.core.health_config import SERVICE_TIERS  # noqa: PLC0415
            SERVICE_TIERS[service_key] = "COLD"

            logger.info(
                "warm_to_cold_demotion_complete",
                extra={"service": service_key},
            )
            return

        # ── Fallback: PID-based kill ────────────────────────────────────
        env_key = f"SERVICE_PID_{service_key.upper().replace('-', '_')}"
        pid_str = _os.environ.get(env_key)
        if not pid_str:
            from src.core.health_config import SERVICE_TIERS  # noqa: PLC0415
            SERVICE_TIERS[service_key] = "COLD"
            logger.info(
                "warm_to_cold_demotion_complete",
                extra={"service": service_key, "method": "noop_no_pid_no_command"},
            )
            return

        try:
            pid = int(pid_str)
        except (ValueError, TypeError):
            from src.core.health_config import SERVICE_TIERS  # noqa: PLC0415
            SERVICE_TIERS[service_key] = "COLD"
            logger.info(
                "warm_to_cold_demotion_complete",
                extra={"service": service_key, "method": "noop_invalid_pid"},
            )
            return

        # SIGTERM
        try:
            _os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # Already dead
        except PermissionError:
            return

        # Drain wait
        from src.core.health_config import SIGTERM_DRAIN_SECS  # noqa: PLC0415
        _time.sleep(SIGTERM_DRAIN_SECS)

        # SIGKILL if still alive
        try:
            _os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

        # Mark COLD
        from src.core.health_config import SERVICE_TIERS  # noqa: PLC0415
        SERVICE_TIERS[service_key] = "COLD"

        logger.info(
            "warm_to_cold_demotion_complete",
            extra={"service": service_key, "method": "pid_kill"},
        )


# Singleton
_warm_cold_actuator: Optional[WarmColdActuator] = None


def get_warm_cold_actuator() -> WarmColdActuator:
    """Get the global WarmColdActuator singleton."""
    global _warm_cold_actuator
    if _warm_cold_actuator is None:
        _warm_cold_actuator = WarmColdActuator()
    return _warm_cold_actuator


def reset_warm_cold_actuator() -> None:
    """Reset the global WarmColdActuator singleton (useful for testing)."""
    global _warm_cold_actuator
    _warm_cold_actuator = None


def on_service_idle(service_key: str, idle_secs: float) -> None:
    """Callback fired when a service is detected as idle.

    Delegates to ``WarmColdActuator.check_and_demote()``.
    """
    get_warm_cold_actuator().check_and_demote(service_key, idle_secs)


class IdleTimeoutTracker:
    """Tracks service idle timeouts and enforces shutdown."""
    
    # Service-specific timeout overrides (from env vars)
    SERVICE_TIMEOUTS = {
        "unified-search-service": "UNIFIED_SEARCH_IDLE_TIMEOUT",
        "unified-search-rs": "UNIFIED_SEARCH_RS_IDLE_TIMEOUT",
        "inference-service-cpp": "INFERENCE_SERVICE_IDLE_TIMEOUT",
        "llm-gateway": "LLM_GATEWAY_IDLE_TIMEOUT",
        "code-orchestrator": "CODE_ORCHESTRATOR_IDLE_TIMEOUT",
        "ai-agents": "AI_AGENTS_IDLE_TIMEOUT",
        "audit-service": "AUDIT_SERVICE_IDLE_TIMEOUT",
        "context-management-service": "CMS_IDLE_TIMEOUT",
        "amve": "AMVE_IDLE_TIMEOUT",
        "struct-analyzer-service": "STRUCT_ANALYZER_IDLE_TIMEOUT",
    }
    
    def __init__(self, default_timeout_seconds: int | None = None):
        """
        Initialize tracker.
        
        Args:
            default_timeout_seconds: Default idle timeout (default: from env or 600s)
        """
        if default_timeout_seconds is None:
            default_timeout_seconds = self._load_default_timeout()
        
        self.default_timeout = default_timeout_seconds
        self._service_states: dict[str, dict] = {}
        self._custom_timeouts: dict[str, int] = {}
        self._on_idle_callbacks: list[Callable[[str, float], None]] = []
        
        # Load custom timeouts from environment
        self._load_custom_timeouts()
    
    def _load_default_timeout(self) -> int:
        """Load default timeout from environment or use 600s (10 min)."""
        env_value = os.getenv("MCP_GATEWAY_DEFAULT_IDLE_TIMEOUT")
        if env_value:
            try:
                return int(env_value)
            except ValueError:
                pass
        return 600  # 10 minutes default
    
    def _load_custom_timeouts(self) -> None:
        """Load custom timeouts from environment variables."""
        for service_id, env_var in self.SERVICE_TIMEOUTS.items():
            env_value = os.getenv(env_var)
            if env_value:
                try:
                    timeout = int(env_value)
                    self._custom_timeouts[service_id] = timeout
                except ValueError:
                    pass
    
    def record_request(self, service_id: str) -> None:
        """
        Record that a service received a request.
        
        Args:
            service_id: Service identifier (e.g., "semantic-search")
        """
        service_id = normalize_service_key(service_id)
        self._service_states[service_id] = {
            "last_request": datetime.now(timezone.utc),
            "total_requests": self._service_states.get(service_id, {}).get("total_requests", 0) + 1,
        }
    
    def register_idle_callback(self, callback: Callable[[str, float], None]) -> None:
        """Register a callback fired when a service is detected as idle.

        Args:
            callback: A callable accepting ``(service_key, idle_seconds)``.
        """
        self._on_idle_callbacks.append(callback)

    def _fire_idle_callbacks(self, service_key: str, idle_secs: float) -> None:
        """Fire all registered idle callbacks, catching individual failures."""
        for cb in self._on_idle_callbacks:
            try:
                cb(service_key, idle_secs)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Idle callback failed for %s", service_key
                )

    def get_idle_time(self, service_id: str) -> Optional[float]:
        """
        Get seconds since last request for a service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Seconds since last request, or None if service never seen
        """
        service_id = normalize_service_key(service_id)
        if service_id not in self._service_states:
            return None
        
        last_request = self._service_states[service_id]["last_request"]
        now = datetime.now(timezone.utc)
        delta = now - last_request
        return delta.total_seconds()
    
    def is_idle(self, service_id: str) -> bool:
        """
        Check if service has exceeded idle timeout.

        Args:
            service_id: Service identifier

        Returns:
            True if service has exceeded its idle timeout
        """
        idle_time = self.get_idle_time(service_id)
        if idle_time is None:
            return False

        timeout = self.get_timeout_config(service_id)
        return idle_time >= timeout
    
    def get_timeout_config(self, service_id: str) -> int:
        """
        Get timeout configuration for a specific service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Timeout in seconds (custom or default)
        """
        service_id = normalize_service_key(service_id)
        return self._custom_timeouts.get(service_id, self.default_timeout)
    
    def set_custom_timeout(self, service_id: str, timeout_seconds: int) -> None:
        """
        Set custom timeout for a service (runtime override).
        
        Args:
            service_id: Service identifier
            timeout_seconds: Custom timeout in seconds
        """
        service_id = normalize_service_key(service_id)
        self._custom_timeouts[service_id] = timeout_seconds
    
    def get_services_needing_shutdown(self) -> list[str]:
        """
        Get list of services that have exceeded idle timeout.

        Returns:
            List of service IDs that should be shut down
        """
        return [
            service_id
            for service_id in self._service_states
            if self.is_idle(service_id)
        ]
    
    def get_status(self, service_id: str) -> dict:
        """
        Get status information for a service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Dict with last_request, idle_seconds, timeout, is_idle, total_requests
        """
        service_id = normalize_service_key(service_id)
        if service_id not in self._service_states:
            return {
                "last_request": None,
                "idle_seconds": None,
                "timeout": self.get_timeout_config(service_id),
                "is_idle": False,
                "total_requests": 0,
            }
        
        idle_seconds = self.get_idle_time(service_id) or 0
        timeout = self.get_timeout_config(service_id)
        state = self._service_states[service_id]
        
        return {
            "last_request": state["last_request"].isoformat(),
            "idle_seconds": idle_seconds,
            "timeout": timeout,
            "is_idle": idle_seconds >= timeout,
            "total_requests": state.get("total_requests", 1),
        }
    
    def get_all_statuses(self) -> dict[str, dict]:
        """
        Get status for all tracked services.
        
        Returns:
            Dict mapping service_id to status dict
        """
        return {
            service_id: self.get_status(service_id)
            for service_id in self._service_states
        }
    
    def cleanup_expired(self) -> list[str]:
        """
        Remove entries for services that have been idle for 2x timeout.
        
        Returns:
            List of removed service IDs
        """
        to_remove = []
        
        for service_id in self._service_states:
            idle_seconds = self.get_idle_time(service_id) or 0
            timeout = self.get_timeout_config(service_id)
            
            if idle_seconds >= timeout * 2:
                to_remove.append(service_id)
        
        for service_id in to_remove:
            del self._service_states[service_id]
        
        return to_remove


# =============================================================================
# Singleton Pattern
# =============================================================================

_tracker_instance: Optional[IdleTimeoutTracker] = None


def get_tracker() -> IdleTimeoutTracker:
    """
    Get the global IdleTimeoutTracker instance (singleton).
    
    Returns:
        Global tracker instance
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = IdleTimeoutTracker()
    return _tracker_instance


def reset_tracker() -> None:
    """
    Reset the global tracker instance (useful for testing).
    """
    global _tracker_instance
    _tracker_instance = None
