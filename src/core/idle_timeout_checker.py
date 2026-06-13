"""
Idle Timeout Checker
====================

Background task that periodically checks for idle services and shuts them down.

Runs every 60 seconds (configurable) and:
1. Checks all tracked services for idle timeout
2. Shuts down idle services gracefully
3. Logs shutdown events

Usage:
    from src.core.idle_timeout_checker import start_idle_timeout_checker
    
    # In FastAPI lifespan
    async with lifespan(app):
        await start_idle_timeout_checker()
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Optional

from src.core.config import ServiceKey, Settings
from src.core.idle_timeout import get_tracker

logger = logging.getLogger(__name__)


def _get_settings() -> Settings:
    """Lazy-access Settings to avoid circular imports at module level."""
    return Settings()  # pydantic-settings loads env vars; no cached singleton needed


def _build_shutdown_commands() -> dict[str, str]:
    """Build flat command dict from Settings structured SERVICE_SHUTDOWN_COMMANDS.
    
    Composes ``command`` + grace-period sleep + ``health_check`` into a single
    shell string for backward compatibility with ``_shutdown_service()``.
    """
    settings = _get_settings()
    cmds: dict[str, str] = {}
    for service_id, cfg in settings.SERVICE_SHUTDOWN_COMMANDS.items():
        cmd = cfg["command"]
        grace = cfg.get("grace_period_seconds", 30)
        health = cfg.get("health_check", "")
        if health:
            cmds[service_id] = f"{cmd}; sleep {grace}; {health}"
        else:
            cmds[service_id] = cmd
    return cmds


class IdleTimeoutChecker:
    """
    Background checker that monitors service idle times and shuts down idle services.
    """

    def __init__(self, check_interval: int = 60):
        """
        Initialize checker.
        
        Args:
            check_interval: How often to check for idle services (seconds)
        """
        self.check_interval = check_interval
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    def start(self) -> None:
        """Start the background checker."""
        if self._task is not None:
            logger.warning("Idle timeout checker already running")
            return

        # Honour the central enabled flag
        settings = _get_settings()
        if not settings.IDLE_TIMEOUT_ENABLED:
            logger.info("Idle timeout checker disabled via IDLE_TIMEOUT_ENABLED=False")
            return

        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info("Idle timeout checker started (interval: %ds)", self.check_interval)
    
    async def stop(self) -> None:
        """Stop the background checker."""
        if self._task is None:
            return
        
        self._running = False
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            logger.debug("Idle timeout checker task cancelled")
            raise
        
        self._task = None
        logger.info("Idle timeout checker stopped")
    
    async def _check_loop(self) -> None:
        """Main check loop."""
        while self._running:
            try:
                await self._check_and_shutdown_idle()
            except Exception:
                logger.exception("Error in idle timeout check")
            
            await asyncio.sleep(self.check_interval)
    
    async def _check_and_shutdown_idle(self) -> None:
        """Check for idle services and shut them down."""
        tracker = get_tracker()
        idle_services = tracker.get_services_needing_shutdown()
        
        if not idle_services:
            return
        
        commands = _build_shutdown_commands()
        for service_id in idle_services:
            await self._shutdown_service(service_id, commands)
    
    async def _shutdown_service(self, service_id: str, commands: dict[str, str]) -> None:
        """
        Shut down a service gracefully.
        
        Args:
            service_id: Service identifier
            commands: Flat command dict built from Settings
        """
        logger.info("Shutting down idle service: %s", service_id)

        # Normalize to canonical key (hyphen form) for shutdown lookup
        svc = str(ServiceKey(service_id))
        command = commands.get(svc)
        if command is None:
            logger.warning("No shutdown command for service: %s", service_id)
            return
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate(timeout=30)
            
            if process.returncode == 0:
                logger.info("Successfully shut down %s", service_id)
            else:
                logger.warning(
                    "Shutdown process for %s returned %d: %s",
                    service_id,
                    process.returncode,
                    stderr.decode() if stderr else "",
                )
        except Exception:
            logger.exception("Error shutting down %s", service_id)


# =============================================================================
# Global Checker Instance
# =============================================================================

_checker_instance: Optional[IdleTimeoutChecker] = None


def get_checker() -> IdleTimeoutChecker:
    """
    Get the global IdleTimeoutChecker instance (singleton).
    
    Returns:
        Global checker instance
    """
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = IdleTimeoutChecker()
    return _checker_instance


def reset_checker() -> None:
    """
    Reset the global checker instance (useful for testing).
    """
    global _checker_instance
    _checker_instance = None
