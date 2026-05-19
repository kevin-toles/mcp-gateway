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

from src.core.idle_timeout import get_tracker

logger = logging.getLogger(__name__)


class IdleTimeoutChecker:
    """
    Background checker that monitors service idle times and shuts down idle services.
    """
    
    # Service shutdown commands
    SERVICE_SHUTDOWN_COMMANDS = {
        "unified-search-service": "pkill -f 'uvicorn.*8081'",
        "unified-search-rs": "pkill -f 'uvicorn.*8089'",
        "code-orchestrator": "pkill -f 'uvicorn.*8083'",
        "llm-gateway": "pkill -f 'uvicorn.*8080'",
        "ai-agents": "pkill -f 'uvicorn.*8082'",
        "audit-service": "pkill -f 'uvicorn.*8084'",
        "context-management-service": "pkill -f 'uvicorn.*8086'",
        "amve": "pkill -f 'uvicorn.*8088'",
    }
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize checker.
        
        Args:
            check_interval: How often to check for idle services (seconds)
        """
        self.check_interval = check_interval
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the background checker."""
        if self._task is not None:
            logger.warning("Idle timeout checker already running")
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
            pass
        
        self._task = None
        logger.info("Idle timeout checker stopped")
    
    async def _check_loop(self) -> None:
        """Main check loop."""
        while self._running:
            try:
                await self._check_and_shutdown_idle()
            except Exception as e:
                logger.error("Error in idle timeout check: %s", e)
            
            await asyncio.sleep(self.check_interval)
    
    async def _check_and_shutdown_idle(self) -> None:
        """Check for idle services and shut them down."""
        tracker = get_tracker()
        idle_services = tracker.get_services_needing_shutdown()
        
        if not idle_services:
            return
        
        for service_id in idle_services:
            await self._shutdown_service(service_id)
    
    async def _shutdown_service(self, service_id: str) -> None:
        """
        Shut down a service gracefully.
        
        Args:
            service_id: Service identifier
        """
        logger.info("Shutting down idle service: %s", service_id)
        
        # Get shutdown command
        command = self.SERVICE_SHUTDOWN_COMMANDS.get(service_id)
        if command is None:
            logger.warning("No shutdown command for service: %s", service_id)
            return
        
        # Execute shutdown
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("Successfully shut down %s", service_id)
            else:
                logger.error(
                    "Failed to shut down %s: %s",
                    service_id,
                    stderr.decode() if stderr else "unknown error",
                )
        except Exception as e:
            logger.error("Error shutting down %s: %s", service_id, e)


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
