"""
Health-Aware Proxy Middleware
==============================

Intercepts tool calls, checks service health, and auto-restarts dead services.

Implements the agentic proxy pattern:
- Detects ConnectionRefused on tool calls
- Silently restarts service (if not already restarting)
- Polls health until ready
- Retries original request
- User sees zero errors, imperceptible delay

Usage:
    from src.middleware.health_proxy import HealthAwareProxy
    
    proxy = HealthAwareProxy()
    result = await proxy.call_tool("semantic_search", {"query": "..."})
"""

from __future__ import annotations

import asyncio
import copy
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from src.core.idle_timeout import get_tracker
from src.core.config import normalize_service_key
from src.config.health_config import auto_warm_service, health_timeout_for
from src.core.config import SLA_TIMEOUT, Settings

logger = logging.getLogger(__name__)


class ServiceRestartError(Exception):
    """Failed to restart service."""
    pass


class HealthAwareProxy:
    """
    Health-aware proxy that auto-restarts dead backend services.
    
    Strategy: Per-service SLA timeouts based on tier classification.
    
    HOT services get 2s SLA — any backend call exceeding this is a
    tier violation. WARM/COLD services receive proportionally larger
    timeouts to accommodate cold-start latency.
    
    The user NEVER sees errors or "processing" status. The system:
    1. Detects dead service
    2. Starts it silently
    3. Polls health every 100ms
    4. Returns result when ready (~500ms for Python, ~150ms for Rust/C++)
    5. Uses per-service sla_timeout from config (not a single 30s default)
    
    This is the core of the agentic MCP gateway - it absorbs service
    downtime and makes it transparent to the user.
    """
    
    # Service configuration is injected from Settings at runtime.
    SERVICE_CONFIG: dict[str, dict[str, Any]] = {}
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        """
        Initialize proxy.
        
        Args:
            http_client: Optional shared httpx.AsyncClient
        """
        self.http_client = http_client
        self._restarting: set[str] = set()  # Services currently restarting
        self._tracker = get_tracker()
        self._settings = Settings()
        self.SERVICE_CONFIG = copy.deepcopy(self._settings.HEALTH_PROXY_SERVICE_CONFIG)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx.AsyncClient."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        return self.http_client
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        original_url: str,
    ) -> Any:
        """
        Call a tool with health-aware auto-restart.
        
        Uses per-service sla_timeout from config (tier-respecting), not
        a blanket 30s timeout.  HOT services get ~2s, WARM ~8-15s,
        COLD ~30-60s.
        
        Strategy:
        1. Detect dead service
        2. Start it silently
        3. Poll health every 100ms
        4. Return result with per-service timeout
        5. On timeout: fail fast with clear error
        
        User NEVER sees errors or "processing" status.
        """
        # Extract service name from tool name
        service_key = self._tool_to_service(tool_name)
        if service_key is None:
            # Not a managed service, call directly
            return await self._call_direct(original_url, arguments)
        
        # Record request for idle tracking
        self._tracker.record_request(service_key)
        
        # Auto-warm cold/warm services before first probe
        # Hot services are skipped internally — this is a no-op for them.
        await auto_warm_service(service_key)
        
        config = self.SERVICE_CONFIG.get(service_key, {})
        timeout = config.get("timeout", 2.0)
        
        # Try to call, auto-restart if dead
        max_retries = 2  # Only 2 retries (keep it fast)
        
        for attempt in range(max_retries):
            try:
                # Check if service is dead
                if await self._is_service_dead(service_key):
                    logger.info(
                        "%s service dead, auto-restarting (timeout=%.1fs)...",
                        service_key,
                        timeout,
                    )
                    await self._restart_service(service_key, timeout=timeout)
                
                # Call the tool (with per-service sla_timeout)
                result = await self._call_backend(original_url, arguments, service_key=service_key)
                logger.debug("%s call succeeded", tool_name)
                return result
                
            except httpx.ConnectError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "%s attempt %d failed: %s, retrying...",
                        tool_name,
                        attempt + 1,
                        e,
                    )
                    await self._restart_service(service_key, timeout=timeout)
                    await asyncio.sleep(0.1)  # Minimal delay
                else:
                    raise
        
        raise ServiceRestartError(f"Failed to restart {service_key} after {max_retries} attempts")
    
    async def _is_service_dead(self, service_key: str) -> bool:
        """
        Check if service is dead (not responding to health check).
        
        Args:
            service_key: Service identifier
            
        Returns:
            True if service is dead
        """
        if service_key not in self.SERVICE_CONFIG:
            return False
        
        config = self.SERVICE_CONFIG[service_key]
        health_url = f"{config['url']}{config['health_endpoint']}"
        
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(health_url)
                return response.status_code != 200
        except Exception:
            return True
    
    async def _restart_service(self, service_key: str, timeout: float = 2.0) -> None:
        """
        Restart a service (silently, in background).
        
        This is NON-BLOCKING - returns immediately while service starts.
        
        Args:
            service_key: Service identifier
            timeout: Maximum time to wait for service to be healthy (default: 2s)
            
        Raises:
            ServiceRestartError: If restart command fails to start
        """
        if service_key not in self.SERVICE_CONFIG:
            raise ServiceRestartError(f"Unknown service: {service_key}")
        
        # Prevent concurrent restarts
        if service_key in self._restarting:
            logger.debug("%s already restarting, waiting...", service_key)
            await self._wait_for_service(service_key, timeout=timeout)
            return
        
        self._restarting.add(service_key)
        
        try:
            config = self.SERVICE_CONFIG[service_key]
            command = config["restart_command"]
            
            logger.info("Starting %s in background: %s", service_key, command)
            
            # Start service in background (non-blocking)
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            
            # Verify it started
            if process.returncode is not None:
                # Process exited immediately - command failed
                raise ServiceRestartError(
                    f"Failed to start {service_key}: exit code {process.returncode}"
                )
            
            # Wait for service to be healthy (up to timeout seconds)
            await self._wait_for_service(service_key, timeout=timeout)
            
            logger.info("%s started successfully", service_key)
            
        finally:
            self._restarting.discard(service_key)
    
    async def _wait_for_service(self, service_key: str, timeout: float | None = None) -> None:
        """
        Wait for service to become healthy.
        
        Polls health endpoint every 100ms until healthy or timeout.
        Designed for near-instant user experience:
        - Python services: ~500ms cold start
        - Rust/C++ services: ~150ms cold start
        - Max wait: 2 seconds for sync, 10s for async (then fails fast)
        
        Args:
            service_key: Service identifier
            timeout: Maximum time to wait (default: 2s for sync)
        """
        if service_key not in self.SERVICE_CONFIG:
            return
        
        config = self.SERVICE_CONFIG[service_key]
        health_url = f"{config['url']}{config['health_endpoint']}"
        
        effective_timeout = timeout if timeout is not None else health_timeout_for(service_key)
        start_time = datetime.now(timezone.utc)
        poll_interval = 0.1  # 100ms polling
        
        while True:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed >= effective_timeout:
                raise ServiceRestartError(
                    f"Service {service_key} did not become healthy within {effective_timeout}s"
                )
            
            try:
                async with httpx.AsyncClient(timeout=0.5) as client:
                    response = await client.get(health_url)
                    if response.status_code == 200:
                        logger.debug(
                            "%s became healthy in %.2fs",
                            service_key,
                            elapsed,
                        )
                        return
            except Exception:
                pass
            
            await asyncio.sleep(poll_interval)
    
    async def _call_backend(
        self,
        url: str,
        arguments: dict[str, Any],
        service_key: str | None = None,
    ) -> Any:
        """
        Call backend service directly.
        
        Args:
            url: Backend URL
            arguments: Tool arguments
            service_key: Optional service key for per-service timeout.
                         Falls back to SLA_TIMEOUT (30.0s) when None.
            
        Returns:
            Tool response
        """
        timeout = self._resolve_timeout(service_key)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=arguments,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
    
    async def _call_direct(
        self,
        url: str,
        arguments: dict[str, Any],
        service_key: str | None = None,
    ) -> Any:
        """
        Call backend without health-aware proxy.
        
        Args:
            url: Backend URL
            arguments: Tool arguments
            service_key: Optional service key for per-service timeout.
                         Falls back to SLA_TIMEOUT (30.0s) when None.
            
        Returns:
            Tool response
        """
        timeout = self._resolve_timeout(service_key)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=arguments,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()

    def _resolve_timeout(self, service_key: str | None) -> float:
        """Resolve the effective timeout for a service.

        Priority:
        1. Per-service ``sla_timeout`` from config (if set)
        2. Per-service ``timeout`` from config (restart-wait fallback)
        3. Module-level ``SLA_TIMEOUT`` (30.0s)

        The sla_timeout should be set to the SLA commitment for the
        backend (e.g. 2.0s for hot services).  The plain timeout is
        the restart-wait budget, which can be much larger.
        """
        if service_key and service_key in self.SERVICE_CONFIG:
            cfg = self.SERVICE_CONFIG[service_key]
            # Prefer sla_timeout if explicitly set; fall back to timeout
            sla = cfg.get("sla_timeout")
            if sla is not None:
                return float(sla)
            return float(cfg.get("timeout", SLA_TIMEOUT))
        return SLA_TIMEOUT
    
    def _tool_to_service(self, tool_name: str) -> Optional[str]:
        """
        Map MCP tool name to service key.
        
        Args:
            tool_name: MCP tool name
            
        Returns:
            Service key or None if not managed
        """
        # Map tool names to service keys
        # Normalise the lookup key so both "semantic_search" and "semantic-search" resolve
        tool_name = normalize_service_key(tool_name)
        tool_to_service = {
            "semantic-search": "semantic-search",
            "hybrid-search": "semantic-search",
            "knowledge-search": "semantic-search",
            "knowledge-refine": "semantic-search",
            "pattern-search": "semantic-search",
            "diagram-search": "semantic-search",
            "graph-query": "semantic-search",
            "graph-traverse": "semantic-search",
            "code-analyze": "code-analyze",
            "code-pattern-audit": "code-analyze",
            "llm-complete": "llm-complete",
            "a2a-send-message": "ai-agents",
            "a2a-get-task": "ai-agents",
            "a2a-cancel-task": "ai-agents",
            "enhance-guideline": "ai-agents",
            "audit-security-scan": "audit-service",
            "audit-code-metrics": "audit-service",
            "audit-corpus-search": "audit-service",
            "audit-dependency-assess": "audit-service",
            "audit-resolve-lookup": "audit-service",
            "audit-search-exploits": "audit-service",
            "audit-search-cves": "audit-service",
            "audit-quality-scan": "audit-service",
            "generate-taxonomy": "code-orchestrator",
            "extract-book-metadata": "code-orchestrator",
            "enrich-book-metadata": "code-orchestrator",
            "batch-extract-metadata": "code-orchestrator",
            "batch-enrich-metadata": "code-orchestrator",
            "analyze-taxonomy-coverage": "code-orchestrator",
            "run-agent-function": "run-agent-function",
            "run-discussion": "run-agent-function",
            "agent-execute": "run-agent-function",
            "context-management": "context-management",
            "amve-evaluate-fitness": "amve",
            "foundation-search": "foundation-search",
        }
        
        raw = tool_to_service.get(tool_name)
        return normalize_service_key(raw) if raw else None
    
    def get_service_status(self, service_key: str) -> dict[str, Any]:
        """
        Get status information for a service.
        
        Args:
            service_key: Service identifier
            
        Returns:
            Status dict with health, idle_time, restarting flags
        """
        if service_key not in self.SERVICE_CONFIG:
            return {"error": "Unknown service"}
        
        config = self.SERVICE_CONFIG[service_key]
        is_restarting = service_key in self._restarting
        
        # Get idle status from tracker
        idle_status = self._tracker.get_status(service_key)
        
        return {
            "name": config["name"],
            "url": config["url"],
            "is_restarting": is_restarting,
            "is_idle": idle_status.get("is_idle", False),
            "idle_seconds": idle_status.get("idle_seconds"),
            "timeout": idle_status.get("timeout"),
            "total_requests": idle_status.get("total_requests", 0),
        }


# =============================================================================
# Global Proxy Instance
# =============================================================================

_proxy_instance: Optional[HealthAwareProxy] = None


def get_proxy() -> HealthAwareProxy:
    """
    Get the global HealthAwareProxy instance (singleton).
    
    Returns:
        Global proxy instance
    """
    global _proxy_instance
    if _proxy_instance is None:
        _proxy_instance = HealthAwareProxy()
    return _proxy_instance


def reset_proxy() -> None:
    """
    Reset the global proxy instance (useful for testing).
    """
    global _proxy_instance
    _proxy_instance = None
