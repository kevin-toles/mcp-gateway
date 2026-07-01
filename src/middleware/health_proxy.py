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
from src.core.config import Settings

logger = logging.getLogger(__name__)


class ServiceRestartError(Exception):
    """Failed to restart service."""
    pass


class HealthAwareProxy:
    """
    Health-aware proxy that auto-restarts dead backend services.
    
    Strategy: Pre-warming + 2s strict timeout for ALL services.
    
    The user NEVER sees errors or "processing" status. The system:
    1. Detects dead service
    2. Starts it silently
    3. Polls health every 100ms
    4. Returns result when ready (~500ms for Python, ~150ms for Rust/C++)
    5. If >2s: fails gracefully (but this should never happen in practice)
    
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
        
        ALL services use the same strategy:
        1. Detect dead service
        2. Start it silently
        3. Poll health every 100ms
        4. Return result when ready (~500ms Python, ~150ms Rust/C++)
        5. If >2s: fail gracefully (safety net)
        
        User NEVER sees errors or "processing" status.
        """
        # Extract service name from tool name
        service_key = self._tool_to_service(tool_name)
        if service_key is None:
            # Not a managed service, call directly
            return await self._call_direct(original_url, arguments)
        
        # Record request for idle tracking
        self._tracker.record_request(service_key)
        
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
                
                # Call the tool
                result = await self._call_backend(original_url, arguments)
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
    
    async def _wait_for_service(self, service_key: str, timeout: float = 2.0) -> None:
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
        
        start_time = datetime.now(timezone.utc)
        poll_interval = 0.1  # 100ms polling
        
        while True:
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed >= timeout:
                raise ServiceRestartError(
                    f"Service {service_key} did not become healthy within {timeout}s"
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
            
            await asyncio.sleep(poll_interval)
    
    async def _call_backend(
        self,
        url: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Call backend service directly.
        
        Args:
            url: Backend URL
            arguments: Tool arguments
            
        Returns:
            Tool response
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json=arguments,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
    
    async def _call_direct(self, url: str, arguments: dict[str, Any]) -> Any:
        """
        Call backend without health-aware proxy.
        
        Args:
            url: Backend URL
            arguments: Tool arguments
            
        Returns:
            Tool response
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json=arguments,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
    
    def _tool_to_service(self, tool_name: str) -> Optional[str]:
        """
        Map MCP tool name to service key.
        
        Args:
            tool_name: MCP tool name
            
        Returns:
            Service key or None if not managed
        """
        # Map tool names to service keys
        tool_to_service = {
            "semantic_search": "semantic_search",
            "hybrid_search": "semantic_search",
            "knowledge_search": "semantic_search",
            "knowledge_refine": "semantic_search",
            "pattern_search": "semantic_search",
            "diagram_search": "semantic_search",
            "graph_query": "semantic_search",
            "graph_traverse": "semantic_search",
            "code_analyze": "code_analyze",
            "code_pattern_audit": "code_analyze",
            "llm_complete": "llm_complete",
            "a2a_send_message": "ai_agents",
            "a2a_get_task": "ai_agents",
            "a2a_cancel_task": "ai_agents",
            "enhance_guideline": "ai_agents",
            "audit_security_scan": "audit_service",
            "audit_code_metrics": "audit_service",
            "audit_corpus_search": "audit_service",
            "audit_dependency_assess": "audit_service",
            "audit_resolve_lookup": "audit_service",
            "audit_search_exploits": "audit_service",
            "audit_search_cves": "audit_service",
            "audit_quality_scan": "audit_service",
            "generate_taxonomy": "code_orchestrator",
            "extract_book_metadata": "code_orchestrator",
            "enrich_book_metadata": "code_orchestrator",
            "batch_extract_metadata": "code_orchestrator",
            "batch_enrich_metadata": "code_orchestrator",
            "analyze_taxonomy_coverage": "code_orchestrator",
            "run_agent_function": "run_agent_function",
            "run_discussion": "run_agent_function",
            "agent_execute": "run_agent_function",
            "context_management": "context_management",
            "amve_evaluate_fitness": "amve",
            "foundation_search": "foundation_search",
        }
        
        return tool_to_service.get(tool_name)
    
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
