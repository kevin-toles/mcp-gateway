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

import os
from datetime import datetime, timezone
from typing import Optional
from functools import lru_cache


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
        self._service_states[service_id] = {
            "last_request": datetime.now(timezone.utc),
            "total_requests": self._service_states.get(service_id, {}).get("total_requests", 0) + 1,
        }
    
    def get_idle_time(self, service_id: str) -> Optional[float]:
        """
        Get seconds since last request for a service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Seconds since last request, or None if service never seen
        """
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
            True if service is idle (exceeded timeout)
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
        return self._custom_timeouts.get(service_id, self.default_timeout)
    
    def set_custom_timeout(self, service_id: str, timeout_seconds: int) -> None:
        """
        Set custom timeout for a service (runtime override).
        
        Args:
            service_id: Service identifier
            timeout_seconds: Custom timeout in seconds
        """
        self._custom_timeouts[service_id] = timeout_seconds
    
    def get_services_needing_shutdown(self) -> list[str]:
        """
        Get list of services that have exceeded idle timeout.
        
        Returns:
            List of service IDs that should be shut down
        """
        idle_services = []
        
        for service_id in self._service_states:
            if self.is_idle(service_id):
                idle_services.append(service_id)
        
        return idle_services
    
    def get_status(self, service_id: str) -> dict:
        """
        Get status information for a service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Dict with last_request, idle_seconds, timeout, is_idle, total_requests
        """
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
