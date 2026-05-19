#!/usr/bin/env python3
"""
Idle Timeout Service Tracker
=============================

Tracks service last-request timestamps and enforces idle timeouts.

TDD Test Suite - RED Phase
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path
import json


# =============================================================================
# System Under Test (SUT) - To be implemented
# =============================================================================

class IdleTimeoutTracker:
    """Tracks service idle timeouts and enforces shutdown."""
    
    def __init__(self, default_timeout_seconds: int = 600):
        """
        Initialize tracker.
        
        Args:
            default_timeout_seconds: Default idle timeout (default: 10 min = 600s)
        """
        self.default_timeout = default_timeout_seconds
        self._service_states: dict[str, dict] = {}
    
    def record_request(self, service_id: str) -> None:
        """Record that a service received a request."""
        pass
    
    def get_idle_time(self, service_id: str) -> Optional[float]:
        """
        Get seconds since last request for a service.
        
        Returns:
            Seconds since last request, or None if service never seen
        """
        pass
    
    def is_idle(self, service_id: str) -> bool:
        """
        Check if service has exceeded idle timeout.
        
        Args:
            service_id: Service identifier
            
        Returns:
            True if service is idle (exceeded timeout)
        """
        pass
    
    def get_timeout_config(self, service_id: str) -> int:
        """
        Get timeout configuration for a specific service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Timeout in seconds (custom or default)
        """
        pass
    
    def set_custom_timeout(self, service_id: str, timeout_seconds: int) -> None:
        """
        Set custom timeout for a service.
        
        Args:
            service_id: Service identifier
            timeout_seconds: Custom timeout in seconds
        """
        pass
    
    def get_services_needing_shutdown(self) -> list[str]:
        """
        Get list of services that have exceeded idle timeout.
        
        Returns:
            List of service IDs that should be shut down
        """
        pass
    
    def get_status(self, service_id: str) -> dict:
        """
        Get status information for a service.
        
        Args:
            service_id: Service identifier
            
        Returns:
            Dict with last_request, idle_seconds, timeout, is_idle
        """
        pass


# =============================================================================
# RED Phase Tests - Should fail initially
# =============================================================================

class TestIdleTimeoutTrackerRed:
    """RED phase: Tests that should fail before implementation."""
    
    def test_tracker_initializes_with_default_timeout(self):
        """Tracker should initialize with 600s (10 min) default."""
        tracker = IdleTimeoutTracker()
        assert tracker.default_timeout == 600
    
    def test_tracker_initializes_with_custom_timeout(self):
        """Tracker should accept custom default timeout."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=300)
        assert tracker.default_timeout == 300
    
    def test_record_request_adds_service(self):
        """Recording a request should add service to tracker."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        
        # Service should now exist
        assert "test-service" in tracker._service_states
    
    def test_record_request_sets_timestamp(self):
        """Recording a request should set last_request timestamp."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        
        state = tracker._service_states["test-service"]
        assert "last_request" in state
        assert state["last_request"] is not None
    
    def test_get_idle_time_returns_none_for_unknown_service(self):
        """Getting idle time for unknown service should return None."""
        tracker = IdleTimeoutTracker()
        result = tracker.get_idle_time("unknown-service")
        assert result is None
    
    def test_get_idle_time_returns_zero_after_record(self):
        """Getting idle time immediately after record should be ~0."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        
        idle_time = tracker.get_idle_time("test-service")
        assert idle_time is not None
        assert idle_time >= 0  # Should be >= 0 (may be small positive)
    
    def test_is_idle_returns_false_immediately_after_request(self):
        """Service should not be idle immediately after request."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("test-service")
        
        assert tracker.is_idle("test-service") is False
    
    def test_is_idle_returns_true_after_timeout(self):
        """Service should be idle after exceeding timeout."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("test-service")
        
        # Simulate time passing by manipulating timestamp
        tracker._service_states["test-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=700)
        )
        
        assert tracker.is_idle("test-service") is True
    
    def test_get_timeout_config_returns_default(self):
        """Getting timeout for unknown service should return default."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        
        assert tracker.get_timeout_config("unknown-service") == 600
    
    def test_get_timeout_config_returns_custom(self):
        """Getting timeout for configured service should return custom value."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.set_custom_timeout("fast-service", 300)
        
        assert tracker.get_timeout_config("fast-service") == 300
        assert tracker.get_timeout_config("other-service") == 600  # default
    
    def test_set_custom_timeout_persists(self):
        """Custom timeout should persist across calls."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.set_custom_timeout("api-service", 900)
        
        assert tracker.get_timeout_config("api-service") == 900
        assert tracker.get_timeout_config("other") == 600  # default unchanged
    
    def test_get_services_needing_shutdown_returns_idle(self):
        """Should return services that have exceeded timeout."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        
        # Service 1: idle
        tracker.record_request("idle-service")
        tracker._service_states["idle-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=700)
        )
        
        # Service 2: not idle
        tracker.record_request("active-service")
        
        idle_services = tracker.get_services_needing_shutdown()
        
        assert "idle-service" in idle_services
        assert "active-service" not in idle_services
    
    def test_get_status_returns_complete_info(self):
        """Status should include all required fields."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("test-service")
        
        status = tracker.get_status("test-service")
        
        assert "last_request" in status
        assert "idle_seconds" in status
        assert "timeout" in status
        assert "is_idle" in status
        assert isinstance(status["last_request"], datetime)
        assert isinstance(status["idle_seconds"], (int, float))
        assert isinstance(status["timeout"], int)
        assert isinstance(status["is_idle"], bool)


# =============================================================================
# GREEN Phase - Implementation
# =============================================================================

from datetime import datetime, timezone


class IdleTimeoutTracker:
    """Tracks service idle timeouts and enforces shutdown."""
    
    def __init__(self, default_timeout_seconds: int = 600):
        """
        Initialize tracker.
        
        Args:
            default_timeout_seconds: Default idle timeout (default: 10 min = 600s)
        """
        self.default_timeout = default_timeout_seconds
        self._service_states: dict[str, dict] = {}
        self._custom_timeouts: dict[str, int] = {}
    
    def record_request(self, service_id: str) -> None:
        """Record that a service received a request."""
        self._service_states[service_id] = {
            "last_request": datetime.now(timezone.utc)
        }
    
    def get_idle_time(self, service_id: str) -> Optional[float]:
        """
        Get seconds since last request for a service.
        
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
        Set custom timeout for a service.
        
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
            Dict with last_request, idle_seconds, timeout, is_idle
        """
        if service_id not in self._service_states:
            return {
                "last_request": None,
                "idle_seconds": None,
                "timeout": self.get_timeout_config(service_id),
                "is_idle": False,
            }
        
        idle_seconds = self.get_idle_time(service_id) or 0
        timeout = self.get_timeout_config(service_id)
        
        return {
            "last_request": self._service_states[service_id]["last_request"],
            "idle_seconds": idle_seconds,
            "timeout": timeout,
            "is_idle": idle_seconds >= timeout,
        }


# =============================================================================
# REFACTOR Phase - Will add after GREEN passes
# =============================================================================

# Will add:
# - JSON persistence
# - Redis backend
# - Metrics export
# - Graceful shutdown hooks
