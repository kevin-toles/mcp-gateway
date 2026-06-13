#!/usr/bin/env python3
"""
Idle Timeout Service Tracker
=============================

Tests for the production IdleTimeoutTracker from src.core.idle_timeout.

Covers:
  - Default and custom timeouts
  - ServiceKey normalization (underscore→hyphen)
  - record_request / get_idle_time / is_idle
  - Custom timeout overrides
  - get_services_needing_shutdown / get_all_statuses
  - cleanup_expired
  - Singleton via get_tracker() / reset_tracker()
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import patch

from src.core.idle_timeout import IdleTimeoutTracker, get_tracker, reset_tracker


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test to avoid cross-test pollution."""
    reset_tracker()
    yield
    reset_tracker()


# =============================================================================
# Timeout & Initialization
# =============================================================================

class TestInit:
    """Tracker initialization."""

    def test_default_timeout_is_600(self):
        """Default timeout should be 600s (10 min)."""
        tracker = IdleTimeoutTracker()
        assert tracker.default_timeout == 600

    def test_custom_default_timeout(self):
        """Tracker should accept custom default timeout."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=300)
        assert tracker.default_timeout == 300

    def test_negative_timeout_accepted(self):
        """Negative timeout accepted (caller responsibility)."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=-1)
        assert tracker.default_timeout == -1


# =============================================================================
# ServiceKey Normalization
# =============================================================================

class TestServiceKeyNormalization:
    """record_request normalizes service_id via ServiceKey (underscore→hyphen)."""

    def test_record_with_underscore_stored_as_hyphen(self):
        """Underscore in service_id should be normalized to hyphen."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test_service")
        assert "test_service" not in tracker._service_states
        assert "test-service" in tracker._service_states

    def test_record_with_hyphen_stored_as_hyphen(self):
        """Hyphen in service_id should remain unchanged."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        assert "test-service" in tracker._service_states

    def test_get_idle_time_normalizes_key(self):
        """get_idle_time should normalize underscore to hyphen."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        idle_time = tracker.get_idle_time("test_service")
        assert idle_time is not None
        assert idle_time >= 0

    def test_get_timeout_config_normalizes_key(self):
        """get_timeout_config should normalize underscore to hyphen."""
        tracker = IdleTimeoutTracker()
        tracker.set_custom_timeout("my-service", 100)
        assert tracker.get_timeout_config("my_service") == 100

    def test_set_custom_timeout_normalizes_key(self):
        """set_custom_timeout should normalize underscore to hyphen."""
        tracker = IdleTimeoutTracker()
        tracker.set_custom_timeout("my_service", 100)
        assert tracker.get_timeout_config("my-service") == 100

    def test_get_status_normalizes_key(self):
        """get_status should normalize underscore to hyphen."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("my-service")
        status = tracker.get_status("my_service")
        assert status["total_requests"] == 1


# =============================================================================
# record_request
# =============================================================================

class TestRecordRequest:
    """Recording requests."""

    def test_adds_service_to_states(self):
        """Recording a request should add service to tracker."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        assert "test-service" in tracker._service_states

    def test_sets_last_request_timestamp(self):
        """Recording should set last_request to a datetime."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        state = tracker._service_states["test-service"]
        assert "last_request" in state
        assert isinstance(state["last_request"], datetime)

    def test_increments_total_requests(self):
        """Repeated calls should increment total_requests."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        tracker.record_request("test-service")
        assert tracker._service_states["test-service"]["total_requests"] == 2

    def test_ten_requests(self):
        """Ten calls should yield total_requests == 10."""
        tracker = IdleTimeoutTracker()
        for _ in range(10):
            tracker.record_request("test-service")
        assert tracker._service_states["test-service"]["total_requests"] == 10


# =============================================================================
# get_idle_time
# =============================================================================

class TestGetIdleTime:
    """Idle time querying."""

    def test_returns_none_for_unknown_service(self):
        """Unknown service should return None."""
        tracker = IdleTimeoutTracker()
        assert tracker.get_idle_time("unknown") is None

    def test_returns_zero_after_record(self):
        """Immediately after record, idle time should be ~0."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        idle = tracker.get_idle_time("test-service")
        assert idle is not None
        assert idle >= 0
        assert idle < 1  # should be near-instant

    def test_increases_with_simulated_time(self):
        """Simulated older timestamp should return larger idle time."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        tracker._service_states["test-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=100)
        )
        idle = tracker.get_idle_time("test-service")
        assert idle is not None
        assert idle >= 100


# =============================================================================
# is_idle
# =============================================================================

class TestIsIdle:
    """Idle status checking."""

    def test_false_immediately_after_request(self):
        """Service should not be idle immediately after request."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("test-service")
        assert tracker.is_idle("test-service") is False

    def test_true_after_timeout_exceeded(self):
        """Service should be idle after exceeding timeout."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("test-service")
        tracker._service_states["test-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=700)
        )
        assert tracker.is_idle("test-service") is True

    def test_false_for_unknown_service(self):
        """Unknown service should not be considered idle."""
        tracker = IdleTimeoutTracker()
        assert tracker.is_idle("unknown") is False

    def test_false_when_not_yet_at_timeout(self):
        """Service just below timeout should not be idle."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=100)
        tracker.record_request("test-service")
        tracker._service_states["test-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=99)
        )
        assert tracker.is_idle("test-service") is False


# =============================================================================
# Custom Timeouts
# =============================================================================

class TestCustomTimeouts:
    """Per-service timeout overrides."""

    def test_get_timeout_config_returns_default(self):
        """Unknown service should get default timeout."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        assert tracker.get_timeout_config("unknown") == 600

    def test_set_and_get_custom_timeout(self):
        """Custom timeout should be retrievable."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.set_custom_timeout("fast-service", 300)
        assert tracker.get_timeout_config("fast-service") == 300

    def test_custom_timeout_does_not_affect_default(self):
        """Setting custom timeout should not change default for others."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.set_custom_timeout("fast-service", 300)
        assert tracker.get_timeout_config("other-service") == 600

    def test_set_custom_timeout_overwrites(self):
        """Previously set custom timeout should be overwritable."""
        tracker = IdleTimeoutTracker()
        tracker.set_custom_timeout("svc", 300)
        tracker.set_custom_timeout("svc", 900)
        assert tracker.get_timeout_config("svc") == 900


# =============================================================================
# get_services_needing_shutdown
# =============================================================================

class TestServicesNeedingShutdown:
    """Idle service enumeration."""

    def test_returns_idle_services(self):
        """Should return services exceeding timeout."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("idle-service")
        tracker._service_states["idle-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=700)
        )
        tracker.record_request("active-service")

        idle = tracker.get_services_needing_shutdown()
        assert "idle-service" in idle
        assert "active-service" not in idle

    def test_empty_when_all_active(self):
        """Should return empty list when no services idle."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("svc-a")
        tracker.record_request("svc-b")
        assert tracker.get_services_needing_shutdown() == []

    def test_respects_custom_timeout(self):
        """Should respect per-service custom timeout."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("slow-service")
        tracker.set_custom_timeout("slow-service", 9999)
        tracker._service_states["slow-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=700)
        )
        idle = tracker.get_services_needing_shutdown()
        assert "slow-service" not in idle  # custom timeout protects it


# =============================================================================
# get_status
# =============================================================================

class TestGetStatus:
    """Status querying."""

    def test_returns_all_required_fields(self):
        """Status should include all fields."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("test-service")
        status = tracker.get_status("test-service")

        assert "last_request" in status
        assert "idle_seconds" in status
        assert "timeout" in status
        assert "is_idle" in status
        assert "total_requests" in status

    def test_last_request_is_isoformat_string(self):
        """last_request should be ISO format string."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        status = tracker.get_status("test-service")
        assert isinstance(status["last_request"], str)
        assert "T" in status["last_request"]  # ISO format

    def test_idle_seconds_is_float(self):
        """idle_seconds should be a number."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        status = tracker.get_status("test-service")
        assert isinstance(status["idle_seconds"], (int, float))

    def test_timeout_is_int(self):
        """timeout should be an int."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        status = tracker.get_status("test-service")
        assert isinstance(status["timeout"], int)

    def test_is_idle_is_bool(self):
        """is_idle should be bool."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        status = tracker.get_status("test-service")
        assert isinstance(status["is_idle"], bool)

    def test_total_requests_tracked(self):
        """total_requests should reflect call count."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("test-service")
        tracker.record_request("test-service")
        tracker.record_request("test-service")
        status = tracker.get_status("test-service")
        assert status["total_requests"] == 3

    def test_unknown_service_returns_empty_status(self):
        """Unknown service should return status with None/0."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        status = tracker.get_status("unknown")
        assert status["last_request"] is None
        assert status["idle_seconds"] is None
        assert status["timeout"] == 600
        assert status["is_idle"] is False
        assert status["total_requests"] == 0


# =============================================================================
# get_all_statuses
# =============================================================================

class TestGetAllStatuses:
    """Bulk status query."""

    def test_returns_all_tracked_services(self):
        """Should return status for all tracked services."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("svc-a")
        tracker.record_request("svc-b")
        all_statuses = tracker.get_all_statuses()
        assert set(all_statuses.keys()) == {"svc-a", "svc-b"}

    def test_each_entry_is_valid_status(self):
        """Each entry should be a valid status dict."""
        tracker = IdleTimeoutTracker()
        tracker.record_request("svc-a")
        all_statuses = tracker.get_all_statuses()
        status = all_statuses["svc-a"]
        assert "last_request" in status
        assert "idle_seconds" in status
        assert "total_requests" in status

    def test_empty_when_no_services(self):
        """Should return empty dict when no services tracked."""
        tracker = IdleTimeoutTracker()
        assert tracker.get_all_statuses() == {}


# =============================================================================
# cleanup_expired
# =============================================================================

class TestCleanupExpired:
    """Expired entry cleanup."""

    def test_removes_entries_past_2x_timeout(self):
        """Services idle for >= 2x timeout should be removed."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("expired-service")
        tracker._service_states["expired-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1201)
        )
        removed = tracker.cleanup_expired()
        assert "expired-service" in removed
        assert "expired-service" not in tracker._service_states

    def test_does_not_remove_recent_services(self):
        """Active services should not be removed."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("active-service")
        removed = tracker.cleanup_expired()
        assert "active-service" not in removed

    def test_removes_multiple_expired(self):
        """Should remove all services past 2x timeout."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("svc-a")
        tracker.record_request("svc-b")
        tracker.record_request("svc-c")

        # Two expired, one active
        tracker._service_states["svc-a"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1300)
        )
        tracker._service_states["svc-b"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1300)
        )
        # svc-c stays as-is (active)

        removed = tracker.cleanup_expired()
        assert "svc-a" in removed
        assert "svc-b" in removed
        assert "svc-c" not in removed

    def test_returns_empty_list_when_nothing_expired(self):
        """No expired services should return empty list."""
        tracker = IdleTimeoutTracker()
        assert tracker.cleanup_expired() == []

    def test_respects_custom_timeout_for_cleanup(self):
        """Custom timeout should affect 2x calculation."""
        tracker = IdleTimeoutTracker(default_timeout_seconds=600)
        tracker.record_request("slow-service")
        tracker.set_custom_timeout("slow-service", 9999)
        tracker._service_states["slow-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1200)
        )
        # 1200 < 2*9999 = 19998, so NOT expired
        removed = tracker.cleanup_expired()
        assert "slow-service" not in removed


# =============================================================================
# Singleton: get_tracker / reset_tracker
# =============================================================================

class TestSingleton:
    """Singleton lifecycle."""

    def test_returns_same_instance(self):
        """get_tracker() should return the same instance."""
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_reset_creates_new_instance(self):
        """reset_tracker() should allow a new instance."""
        t1 = get_tracker()
        reset_tracker()
        t2 = get_tracker()
        assert t1 is not t2

    def test_reset_clears_data(self):
        """After reset, tracker should have no state."""
        tracker = get_tracker()
        tracker.record_request("test-service")
        reset_tracker()
        tracker2 = get_tracker()
        assert tracker2._service_states == {}

    def test_autouse_fixture_provides_isolation(self):
        """Autouse fixture resets singleton before each test.
        This runs after test_reset_clears_data, which calls reset_tracker()
        and record_request — that state must not leak into this test.
        """
        fresh_tracker = get_tracker()
        assert fresh_tracker._service_states == {}
