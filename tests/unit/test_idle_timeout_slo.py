"""
F10 (P2) — SLO Stress Test: IdleTimeoutTracker throughput & timing.

Verifies that:
  1. Sequential throughput: 1,000 record_request() calls completes under 200ms.
  2. Bulk reading: get_all_statuses() returns correct counts after batch writes.
  3. Accuracy: recorded timestamps are monotonic non-decreasing.
  4. Concurrent safety: sequential dispatches maintain correct total_requests.
  5. Memory stability: no unexpected state growth over repeated cycles.

Reference: F10 — Dead code in idle-timeout tracking hook
           SLO: 99.9% availability, P95 latency ≤20ms for tracker ops
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import pytest

from src.core.idle_timeout import IdleTimeoutTracker, reset_tracker


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

SERVICE_NAMES = [
    "semantic-search",
    "code-orchestrator",
    "llm-gateway",
    "ai-agents",
    "audit-service",
    "context-management-service",
    "unified-search-service",
    "inference-service-cpp",
    "amve",
]


@pytest.fixture(autouse=True)
def reset_singleton():
    """Fresh tracker before each test."""
    reset_tracker()
    yield
    reset_tracker()


@pytest.fixture
def tracker() -> IdleTimeoutTracker:
    """Return a fresh tracker instance."""
    return IdleTimeoutTracker()


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Sequential Throughput
# ═══════════════════════════════════════════════════════════════════════

class TestSequentialThroughput:
    """1,000 record_request calls must complete under SLO threshold."""

    SLO_MAX_MS = 200
    NUM_CALLS = 1000

    def test_sequential_throughput_single_service(self, tracker):
        """1,000 calls to one service completes within 200ms."""
        start = time.perf_counter()
        for _ in range(self.NUM_CALLS):
            tracker.record_request("semantic-search")
        elapsed_ms = (time.perf_counter() - start) * 1000

        state = tracker._service_states["semantic-search"]
        assert state["total_requests"] == self.NUM_CALLS
        assert elapsed_ms < self.SLO_MAX_MS, (
            f"Throughput SLO exceeded: {elapsed_ms:.1f}ms > {self.SLO_MAX_MS}ms"
        )

    def test_sequential_throughput_round_robin(self, tracker):
        """1,000 calls round-robined across 9 services within 200ms."""
        start = time.perf_counter()
        for i in range(self.NUM_CALLS):
            svc = SERVICE_NAMES[i % len(SERVICE_NAMES)]
            tracker.record_request(svc)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Each service should have either floor or ceil calls
        base = self.NUM_CALLS // len(SERVICE_NAMES)
        remainder = self.NUM_CALLS % len(SERVICE_NAMES)
        for i, svc in enumerate(SERVICE_NAMES):
            expected = base + (1 if i < remainder else 0)
            actual = tracker._service_states[svc]["total_requests"]
            assert actual == expected, f"{svc}: expected {expected}, got {actual}"

        assert elapsed_ms < self.SLO_MAX_MS, (
            f"Round-robin SLO exceeded: {elapsed_ms:.1f}ms > {self.SLO_MAX_MS}ms"
        )


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Bulk Read After Batch Write
# ═══════════════════════════════════════════════════════════════════════

class TestBulkReadAfterBatchWrite:
    """get_all_statuses() returns correct counts after mass writes."""

    CALLS_PER_SERVICE = 500

    def test_get_all_statuses_after_batch_write(self, tracker):
        """All services visible in get_all_statuses with correct counts."""
        for svc in SERVICE_NAMES:
            for _ in range(self.CALLS_PER_SERVICE):
                tracker.record_request(svc)

        statuses = tracker.get_all_statuses()

        assert len(statuses) == len(SERVICE_NAMES)
        for svc in SERVICE_NAMES:
            assert svc in statuses, f"{svc} missing from statuses"
            assert statuses[svc]["total_requests"] == self.CALLS_PER_SERVICE

    def test_bulk_read_latency(self, tracker):
        """get_all_statuses() on 9 services completes under 10ms."""
        for svc in SERVICE_NAMES:
            tracker.record_request(svc)

        start = time.perf_counter()
        for _ in range(100):
            tracker.get_all_statuses()
        elapsed_ms = (time.perf_counter() - start) * 1000

        per_call_us = (elapsed_ms / 100) * 1000
        assert per_call_us < 100, (
            f"get_all_statuses too slow: {per_call_us:.1f}µs/call"
        )


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Timestamp Monotonicity
# ═══════════════════════════════════════════════════════════════════════

class TestTimestampMonotonicity:
    """record_request timestamps must be strictly non-decreasing."""

    NUM_CALLS = 100

    def test_timestamps_are_monotonic(self, tracker):
        """Each record_request produces a timestamp >= previous."""
        previous: datetime | None = None
        for _ in range(self.NUM_CALLS):
            tracker.record_request("monotonic-test")
            current = tracker._service_states["monotonic-test"]["last_request"]
            if previous is not None:
                assert current >= previous, (
                    f"Timestamp went backwards: {current} < {previous}"
                )
            previous = current

    def test_timestamps_advance_over_time(self, tracker):
        """Timestamps separated by real time show increasing values."""
        tracker.record_request("time-test")
        t1 = tracker._service_states["time-test"]["last_request"]

        time.sleep(0.01)  # 10ms delay

        tracker.record_request("time-test")
        t2 = tracker._service_states["time-test"]["last_request"]

        delta = (t2 - t1).total_seconds()
        assert delta >= 0.005, (
            f"Timestamp delta too small: {delta:.4f}s (expected ≥0.005s)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Sequential Dispatch Accuracy
# ═══════════════════════════════════════════════════════════════════════

class TestSequentialDispatchAccuracy:
    """Simulate the real dispatch pattern — record per service."""

    def test_alternating_services(self, tracker):
        """Alternating between two services maintains correct counts."""
        for _ in range(100):
            tracker.record_request("svc-a")
            tracker.record_request("svc-b")

        assert tracker._service_states["svc-a"]["total_requests"] == 100
        assert tracker._service_states["svc-b"]["total_requests"] == 100

    def test_burst_then_verify(self, tracker):
        """Burst of calls then verify no state corruption."""
        for i in range(500):
            svc = SERVICE_NAMES[i % len(SERVICE_NAMES)]
            tracker.record_request(svc)

        # Verify all totals
        base = 500 // len(SERVICE_NAMES)
        remainder = 500 % len(SERVICE_NAMES)
        for i, svc in enumerate(SERVICE_NAMES):
            expected = base + (1 if i < remainder else 0)
            assert tracker._service_states[svc]["total_requests"] == expected

        # Verify idle times are sane
        for svc in SERVICE_NAMES:
            idle = tracker.get_idle_time(svc)
            assert idle is not None
            assert idle >= 0
            assert idle < 5  # all within last 5s


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Memory Stability Over Repeated Cycles
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryStability:
    """No unexpected state growth over repeated reset-record cycles."""

    NUM_CYCLES = 50

    def test_state_does_not_leak_across_cycles(self):
        """After each reset, tracker has exactly the services recorded."""
        for cycle in range(self.NUM_CYCLES):
            tracker = IdleTimeoutTracker()
            # Record a subset of services each cycle
            for svc in SERVICE_NAMES[: (cycle % len(SERVICE_NAMES)) + 1]:
                tracker.record_request(svc)

            expected_count = (cycle % len(SERVICE_NAMES)) + 1
            assert len(tracker._service_states) == expected_count, (
                f"Cycle {cycle}: expected {expected_count} services, "
                f"got {len(tracker._service_states)}"
            )
            reset_tracker()

    def test_repeated_record_same_service_stable(self, tracker):
        """1,000 calls to same service — no negative counts, no errors."""
        for _ in range(1000):
            tracker.record_request("stable-svc")

        state = tracker._service_states["stable-svc"]
        assert state["total_requests"] == 1000
        assert isinstance(state["last_request"], datetime)

    def test_many_services_no_crosstalk(self, tracker):
        """Recording on one service doesn't affect another."""
        tracker.record_request("svc-a")
        for _ in range(100):
            tracker.record_request("svc-b")

        assert tracker._service_states["svc-a"]["total_requests"] == 1
        assert tracker._service_states["svc-b"]["total_requests"] == 100
