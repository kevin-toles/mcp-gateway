"""
Cold Warm Promotion Tests — HWC-PY-1
=====================================

RED tests for the new deque-based ``ColdWarmPromoter`` in
``src.core.idle_timeout`` (distinct from the old HTTP-based promoter
in ``src.core.cold_warm_promoter``).

Tests assert COLD→WARM promotion via the sliding window of N requests
in M minutes, **not** the legacy HTTP health-check promoter.

Covers:
  - Threshold-based promotion (5 requests in 10 min)
  - Counter increment tracking
  - Window expiry reset (11 min gap)
  - Configurable threshold via env var
  - WARM-service no-re-promotion
  - Tier registry update
  - Structured logging on promotion
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.core.idle_timeout import (
    ColdWarmPromoter,
)
from src.core.health_config import (
    SERVICE_TIERS,
    COLD_PROMOTION_REQUESTS,
    COLD_PROMOTION_WINDOW_SECS,
    reset_service_tiers,
)


# ── Helpers ─────────────────────────────────────────────────────────────

_FROZEN = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _dt(offset_minutes: int = 0) -> datetime:
    """Return *offset_minutes* past the frozen base time."""
    return _FROZEN.replace(tzinfo=timezone.utc) + __import__(
        "datetime"
    ).timedelta(minutes=offset_minutes)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_tiers() -> None:
    """Reset every test to a clean tier slate."""
    reset_service_tiers()


@pytest.fixture
def promoter() -> ColdWarmPromoter:
    """Fresh promoter per test — never share the global singleton."""
    return ColdWarmPromoter()


# ── Tests ───────────────────────────────────────────────────────────────


class TestColdToWarmPromotion:
    """HWC-PY-1: COLD→WARM promotion via sliding-window request counting."""

    def test_cold_to_warm_after_n_requests(self, promoter: ColdWarmPromoter) -> None:
        """5 tool dispatches to a Cold service within a 10-minute window
        promote the service to WARM."""
        service = "semantic-search"
        assert SERVICE_TIERS[service] == "COLD"

        for i in range(5):
            result = promoter.record_dispatch(service, _now=_dt(i))
            # Only the last call should return True
            if i < 4:
                assert result is False, f"Early promotion at request {i + 1}"
            else:
                assert result is True, "Promotion should fire at request 5"

        assert SERVICE_TIERS[service] == "WARM"

    def test_cold_request_counter_increments(self, promoter: ColdWarmPromoter) -> None:
        """Each dispatch to a Cold service increments the internal request
        counter for that service."""
        service = "code-orchestrator"
        assert SERVICE_TIERS[service] == "COLD"

        # Verify that repeated calls add up
        for i in range(3):
            promoter.record_dispatch(service, _now=_dt(i))
            # Inspect private deque to confirm entries
            win = promoter._request_windows[service]
            assert len(win) == i + 1, (
                f"Expected {i + 1} entries, got {len(win)}"
            )

    def test_cold_promotion_window_resets(self, promoter: ColdWarmPromoter) -> None:
        """4 dispatches, then an 11-minute gap, then 1 more:
        the window should have expired and promotion should NOT fire."""
        service = "audit-service"
        assert SERVICE_TIERS[service] == "COLD"

        # Dispatch 4 times at t=0, 2, 4, 6 (all within window)
        for i in range(4):
            result = promoter.record_dispatch(service, _now=_dt(i * 2))
            assert result is False, f"Unexpected promotion at request {i + 1}"

        # 11 minutes after the last entry (t=6+11=17) — all entries expired.
        # Cutoff at t=17-10=7. Entries at t=0,2,4,6 are all <7 → pruned.
        future = _dt(17)

        result = promoter.record_dispatch(service, _now=future)
        assert result is False, (
            "Should not promote after window expiry with only 1 new request"
        )

        # Window should now have only the single new entry (old 4 pruned)
        win = promoter._request_windows.get(service)
        assert win is not None
        assert len(win) == 1, (
            "Window should have 1 remaining entry (old entries pruned)"
        )

        assert SERVICE_TIERS[service] == "COLD", "Tier must remain COLD"

    def test_cold_promotion_configurable_threshold(
        self, promoter: ColdWarmPromoter
    ) -> None:
        """COLD_PROMOTION_REQUESTS=3 → promoted after 3 requests."""
        service = "llm-gateway"
        assert SERVICE_TIERS[service] == "COLD"

        with patch("src.core.health_config.COLD_PROMOTION_REQUESTS", 3):
            for i in range(3):
                result = promoter.record_dispatch(service, _now=_dt(i))
                if i < 2:
                    assert result is False
                else:
                    assert result is True

        assert SERVICE_TIERS[service] == "WARM"

    def test_warm_service_not_re_promoted(self, promoter: ColdWarmPromoter) -> None:
        """Service already WARM → N more requests do not fire promotion
        logic (returns False immediately)."""
        service = "ai-agents"
        # Pre-set to WARM
        SERVICE_TIERS[service] = "WARM"

        for i in range(10):
            result = promoter.record_dispatch(service, _now=_dt(i))
            assert result is False, (
                "WARM service must never return True from record_dispatch"
            )

        # Tier must remain WARM (not change to something else)
        assert SERVICE_TIERS[service] == "WARM"

    def test_cold_promotion_updates_tier_in_registry(
        self, promoter: ColdWarmPromoter
    ) -> None:
        """Promotion fires → SERVICE_TIERS dict updated to ``"WARM"`` for
        that service."""
        service = "context-management-service"
        assert SERVICE_TIERS[service] == "COLD"

        promoter.record_dispatch(service, _now=_dt(0))
        promoter.record_dispatch(service, _now=_dt(1))
        promoter.record_dispatch(service, _now=_dt(2))
        promoter.record_dispatch(service, _now=_dt(3))
        promoter.record_dispatch(service, _now=_dt(4))

        assert SERVICE_TIERS[service] == "WARM", (
            "SERVICE_TIERS must be updated to WARM after promotion"
        )

    def test_cold_promotion_logs_transition(
        self, promoter: ColdWarmPromoter, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Promotion fires → structured log entry with service name and
        request count."""
        service = "struct-analyzer-service"
        assert SERVICE_TIERS[service] == "COLD"

        caplog.set_level(logging.INFO)

        for i in range(5):
            promoter.record_dispatch(service, _now=_dt(i))

        # Check for the exact structured log message
        matching = [
            r for r in caplog.records
            if r.msg == "cold_to_warm_promotion"
        ]
        assert len(matching) == 1, (
            f"Expected exactly 1 'cold_to_warm_promotion' log, "
            f"got {len(matching)}"
        )
        record = matching[0]
        assert record.service == service, (
            f"Expected service={service}, got {record.service}"
        )
        # The request_count extra field should be present
        assert hasattr(record, "request_count"), (
            "Log record must include request_count extra field"
        )
