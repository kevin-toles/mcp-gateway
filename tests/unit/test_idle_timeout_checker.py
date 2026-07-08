"""
Idle Timeout Checker
====================

Tests for the WarmColdActuator, ColdWarmPromoter, and singleton accessors
from src.core.idle_timeout_checker and src.core.idle_timeout.

RED phase (GAP-1): Verify importability, callability, and structural wiring.

Coverage:
  - get_checker() is importable
  - get_checker() returns a callable WarmColdActuator instance
  - IdleTimeoutTracker callback registration wiring
  - ColdWarmPromoter is importable and instantiable
  - WarmColdActuator is importable and instantiable
"""

import pytest

from src.core.idle_timeout_checker import get_checker, reset_checker, WarmColdActuator
from src.core.idle_timeout import (
    ColdWarmPromoter,
    get_warm_cold_actuator,
    reset_warm_cold_actuator,
    get_tracker,
    reset_tracker,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons before each test to avoid cross-test pollution."""
    reset_tracker()
    reset_checker()
    reset_warm_cold_actuator()
    yield
    reset_tracker()
    reset_checker()
    reset_warm_cold_actuator()


# =============================================================================
# get_checker — importable & callable
# =============================================================================

class TestCheckerImportable:
    """GAP-1 RED: get_checker() importable."""

    def test_get_checker_is_importable(self):
        """get_checker should be importable from idle_timeout_checker."""
        from src.core.idle_timeout_checker import get_checker as _gc
        assert _gc is not None

    def test_get_checker_returns_callable(self):
        """get_checker() should return a WarmColdActuator instance."""
        checker = get_checker()
        assert checker is not None
        assert callable(checker.start)
        assert callable(checker.stop)
        assert hasattr(checker, "is_running")


# =============================================================================
# get_checker — callable
# =============================================================================

class TestCheckerCallable:
    """GAP-1 RED: get_checker() returns a running instance."""

    @pytest.mark.asyncio
    async def test_checker_start_stop_cycle(self):
        """Checker should start and stop cleanly."""
        checker = get_checker()
        assert not checker.is_running

        await checker.start()
        assert checker.is_running

        await checker.stop()
        assert not checker.is_running

    @pytest.mark.asyncio
    async def test_checker_start_is_idempotent(self):
        """Starting an already-running checker should not raise."""
        checker = get_checker()
        await checker.start()
        await checker.start()  # second start — should warn, not crash
        assert checker.is_running
        await checker.stop()

    @pytest.mark.asyncio
    async def test_checker_stop_is_idempotent(self):
        """Stopping a non-running checker should not raise."""
        checker = get_checker()
        await checker.stop()  # never started — should be no-op
        assert not checker.is_running


# =============================================================================
# IdleTimeoutTracker callback registration
# =============================================================================

class TestCallbackRegistration:
    """GAP-1 RED: idle callback registration is wired correctly."""

    def test_register_idle_callback_is_callable(self):
        """get_tracker().register_idle_callback should be callable."""
        tracker = get_tracker()
        assert callable(tracker.register_idle_callback)

    def test_on_service_idle_is_importable(self):
        """on_service_idle should be importable."""
        from src.core.idle_timeout import on_service_idle
        assert callable(on_service_idle)


# =============================================================================
# ColdWarmPromoter — importable
# =============================================================================

class TestColdWarmPromoterImportable:
    """GAP-1 RED: ColdWarmPromoter is importable and instantiable."""

    def test_cold_warm_promoter_class_importable(self):
        """ColdWarmPromoter should be importable."""
        from src.core.idle_timeout import ColdWarmPromoter
        assert ColdWarmPromoter is not None

    def test_cold_warm_promoter_instantiable(self):
        """ColdWarmPromoter should be instantiable."""
        promoter = ColdWarmPromoter()
        assert promoter is not None
        assert callable(promoter.record_dispatch)


# =============================================================================
# WarmColdActuator — importable
# =============================================================================

class TestWarmColdActuatorImportable:
    """GAP-1 RED: WarmColdActuator is importable and instantiable."""

    def test_warm_cold_actuator_class_importable(self):
        """WarmColdActuator should be importable from idle_timeout."""
        from src.core.idle_timeout import WarmColdActuator
        assert WarmColdActuator is not None

    def test_warm_cold_actuator_instantiable(self):
        """WarmColdActuator should be instantiable with default params."""
        from src.core.idle_timeout import WarmColdActuator
        actuator = WarmColdActuator()
        assert actuator is not None
        assert callable(actuator.check_and_demote)

    def test_checker_actuator_class_importable(self):
        """WarmColdActuator should also be importable from idle_timeout_checker."""
        from src.core.idle_timeout_checker import WarmColdActuator
        assert WarmColdActuator is not None

    def test_checker_actuator_instantiable(self):
        """WarmColdActuator from checker should be instantiable."""
        actuator = WarmColdActuator()
        assert actuator is not None
        assert callable(actuator.start)
        assert callable(actuator.stop)
        assert hasattr(actuator, "is_running")
