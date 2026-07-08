"""
GAP-1 RED: Tests for idle_timeout_checker.py startup blocker.

File: tests/test_idle_timeout_checker.py (new)
"""

import pytest


class TestStartIdleTimeoutCheckerImportable:
    """from src.core.idle_timeout_checker import start_idle_timeout_checker succeeds."""

    def test_start_idle_timeout_checker_importable(self):
        from src.core.idle_timeout_checker import start_idle_timeout_checker
        assert start_idle_timeout_checker is not None


class TestStartIdleTimeoutCheckerCallable:
    """start_idle_timeout_checker() is callable and returns without raising."""

    def test_start_idle_timeout_checker_callable(self):
        from src.core.idle_timeout_checker import start_idle_timeout_checker
        assert callable(start_idle_timeout_checker)
        result = start_idle_timeout_checker()
        # Should return without raising — no further assertion needed


class TestIdleTimeoutCheckerRegistersCallback:
    """After start_idle_timeout_checker(), IdleTimeoutTracker has at least one callback."""

    def test_idle_timeout_checker_registers_callback(self):
        from src.core.idle_timeout_checker import start_idle_timeout_checker
        from src.core.idle_timeout import get_tracker, reset_tracker
        import src.core.idle_timeout_checker as _checker_mod

        # Reset both the module flag and the tracker so we can verify
        # that start_idle_timeout_checker() actually registers a callback.
        # NOTE: We must set the module attribute directly (not via from X import)
        # because 'from X import' creates a local binding; assigning = False
        # would rebind the local name, not the module's global.
        reset_tracker()
        _checker_mod._IDLE_CHECKER_CALLBACK_REGISTERED = False
        tracker = get_tracker()
        assert len(tracker._on_idle_callbacks) == 0

        start_idle_timeout_checker()

        assert len(tracker._on_idle_callbacks) >= 1
        reset_tracker()


class TestColdWarmPromoterImportable:
    """from src.core.idle_timeout_checker import ColdWarmPromoter succeeds."""

    def test_cold_warm_promoter_importable(self):
        from src.core.idle_timeout_checker import ColdWarmPromoter
        assert ColdWarmPromoter is not None


class TestWarmColdActuatorImportable:
    """from src.core.idle_timeout_checker import WarmColdActuator succeeds."""

    def test_warm_cold_actuator_importable(self):
        from src.core.idle_timeout_checker import WarmColdActuator
        assert WarmColdActuator is not None
