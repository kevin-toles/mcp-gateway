"""
ServiceKey Type Tests — HWC F1
===============================

RED tests for the ``make_service_key()`` function in
``src.core.service_key``.

Tests assert:
  - Underscore→hyphen normalization
  - URI format with port
  - Idempotency (double-wrapping)
  - Lookup with underscore finds hyphen-keyed entry
  - All ``_TOOL_SERVICE_NAMES`` values are valid
  - All ``SERVICE_SHUTDOWN_COMMANDS`` keys are valid
  - ``IdleTimeoutTracker.record_request()`` uses normalized key

Spec source: TDD_AMENDMENT_PLAN_CONSOLIDATED (3).md §HWC F1
"""

from __future__ import annotations

import pytest

from src.core.service_key import make_service_key
from src.core.config import settings


# ── Test 1: Underscore→hyphen normalization ────────────────────────────


class TestNormalization:
    """ServiceKey normalization behaviour."""

    def test_normalizes_underscore_to_hyphen(self) -> None:
        """Underscores become hyphens."""
        assert make_service_key("semantic_search") == "semantic-search"

    def test_uri_format_with_port(self) -> None:
        """With port=8081 → '{name}:{port}'."""
        assert make_service_key("semantic-search", port=8081) == "semantic-search:8081"

    def test_idempotent(self) -> None:
        """make_service_key(make_service_key(x)) == make_service_key(x)."""
        once = make_service_key("a_b")
        twice = make_service_key(once)
        assert twice == once
        assert twice == "a-b"

    def test_lookup_underscore_matches_hyphen(self) -> None:
        """A registry lookup with underscore key finds a hyphen-keyed entry."""
        registry = {"semantic-search": "value"}
        key = make_service_key("semantic_search")
        assert registry[key] == "value"


# ── Test 2: Consumer validation ────────────────────────────────────────


class TestConsumerValidation:
    """Validate that existing consumers use valid make_service_key() values."""

    def test_in_tool_service_names(self) -> None:
        """All values in _TOOL_SERVICE_NAMES are valid make_service_key() instances."""
        # Import here to avoid circular import at module level
        from src.tool_dispatcher import _TOOL_SERVICE_NAMES  # type: ignore[private-code]

        for name, svc in _TOOL_SERVICE_NAMES.items():
            assert isinstance(svc, str), f"{name} value is not a str: {type(svc)}"
            # Double-wrapping must be idempotent
            assert make_service_key(svc) == svc, (
                f"{name}={svc!r} is not in canonical form"
            )
            # No underscores
            assert "_" not in svc, f"{name}={svc!r} contains underscores"

    def test_in_shutdown_commands(self) -> None:
        """All keys in SERVICE_SHUTDOWN_COMMANDS are valid make_service_key() instances."""
        for key in settings.SERVICE_SHUTDOWN_COMMANDS:
            assert isinstance(key, str), f"Shutdown key is not a str: {type(key)}"
            assert make_service_key(key) == key, (
                f"Shutdown key {key!r} is not in canonical form"
            )
            assert "_" not in key, f"Shutdown key {key!r} contains underscores"

    def test_in_idle_tracker(self) -> None:
        """IdleTimeoutTracker.record_request(make_service_key(...)) uses normalized key."""
        from src.core.idle_timeout import get_tracker

        tracker = get_tracker()
        # Record with underscore key
        raw_key = "code_orchestrator"
        canonical = make_service_key(raw_key)

        tracker.record_request(canonical)

        # After recording, the internal state must use hyphen form
        state = tracker._service_states  # type: ignore[private-member]
        assert "_" not in state, (
            f"Tracker state contains underscore keys: {list(state.keys())}"
        )
        assert "code-orchestrator" in state, (
            f"Expected hyphen key 'code-orchestrator' in state, got {list(state.keys())}"
        )
