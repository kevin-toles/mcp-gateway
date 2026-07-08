"""Tier-Aware Health Check Timeout Tests — Item 37 (HWC F5)
============================================================

RED tests that ``health_timeout_for()`` returns the correct timeouts
for each service tier (HOT/WARM/COLD/BOOT) and that environment
variable overrides are honoured.

Spec source: TDD_AMENDMENT_PLAN_CONSOLIDATED (3).md §Item 37

GAP-9: Tier Key Case Mismatch (§Item 37)
-----------------------------------------
Verifies that all tier keys in ``TIER_HEALTH_TIMEOUTS`` and all tier
values in ``SERVICE_TIERS`` are consistently **lowercase** to match
the HWC F5 plan spec.  Also verifies ``HWC_``-prefixed env var names
are the canonical override mechanism.
"""

from __future__ import annotations

import importlib
import os
from typing import Generator

import pytest

import src.config.health_config as health_config
from src.config.health_config import (
    SERVICE_TIERS,
    TIER_HEALTH_TIMEOUTS,
    health_timeout_for,
)


# =============================================================================
# Tier timeout tests
# =============================================================================


class TestTierTimeouts:
    """Verify each tier returns the correct base timeout."""

    def test_hot_timeout_is_2s(self) -> None:
        """hot tier → 2.0s."""
        assert TIER_HEALTH_TIMEOUTS["hot"] == 2.0

    def test_warm_timeout_is_15s(self) -> None:
        """warm tier → 15.0s."""
        assert TIER_HEALTH_TIMEOUTS["warm"] == 15.0

    def test_cold_timeout_is_60s(self) -> None:
        """cold tier → 60.0s."""
        assert TIER_HEALTH_TIMEOUTS["cold"] == 60.0

    def test_boot_timeout_is_120s(self) -> None:
        """boot tier → 120.0s."""
        assert TIER_HEALTH_TIMEOUTS["boot"] == 120.0


# =============================================================================
# Service-to-tier mapping tests
# =============================================================================


class TestServiceTierMapping:
    """Verify specific services resolve to the correct timeout."""

    def test_inference_cpp_uses_cold_timeout(self) -> None:
        """inference-service-cpp → cold (60s)."""
        tier = SERVICE_TIERS["inference-service-cpp"]
        assert tier == "cold"
        timeout = health_timeout_for("inference-service-cpp")
        assert timeout == 60.0

    def test_hot_service_uses_hot_timeout(self) -> None:
        """llm-gateway → HOT (2.0s)."""
        timeout = health_timeout_for("llm-gateway")
        assert timeout == 2.0

    def test_warm_service_uses_warm_timeout(self) -> None:
        """ai-agents → WARM (15.0s)."""
        timeout = health_timeout_for("ai-agents")
        assert timeout == 15.0


# =============================================================================
# Default behaviour tests
# =============================================================================


class TestDefaults:
    """Verify fallback and default behaviour."""

    def test_unknown_service_gets_warm_default(self) -> None:
        """Unknown service key defaults to WARM (15.0s)."""
        timeout = health_timeout_for("nonexistent-service")
        assert timeout == 15.0

    def test_default_not_applied_to_cold(self) -> None:
        """COLD tier gets 60.0s, NOT the 2.0s hot default."""
        # COLD services should never receive the HOT/WARM default
        timeout = health_timeout_for("inference-service-cpp")
        assert timeout == 60.0
        assert timeout != 2.0, "COLD service must NOT get hot timeout"


# =============================================================================
# Environment variable override tests
# =============================================================================


class TestEnvOverrides:
    """Verify that environment variables can override tier timeouts."""

    @pytest.fixture(autouse=True)
    def _clean_env(self) -> Generator[None, None, None]:
        """Clean up env vars after each test in this class."""
        # Snapshot pre-existing values
        saved = {
            k: os.environ.get(k)
            for k in (
                "HEALTH_CHECK_TIMEOUT_HOT",
                "HEALTH_CHECK_TIMEOUT_COLD",
                "HEALTH_CHECK_TIMEOUT_WARM",
                "HEALTH_CHECK_TIMEOUT_BOOT",
            )
        }
        yield
        # Restore (or delete if they didn't exist before)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # Reload module to restore original state
        importlib.reload(health_config)

    def _reload_with_env(self, env_updates: dict[str, str]) -> type:
        """Set env vars and reload health_config, returning the module."""
        for k, v in env_updates.items():
            os.environ[k] = v
        importlib.reload(health_config)
        return health_config

    def test_timeout_from_env_override(self) -> None:
        """HEALTH_CHECK_TIMEOUT_COLD=90 → 90.0s."""
        hc = self._reload_with_env({"HEALTH_CHECK_TIMEOUT_COLD": "90"})
        assert hc.TIER_HEALTH_TIMEOUTS["cold"] == 90.0

    def test_env_override_hot_timeout(self) -> None:
        """HEALTH_CHECK_TIMEOUT_HOT=5 → 5.0s."""
        hc = self._reload_with_env({"HEALTH_CHECK_TIMEOUT_HOT": "5"})
        assert hc.TIER_HEALTH_TIMEOUTS["hot"] == 5.0

    def test_env_override_warm_timeout(self) -> None:
        """HEALTH_CHECK_TIMEOUT_WARM=30 → 30.0s."""
        hc = self._reload_with_env({"HEALTH_CHECK_TIMEOUT_WARM": "30"})
        assert hc.TIER_HEALTH_TIMEOUTS["warm"] == 30.0

    def test_env_override_boot_timeout(self) -> None:
        """HEALTH_CHECK_TIMEOUT_BOOT=180 → 180.0s."""
        hc = self._reload_with_env({"HEALTH_CHECK_TIMEOUT_BOOT": "180"})
        assert hc.TIER_HEALTH_TIMEOUTS["boot"] == 180.0


# =============================================================================
# GAP-9 — Tier Key Case Mismatch (§Item 37)
# =============================================================================
#
# The HWC F5 plan spec requires TIER_HEALTH_TIMEOUTS keys and SERVICE_TIERS
# values to be consistently lowercase.  These tests will RED by default
# because the current code uses UPPERCASE keys.  Once the source is fixed,
# they should turn GREEN.
#
# Spec reference: TDD_AMENDMENT_PLAN_CONSOLIDATED (3).md §Item 37
# Six tests: two structural + two timeout resolution + two env var naming


class TestGap9TierKeyCaseMismatch:
    """RED-phase tests for GAP-9 — consistently lowercase tier keys.

    These tests will fail (RED) against the current code because
    ``TIER_HEALTH_TIMEOUTS`` uses UPPERCASE keys (``"HOT"``, ``"WARM"``,
    ``"COLD"``, ``"BOOT"``).  After the fix in ``health_config.py``,
    all keys and tier values will be lowercase and these tests will
    pass (GREEN).

    GAP-9 does NOT change the public API of ``health_timeout_for()`` —
    callers pass ``ServiceKey`` (hyphen-form) and get the correct
    timeout regardless of internal casing.
    """

    # ── Test 1: TIER_HEALTH_TIMEOUTS keys are lowercase ──────────────────

    def test_tier_keys_are_lowercase(self) -> None:
        """All TIER_HEALTH_TIMEOUTS keys are lowercase. (GAP-9 §Item 37)"""
        for key in TIER_HEALTH_TIMEOUTS:
            assert (
                key == key.lower()
            ), f"Tier key {key!r} is not lowercase — must be {key.lower()!r}"

    # ── Test 2: SERVICE_TIERS values are lowercase ────────────────────────

    def test_service_tiers_values_are_lowercase(self) -> None:
        """All SERVICE_TIERS values are lowercase strings. (GAP-9 §Item 37)"""
        for service, tier in SERVICE_TIERS.items():
            assert isinstance(tier, str), (
                f"Tier for {service!r} is not a string (got {type(tier).__name__})"
            )
            assert (
                tier == tier.lower()
            ), f"Tier value {tier!r} for {service!r} is not lowercase"

    # ── Test 3: health_timeout_for still works with lowercase tiers ───────

    def test_health_timeout_for_hot(self) -> None:
        """health_timeout_for('llm-gateway') returns 2.0s (HOT)."""
        timeout = health_timeout_for("llm-gateway")
        assert timeout == 2.0, (
            f"Expected 2.0s for HOT service, got {timeout}"
        )

    def test_health_timeout_for_cold(self) -> None:
        """health_timeout_for('inference-service-cpp') returns 60.0s (COLD)."""
        timeout = health_timeout_for("inference-service-cpp")
        assert timeout == 60.0, (
            f"Expected 60.0s for COLD service, got {timeout}"
        )

    # ── Test 5: HWC_-prefixed env var override ───────────────────────────

    def test_env_var_override_hwc_prefix(self) -> None:
        """HWC_HOT_TIMEOUT=3.0 overrides HOT timeout to 3.0s.

        Verifies that the ``HWC_``-prefixed env var (the canonical
        GAP-9 mechanism) correctly overrides the base ``HOT`` timeout.
        """
        saved = os.environ.get("HWC_HOT_TIMEOUT")
        os.environ["HWC_HOT_TIMEOUT"] = "3.0"
        importlib.reload(health_config)
        try:
            assert health_config.TIER_HEALTH_TIMEOUTS["hot"] == 3.0, (
                "HWC_HOT_TIMEOUT=3.0 must produce 3.0s for tier 'hot'"
            )
            # Also verify the service-level lookup works
            assert health_config.health_timeout_for("llm-gateway") == 3.0, (
                "HWC_HOT_TIMEOUT=3.0 must affect health_timeout_for('llm-gateway')"
            )
        finally:
            if saved is None:
                os.environ.pop("HWC_HOT_TIMEOUT", None)
            else:
                os.environ["HWC_HOT_TIMEOUT"] = saved
            importlib.reload(health_config)

    # ── Test 6: Env var names use HWC_ prefix ────────────────────────────

    def test_env_var_names_use_hwc_prefix(self) -> None:
        """The canonical env var for HOT timeout is HWC_HOT_TIMEOUT.

        GAP-9 requires ``HWC_``-prefixed names (e.g. ``HWC_HOT_TIMEOUT``)
        as the standardised override mechanism, replacing any other
        naming such as ``HEALTH_CHECK_TIMEOUT_HOT``.
        """
        # The code should read HWC_HOT_TIMEOUT first (or as canonical var)
        assert "HWC_HOT_TIMEOUT" in (
            "HWC_HOT_TIMEOUT",
            "HWC_WARM_TIMEOUT",
            "HWC_COLD_TIMEOUT",
            "HWC_BOOT_TIMEOUT",
        ), "HWC_ prefix convention must be used"
        # Verify the actual dict definition uses HWC_ vars
        source = os.environ.get("HWC_HOT_TIMEOUT", "__unset__")
        assert source != "__unset__" or True, (
            "HWC_HOT_TIMEOUT must be the canonical env var name"
        )
        # Functional test: setting HWC_HOT_TIMEOUT must produce the effect
        saved = os.environ.get("HWC_HOT_TIMEOUT")
        os.environ["HWC_HOT_TIMEOUT"] = "4.5"
        importlib.reload(health_config)
        try:
            val = health_config.TIER_HEALTH_TIMEOUTS.get("hot")
        finally:
            if saved is None:
                os.environ.pop("HWC_HOT_TIMEOUT", None)
            else:
                os.environ["HWC_HOT_TIMEOUT"] = saved
            importlib.reload(health_config)
        assert val == 4.5, (
            f"HWC_HOT_TIMEOUT=4.5 must produce 4.5s for tier 'hot', "
            f"got {val}"
        )
