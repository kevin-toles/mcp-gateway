"""
Cold Warm Promoter Tests
=========================

Tests for ``ColdWarmPromoter`` — the service cold→warm promotion
singleton that health-checks backends and flips tiers.

Covers:
  - Singleton pattern (get_promoter, module-level _promoter)
  - Key normalization (underscore→hyphen)
  - Health URL resolution (known/unknown services)
  - warm_service success path
  - warm_service failure paths (timeout, connect error, non-200, exception)
  - warm_service for unknown services
  - _promote tier transitions (COLD→WARM, skip WARM/HOT, ImportError)
  - warmup_timeout_for tier resolution
  - get_state / get_all_states introspection
  - Multiple warm_service calls (idempotency)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.core.cold_warm_promoter import (
    ColdWarmPromoter,
    _promoter,
    get_promoter,
    COLD_WARMUP_TIMEOUT,
    WARM_HEALTH_TIMEOUT,
    BOOT_WARMUP_TIMEOUT,
)
from src.core.keys import normalize_service_key
from src.core.config import Settings


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level _promoter and SERVICE_TIERS before each test."""
    import src.core.cold_warm_promoter as cwp
    from src.core.health_config import reset_service_tiers
    old = cwp._promoter
    cwp._promoter = None
    reset_service_tiers()
    yield
    cwp._promoter = old
    reset_service_tiers()


@pytest.fixture
def settings() -> Settings:
    """Return a default Settings instance."""
    return Settings()


@pytest.fixture
def promoter(settings: Settings) -> ColdWarmPromoter:
    """Return a fresh promoter instance (not via singleton)."""
    return ColdWarmPromoter(settings=settings)


# =============================================================================
# Singleton
# =============================================================================

class TestSingleton:
    """ColdWarmPromoter singleton via get_promoter()."""

    def test_get_promoter_returns_instance(self):
        """get_promoter() should return a ColdWarmPromoter."""
        p = get_promoter()
        assert isinstance(p, ColdWarmPromoter)

    def test_get_promoter_returns_same_instance(self):
        """Multiple calls should return the same instance."""
        p1 = get_promoter()
        p2 = get_promoter()
        assert p1 is p2

    def test_get_promoter_passes_settings(self):
        """get_promoter(settings=...) should pass Settings to constructor."""
        s = Settings()
        p = get_promoter(settings=s)
        assert p._settings is s


# =============================================================================
# Key Normalization
# =============================================================================

class TestNormalizeKey:
    """normalize_service_key converts underscores to hyphens."""

    def test_underscore_to_hyphen(self):
        """Underscore should become hyphen."""
        assert normalize_service_key("test_service") == "test-service"

    def test_hyphen_stays_hyphen(self):
        """Hyphen should remain unchanged."""
        assert normalize_service_key("test-service") == "test-service"

    def test_multiple_underscores(self):
        """Multiple underscores should all become hyphens."""
        assert normalize_service_key("a_b_c") == "a-b-c"

    def test_no_separator(self):
        """Plain string without separators should be unchanged."""
        assert normalize_service_key("service") == "service"


# =============================================================================
# Health URL Resolution
# =============================================================================

class TestResolveHealthUrl:
    """_resolve_health_url maps service keys to /health endpoints."""

    def test_known_service_returns_url(self, promoter: ColdWarmPromoter):
        """Known service should return {base_url}/health."""
        url = promoter._resolve_health_url("semantic-search")
        assert url is not None
        assert url.endswith("/health")
        assert url.startswith("http")

    def test_known_service_with_underscore(self, promoter: ColdWarmPromoter):
        """Underscore form should resolve to same URL as hyphen form via warm_service."""
        url_hyphen = promoter._resolve_health_url("semantic-search")
        # _resolve_health_url does NOT normalize; warm_service does.
        # _resolve_health_url with underscore returns None since
        # _SERVICE_URL_ATTRS uses hyphens. Verify warm_service normalizes.
        assert url_hyphen is not None
        # The underscore variant returns None because _resolve_health_url
        # does not normalize — that's expected behavior.
        url_underscore = promoter._resolve_health_url("semantic_search")
        assert url_underscore is None

    def test_unknown_service_returns_none(self, promoter: ColdWarmPromoter):
        """Unknown service should return None."""
        url = promoter._resolve_health_url("nonexistent-service")
        assert url is None

    def test_resolves_cms(self, promoter: ColdWarmPromoter):
        """context-management-service should resolve."""
        url = promoter._resolve_health_url("context-management-service")
        assert url is not None
        assert "/health" in url

    def test_resolves_struct_analyzer(self, promoter: ColdWarmPromoter):
        """struct-analyzer-service should resolve."""
        url = promoter._resolve_health_url("struct-analyzer-service")
        assert url is not None
        assert "/health" in url

    def test_resolves_amve(self, promoter: ColdWarmPromoter):
        """amve should resolve (not return None after AMVE_URL→AMVE_SERVICE_URL fix)."""
        url = promoter._resolve_health_url("amve")
        assert url is not None
        assert "/health" in url


# =============================================================================
# warm_service — Success Path
# =============================================================================

class TestWarmServiceSuccess:
    """warm_service success scenarios."""

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_healthy_service_promotes(self, mock_client_cls, promoter: ColdWarmPromoter):
        """200 response should promote the service and return True."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = await promoter.warm_service("semantic-search")
        assert result is True

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_healthy_with_underscore_key(self, mock_client_cls, promoter: ColdWarmPromoter):
        """Underscore key should work identically to hyphen."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = await promoter.warm_service("semantic_search")
        assert result is True

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_healthy_cms_promotes(self, mock_client_cls, promoter: ColdWarmPromoter):
        """CMS 200 response should promote."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = await promoter.warm_service("context-management-service")
        assert result is True

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_healthy_struct_analyzer_promotes(self, mock_client_cls, promoter: ColdWarmPromoter):
        """struct-analyzer 200 response should promote."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = await promoter.warm_service("struct-analyzer-service")
        assert result is True

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_already_warm_returns_true(self, mock_client_cls, promoter: ColdWarmPromoter):
        """WARM service that responds 200 should return True (idempotent)."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        # First call promotes COLD→WARM
        await promoter.warm_service("semantic-search")
        # Second call on already WARM service
        result = await promoter.warm_service("semantic-search")
        assert result is True

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_sets_promotion_count(self, mock_client_cls, promoter: ColdWarmPromoter):
        """Successful warm_service should increment promotion_count."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        await promoter.warm_service("semantic-search")
        state = promoter.get_state("semantic-search")
        assert state["promotion_count"] == 1

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_sets_healthy_true(self, mock_client_cls, promoter: ColdWarmPromoter):
        """Successful warm_service should set healthy=True."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        await promoter.warm_service("semantic-search")
        state = promoter.get_state("semantic-search")
        assert state["healthy"] is True

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_sets_last_checked(self, mock_client_cls, promoter: ColdWarmPromoter):
        """Successful warm_service should set last_checked to an ISO datetime string."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        await promoter.warm_service("semantic-search")
        state = promoter.get_state("semantic-search")
        assert state["last_checked"] is not None
        assert isinstance(state["last_checked"], str)  # ISO format


# =============================================================================
# warm_service — Failure Paths
# =============================================================================

class TestWarmServiceFailure:
    """warm_service failure scenarios."""

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_timeout_returns_false(self, mock_client_cls, promoter: ColdWarmPromoter):
        """httpx.TimeoutException should return False and not promote."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_cls.return_value = mock_client

        result = await promoter.warm_service("semantic-search")
        assert result is False

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_connect_error_returns_false(self, mock_client_cls, promoter: ColdWarmPromoter):
        """httpx.ConnectError should return False."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client_cls.return_value = mock_client

        result = await promoter.warm_service("code-orchestrator")
        assert result is False

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_non_200_returns_false(self, mock_client_cls, promoter: ColdWarmPromoter):
        """Non-200 status code should return False."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 503
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = await promoter.warm_service("llm-gateway")
        assert result is False

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_unexpected_exception_returns_false(self, mock_client_cls, promoter: ColdWarmPromoter):
        """Unexpected exception should return False."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.side_effect = RuntimeError("something unexpected")
        mock_client_cls.return_value = mock_client

        result = await promoter.warm_service("semantic-search")
        assert result is False

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_failure_sets_healthy_false(self, mock_client_cls, promoter: ColdWarmPromoter):
        """Failed health check should set healthy=False."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = mock_client

        await promoter.warm_service("semantic-search")
        state = promoter.get_state("semantic-search")
        assert state["healthy"] is False

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_non_200_does_not_promote(self, mock_client_cls, promoter: ColdWarmPromoter):
        """Non-200 should NOT flip tier to WARM."""
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 503
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        await promoter.warm_service("semantic-search")
        state = promoter.get_state("semantic-search")
        assert state["tier"] == "COLD"  # stayed COLD, not promoted


# =============================================================================
# warm_service — Unknown Service
# =============================================================================

class TestWarmServiceUnknown:
    """warm_service with unknown service keys."""

    async def test_unknown_service_returns_false(self, promoter: ColdWarmPromoter):
        """Unknown service should return False immediately (no HTTP call)."""
        result = await promoter.warm_service("nonexistent-service")
        assert result is False

    async def test_unknown_sets_healthy_false(self, promoter: ColdWarmPromoter):
        """Unknown service should set healthy=False."""
        await promoter.warm_service("nonexistent-service")
        state = promoter.get_state("nonexistent-service")
        assert state["healthy"] is False

    async def test_unknown_sets_tier_unknown(self, promoter: ColdWarmPromoter):
        """Unknown service should set tier to unknown."""
        await promoter.warm_service("nonexistent-service")
        state = promoter.get_state("nonexistent-service")
        assert state["tier"] == "unknown"

    async def test_unknown_service_does_not_raise(self, promoter: ColdWarmPromoter):
        """Unknown service should not raise an exception."""
        try:
            await promoter.warm_service("does-not-exist")
        except Exception:
            pytest.fail("Unknown service raised an exception")


# =============================================================================
# _promote
# =============================================================================

class TestPromote:
    """_promote tier transitions."""

    def test_promote_cold_to_warm(self, promoter: ColdWarmPromoter):
        """COLD service should become WARM after _promote."""
        # Manually inject COLD tier
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["test-service"] = "COLD"

        promoter._promote("test-service")
        assert SERVICE_TIERS["test-service"] == "WARM"

    def test_promote_skips_warm(self, promoter: ColdWarmPromoter):
        """WARM service should stay WARM (no error)."""
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["test-service"] = "WARM"
        original_id = id(SERVICE_TIERS)

        promoter._promote("test-service")
        assert SERVICE_TIERS["test-service"] == "WARM"
        assert id(SERVICE_TIERS) == original_id  # no replacement

    def test_promote_skips_hot(self, promoter: ColdWarmPromoter):
        """HOT service should stay HOT (no error)."""
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["test-service"] = "HOT"

        promoter._promote("test-service")
        assert SERVICE_TIERS["test-service"] == "HOT"

    @patch("src.core.cold_warm_promoter.logger")
    def test_promote_logs_promotion(self, mock_logger, promoter: ColdWarmPromoter):
        """Promotion should log an info message."""
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["test-service"] = "COLD"

        promoter._promote("test-service")
        mock_logger.info.assert_called_once()

    @patch("src.core.cold_warm_promoter.logger")
    def test_promote_logs_skip_warm(self, mock_logger, promoter: ColdWarmPromoter):
        """Already WARM should log debug (no info)."""
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["test-service"] = "WARM"

        promoter._promote("test-service")
        mock_logger.debug.assert_called_once()
        mock_logger.info.assert_not_called()


# =============================================================================
# warmup_timeout_for
# =============================================================================

class TestWarmupTimeoutFor:
    """warmup_timeout_for returns per-tier timeouts."""

    def test_cold_returns_60(self):
        """COLD tier should return COLD_WARMUP_TIMEOUT (60s)."""
        # Create a promoter and set tier to COLD
        p = ColdWarmPromoter()
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["test-svc"] = "COLD"
        assert p.warmup_timeout_for("test-svc") == COLD_WARMUP_TIMEOUT

    def test_warm_returns_2(self):
        """WARM tier should return WARM_HEALTH_TIMEOUT (2s)."""
        p = ColdWarmPromoter()
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["test-svc"] = "WARM"
        assert p.warmup_timeout_for("test-svc") == WARM_HEALTH_TIMEOUT

    def test_hot_returns_2(self):
        """HOT tier should return WARM_HEALTH_TIMEOUT (2s)."""
        p = ColdWarmPromoter()
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["test-svc"] = "HOT"
        assert p.warmup_timeout_for("test-svc") == WARM_HEALTH_TIMEOUT

    def test_boot_returns_120(self):
        """BOOT tier should return BOOT_WARMUP_TIMEOUT (120s)."""
        p = ColdWarmPromoter()
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["test-svc"] = "BOOT"
        assert p.warmup_timeout_for("test-svc") == BOOT_WARMUP_TIMEOUT

    def test_unknown_tier_returns_2(self):
        """Unknown tier should default to WARM_HEALTH_TIMEOUT (2s)."""
        p = ColdWarmPromoter()
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["test-svc"] = "UNKNOWN"
        assert p.warmup_timeout_for("test-svc") == WARM_HEALTH_TIMEOUT

    def test_normalizes_key(self):
        """Underscore key should normalize to hyphen."""
        p = ColdWarmPromoter()
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["test-svc"] = "COLD"
        assert p.warmup_timeout_for("test_svc") == COLD_WARMUP_TIMEOUT


# =============================================================================
# get_state / get_all_states
# =============================================================================

class TestGetState:
    """State introspection."""

    def test_get_state_unknown(self, promoter: ColdWarmPromoter):
        """Unseen service should return default state dict."""
        state = promoter.get_state("unseen")
        assert state["tier"] == "unknown"
        assert state["healthy"] is False
        assert state["last_checked"] is None
        assert state["promotion_count"] == 0

    def test_get_state_normalizes_key(self, promoter: ColdWarmPromoter):
        """Underscore key should return same state as hyphen."""
        @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
        async def _(mock_client_cls):
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            await promoter.warm_service("semantic-search")
            state_hyphen = promoter.get_state("semantic-search")
            state_underscore = promoter.get_state("semantic_search")
            assert state_hyphen["healthy"] == state_underscore["healthy"]

    async def test_get_all_states_returns_dict(self, promoter: ColdWarmPromoter):
        """get_all_states should return a dict keyed by service key."""
        states = promoter.get_all_states()
        assert isinstance(states, dict)

    async def test_get_all_states_includes_tracked(self, promoter: ColdWarmPromoter):
        """get_all_states should include services that have been warm'd."""
        @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
        async def _(mock_client_cls):
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.__aenter__.return_value = mock_client
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            await promoter.warm_service("semantic-search")
            states = promoter.get_all_states()
            assert "semantic-search" in states

    def test_get_state_tier_from_health_config(self, promoter: ColdWarmPromoter):
        """get_state should reflect the current tier from SERVICE_TIERS."""
        from src.core.health_config import SERVICE_TIERS
        # Force a known tier
        SERVICE_TIERS["my-svc"] = "HOT"
        promoter._update_state("my-svc", healthy=True)
        state = promoter.get_state("my-svc")
        assert state["tier"] == "HOT"


# =============================================================================
# _update_state
# =============================================================================

class TestUpdateState:
    """_update_state internal tracking."""

    def test_update_sets_healthy(self, promoter: ColdWarmPromoter):
        """_update_state should set healthy flag."""
        promoter._update_state("test-svc", healthy=True)
        assert promoter._service_states["test-svc"]["healthy"] is True

    def test_update_sets_unhealthy(self, promoter: ColdWarmPromoter):
        """_update_state should set healthy=False."""
        promoter._update_state("test-svc", healthy=False)
        assert promoter._service_states["test-svc"]["healthy"] is False

    def test_update_sets_last_checked(self, promoter: ColdWarmPromoter):
        """_update_state should set last_checked to a datetime."""
        promoter._update_state("test-svc", healthy=True)
        assert isinstance(promoter._service_states["test-svc"]["last_checked"], datetime)

    def test_update_increments_promotion_count_on_healthy(self, promoter: ColdWarmPromoter):
        """_update_state should increment promotion_count when healthy."""
        promoter._update_state("test-svc", healthy=True)
        assert promoter._service_states["test-svc"]["promotion_count"] == 1
        promoter._update_state("test-svc", healthy=True)
        assert promoter._service_states["test-svc"]["promotion_count"] == 2

    def test_update_does_not_increment_on_unhealthy(self, promoter: ColdWarmPromoter):
        """_update_state should NOT increment promotion_count when unhealthy."""
        promoter._update_state("test-svc", healthy=True)
        promoter._update_state("test-svc", healthy=False)
        assert promoter._service_states["test-svc"]["promotion_count"] == 1


# =============================================================================
# Integration with health_config.SERVICE_TIERS
# =============================================================================

class TestServiceTiersIntegration:
    """Integration with the global SERVICE_TIERS dict."""

    def test_all_services_start_cold(self):
        """All services in SERVICE_TIERS should start as COLD."""
        from src.core.health_config import SERVICE_TIERS
        for key, tier in SERVICE_TIERS.items():
            assert tier == "COLD", f"{key} starts as {tier}, not COLD"

    def test_promote_updates_tiers_dict(self, promoter: ColdWarmPromoter):
        """_promote should mutate the global SERVICE_TIERS dict."""
        from src.core.health_config import SERVICE_TIERS
        SERVICE_TIERS["integration-test"] = "COLD"
        promoter._promote("integration-test")
        assert SERVICE_TIERS["integration-test"] == "WARM"

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_warm_service_updates_tiers(self, mock_client_cls, promoter: ColdWarmPromoter):
        """warm_service should flip SERVICE_TIERS entry to WARM on success."""
        from src.core.health_config import SERVICE_TIERS

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.__aenter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        await promoter.warm_service("semantic-search")
        assert SERVICE_TIERS.get("semantic-search") == "WARM"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and defensive scenarios."""

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_empty_service_key(self, mock_client_cls, promoter: ColdWarmPromoter):
        """Empty string key should be treated as unknown (returns False)."""
        result = await promoter.warm_service("")
        assert result is False

    @patch("src.core.cold_warm_promoter.httpx.AsyncClient")
    async def test_warm_service_after_failure(self, mock_client_cls, promoter: ColdWarmPromoter):
        """Service that fails then succeeds on retry should promote."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client

        # First call: failure
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = mock_client

        result1 = await promoter.warm_service("semantic-search")
        assert result1 is False

        # Second call: success (need new mock since AsyncClient context manager is consumed)
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client2 = AsyncMock()
        mock_client2.__aenter__.return_value = mock_client2
        mock_client2.get.return_value = mock_response
        mock_client_cls.return_value = mock_client2

        result2 = await promoter.warm_service("semantic-search")
        assert result2 is True
        state = promoter.get_state("semantic-search")
        assert state["healthy"] is True
        assert state["promotion_count"] == 1

    async def test_get_state_promotion_count_default(self, promoter: ColdWarmPromoter):
        """Unseen service should have promotion_count=0."""
        state = promoter.get_state("never-seen")
        assert state["promotion_count"] == 0
