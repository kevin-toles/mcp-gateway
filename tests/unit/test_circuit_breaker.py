"""Tests for C-5 circuit breaker and retry resilience in mcp-gateway.

Covers:
- CircuitBreaker state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
- CircuitBreakerRegistry per-backend isolation
- ToolDispatcher retry with exponential backoff
- ToolDispatcher circuit breaker integration
- CircuitOpenError in StructuredErrorResponse
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from src.core.config import Settings
from src.core.errors import CircuitOpenError, StructuredErrorResponse
from src.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
)
from src.resilience.circuit_breaker import (
    CircuitOpenError as CBOpenError,
)
from src.tool_dispatcher import ToolDispatcher

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CircuitBreaker State Machine Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCircuitBreakerStates:
    """Test the three-state circuit breaker transitions."""

    async def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test-backend")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    async def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        await cb.on_failure()
        await cb.on_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 2

    async def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            await cb.on_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

    async def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        await cb.on_failure()
        await cb.on_failure()
        await cb.on_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    async def test_open_rejects_calls(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60.0)
        await cb.on_failure()
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CBOpenError, match="Circuit open for 'test'"):
            await cb.pre_check()
        assert cb.total_rejections == 1

    async def test_open_transitions_to_half_open_after_recovery(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        await cb.on_failure()
        assert cb._state == CircuitState.OPEN
        await asyncio.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    async def test_half_open_probe_success_closes(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        await cb.on_failure()
        await asyncio.sleep(0.02)
        # Should allow one probe
        await cb.pre_check()
        await cb.on_success()
        assert cb.state == CircuitState.CLOSED

    async def test_half_open_probe_failure_reopens(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        await cb.on_failure()
        await asyncio.sleep(0.02)
        await cb.pre_check()
        await cb.on_failure()
        assert cb._state == CircuitState.OPEN

    async def test_half_open_limits_concurrent_probes(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01, half_open_max=1)
        await cb.on_failure()
        await asyncio.sleep(0.02)
        await cb.pre_check()  # First probe allowed
        with pytest.raises(CBOpenError):
            await cb.pre_check()  # Second probe blocked

    async def test_force_reset(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        await cb.on_failure()
        assert cb.state == CircuitState.OPEN
        await cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerMetrics:
    """Test the metrics/snapshot reporting."""

    async def test_snapshot_structure(self):
        cb = CircuitBreaker("my-backend")
        snap = cb.snapshot()
        assert snap["name"] == "my-backend"
        assert snap["state"] == "closed"
        assert snap["failure_count"] == 0
        assert snap["total_calls"] == 0

    async def test_metrics_track_correctly(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        await cb.pre_check()
        await cb.on_success()
        await cb.pre_check()
        await cb.on_failure()
        assert cb.total_calls == 2
        assert cb.total_successes == 1
        assert cb.total_failures == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CircuitBreakerRegistry Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCircuitBreakerRegistry:
    """Test per-backend isolation and registry management."""

    def test_creates_breakers_on_demand(self):
        reg = CircuitBreakerRegistry()
        cb = reg.get("semantic-search")
        assert isinstance(cb, CircuitBreaker)
        assert cb.name == "semantic-search"

    def test_returns_same_breaker_for_same_name(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get("semantic-search")
        cb2 = reg.get("semantic-search")
        assert cb1 is cb2

    def test_different_names_get_different_breakers(self):
        reg = CircuitBreakerRegistry()
        cb1 = reg.get("semantic-search")
        cb2 = reg.get("code-orchestrator")
        assert cb1 is not cb2

    def test_all_snapshots(self):
        reg = CircuitBreakerRegistry()
        reg.get("a")
        reg.get("b")
        snaps = reg.all_snapshots()
        assert len(snaps) == 2
        assert {s["name"] for s in snaps} == {"a", "b"}

    async def test_reset_all(self):
        reg = CircuitBreakerRegistry(failure_threshold=1)
        cb1 = reg.get("a")
        cb2 = reg.get("b")
        await cb1.on_failure()
        await cb2.on_failure()
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN
        await reg.reset_all()
        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED

    def test_passes_config_to_breakers(self):
        reg = CircuitBreakerRegistry(failure_threshold=10, recovery_timeout=60.0)
        cb = reg.get("test")
        assert cb.failure_threshold == 10
        assert cb.recovery_timeout == 60.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ToolDispatcher Retry + Circuit Breaker Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestDispatcherRetry:
    """Test retry with exponential backoff on transient failures."""

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings(
            DISPATCH_MAX_RETRIES=2,
            DISPATCH_RETRY_BASE_DELAY=0.01,  # Fast tests
            CIRCUIT_BREAKER_THRESHOLD=10,  # High so CB doesn't interfere
        )

    async def test_retries_on_timeout(self, settings: Settings):
        """Timeout on first attempt, success on second."""
        dispatcher = ToolDispatcher(settings)
        call_count = 0

        async def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ReadTimeout("read timed out")
            return httpx.Response(200, json={"ok": True})

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = mock_post
        dispatcher._client = mock_client

        result = await dispatcher.dispatch("semantic_search", {"query": "test"})
        assert result.status_code == 200
        assert call_count == 2

    async def test_retries_on_connect_error(self, settings: Settings):
        """Connection error on first attempt, success on retry."""
        dispatcher = ToolDispatcher(settings)
        call_count = 0

        async def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            return httpx.Response(200, json={"ok": True})

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = mock_post
        dispatcher._client = mock_client

        result = await dispatcher.dispatch("semantic_search", {"query": "test"})
        assert result.status_code == 200
        assert call_count == 2

    async def test_retries_on_503(self, settings: Settings):
        """503 on first attempt, success on retry."""
        dispatcher = ToolDispatcher(settings)
        call_count = 0

        async def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(503, json={"error": "unavailable"})
            return httpx.Response(200, json={"ok": True})

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = mock_post
        dispatcher._client = mock_client

        result = await dispatcher.dispatch("semantic_search", {"query": "test"})
        assert result.status_code == 200
        assert call_count == 2

    async def test_exhausts_retries_then_raises(self, settings: Settings):
        """All attempts fail — should raise after exhausting retries."""
        dispatcher = ToolDispatcher(settings)

        async def always_timeout(url, **kwargs):
            raise httpx.ReadTimeout("read timed out")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = always_timeout
        dispatcher._client = mock_client

        from src.core.errors import ToolTimeoutError

        with pytest.raises(ToolTimeoutError):
            await dispatcher.dispatch("semantic_search", {"query": "test"})

    async def test_no_retry_on_4xx(self, settings: Settings):
        """Client errors (4xx) should NOT be retried."""
        dispatcher = ToolDispatcher(settings)
        call_count = 0

        async def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, json={"error": "bad request"})

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = mock_post
        dispatcher._client = mock_client

        result = await dispatcher.dispatch("semantic_search", {"query": "test"})
        assert result.status_code == 400
        assert call_count == 1  # No retries


class TestDispatcherCircuitBreaker:
    """Test circuit breaker integration in ToolDispatcher."""

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings(
            CIRCUIT_BREAKER_THRESHOLD=2,
            CIRCUIT_BREAKER_RECOVERY_SECONDS=60.0,
            DISPATCH_MAX_RETRIES=0,  # No retries — test pure CB behavior
            DISPATCH_RETRY_BASE_DELAY=0.01,
        )

    async def test_circuit_opens_after_threshold_failures(self, settings: Settings):
        dispatcher = ToolDispatcher(settings)

        async def always_fail(url, **kwargs):
            raise httpx.ConnectError("refused")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = always_fail
        dispatcher._client = mock_client

        # Fail twice to trip the circuit (threshold=2)
        from src.core.errors import BackendUnavailableError

        for _ in range(2):
            with pytest.raises(BackendUnavailableError):
                await dispatcher.dispatch("semantic_search", {"query": "test"})

        # Third call should be rejected by circuit breaker
        with pytest.raises(CircuitOpenError, match="Circuit open"):
            await dispatcher.dispatch("semantic_search", {"query": "test"})

    async def test_different_backends_have_independent_breakers(self, settings: Settings):
        dispatcher = ToolDispatcher(settings)

        async def always_fail(url, **kwargs):
            raise httpx.ConnectError("refused")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = always_fail
        dispatcher._client = mock_client

        from src.core.errors import BackendUnavailableError

        # Trip semantic-search circuit
        for _ in range(2):
            with pytest.raises(BackendUnavailableError):
                await dispatcher.dispatch("semantic_search", {"query": "test"})

        # semantic-search should be open
        with pytest.raises(CircuitOpenError):
            await dispatcher.dispatch("semantic_search", {"query": "test"})

        # code-orchestrator should still be CLOSED
        with pytest.raises(BackendUnavailableError):
            await dispatcher.dispatch("code_analyze", {"code": "x"})

    async def test_circuit_breakers_exposed_via_property(self, settings: Settings):
        dispatcher = ToolDispatcher(settings)
        assert isinstance(dispatcher.circuit_breakers, CircuitBreakerRegistry)
        snaps = dispatcher.circuit_breakers.all_snapshots()
        assert isinstance(snaps, list)


class TestDispatcherRetryWithCircuitBreaker:
    """Test retry + circuit breaker working together."""

    async def test_retries_count_toward_circuit_breaker(self):
        settings = Settings(
            CIRCUIT_BREAKER_THRESHOLD=3,
            DISPATCH_MAX_RETRIES=2,
            DISPATCH_RETRY_BASE_DELAY=0.01,
        )
        dispatcher = ToolDispatcher(settings)

        async def always_fail(url, **kwargs):
            raise httpx.ConnectError("refused")

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = always_fail
        dispatcher._client = mock_client

        from src.core.errors import BackendUnavailableError

        # First dispatch: 1 initial + 2 retries = 3 failures → trips circuit
        with pytest.raises(BackendUnavailableError):
            await dispatcher.dispatch("semantic_search", {"query": "test"})

        # Circuit should now be open (3 failures = threshold)
        with pytest.raises(CircuitOpenError):
            await dispatcher.dispatch("semantic_search", {"query": "test"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Error Response Integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCircuitOpenErrorResponse:
    """Test CircuitOpenError maps to structured error responses."""

    def test_circuit_open_error_code(self):
        exc = CircuitOpenError("semantic-search", 15.0)
        resp = StructuredErrorResponse.from_exception(exc, "req-123")
        assert resp.code == "CIRCUIT_OPEN"
        assert "semantic-search" in resp.error
        assert resp.request_id == "req-123"

    def test_circuit_open_error_attributes(self):
        exc = CircuitOpenError("code-orchestrator", 25.5)
        assert exc.backend_name == "code-orchestrator"
        assert exc.retry_after == 25.5

    def test_circuit_open_negative_retry_clamped(self):
        exc = CircuitOpenError("test", -5.0)
        assert exc.retry_after == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Settings Defaults
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestResilienceSettings:
    """Verify resilience config has sane defaults."""

    def test_default_circuit_breaker_threshold(self):
        s = Settings()
        assert s.CIRCUIT_BREAKER_THRESHOLD == 5

    def test_default_circuit_breaker_recovery(self):
        s = Settings()
        assert s.CIRCUIT_BREAKER_RECOVERY_SECONDS == 30.0

    def test_default_retry_config(self):
        s = Settings()
        assert s.DISPATCH_MAX_RETRIES == 2
        assert s.DISPATCH_RETRY_BASE_DELAY == 0.5
