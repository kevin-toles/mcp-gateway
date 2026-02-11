"""Async circuit breaker — C-5 resilience pattern.

Implements the standard three-state circuit breaker:

    CLOSED  →  (failure_threshold exceeded)  →  OPEN
    OPEN    →  (recovery_timeout elapsed)    →  HALF_OPEN
    HALF_OPEN → (probe succeeds)             →  CLOSED
    HALF_OPEN → (probe fails)                →  OPEN

Each backend service gets its own ``CircuitBreaker`` instance via
``CircuitBreakerRegistry``.  This prevents a single failing backend
from consuming retry budget across all tools.

Reference: PLATFORM_CONSOLIDATED_ISSUES C-5
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open.

    Attributes:
        backend_name: Friendly name of the failing backend.
        retry_after: Seconds until the circuit transitions to HALF_OPEN.
    """

    def __init__(self, backend_name: str, retry_after: float) -> None:
        self.backend_name = backend_name
        self.retry_after = max(0.0, retry_after)
        super().__init__(f"Circuit open for '{backend_name}' — retry after {self.retry_after:.1f}s")


class CircuitBreaker:
    """Async-safe circuit breaker for a single backend.

    Args:
        name:               Human-readable backend name (for logging/errors).
        failure_threshold:  Consecutive failures before opening the circuit.
        recovery_timeout:   Seconds the circuit stays OPEN before probing.
        half_open_max:      Max concurrent probes in HALF_OPEN state.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.total_rejections = 0
        self.total_successes = 0

    # ── Public properties ────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        """Return the current state, auto-transitioning OPEN → HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    # ── Core call wrapper ────────────────────────────────────────────

    async def pre_check(self) -> None:
        """Check whether a call is allowed; raise if circuit is open.

        Must be called **before** the actual HTTP dispatch.
        """
        async with self._lock:
            current = self.state

            if current == CircuitState.OPEN:
                retry_after = self.recovery_timeout - (time.monotonic() - self._last_failure_time)
                self.total_rejections += 1
                raise CircuitOpenError(self.name, retry_after)

            if current == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max:
                    self.total_rejections += 1
                    raise CircuitOpenError(self.name, 1.0)
                self._half_open_calls += 1

            self.total_calls += 1

    async def on_success(self) -> None:
        """Record a successful call — close the circuit if probing."""
        async with self._lock:
            self.total_successes += 1
            if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                # Probe succeeded — back to CLOSED
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
            else:
                # Normal success in CLOSED state
                self._failure_count = 0

    async def on_failure(self) -> None:
        """Record a failed call — potentially open the circuit."""
        async with self._lock:
            self._failure_count += 1
            self.total_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — reopen
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    async def reset(self) -> None:
        """Force-reset the circuit breaker to CLOSED state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot for health/metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "total_successes": self.total_successes,
        }


class CircuitBreakerRegistry:
    """Manages per-backend ``CircuitBreaker`` instances.

    Usage::

        registry = CircuitBreakerRegistry(threshold=5, recovery=30.0)
        cb = registry.get("semantic-search")
        await cb.pre_check()
        # ... dispatch ...
        await cb.on_success()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 1,
    ) -> None:
        self._threshold = failure_threshold
        self._recovery = recovery_timeout
        self._half_open_max = half_open_max
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, backend_name: str) -> CircuitBreaker:
        """Return (or create) the circuit breaker for *backend_name*."""
        if backend_name not in self._breakers:
            self._breakers[backend_name] = CircuitBreaker(
                name=backend_name,
                failure_threshold=self._threshold,
                recovery_timeout=self._recovery,
                half_open_max=self._half_open_max,
            )
        return self._breakers[backend_name]

    def all_snapshots(self) -> list[dict]:
        """Return snapshots for every registered breaker."""
        return [cb.snapshot() for cb in self._breakers.values()]

    async def reset_all(self) -> None:
        """Reset every circuit breaker to CLOSED."""
        for cb in self._breakers.values():
            await cb.reset()
