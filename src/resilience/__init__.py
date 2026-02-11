"""Resilience patterns — circuit breaker and retry for backend dispatch.

Implements C-5 (Missing Circuit Breakers Across All HTTP Clients).
Provides per-backend circuit breakers and exponential-backoff retry
to protect the mcp-gateway from cascading backend failures.
"""

from src.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitState",
]
