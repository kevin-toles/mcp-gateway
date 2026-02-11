"""ToolDispatcher — WBS-MCP2 (GREEN), C-5 (circuit breakers + retry).

HTTP dispatch to backend services.  Each platform tool maps to a
``DispatchRoute`` containing a backend base_url, path, and per-tool
timeout.  ``ToolDispatcher.dispatch()`` sends an HTTP request with
per-backend circuit breaker protection and exponential-backoff retry.

Reference: Strategy §4.1 (dispatcher pattern), AC-2.1 through AC-2.5
           C-5 (Missing Circuit Breakers Across All HTTP Clients)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from src.core.config import Settings
from src.core.errors import BackendUnavailableError, CircuitOpenError, ToolTimeoutError
from src.resilience.circuit_breaker import CircuitBreakerRegistry

logger = logging.getLogger(__name__)

# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class DispatchRoute:
    """Route definition for a single tool.

    Attributes:
        base_url: Backend service origin (e.g. ``http://localhost:8081``).
        path:     API path on that service  (e.g. ``/v1/search``).
        timeout:  Per-tool timeout in seconds (default 30).
    """

    base_url: str
    path: str
    timeout: float = 30.0


@dataclass
class DispatchResult:
    """Structured response from a backend dispatch.

    Attributes:
        status_code: HTTP status code from the backend.
        body:        Parsed JSON body (empty dict if no body).
        headers:     Response headers as a plain dict.
        elapsed_ms:  Round-trip time in milliseconds.
    """

    status_code: int
    body: dict
    headers: dict
    elapsed_ms: float


# ── Route table builder ────────────────────────────────────────────────

# Maps tool_name → human-friendly service name (for error messages)
_TOOL_SERVICE_NAMES: dict[str, str] = {
    "semantic_search": "semantic-search",
    "hybrid_search": "semantic-search",
    "code_analyze": "code-orchestrator",
    "code_pattern_audit": "code-orchestrator",
    "graph_query": "semantic-search",
    "llm_complete": "llm-gateway",
    "a2a_send_message": "ai-agents",
    "a2a_get_task": "ai-agents",
    "a2a_cancel_task": "ai-agents",
}


def _build_routes(settings: Settings) -> dict[str, DispatchRoute]:
    """Build the full route table from Settings backend URLs.

    Each entry uses the default 30s timeout.  Per-tool overrides can be
    added later without breaking the route table structure.
    """
    return {
        "semantic_search": DispatchRoute(
            base_url=settings.SEMANTIC_SEARCH_URL,
            path="/v1/search",
        ),
        "hybrid_search": DispatchRoute(
            base_url=settings.SEMANTIC_SEARCH_URL,
            path="/v1/search/hybrid",
        ),
        "code_analyze": DispatchRoute(
            base_url=settings.CODE_ORCHESTRATOR_URL,
            path="/v1/analyze",
        ),
        "code_pattern_audit": DispatchRoute(
            base_url=settings.CODE_ORCHESTRATOR_URL,
            path="/v1/audit/patterns",
        ),
        "graph_query": DispatchRoute(
            base_url=settings.SEMANTIC_SEARCH_URL,
            path="/v1/graph/query",
        ),
        "llm_complete": DispatchRoute(
            base_url=settings.LLM_GATEWAY_URL,
            path="/v1/completions",
        ),
        "a2a_send_message": DispatchRoute(
            base_url=settings.AI_AGENTS_URL,
            path="/a2a/v1/message:send",
        ),
        "a2a_get_task": DispatchRoute(
            base_url=settings.AI_AGENTS_URL,
            path="/a2a/v1/tasks",  # /{task_id} appended at dispatch time
        ),
        "a2a_cancel_task": DispatchRoute(
            base_url=settings.AI_AGENTS_URL,
            path="/a2a/v1/tasks",  # /{task_id}:cancel appended at dispatch time
        ),
    }


# ── Dispatcher ──────────────────────────────────────────────────────────


class ToolDispatcher:
    """Dispatches tool calls to backend platform services via HTTP.

    Uses a single ``httpx.AsyncClient`` per unique backend base_url for
    connection pooling (MCP2.11 REFACTOR).  Each backend is protected by
    a circuit breaker, and transient failures are retried with exponential
    backoff (C-5).

    Args:
        settings: Application settings containing backend URLs and
                  resilience configuration.
    """

    # HTTP status codes that are safe to retry
    _RETRYABLE_STATUS_CODES = frozenset({502, 503, 504, 429})

    def __init__(self, settings: Settings) -> None:
        self.routes: dict[str, DispatchRoute] = _build_routes(settings)
        # Connection pool: one AsyncClient per unique backend base_url
        self._clients: dict[str, httpx.AsyncClient] = {}
        # Legacy single-client attribute — tests can override this
        self._client: httpx.AsyncClient | None = None

        # C-5: Per-backend circuit breakers
        self._cb_registry = CircuitBreakerRegistry(
            failure_threshold=settings.CIRCUIT_BREAKER_THRESHOLD,
            recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_SECONDS,
        )
        self._max_retries = settings.DISPATCH_MAX_RETRIES
        self._retry_base_delay = settings.DISPATCH_RETRY_BASE_DELAY

    def _get_client(self, base_url: str) -> httpx.AsyncClient:
        """Return a pooled ``AsyncClient`` for *base_url*.

        If ``_client`` has been explicitly set (e.g. by tests injecting
        a mock transport), that client is used for all backends.
        """
        if self._client is not None:
            return self._client
        if base_url not in self._clients:
            self._clients[base_url] = httpx.AsyncClient(base_url=base_url)
        return self._clients[base_url]

    def get_route(self, tool_name: str) -> DispatchRoute | None:
        """Return the ``DispatchRoute`` for *tool_name*, or ``None``."""
        return self.routes.get(tool_name)

    async def dispatch(
        self,
        tool_name: str,
        payload: dict,
        *,
        method: str = "POST",
        path_override: str | None = None,
    ) -> DispatchResult:
        """Send *payload* to the backend for *tool_name*.

        Includes per-backend circuit breaker protection and exponential
        backoff retry for transient failures (C-5).

        Args:
            tool_name: Registered tool name from the route table.
            payload: JSON-serializable request body (ignored for GET).
            method: HTTP method — ``"POST"`` (default) or ``"GET"``.
            path_override: If given, replaces the route's default path.

        Returns:
            A ``DispatchResult`` with status, body, headers, and timing.

        Raises:
            ValueError: If *tool_name* is not in the route table.
            ToolTimeoutError: If the request exceeds the per-tool timeout.
            BackendUnavailableError: If the backend cannot be reached.
            CircuitOpenError: If the circuit breaker rejects the call.
        """
        route = self.get_route(tool_name)
        if route is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        path = path_override or route.path
        url = f"{route.base_url}{path}"
        service_name = _TOOL_SERVICE_NAMES.get(tool_name, tool_name)
        client = self._get_client(route.base_url)

        # C-5: Get the circuit breaker for this backend
        cb = self._cb_registry.get(service_name)

        last_exc: Exception | None = None
        attempts = 1 + self._max_retries  # 1 initial + N retries

        for attempt in range(attempts):
            # Circuit breaker pre-check (raises CircuitOpenError if open)
            try:
                await cb.pre_check()
            except Exception as exc:
                # Wrap circuit open errors for structured error response
                from src.resilience.circuit_breaker import (
                    CircuitOpenError as _CBOpenError,
                )

                if isinstance(exc, _CBOpenError):
                    raise CircuitOpenError(exc.backend_name, exc.retry_after) from exc
                raise

            start = time.monotonic()
            try:
                if method.upper() == "GET":
                    response = await client.get(
                        url,
                        params=payload or None,
                        timeout=route.timeout,
                    )
                else:
                    response = await client.post(
                        url,
                        json=payload,
                        timeout=route.timeout,
                    )

                elapsed_ms = (time.monotonic() - start) * 1000

                # Check for retryable HTTP status codes
                if response.status_code in self._RETRYABLE_STATUS_CODES:
                    await cb.on_failure()
                    last_exc = BackendUnavailableError(
                        service_name,
                        f"HTTP {response.status_code}",
                    )
                    if attempt < attempts - 1:
                        delay = self._retry_base_delay * (2**attempt)
                        logger.warning(
                            "Retryable status %d from %s (attempt %d/%d), retrying in %.1fs",
                            response.status_code,
                            service_name,
                            attempt + 1,
                            attempts,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    # Exhausted retries
                    raise last_exc

                # Success path
                await cb.on_success()

                try:
                    body = response.json()
                except Exception:
                    body = {}

                return DispatchResult(
                    status_code=response.status_code,
                    body=body,
                    headers=dict(response.headers),
                    elapsed_ms=round(elapsed_ms, 2),
                )

            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
                await cb.on_failure()
                last_exc = ToolTimeoutError(tool_name, route.timeout)
                if attempt < attempts - 1:
                    delay = self._retry_base_delay * (2**attempt)
                    logger.warning(
                        "Timeout for %s (attempt %d/%d), retrying in %.1fs",
                        tool_name,
                        attempt + 1,
                        attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_exc

            except (httpx.ConnectError, httpx.ConnectTimeout):
                await cb.on_failure()
                last_exc = BackendUnavailableError(service_name, "Connection failed")
                if attempt < attempts - 1:
                    delay = self._retry_base_delay * (2**attempt)
                    logger.warning(
                        "Connection failed to %s (attempt %d/%d), retrying in %.1fs",
                        service_name,
                        attempt + 1,
                        attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_exc

        # Should not reach here, but just in case
        if last_exc is not None:
            raise last_exc
        raise BackendUnavailableError(service_name, "All retry attempts exhausted")

    @property
    def circuit_breakers(self) -> CircuitBreakerRegistry:
        """Expose circuit breaker registry for health/metrics endpoints."""
        return self._cb_registry

    async def close(self) -> None:
        """Close all pooled httpx clients."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
