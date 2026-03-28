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
import uuid
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
    timeout: float | None = 30.0


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
    "graph_traverse": "semantic-search",
    # Issue #6: consolidated KB tools
    "knowledge_search": "semantic-search",
    "knowledge_refine": "semantic-search",
    "pattern_search": "semantic-search",
    "code_analyze": "code-orchestrator",
    "code_pattern_audit": "code-orchestrator",
    "graph_query": "semantic-search",
    "llm_complete": "llm-gateway",
    "a2a_send_message": "ai-agents",
    "a2a_get_task": "ai-agents",
    "a2a_cancel_task": "ai-agents",
    # Workflow tools (WBS-WF6)
    "convert_pdf": "code-orchestrator",
    "extract_book_metadata": "code-orchestrator",
    "batch_extract_metadata": "code-orchestrator",
    "generate_taxonomy": "code-orchestrator",
    "enrich_book_metadata": "code-orchestrator",
    "batch_enrich_metadata": "code-orchestrator",
    "enhance_guideline": "ai-agents",
    # Taxonomy Analysis (WBS-TAP9)
    "analyze_taxonomy_coverage": "code-orchestrator",
    # AMVE tools (AEI-7)
    "amve_detect_patterns": "amve",
    "amve_detect_boundaries": "amve",
    "amve_detect_communication": "amve",
    "amve_build_call_graph": "amve",
    "amve_evaluate_fitness": "amve",
    "amve_generate_architecture_log": "amve",
    # AEI-17: Dead code detection
    "amve_detect_dead_code": "amve",
    # Phase 2: Content-Addressed Snapshot Store tools
    "amve_extract_architecture": "amve",
    "amve_detect_drift": "amve",
    # Audit Service (WBS-AEI13)
    "audit_security_scan": "audit-service",
    "audit_code_metrics": "audit-service",
    "audit_corpus_search": "audit-service",
    # AEI-18: Dependency assessment
    "audit_dependency_assess": "audit-service",
    # AEI-20: Resolution lookup
    "audit_resolve_lookup": "audit-service",
    # AEI-23: VRE quarantine tools
    "audit_search_exploits": "audit-service",
    "audit_search_cves": "audit-service",
    # WBS-F7: Foundation search (scientific / theoretical layer)
    "foundation_search": "unified-search-service",
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
        # Issue #6: consolidated KB tools — all route to USS /v1/search/hybrid
        "knowledge_search": DispatchRoute(
            base_url=settings.SEMANTIC_SEARCH_URL,
            path="/v1/search/hybrid",
        ),
        "knowledge_refine": DispatchRoute(
            base_url=settings.SEMANTIC_SEARCH_URL,
            path="/v1/search/hybrid",
        ),
        "pattern_search": DispatchRoute(
            base_url=settings.SEMANTIC_SEARCH_URL,
            path="/v1/search/hybrid",
        ),
        # diagram_search: routes to USS /v1/search/hybrid with collection=ascii_diagrams.
        # USS detects the CLIP collection and uses CLIPEncoder.encode_text() instead of MiniLM.
        "diagram_search": DispatchRoute(
            base_url=settings.SEMANTIC_SEARCH_URL,
            path="/v1/search/hybrid",
        ),
        "code_analyze": DispatchRoute(
            base_url=settings.AUDIT_SERVICE_URL,
            path="/v1/patterns/detect",
        ),
        "code_pattern_audit": DispatchRoute(
            base_url=settings.AUDIT_SERVICE_URL,
            path="/v1/patterns/detect",
        ),
        "graph_query": DispatchRoute(
            base_url=settings.SEMANTIC_SEARCH_URL,
            path="/v1/graph/query",
        ),
        "graph_traverse": DispatchRoute(
            base_url=settings.SEMANTIC_SEARCH_URL,
            path="/v1/graph/traverse",
        ),
        "llm_complete": DispatchRoute(
            base_url=settings.LLM_GATEWAY_URL,
            path="/v1/chat/completions",
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
        # Workflow tools (WBS-WF6) — no timeout (large scanned PDFs can exceed 900s with OCR)
        "convert_pdf": DispatchRoute(
            base_url=settings.CODE_ORCHESTRATOR_URL,
            path="/api/v1/workflows/convert-pdf",
            timeout=None,
        ),
        "extract_book_metadata": DispatchRoute(
            base_url=settings.CODE_ORCHESTRATOR_URL,
            path="/api/v1/workflows/extract-book",
            timeout=None,  # No timeout — matches original script
        ),
        "batch_extract_metadata": DispatchRoute(
            base_url=settings.CODE_ORCHESTRATOR_URL,
            path="/api/v1/workflows/extract-book",
            timeout=None,  # No timeout — batch loops per-book via extract_book_metadata route
        ),
        "generate_taxonomy": DispatchRoute(
            base_url=settings.CODE_ORCHESTRATOR_URL,
            path="/api/v1/workflows/generate-taxonomy-from-enriched",
            timeout=300.0,
        ),
        "enrich_book_metadata": DispatchRoute(
            base_url=settings.CODE_ORCHESTRATOR_URL,
            path="/api/v1/workflows/enrich-book",
            timeout=300.0,
        ),
        "batch_enrich_metadata": DispatchRoute(
            base_url=settings.CODE_ORCHESTRATOR_URL,
            path="/api/v1/workflows/batch-enrich",
            timeout=None,  # No timeout — batch can take 90+ min for 554 books
        ),
        "enhance_guideline": DispatchRoute(
            base_url=settings.AI_AGENTS_URL,
            path="/v1/workflows/enhance-guideline",
            timeout=300.0,
        ),
        # Taxonomy Analysis (WBS-TAP9)
        "analyze_taxonomy_coverage": DispatchRoute(
            base_url=settings.CODE_ORCHESTRATOR_URL,
            path="/api/v1/workflows/analyze-taxonomy-coverage",
            timeout=300.0,
        ),
        # AMVE tools (AEI-7)
        "amve_detect_patterns": DispatchRoute(
            base_url=settings.AMVE_SERVICE_URL,
            path="/v1/analysis/patterns",
        ),
        "amve_detect_boundaries": DispatchRoute(
            base_url=settings.AMVE_SERVICE_URL,
            path="/v1/analysis/boundaries",
        ),
        "amve_detect_communication": DispatchRoute(
            base_url=settings.AMVE_SERVICE_URL,
            path="/v1/analysis/communication",
        ),
        "amve_build_call_graph": DispatchRoute(
            base_url=settings.AMVE_SERVICE_URL,
            path="/v1/analysis/call-graph",
        ),
        "amve_evaluate_fitness": DispatchRoute(
            base_url=settings.AMVE_SERVICE_URL,
            path="/v1/fitness/evaluate",
        ),
        "amve_generate_architecture_log": DispatchRoute(
            base_url=settings.AMVE_SERVICE_URL,
            path="/v1/architecture/batch-scan",
        ),
        # AEI-17: Dead code detection
        "amve_detect_dead_code": DispatchRoute(
            base_url=settings.AMVE_SERVICE_URL,
            path="/v1/analysis/dead-code",
        ),
        # Phase 2: Content-Addressed Snapshot Store tools (G2.12 GREEN)
        "amve_extract_architecture": DispatchRoute(
            base_url=settings.AMVE_SERVICE_URL,
            path="/v1/architecture/extract",
        ),
        "amve_detect_drift": DispatchRoute(
            base_url=settings.AMVE_SERVICE_URL,
            path="/v1/architecture/drift",
        ),
        # Audit Service (WBS-AEI13)
        "audit_security_scan": DispatchRoute(
            base_url=settings.AUDIT_SERVICE_URL,
            path="/v1/audit/security",
        ),
        "audit_code_metrics": DispatchRoute(
            base_url=settings.AUDIT_SERVICE_URL,
            path="/v1/audit/metrics",
        ),
        "audit_corpus_search": DispatchRoute(
            base_url=settings.AUDIT_SERVICE_URL,
            path="/v1/audit/corpus",
        ),
        # AEI-18: Dependency assessment
        "audit_dependency_assess": DispatchRoute(
            base_url=settings.AUDIT_SERVICE_URL,
            path="/v1/audit/dependency",
        ),
        # AEI-20: Resolution lookup
        "audit_resolve_lookup": DispatchRoute(
            base_url=settings.AUDIT_SERVICE_URL,
            path="/v1/audit/resolve",
        ),
        # AEI-23: VRE quarantine tools
        "audit_search_exploits": DispatchRoute(
            base_url=settings.AUDIT_SERVICE_URL,
            path="/v1/audit/exploits",
        ),
        "audit_search_cves": DispatchRoute(
            base_url=settings.AUDIT_SERVICE_URL,
            path="/v1/audit/cves",
        ),
        # Phase 7: Quality audit (pattern compliance + anti-patterns)
        "audit_quality_scan": DispatchRoute(
            base_url=settings.AUDIT_SERVICE_URL,
            path="/v1/audit/quality",
        ),
        # WBS-F7: Foundation search (scientific / theoretical layer)
        "foundation_search": DispatchRoute(
            base_url=settings.SEMANTIC_SEARCH_URL,
            path="/v1/search/foundation",
        ),
    }


# ── Helpers ────────────────────────────────────────────────────────────


def _get_identity_headers(
    request_headers: dict | None,
    *,
    enabled: bool,
    auth_enabled: bool,
) -> dict | None:
    """Return X-Tenant-ID and X-Agent-ID headers for identity propagation.

    When ``enabled`` is ``False`` returns ``None`` — no headers added,
    behaviour is byte-for-bit identical to pre-Phase-3 state (AC-3.4).

    When ``enabled`` is ``True`` and ``auth_enabled`` is ``False``, logs a
    WARNING and falls back to ``tenant_id='anonymous'`` (AC-3.2).

    When ``enabled`` is ``True`` and ``auth_enabled`` is ``True``, extracts
    ``X-Tenant-ID`` (or ``'anonymous'``) and ``X-Agent-ID`` (or ``'unknown'``)
    from *request_headers* (AC-3.3).

    G3.4 (GREEN) — extracted helper mirror to ``_get_or_create_session_id``.
    """
    if not enabled:
        return None
    if not auth_enabled:
        logger.warning(
            "IDENTITY_PROPAGATION=true but AUTH_ENABLED=false — treating all requests as tenant_id='anonymous'"
        )
        tenant_id = "anonymous"
    else:
        headers = request_headers or {}
        tenant_id = headers.get("x-tenant-id") or headers.get("X-Tenant-ID") or "anonymous"
    agent_id = "unknown"
    if request_headers:
        agent_id = request_headers.get("x-agent-id") or request_headers.get("X-Agent-ID") or "unknown"
    return {"x-tenant-id": tenant_id, "x-agent-id": agent_id}


def _get_or_create_session_id(
    request_headers: dict | None,
    *,
    enabled: bool,
) -> str | None:
    """Return an X-Session-ID value to forward to backend services.

    When ``enabled`` is ``False`` returns ``None`` — no header is added and
    behaviour is byte-for-bit identical to the pre-Phase-1 state (AC-1.4).

    When ``enabled`` is ``True``:
    - propagates the existing ``x-session-id`` / ``X-Session-ID`` value from
      *request_headers* when present (AC-1.3, propagation path), or
    - generates a fresh UUID4 string when absent (AC-1.3, generation path).

    G1.7 (REFACTOR) — extracted from ``ToolDispatcher.dispatch()`` so the
    logic can be tested and reused independently.
    """
    if not enabled:
        return None
    if request_headers:
        # Accept both lower-case and title-case header names
        session_id = request_headers.get("x-session-id") or request_headers.get("X-Session-ID")
        if session_id:
            return session_id
    return str(uuid.uuid4())


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
        self._settings = settings
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

    async def _check_circuit_breaker(self, cb) -> None:
        """Pre-check the circuit breaker; translate open-circuit errors."""
        try:
            await cb.pre_check()
        except Exception as exc:
            from src.resilience.circuit_breaker import (
                CircuitOpenError as _CBOpenError,
            )

            if isinstance(exc, _CBOpenError):
                raise CircuitOpenError(exc.backend_name, exc.retry_after) from exc
            raise

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        payload: dict,
        timeout: float | None,
        extra_headers: dict | None = None,
    ) -> httpx.Response:
        """Send a single HTTP request (GET or POST)."""
        headers = extra_headers or None
        if method.upper() == "GET":
            return await client.get(url, params=payload or None, timeout=timeout, headers=headers)
        return await client.post(url, json=payload, timeout=timeout, headers=headers)

    async def _retry_delay(
        self,
        attempt: int,
        attempts: int,
        label: str,
        reason: str,
    ) -> None:
        """Log a warning and sleep for exponential backoff."""
        delay = self._retry_base_delay * (2**attempt)
        logger.warning(
            "%s for %s (attempt %d/%d), retrying in %.1fs",
            reason,
            label,
            attempt + 1,
            attempts,
            delay,
        )
        await asyncio.sleep(delay)

    def _parse_body(self, response: httpx.Response) -> dict:
        """Safely parse a JSON response body."""
        try:
            return response.json()
        except Exception:
            return {}

    async def dispatch(
        self,
        tool_name: str,
        payload: dict,
        *,
        method: str = "POST",
        path_override: str | None = None,
        request_headers: dict | None = None,
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
        cb = self._cb_registry.get(service_name)

        # G1.4 (GREEN) — Phase 1: Session Correlation
        # Build extra_headers with X-Session-ID when CORRELATION_ENABLED.
        # When flag is False the helper returns None and nothing is added.
        session_id = _get_or_create_session_id(request_headers, enabled=self._settings.CORRELATION_ENABLED)
        extra_headers: dict | None = {"x-session-id": session_id} if session_id else None

        # G3.6 (GREEN) — Phase 3: Multi-Tenant Identity Propagation
        # Merge X-Tenant-ID + X-Agent-ID into extra_headers when flag enabled.
        identity_headers = _get_identity_headers(
            request_headers,
            enabled=self._settings.IDENTITY_PROPAGATION,
            auth_enabled=self._settings.AUTH_ENABLED,
        )
        if identity_headers:
            extra_headers = {**(extra_headers or {}), **identity_headers}

        last_exc: Exception | None = None
        attempts = 1 + self._max_retries

        for attempt in range(attempts):
            await self._check_circuit_breaker(cb)
            result = await self._attempt_dispatch(
                client,
                cb,
                method,
                url,
                payload,
                route,
                tool_name,
                service_name,
                attempt,
                attempts,
                extra_headers=extra_headers,
            )
            if result is not None:
                return result

        if last_exc is not None:
            raise last_exc
        raise BackendUnavailableError(service_name, "All retry attempts exhausted")

    async def _attempt_dispatch(
        self,
        client: httpx.AsyncClient,
        cb,
        method: str,
        url: str,
        payload: dict,
        route: DispatchRoute,
        tool_name: str,
        service_name: str,
        attempt: int,
        attempts: int,
        extra_headers: dict | None = None,
    ) -> DispatchResult | None:
        """Execute a single dispatch attempt; return result or None to retry."""
        start = time.monotonic()
        try:
            response = await self._send_request(
                client, method, url, payload, route.timeout, extra_headers=extra_headers
            )
            elapsed_ms = (time.monotonic() - start) * 1000

            if response.status_code in self._RETRYABLE_STATUS_CODES:
                return await self._handle_retryable_status(
                    cb,
                    service_name,
                    response.status_code,
                    attempt,
                    attempts,
                )

            await cb.on_success()
            return DispatchResult(
                status_code=response.status_code,
                body=self._parse_body(response),
                headers=dict(response.headers),
                elapsed_ms=round(elapsed_ms, 2),
            )

        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
            await cb.on_failure()
            exc = ToolTimeoutError(tool_name, route.timeout or 0.0)
            if attempt < attempts - 1:
                await self._retry_delay(attempt, attempts, tool_name, "Timeout")
                return None
            raise exc from None

        except (httpx.ConnectError, httpx.ConnectTimeout):
            await cb.on_failure()
            exc = BackendUnavailableError(service_name, "Connection failed")
            if attempt < attempts - 1:
                await self._retry_delay(attempt, attempts, service_name, "Connection failed")
                return None
            raise exc from None

    async def _handle_retryable_status(
        self,
        cb,
        service_name: str,
        status_code: int,
        attempt: int,
        attempts: int,
    ) -> None:
        """Handle a retryable HTTP status code; raise on final attempt."""
        await cb.on_failure()
        exc = BackendUnavailableError(service_name, f"HTTP {status_code}")
        if attempt < attempts - 1:
            await self._retry_delay(
                attempt,
                attempts,
                service_name,
                f"Retryable status {status_code}",
            )
            return None
        raise exc

    @property
    def circuit_breakers(self) -> CircuitBreakerRegistry:
        """Expose circuit breaker registry for health/metrics endpoints."""
        return self._cb_registry

    async def dispatch_stream(
        self,
        tool_name: str,
        payload: dict,
        *,
        method: str = "POST",
        params: dict | None = None,
    ):
        """Dispatch and yield raw SSE lines from a streaming backend response.

        Uses the same route table and circuit breakers as ``dispatch()``,
        but reads the response as a stream, yielding each line as it arrives.

        Yields:
            Raw text lines from the SSE response (e.g. ``event: chapter\\n``).
        """
        route = self.get_route(tool_name)
        if route is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        url = f"{route.base_url}{route.path}"
        service_name = _TOOL_SERVICE_NAMES.get(tool_name, tool_name)
        client = self._get_client(route.base_url)
        cb = self._cb_registry.get(service_name)

        await self._check_circuit_breaker(cb)

        try:
            async with client.stream(
                method,
                url,
                json=payload,
                params=params,
                timeout=route.timeout,
            ) as response:
                await cb.on_success()
                async for line in response.aiter_lines():
                    yield line
        except (httpx.ConnectError, httpx.ConnectTimeout) as err:
            await cb.on_failure()
            raise BackendUnavailableError(service_name, "Connection failed") from err
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as err:
            await cb.on_failure()
            raise ToolTimeoutError(tool_name, route.timeout or 0.0) from err

    async def close(self) -> None:
        """Close all pooled httpx clients."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
