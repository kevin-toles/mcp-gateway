"""ToolDispatcher — WBS-MCP2 (GREEN).

HTTP dispatch to backend services.  Each of the 6 platform tools maps
to a ``DispatchRoute`` containing a backend base_url, path, and
per-tool timeout.  ``ToolDispatcher.dispatch()`` sends an HTTP POST
and returns a structured ``DispatchResult``.

Reference: Strategy §4.1 (dispatcher pattern), AC-2.1 through AC-2.5
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from src.core.config import Settings
from src.core.errors import BackendUnavailableError, ToolTimeoutError

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
    }


# ── Dispatcher ──────────────────────────────────────────────────────────


class ToolDispatcher:
    """Dispatches tool calls to backend platform services via HTTP POST.

    Uses a single ``httpx.AsyncClient`` per unique backend base_url for
    connection pooling (MCP2.11 REFACTOR).

    Args:
        settings: Application settings containing backend URLs.
    """

    def __init__(self, settings: Settings) -> None:
        self.routes: dict[str, DispatchRoute] = _build_routes(settings)
        # Connection pool: one AsyncClient per unique backend base_url
        self._clients: dict[str, httpx.AsyncClient] = {}
        # Legacy single-client attribute — tests can override this
        self._client: httpx.AsyncClient | None = None

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
    ) -> DispatchResult:
        """Send *payload* as HTTP POST to the backend for *tool_name*.

        Returns:
            A ``DispatchResult`` with status, body, headers, and timing.

        Raises:
            ValueError: If *tool_name* is not in the route table.
            ToolTimeoutError: If the request exceeds the per-tool timeout.
            BackendUnavailableError: If the backend cannot be reached.
        """
        route = self.get_route(tool_name)
        if route is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        url = f"{route.base_url}{route.path}"
        service_name = _TOOL_SERVICE_NAMES.get(tool_name, tool_name)
        client = self._get_client(route.base_url)

        start = time.monotonic()
        try:
            response = await client.post(
                url,
                json=payload,
                timeout=route.timeout,
            )
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
            raise ToolTimeoutError(tool_name, route.timeout)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise BackendUnavailableError(service_name, "Connection failed")

        elapsed_ms = (time.monotonic() - start) * 1000

        # Parse body — gracefully handle empty responses
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

    async def close(self) -> None:
        """Close all pooled httpx clients."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
