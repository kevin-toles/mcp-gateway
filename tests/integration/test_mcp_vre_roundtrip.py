"""Integration tests — PDW-6.7 (mcp-gateway VRE tool roundtrip).

AC-PDW6.4: MCP tool ``audit_search_exploits`` dispatched through mcp-gateway
           returns results from a live audit-service.

The ``ToolDispatcher`` is the production dispatch mechanism inside
mcp-gateway — the same object ``create_mcp_server()`` uses to forward every
tool call to backend services.  Calling it directly here exercises the full
dispatch routing path: route-table lookup → HTTP POST to audit-service
``/v1/audit/exploits`` → parse ``DispatchResult``.

Requires ``INTEGRATION=1`` and all of:
  - audit-service     :8084  (VRE exploit search endpoint)
  - code-orchestrator :8083  (CodeBERT embedding for the query)
  - qdrant-quarantine :6336  (``vuln_exploits`` collection must be seeded
                               before or during this test session)

Run with::

    INTEGRATION=1 pytest tests/integration/test_mcp_vre_roundtrip.py -v -m integration
"""

from __future__ import annotations

import os

import httpx
import pytest

from tests.integration.fixtures.mcp_client import dispatcher, mcp_settings  # noqa: F401 — fixture imports
from tests.integration.utils.health_waiter import wait_for_service

pytestmark = pytest.mark.integration

AUDIT_SERVICE_URL = os.environ.get("AUDIT_SERVICE_URL", "http://localhost:8084")
CODE_ORCHESTRATOR_URL = os.environ.get("CODE_ORCHESTRATOR_URL", "http://localhost:8083")


async def _service_healthy(url: str) -> bool:
    """Return True if the service health endpoint responds with HTTP 200."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, timeout=3.0)
            return r.status_code == 200
    except Exception:
        return False


class TestMCPAuditSearchExploits:
    """PDW-6.7 RED → GREEN: mcp-gateway dispatches audit_search_exploits to audit-service."""

    async def test_audit_search_exploits_returns_dispatch_result(self, dispatcher) -> None:  # noqa: F811
        """AC-PDW6.4: ToolDispatcher.dispatch('audit_search_exploits', …) succeeds.

        Verifies:
          1. The dispatch route ``"audit_search_exploits"`` is registered and
             resolves to ``AUDIT_SERVICE_URL + "/v1/audit/exploits"``.
          2. The dispatcher reaches audit-service and receives a valid HTTP response.
          3. ``result.body`` is a dict (parsed JSON from audit-service).
          4. ``result.elapsed_ms > 0`` (real HTTP call was made).
        """
        if not await wait_for_service(f"{AUDIT_SERVICE_URL}/health", timeout=5.0):
            pytest.skip("audit-service not running on :8084")
        if not await wait_for_service(f"{CODE_ORCHESTRATOR_URL}/health", timeout=5.0):
            pytest.skip("code-orchestrator not running on :8083 (required for CodeBERT)")

        result = await dispatcher.dispatch(
            "audit_search_exploits",
            {
                "query": "SQL injection vulnerability user input",
                "top_k": 5,
                "min_similarity": 0.0,  # accept all hits — we care about the dispatch, not the data
            },
        )

        # Dispatch must succeed (2xx from audit-service)
        assert 200 <= result.status_code < 300, (
            f"audit_search_exploits dispatch returned {result.status_code}. Body: {str(result.body)[:300]}"
        )
        assert isinstance(result.body, dict), f"Expected dict body, got: {type(result.body)} — {str(result.body)[:200]}"
        assert result.elapsed_ms > 0, "elapsed_ms must be > 0 for a real HTTP call"

        # Response must have the 'matches' key from ExploitSearchResponse
        assert "matches" in result.body, f"Response body missing 'matches' key: {result.body}"

    async def test_audit_search_exploits_route_registered(self, dispatcher) -> None:  # noqa: F811
        """ToolDispatcher has 'audit_search_exploits' in its route table.

        Smoke test: verifies PDW-3 routing config is present — the route
        that was verified in unit tests (PDW3.1) is also present at runtime.
        """
        route = dispatcher.get_route("audit_search_exploits")
        assert route is not None, (
            "'audit_search_exploits' not in ToolDispatcher route table. "
            "Check tool_dispatcher.py DispatchRoute registration."
        )
        assert "/v1/audit/exploits" in route.path, (
            f"Route path mismatch: expected '/v1/audit/exploits', got {route.path!r}"
        )

    async def test_audit_search_cves_dispatch_succeeds(self, dispatcher) -> None:  # noqa: F811
        """AC-PDW6.4 (extended): audit_search_cves also dispatches successfully.

        Verifies the second VRE tool route is live, exercising PDW-3 end-to-end.
        """
        if not await wait_for_service(f"{AUDIT_SERVICE_URL}/health", timeout=5.0):
            pytest.skip("audit-service not running on :8084")

        result = await dispatcher.dispatch(
            "audit_search_cves",
            {
                "cwe_id": "CWE-89",
                "limit": 5,
            },
        )

        assert 200 <= result.status_code < 300, (
            f"audit_search_cves dispatch returned {result.status_code}. Body: {str(result.body)[:300]}"
        )
        assert isinstance(result.body, dict)
        assert "records" in result.body, f"Response body missing 'records' key: {result.body}"
