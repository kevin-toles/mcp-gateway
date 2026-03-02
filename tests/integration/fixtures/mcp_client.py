"""MCP client fixture — PDW-6.8 GREEN.

Provides a session-scoped ``ToolDispatcher`` fixture that dispatches tool
calls through the mcp-gateway routing table to live backend services.

The ``ToolDispatcher`` is the canonical gateway dispatch mechanism: it holds
the same ``DispatchRoute`` table as the production mcp-gateway server and
sends real HTTP requests to ``settings.AUDIT_SERVICE_URL``.

Using ``ToolDispatcher`` directly (rather than the SSE/MCP wire protocol)
tests the dispatch configuration end-to-end — the same code path that
``create_mcp_server`` ultimately calls for every tool invocation.

Usage in integration tests::

    from tests.integration.fixtures.mcp_client import dispatcher

    result = await dispatcher.dispatch("audit_search_exploits", {...})
    assert result.status_code == 200
"""

from __future__ import annotations

import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher


@pytest.fixture(scope="session")
def mcp_settings() -> Settings:
    """Real Settings instance pointing at localhost backends."""
    return Settings()


@pytest.fixture(scope="session")
def dispatcher(mcp_settings: Settings) -> ToolDispatcher:
    """Session-scoped ToolDispatcher with live HTTP clients.

    PDW-6.8 GREEN: Provides the mcp-gateway dispatch client for integration
    tests.  Uses a session scope so a single httpx connection pool is reused
    across all VRE roundtrip tests in the session.

    Note: ``ToolDispatcher`` creates httpx clients lazily; no cleanup required
    unless explicit connection draining is needed.
    """
    return ToolDispatcher(mcp_settings)
