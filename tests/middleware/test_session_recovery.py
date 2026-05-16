"""Tests for session recovery middleware.

Verifies that stale session IDs return 410 with recovery instructions
instead of silent 404s that cause clients to hang.

Reference: DESIGN_SESSION_RESILIENCE.md Layer 1
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.middleware.session_recovery import SessionRecoveryMiddleware


@pytest.fixture
def app_with_middleware():
    """Create test FastAPI app with session recovery middleware."""
    app = FastAPI()

    app.add_middleware(SessionRecoveryMiddleware, service_version="test-1.0.0")

    # Mock MCP endpoint that returns 404 for unknown sessions
    @app.post("/mcp/messages/")
    async def mock_mcp_messages(session_id: str):
        # Simulate FastMCP returning 404 for unknown session
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Session not found")

    # Other endpoint that should NOT trigger middleware
    @app.get("/api/data")
    async def mock_api_endpoint():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Data not found")

    return app


def test_unknown_session_returns_410(app_with_middleware):
    """Session 404s should be converted to 410 with recovery info."""
    client = TestClient(app_with_middleware)

    response = client.post("/mcp/messages/", params={"session_id": "unknown-session-abc123"})

    # Should return 410 instead of 404
    assert response.status_code == 410

    # Should have helpful error body
    body = response.json()
    assert body["error"] == "session_expired"
    assert "unknown-session-abc123" in body["message"]
    assert body["recovery_endpoint"] == "/mcp/sse"
    assert body["session_id"] == "unknown-session-abc123"
    assert body["server_version"] == "test-1.0.0"


def test_non_session_404_untouched(app_with_middleware):
    """Regular 404s on non-MCP endpoints should pass through unchanged."""
    client = TestClient(app_with_middleware)

    response = client.get("/api/data")

    # Should remain 404, not converted to 410
    assert response.status_code == 404
    assert response.json()["detail"] == "Data not found"


def test_mcp_endpoint_without_session_id_untouched(app_with_middleware):
    """MCP endpoint 404s without session_id param should pass through."""
    client = TestClient(app_with_middleware)

    response = client.post("/mcp/messages/")

    # Should remain 404 (not 410) because no session_id param
    assert response.status_code == 404


def test_error_message_contains_recovery_instructions(app_with_middleware):
    """Error message should guide user to reconnect."""
    client = TestClient(app_with_middleware)

    response = client.post("/mcp/messages/", params={"session_id": "stale-session"})

    body = response.json()
    message = body["message"].lower()

    # Should mention reconnection
    assert "reconnect" in message or "restart" in message
    assert "/mcp/sse" in body["message"]


def test_middleware_logs_session_expiry(app_with_middleware, caplog):
    """Middleware should log session expiry events for debugging."""
    client = TestClient(app_with_middleware)

    with caplog.at_level("WARNING"):
        client.post("/mcp/messages/", params={"session_id": "test-session-id"})

    # Should log the session expiry
    assert any("test-session-id" in record.message for record in caplog.records)
    assert any("not found" in record.message.lower() for record in caplog.records)
