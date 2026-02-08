"""Structured error tests — WBS-MCP7 (RED).

Covers AC-7.4 (structured error responses, no stack traces),
and error hierarchy from MCP2.
"""

import pytest

from src.core.errors import (
    BackendUnavailableError,
    MCPGatewayError,
    ResourceExhaustedError,
    StructuredErrorResponse,
    ToolTimeoutError,
)


# ── Error hierarchy (MCP2 existing) ────────────────────────────────────


class TestErrorHierarchy:
    """All custom errors inherit from MCPGatewayError."""

    def test_backend_unavailable_inherits(self) -> None:
        assert issubclass(BackendUnavailableError, MCPGatewayError)

    def test_tool_timeout_inherits(self) -> None:
        assert issubclass(ToolTimeoutError, MCPGatewayError)

    def test_resource_exhausted_inherits(self) -> None:
        assert issubclass(ResourceExhaustedError, MCPGatewayError)

    def test_backend_unavailable_message(self) -> None:
        err = BackendUnavailableError("ai-agents", "connection refused")
        assert "ai-agents" in str(err)
        assert err.service_name == "ai-agents"

    def test_tool_timeout_message(self) -> None:
        err = ToolTimeoutError("llm_complete", 30.0)
        assert "llm_complete" in str(err)
        assert err.timeout_seconds == 30.0

    def test_resource_exhausted_message(self) -> None:
        err = ResourceExhaustedError("token_budget", "exceeded")
        assert "token_budget" in str(err)
        assert err.resource == "token_budget"


# ── AC-7.4: StructuredErrorResponse ────────────────────────────────────


class TestStructuredErrorResponse:
    """Structured error response format."""

    def test_has_required_fields(self) -> None:
        resp = StructuredErrorResponse(
            error="Something went wrong",
            code="INTERNAL_ERROR",
            request_id="req-123",
        )
        assert resp.error == "Something went wrong"
        assert resp.code == "INTERNAL_ERROR"
        assert resp.request_id == "req-123"

    def test_serializes_to_dict(self) -> None:
        resp = StructuredErrorResponse(
            error="Not found",
            code="NOT_FOUND",
            request_id="req-456",
        )
        d = resp.model_dump()
        assert set(d.keys()) == {"error", "code", "request_id"}

    def test_no_stack_trace_in_output(self) -> None:
        resp = StructuredErrorResponse(
            error="Bad request",
            code="VALIDATION_ERROR",
            request_id="req-789",
        )
        output = resp.model_dump_json()
        assert "Traceback" not in output
        assert "File \"" not in output

    def test_from_exception_backend(self) -> None:
        exc = BackendUnavailableError("ai-agents", "timeout")
        resp = StructuredErrorResponse.from_exception(exc, request_id="req-be")
        assert resp.code == "BACKEND_UNAVAILABLE"
        assert "ai-agents" in resp.error

    def test_from_exception_timeout(self) -> None:
        exc = ToolTimeoutError("llm_complete", 30.0)
        resp = StructuredErrorResponse.from_exception(exc, request_id="req-to")
        assert resp.code == "TOOL_TIMEOUT"

    def test_from_exception_resource(self) -> None:
        exc = ResourceExhaustedError("token_budget")
        resp = StructuredErrorResponse.from_exception(exc, request_id="req-re")
        assert resp.code == "RESOURCE_EXHAUSTED"

    def test_from_exception_generic(self) -> None:
        exc = MCPGatewayError("unknown")
        resp = StructuredErrorResponse.from_exception(exc, request_id="req-ge")
        assert resp.code == "GATEWAY_ERROR"

    def test_from_exception_unhandled(self) -> None:
        exc = ValueError("bad")
        resp = StructuredErrorResponse.from_exception(exc, request_id="req-uh")
        assert resp.code == "INTERNAL_ERROR"
        assert "internal" in resp.error.lower()  # no raw exception details
