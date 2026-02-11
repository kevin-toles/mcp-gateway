"""Structured error responses — WBS-MCP2 (errors), MCP7 (full implementation).

Custom exception hierarchy for the mcp-gateway service.
MCP2 introduces BackendUnavailableError and ToolTimeoutError.
MCP7 adds StructuredErrorResponse and ResourceExhaustedError.
"""

from pydantic import BaseModel


class MCPGatewayError(Exception):
    """Base exception for all mcp-gateway errors."""


class BackendUnavailableError(MCPGatewayError):
    """Raised when an HTTP connection to a backend service fails.

    Reference: AC-2.4 — Connection errors raise BackendUnavailableError
    with service name.
    """

    def __init__(self, service_name: str, detail: str = "") -> None:
        self.service_name = service_name
        self.detail = detail
        msg = f"Backend unavailable: {service_name}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


class ToolTimeoutError(MCPGatewayError):
    """Raised when a tool dispatch exceeds its configured timeout.

    Reference: AC-2.5 — Timeout errors raise ToolTimeoutError with
    tool name and timeout value.
    """

    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Tool '{tool_name}' timed out after {timeout_seconds}s")


class CircuitOpenError(MCPGatewayError):
    """Raised when a circuit breaker is open and the call is rejected.

    Reference: C-5 — Circuit breaker rejects calls to unhealthy backends
    to prevent cascading failures.
    """

    def __init__(self, backend_name: str, retry_after: float) -> None:
        self.backend_name = backend_name
        self.retry_after = max(0.0, retry_after)
        super().__init__(f"Circuit open for '{backend_name}' — retry after {self.retry_after:.1f}s")


class ResourceExhaustedError(MCPGatewayError):
    """Raised when a resource budget is exceeded.

    Reference: AC-6.4 — Token budget, tool call count, or memory limit exceeded.
    """

    def __init__(self, resource: str, detail: str = "") -> None:
        self.resource = resource
        self.detail = detail
        msg = f"Resource exhausted: {resource}"
        if detail:
            msg += f" — {detail}"
        super().__init__(msg)


class StructuredErrorResponse(BaseModel):
    """Structured error response — AC-7.4.

    Returns ``{"error": str, "code": str, "request_id": str}`` — no stack traces.
    """

    error: str
    code: str
    request_id: str

    @classmethod
    def from_exception(cls, exc: Exception, request_id: str) -> "StructuredErrorResponse":
        """Create from an exception, mapping to machine-readable codes.

        Never leaks internal details for unhandled exceptions.
        """
        if isinstance(exc, CircuitOpenError):
            return cls(
                error=str(exc),
                code="CIRCUIT_OPEN",
                request_id=request_id,
            )
        if isinstance(exc, BackendUnavailableError):
            return cls(
                error=str(exc),
                code="BACKEND_UNAVAILABLE",
                request_id=request_id,
            )
        if isinstance(exc, ToolTimeoutError):
            return cls(
                error=str(exc),
                code="TOOL_TIMEOUT",
                request_id=request_id,
            )
        if isinstance(exc, ResourceExhaustedError):
            return cls(
                error=str(exc),
                code="RESOURCE_EXHAUSTED",
                request_id=request_id,
            )
        if isinstance(exc, MCPGatewayError):
            return cls(
                error=str(exc),
                code="GATEWAY_ERROR",
                request_id=request_id,
            )
        # Unhandled — never expose internal details
        return cls(
            error="An internal error occurred",
            code="INTERNAL_ERROR",
            request_id=request_id,
        )
