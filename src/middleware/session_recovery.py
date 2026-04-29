"""Session recovery middleware — prevents silent hangs on stale sessions.

When the MCP server restarts (code reload, deployment, crash), in-memory
sessions are lost but clients retain stale session IDs. This middleware
intercepts 404 errors on session endpoints and returns a clear 410 response
with recovery instructions.

Reference: DESIGN_SESSION_RESILIENCE.md Layer 1
"""

import json
import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SessionRecoveryMiddleware(BaseHTTPMiddleware):
    """Convert session 404s to 410 Gone with recovery instructions.

    This middleware intercepts requests to /mcp/messages that fail with 404
    due to unknown session IDs. Instead of returning a generic 404 (which
    causes MCP clients to hang indefinitely), it returns:

    - HTTP 410 Gone (more semantic than 404)
    - JSON body with clear error message
    - Recovery endpoint for reconnection
    - Session ID that failed

    This allows MCP clients to detect stale sessions and reconnect
    automatically, preventing the "tool appears to hang forever" UX issue.

    Example response:
        {
            "error": "session_expired",
            "message": "Session abc123... no longer exists. Please reconnect via SSE at /mcp/sse",
            "recovery_endpoint": "/mcp/sse",
            "session_id": "abc123...",
            "server_version": "1.0.0"
        }
    """

    def __init__(self, app, service_version: str = "unknown"):
        """Initialize middleware with service version for client debugging.

        Args:
            app: FastAPI application instance
            service_version: Service version to include in error responses
        """
        super().__init__(app)
        self.service_version = service_version

    async def dispatch(self, request: Request, call_next):
        """Intercept response and handle session 404s gracefully.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain

        Returns:
            Original response, or 410 error if session not found
        """
        response = await call_next(request)

        # Only intercept 404s on MCP session endpoints
        if (
            response.status_code == 404
            and request.url.path.startswith("/mcp/messages")
            and "session_id" in request.query_params
        ):
            session_id = request.query_params["session_id"]

            logger.warning(
                f"Session {session_id} not found - likely server restart. Client should reconnect via /mcp/sse"
            )

            error_body = {
                "error": "session_expired",
                "message": (
                    f"Session {session_id} no longer exists. "
                    f"The server may have restarted. "
                    f"Please reconnect via SSE at /mcp/sse"
                ),
                "recovery_endpoint": "/mcp/sse",
                "session_id": session_id,
                "server_version": self.service_version,
            }

            return Response(
                status_code=410,  # Gone - more semantic than 404
                content=json.dumps(error_body),
                media_type="application/json",
            )

        return response
