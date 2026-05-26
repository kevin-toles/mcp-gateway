"""Session recovery middleware — prevents silent hangs on stale sessions.

When the MCP server restarts (code reload, deployment, crash), in-memory
sessions are lost but clients retain stale session IDs. This middleware
intercepts 404 errors on session endpoints and returns a clear 410 response
with recovery instructions.

Reference: DESIGN_SESSION_RESILIENCE.md Layer 1
"""

import json
import logging
from urllib.parse import parse_qs

from fastapi import Response

logger = logging.getLogger(__name__)


class SessionRecoveryMiddleware:
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
        self.app = app
        self.service_version = service_version

    _HTTP_RESP_BODY = "http.response.body"
    _HTTP_RESP_START = "http.response.start"

    async def __call__(self, scope, receive, send):
        """ASGI middleware implementation safe for streaming endpoints.

        Uses a low-level send wrapper for `/mcp/messages` requests only,
        avoiding BaseHTTPMiddleware issues with SSE/streaming paths.
        """
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        session_path_prefix = "/mcp/messages"
        if not path.startswith(session_path_prefix):
            await self.app(scope, receive, send)
            return

        query_string = scope.get("query_string", b"").decode()
        session_id = parse_qs(query_string).get("session_id", [None])[0]
        if not session_id:
            await self.app(scope, receive, send)
            return

        response_start = None
        response_body_chunks = []

        async def send_wrapper(message):
            nonlocal response_start

            message_type = message.get("type")
            if message_type == _HTTP_RESP_START:
                response_start = message
                return

            if message_type == _HTTP_RESP_BODY and response_start is not None:
                response_body_chunks.append(message.get("body", b""))
                if message.get("more_body", False):
                    return

                status_code = response_start.get("status", 200)
                if status_code == 404:
                    logger.warning(
                        "Session %s not found - likely server restart. Client should reconnect via /mcp/sse",
                        session_id,
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
                    encoded = json.dumps(error_body).encode("utf-8")
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 410,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(encoded)).encode("ascii")),
                            ],
                        }
                    )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": encoded,
                            "more_body": False,
                        }
                    )
                    return

                await send(response_start)
                for idx, chunk in enumerate(response_body_chunks):
                    await send(
                        {
                            "type": "http.response.body",
                            "body": chunk,
                            "more_body": idx < len(response_body_chunks) - 1,
                        }
                    )
                return

            await send(message)

        await self.app(scope, receive, send_wrapper)
