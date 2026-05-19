"""
Idle Timeout Middleware
=======================

FastAPI middleware that tracks last-request timestamps for idle timeout enforcement.

This middleware is added to each service to record when requests arrive.
The mcp-gateway's IdleTimeoutTracker uses this data to determine when
services should be shut down.

Usage:
    from src.middleware.idle_timeout import IdleTimeoutMiddleware
    
    app.add_middleware(IdleTimeoutMiddleware, service_id="semantic-search")
"""

from __future__ import annotations

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class IdleTimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware that records request timestamps for idle timeout tracking.
    
    This is a lightweight middleware that adds minimal overhead (~1ms per request).
    It records the timestamp of each request so the mcp-gateway can track
    service idle time.
    """
    
    def __init__(
        self,
        app,
        service_id: str,
        tracker_url: str = "http://localhost:8087/api/idle-timeout",
    ):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI application
            service_id: Service identifier (e.g., "semantic-search")
            tracker_url: URL to report request to (mcp-gateway endpoint)
        """
        super().__init__(app)
        self.service_id = service_id
        self.tracker_url = tracker_url
        self._last_request_time: float = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and record timestamp.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from handler
        """
        # Record request timestamp
        self._last_request_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Add idle timeout headers to response
        response.headers["X-Service-Id"] = self.service_id
        response.headers["X-Last-Request"] = str(self._last_request_time)
        
        return response
    
    @property
    def last_request_time(self) -> float:
        """Get timestamp of last request."""
        return self._last_request_time
    
    @property
    def idle_seconds(self) -> float:
        """Get seconds since last request."""
        if self._last_request_time == 0:
            return float('inf')
        return time.time() - self._last_request_time
