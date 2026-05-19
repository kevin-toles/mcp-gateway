"""Tools REST API — LLM-friendly HTTP endpoint for tool calls.

Adds a REST endpoint at ``POST /api/v1/tools/{tool_name}`` that wraps
the existing ``ToolDispatcher`` dispatch pipeline.  This allows LLM
agents to call tools via simple HTTP POST instead of MCP's SSE protocol.

Usage via cURL::

    curl -X POST "http://localhost:8087/api/v1/tools/convert_pdf" \\
      -H "Content-Type: application/json" \\
      -d '{"input_path": "...", "output_path": "..."}'
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from src.tool_dispatcher import ToolDispatcher
from src.tools import (
    batch_enrich_metadata as batch_enrich_metadata_tool,
    batch_extract_metadata as batch_extract_metadata_tool,
    convert_pdf as convert_pdf_tool,
)
from src.security.output_sanitizer import OutputSanitizer

logger = logging.getLogger(__name__)


def create_tools_router(dispatcher: ToolDispatcher) -> APIRouter:
    """Create a router with a REST endpoint for each tool in the dispatcher.

    Args:
        dispatcher: The ToolDispatcher instance with a populated route table.

    Returns:
        FastAPI APIRouter mounted at ``/api/v1/tools``.
    """
    router = APIRouter(prefix="/api/v1/tools", tags=["tools"])

    @router.get("/")
    async def list_tools() -> dict:
        """List all available tools with their backend routes."""
        routes = dispatcher.routes
        return {
            "tools": [
                {
                    "name": name,
                    "backend_url": f"{route.base_url}{route.path}",
                    "timeout": route.timeout,
                }
                for name, route in sorted(routes.items())
            ],
            "count": len(routes),
        }

    @router.post("/{tool_name:path}")
    async def call_tool(tool_name: str, payload: dict[str, Any]) -> dict:
        """Call a tool by name via REST (LLM-friendly).

        Args:
            tool_name: Tool name (e.g. ``convert_pdf``, ``semantic_search``).
            payload:  JSON body forwarded as tool arguments.

        Returns:
            The backend's JSON response wrapped in a result envelope.
        """
        route = dispatcher.get_route(tool_name)
        if route is None:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{tool_name}' not found. "
                       f"Available: {', '.join(sorted(dispatcher.routes.keys()))}",
            )

        logger.info("REST tool call: %s (payload: %s)", tool_name, payload)

        # Intercept tools that use local handlers (not backend dispatch)
        # These launch Terminal.app for batch operations
        _local_handlers = {
            "convert_pdf": convert_pdf_tool,
            "batch_extract_metadata": batch_extract_metadata_tool,
            "batch_enrich_metadata": batch_enrich_metadata_tool,
        }
        if tool_name in _local_handlers:
            sanitizer = OutputSanitizer()
            handler = _local_handlers[tool_name].create_handler(dispatcher, sanitizer)
            body = await handler(**payload)
            return {
                "tool": tool_name,
                "status_code": 200,
                "body": body,
                "elapsed_ms": 0,
            }

        try:
            result = await dispatcher.dispatch(
                tool_name=tool_name,
                payload=payload,
                method="POST",
            )
        except Exception as exc:
            logger.error("Tool dispatch failed for %s: %s", tool_name, exc)
            raise HTTPException(
                status_code=502,
                detail=f"Tool '{tool_name}' dispatch failed: {exc}",
            ) from exc

        return {
            "tool": tool_name,
            "status_code": result.status_code,
            "body": result.body,
            "elapsed_ms": result.elapsed_ms,
        }

    return router
