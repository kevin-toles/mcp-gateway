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
from src.tool_registry import _INPUT_MODELS
from src.tools import (
    batch_enrich_metadata as batch_enrich_metadata_tool,
    batch_extract_metadata as batch_extract_metadata_tool,
    convert_pdf_to_json as convert_pdf_to_json_tool,
)
from src.security.output_sanitizer import OutputSanitizer

logger = logging.getLogger(__name__)


def _apply_first_alias(payload: dict[str, Any], canonical: str, aliases: list[str]) -> None:
    """Copy first present alias value into canonical key (and remove alias key)."""
    if canonical in payload:
        return
    for alias in aliases:
        if alias in payload:
            payload[canonical] = payload.pop(alias)
            return


def _path_aliases(field_name: str) -> list[str]:
    """Return alias candidates for *_path fields."""
    prefix = field_name[: -len("_path")]
    aliases = [f"{prefix}_file", f"{prefix}_json_path", "path", "file_path"]
    if field_name == "input_path":
        aliases.extend(["json_path", "source_path", "input_file"])
    elif field_name == "output_path":
        aliases.extend(["out_path", "dest_path", "destination_path", "output_file"])
    return aliases


def _dir_aliases(field_name: str) -> list[str]:
    """Return alias candidates for *_dir fields."""
    prefix = field_name[: -len("_dir")]
    aliases = [f"{prefix}_directory", f"{prefix}_path", "dir_path", "directory"]
    if field_name == "input_dir":
        aliases.append("source_dir")
    elif field_name == "output_dir":
        aliases.extend(["dest_dir", "destination_dir", "out_dir"])
    return aliases


def _aliases_for_field(field_name: str) -> list[str]:
    """Return convention-based alias candidates for a canonical field name."""
    aliases: list[str] = []

    # Result count compatibility used broadly across search-like tools.
    if field_name == "top_k":
        aliases.extend(["limit", "k", "max_results"])
    elif field_name == "limit":
        aliases.extend(["top_k", "k", "max_results"])

    # Query compatibility for conversational callers.
    if field_name == "query":
        aliases.extend(["q", "question", "prompt"])

    # Generic path conventions.
    if field_name.endswith("_path"):
        aliases.extend(_path_aliases(field_name))

    # Generic directory conventions.
    if field_name.endswith("_dir"):
        aliases.extend(_dir_aliases(field_name))

    # De-duplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for item in aliases:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _normalize_rest_payload(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize common legacy/alias REST payload keys before dispatch.

    This keeps tool calls resilient to natural-language key drift
    (e.g., json_path vs input_path) without changing backend contracts.
    """
    normalized = dict(payload)
    input_model = _INPUT_MODELS.get(tool_name)
    model_fields = set(input_model.model_fields.keys()) if input_model else set()

    for canonical in model_fields:
        _apply_first_alias(normalized, canonical, _aliases_for_field(canonical))

    # Common paging/result-count compatibility.
    if "top_k" in model_fields and "top_k" not in normalized and "limit" in normalized:
        normalized["top_k"] = normalized.get("limit")
    if "limit" in model_fields and "limit" not in normalized and "top_k" in normalized:
        normalized["limit"] = normalized.get("top_k")

    # Common list-vs-scalar normalization.
    if "collections" in model_fields and isinstance(normalized.get("collections"), str):
        normalized["collections"] = [normalized["collections"]]

    # Backward compatibility for older clients/prompts.
    if tool_name == "extract_book_metadata":
        if "input_path" not in normalized and "json_path" in normalized:
            normalized["input_path"] = normalized.pop("json_path")

    return normalized


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


    @router.post(
        "/{tool_name:path}",
        responses={
            404: {"description": "Tool not found"},
            502: {"description": "Tool dispatch failed"},
            503: {"description": "Preflight dependency check failed"},
        },
    )
    async def call_tool(tool_name: str, payload: dict[str, Any], diagnostics: bool = False) -> dict:
        """Call a tool by name via REST (LLM-friendly). Preflight check is diagnostics-only."""
        from src.core.config import Settings
        from src.preflight import preflight_check

        route = dispatcher.get_route(tool_name)
        if route is None:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{tool_name}' not found. "
                       f"Available: {', '.join(sorted(dispatcher.routes.keys()))}",
            )

        # Preflight dependency-state check: diagnostics only
        if diagnostics:
            try:
                await preflight_check(tool_name, Settings())
            except RuntimeError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e

        payload = _normalize_rest_payload(tool_name, payload)
        logger.info("REST tool call: %s (payload: %s)", tool_name, payload)

        # Intercept tools that use local handlers (not backend dispatch)
        _local_handlers = {
              "convert_pdf_to_json": convert_pdf_to_json_tool,
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
            logger.exception("Tool dispatch failed for %s", tool_name)
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
