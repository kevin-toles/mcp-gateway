"""
Preflight dependency-state check for MCP gateway tool calls.

Checks if the backend service for a tool is healthy before dispatching.
If not healthy, raises a fast, explicit error.
"""
import httpx
from src.tool_dispatcher import _TOOL_SERVICE_NAMES, _build_routes
from src.config.health_config import health_timeout_for
from src.core.config import Settings

# Map service name to health endpoint
SERVICE_HEALTH_ENDPOINTS = {
    "unified-search-service": ("http://localhost:8081", "/health"),
    "code-orchestrator": ("http://localhost:8083", "/health"),
    "llm-gateway": ("http://localhost:8080", "/health"),
    "ai-agents": ("http://localhost:8082", "/health"),
    "audit-service": ("http://localhost:8084", "/health"),
    "amve": ("http://localhost:8088", "/health"),
    # unified-search-rs (port 8089) removed: the Go/Rust rewrite does not exist yet.
    # When it's built on a new port (e.g. 8090), add it here.
    # Item 36: inference-service-cpp (HWC F2, P0)
    "inference-service-cpp": ("http://localhost:8085", "/health"),
}

async def preflight_check(tool_name: str, settings: Settings) -> None:
    """
    Checks if the backend service for the tool is healthy.
    Raises RuntimeError if not healthy.
    """
    service = _TOOL_SERVICE_NAMES.get(tool_name)
    if not service:
        return  # No backend dependency (local handler or unknown)
    health_info = SERVICE_HEALTH_ENDPOINTS.get(str(service))
    if not health_info:
        return  # No health check defined
    url, endpoint = health_info
    health_url = f"{url}{endpoint}"
    timeout = health_timeout_for(service)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(health_url)
            if resp.status_code != 200:
                raise RuntimeError(f"Dependency service '{service}' is not running (HTTP {resp.status_code})")
    except Exception:
        raise RuntimeError(f"Dependency service '{service}' is not running (connection failed)")
