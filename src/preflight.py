"""
Preflight dependency-state check for MCP gateway tool calls.

Checks if the backend service for a tool is healthy before dispatching.
If not healthy, raises a fast, explicit error.

Service URLs are resolved from Settings (env-var driven) so the same code
works in hybrid mode (localhost) and Docker mode (container DNS).
"""
import httpx

from src.config.health_config import health_timeout_for
from src.core.config import Settings
from src.tool_dispatcher import _TOOL_SERVICE_NAMES

# Health endpoint suffix per service. URL base comes from Settings at call time.
_HEALTH_PATHS = {
    "unified-search-service": "/health",
    "code-orchestrator": "/health",
    "llm-gateway": "/health",
    "ai-agents": "/health",
    "audit-service": "/health",
    "struct-analyzer": "/health",  # Go service — replaces amve
    "amve": "/health",  # kept for legacy tool routing during migration
    "inference-service-cpp": "/health",
    "context-management-service": "/health",
    "unified-search-rs": "/health",
}


def _service_base_url(service: str, settings: Settings) -> str | None:
    """Resolve a service name to its base URL from Settings.

    Returns None if the service has no configured URL. Settings fields are
    env-overridable via the MCP_GATEWAY_ prefix (e.g. MCP_GATEWAY_LLM_GATEWAY_URL).
    """
    mapping = {
        "unified-search-service": settings.UNIFIED_SEARCH_URL,
        "unified-search-rs": settings.UNIFIED_SEARCH_RS_URL,
        "code-orchestrator": settings.CODE_ORCHESTRATOR_URL,
        "llm-gateway": settings.LLM_GATEWAY_URL,
        "ai-agents": settings.AI_AGENTS_URL,
        "audit-service": settings.AUDIT_SERVICE_URL,
        "amve": settings.AMVE_SERVICE_URL,
        "struct-analyzer": settings.STRUCT_ANALYZER_URL,
        "inference-service-cpp": settings.INFERENCE_SERVICE_URL,
        "context-management-service": settings.CONTEXT_MANAGEMENT_URL,
    }
    return mapping.get(service)


async def preflight_check(tool_name: str, settings: Settings) -> None:
    """Check that the backend service for the tool is healthy.

    Raises RuntimeError if the dependency is not reachable or returns non-200.
    """
    service = _TOOL_SERVICE_NAMES.get(tool_name)
    if not service:
        return  # No backend dependency (local handler or unknown)

    service_name = str(service)
    base_url = _service_base_url(service_name, settings)
    if not base_url:
        return  # No URL configured for this service

    path = _HEALTH_PATHS.get(service_name, "/health")
    health_url = f"{base_url}{path}"
    timeout = health_timeout_for(service)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(health_url)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Dependency service '{service}' is not running (HTTP {resp.status_code})"
                )
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError(
            f"Dependency service '{service}' is not running (connection failed)"
        )
