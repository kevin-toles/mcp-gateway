"""Integration test configuration — WBS-MCP9 + AEI-8.

Shared fixtures for tests requiring live backend services.
All integration tests are marked with ``@pytest.mark.integration``.
Run them with: ``INTEGRATION=1 pytest tests/integration/ -m integration``
"""

import os
from pathlib import Path

import httpx
import pytest

from src.core.config import Settings
from src.security.output_sanitizer import OutputSanitizer
from src.server import create_mcp_server
from src.tool_dispatcher import ToolDispatcher
from src.tool_registry import ToolRegistry

# ── Auto-skip when INTEGRATION env not set ──────────────────────────────────


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests when INTEGRATION env var is not set."""
    if os.environ.get("INTEGRATION", "").lower() in ("1", "true", "yes"):
        return
    skip_marker = pytest.mark.skip(reason="Set INTEGRATION=1 to run integration tests with live services")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


# ── Session-scoped fixtures ─────────────────────────────────────────────────


@pytest.fixture(scope="session")
def gateway_url() -> str:
    """Base URL of the running mcp-gateway."""
    return os.environ.get("MCP_GATEWAY_URL", "http://localhost:8087")


@pytest.fixture(scope="session")
def amve_url() -> str:
    """Base URL of the running AMVE service."""
    return os.environ.get("MCP_GATEWAY_AMVE_SERVICE_URL", "http://localhost:8088")


@pytest.fixture(scope="session")
def audit_log_path(tmp_path_factory) -> Path:
    """Temp path for audit JSONL during tests."""
    return tmp_path_factory.mktemp("audit") / "test_audit.jsonl"


@pytest.fixture(scope="session")
def backend_urls() -> dict[str, str]:
    """Map of service name -> health endpoint."""
    return {
        "llm-gateway": "http://localhost:8080/health",
        "semantic-search": "http://localhost:8081/health",
        "ai-agents": "http://localhost:8082/health",
        "code-orchestrator": "http://localhost:8083/health",
        "amve": "http://localhost:8088/v1/health",
    }


# ── AMVE MCP Integration Fixtures (AEI-8) ──────────────────────────────────


async def _check_backend(url: str, timeout: float = 3.0) -> bool:
    """Return True if a health endpoint responds with 200."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=timeout)
            return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture
def settings() -> Settings:
    """Real settings pointing at localhost backends."""
    return Settings()


@pytest.fixture
def dispatcher(settings) -> ToolDispatcher:
    """Real ToolDispatcher with live HTTP clients."""
    return ToolDispatcher(settings)


@pytest.fixture
def sanitizer() -> OutputSanitizer:
    return OutputSanitizer()


@pytest.fixture
def registry() -> ToolRegistry:
    """Real tool registry from config/tools.yaml."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"
    if not config_path.exists():
        pytest.skip("config/tools.yaml not found")
    return ToolRegistry(config_path)


@pytest.fixture
def mcp_server(registry, dispatcher, sanitizer):
    """Create a real MCP server — same as production."""
    return create_mcp_server(registry, dispatcher, sanitizer)


# ── Real codebase paths for analysis ────────────────────────────────────────


INFERENCE_SERVICE_SRC = "/Users/kevintoles/POC/inference-service/src"
LLM_GATEWAY_SRC = "/Users/kevintoles/POC/llm-gateway/src"
AI_AGENTS_SRC = "/Users/kevintoles/POC/ai-agents/src"
AMVE_SRC = "/Users/kevintoles/POC/architecture-mapping-validation-engine/src"


def _extract_body(result) -> dict:
    """Extract the JSON body from a FastMCP tool call result.

    FastMCP's Client.call_tool() returns a CallToolResult (Pydantic model)
    with .content (list of TextContent) and .isError.  The first TextContent
    has a .text attribute holding the JSON payload.
    """
    import json

    # CallToolResult — the normal return type from client.call_tool()
    if hasattr(result, "content"):
        text = result.content[0].text
        return json.loads(text)
    # List of TextContent (older FastMCP versions)
    if isinstance(result, list):
        text = result[0].text if hasattr(result[0], "text") else str(result[0])
        return json.loads(text)
    # Already a dict
    if isinstance(result, dict):
        return result
    # Fallback
    return json.loads(str(result))
