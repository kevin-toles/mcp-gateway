"""Integration test configuration — WBS-MCP9.

Shared fixtures for tests requiring live backend services.
All integration tests are marked with ``@pytest.mark.integration``.
Run them with: ``pytest tests/integration/ -m integration``
"""

import os
from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests when INTEGRATION env var is not set."""
    if os.environ.get("INTEGRATION", "").lower() in ("1", "true", "yes"):
        return
    skip_marker = pytest.mark.skip(reason="Set INTEGRATION=1 to run integration tests with live services")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def gateway_url() -> str:
    """Base URL of the running mcp-gateway."""
    return os.environ.get("MCP_GATEWAY_URL", "http://localhost:8087")


@pytest.fixture(scope="session")
def audit_log_path(tmp_path_factory) -> Path:
    """Temp path for audit JSONL during tests."""
    return tmp_path_factory.mktemp("audit") / "test_audit.jsonl"


@pytest.fixture(scope="session")
def backend_urls() -> dict[str, str]:
    """Map of service name → health endpoint."""
    return {
        "llm-gateway": "http://localhost:8080/health",
        "semantic-search": "http://localhost:8081/health",
        "ai-agents": "http://localhost:8082/health",
        "code-orchestrator": "http://localhost:8083/health",
    }
