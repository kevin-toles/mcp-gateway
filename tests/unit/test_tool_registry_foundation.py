"""
WBS-F7.1 – F7.3 RED tests: foundation_search tool in mcp-gateway registry + dispatcher.

These tests will FAIL (RED) until:
  - FoundationSearchInput is added to src/models/schemas.py  (F7.4b)
  - 'foundation_search' is registered in src/tool_registry.py _INPUT_MODELS  (F7.4b)
  - config/tools.yaml receives the foundation_search YAML stanza  (F7.4)
  - src/tool_dispatcher.py adds the route + service name entries  (F7.5)
"""

from pathlib import Path

import httpx
import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher
from src.tool_registry import ToolRegistry

# ---------------------------------------------------------------------------
# YAML stanza used by F7.1 / F7.3 registry fixtures
# ---------------------------------------------------------------------------
FOUNDATION_YAML = """\
tools:
  - name: foundation_search
    description: >
      Search the scientific foundation layer for mathematical, statistical, or theoretical
      underpinnings of software concepts. Use when a user asks WHY something works, asks
      about formal proofs or convergence, or when software KB results are
      insufficient for understanding first-principles reasoning.
    when_to_use: >
      Use when the question requires first-principles reasoning, formal proofs, or
      convergence analysis beyond what the software KB provides.
    tier: bronze
    tags: [search, foundation, scientific]
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def foundation_registry(tmp_path: Path) -> ToolRegistry:
    yaml_file = tmp_path / "tools.yaml"
    yaml_file.write_text(FOUNDATION_YAML)
    return ToolRegistry(config_path=yaml_file)


@pytest.fixture
def dispatcher() -> ToolDispatcher:
    return ToolDispatcher(Settings())


# ---------------------------------------------------------------------------
# F7.1 — Registry loads foundation_search correctly
# ---------------------------------------------------------------------------


def test_foundation_search_in_registry(foundation_registry: ToolRegistry) -> None:
    assert "foundation_search" in foundation_registry.tool_names()


def test_foundation_search_name(foundation_registry: ToolRegistry) -> None:
    tool = foundation_registry.get("foundation_search")
    assert tool is not None
    assert tool.name == "foundation_search"


def test_foundation_search_tier_is_bronze(foundation_registry: ToolRegistry) -> None:
    tool = foundation_registry.get("foundation_search")
    assert tool is not None
    assert tool.tier == "bronze"


def test_foundation_search_tags(foundation_registry: ToolRegistry) -> None:
    tool = foundation_registry.get("foundation_search")
    assert tool is not None
    assert "search" in tool.tags
    assert "foundation" in tool.tags
    assert "scientific" in tool.tags


# ---------------------------------------------------------------------------
# F7.2 — Dispatcher routes foundation_search to correct URL / path
# ---------------------------------------------------------------------------


def test_foundation_search_route_exists(dispatcher: ToolDispatcher) -> None:
    route = dispatcher.get_route("foundation_search")
    assert route is not None


def test_foundation_search_route_base_url(dispatcher: ToolDispatcher) -> None:
    route = dispatcher.get_route("foundation_search")
    assert route is not None
    assert route.base_url == "http://localhost:8081"


def test_foundation_search_route_path(dispatcher: ToolDispatcher) -> None:
    route = dispatcher.get_route("foundation_search")
    assert route is not None
    assert route.path == "/v1/search/foundation"


@pytest.mark.asyncio
async def test_foundation_search_dispatches_to_correct_url(
    dispatcher: ToolDispatcher,
) -> None:
    captured: dict = {}

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(200, json={"results": []})

    transport = httpx.MockTransport(mock_handler)
    dispatcher._client = httpx.AsyncClient(transport=transport)

    await dispatcher.dispatch("foundation_search", {"query": "entropy", "limit": 3})

    assert captured["url"] == "http://localhost:8081/v1/search/foundation"
    assert captured["method"] == "POST"


# ---------------------------------------------------------------------------
# F7.3 — Description contains required phrases
# ---------------------------------------------------------------------------


def test_description_contains_why_phrase(foundation_registry: ToolRegistry) -> None:
    tool = foundation_registry.get("foundation_search")
    assert tool is not None
    assert "WHY something works" in tool.description


def test_description_contains_proofs_phrase(foundation_registry: ToolRegistry) -> None:
    tool = foundation_registry.get("foundation_search")
    assert tool is not None
    assert "formal proofs or convergence" in tool.description


def test_description_contains_insufficient_phrase(
    foundation_registry: ToolRegistry,
) -> None:
    tool = foundation_registry.get("foundation_search")
    assert tool is not None
    assert "software KB results are" in tool.description
    assert "insufficient for understanding first-principles reasoning" in tool.description
