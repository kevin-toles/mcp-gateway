"""Tests for ToolRegistry — WBS-MCP8 (RED).

AC-8.4 (tool registry from config), AC-8.2 (9 tools with JSON Schema).
"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from src.tool_registry import ToolDefinition, ToolRegistry

# ── Constants ───────────────────────────────────────────────────────────

EXPECTED_TOOL_NAMES = {
    "semantic_search",
    "hybrid_search",
    "code_analyze",
    "code_pattern_audit",
    "graph_query",
    "llm_complete",
    "run_agent_function",
    "run_discussion",
    "agent_execute",
}

VALID_TIERS = {"bronze", "silver", "gold", "enterprise"}

# ── YAML Fixture ────────────────────────────────────────────────────────

VALID_TOOLS_YAML = """\
tools:
  - name: semantic_search
    description: "Search using semantic similarity"
    tier: bronze
    tags: [search, rag]
  - name: hybrid_search
    description: "Hybrid semantic + keyword search"
    tier: bronze
    tags: [search, rag]
  - name: code_analyze
    description: "Analyze code quality"
    tier: silver
    tags: [code, analysis]
  - name: code_pattern_audit
    description: "Audit code anti-patterns"
    tier: silver
    tags: [code, audit]
  - name: graph_query
    description: "Query Neo4j knowledge graph"
    tier: gold
    tags: [graph, query]
  - name: llm_complete
    description: "LLM completion with fallback"
    tier: gold
    tags: [llm, completion]
  - name: run_agent_function
    description: "Run single agent function"
    tier: gold
    tags: [agent, function]
  - name: run_discussion
    description: "Multi-LLM discussion"
    tier: enterprise
    tags: [agent, discussion]
  - name: agent_execute
    description: "Autonomous agent execution"
    tier: enterprise
    tags: [agent, execution]
"""


# ── Helpers ─────────────────────────────────────────────────────────────


@pytest.fixture()
def yaml_path(tmp_path: Path) -> Path:
    p = tmp_path / "tools.yaml"
    p.write_text(VALID_TOOLS_YAML)
    return p


@pytest.fixture()
def registry(yaml_path: Path) -> ToolRegistry:
    return ToolRegistry(yaml_path)


# ── ToolDefinition ──────────────────────────────────────────────────────


class TestToolDefinition:
    def test_creation(self):
        class Dummy(BaseModel):
            x: int = 1

        td = ToolDefinition(
            name="test_tool",
            description="A test tool",
            tier="bronze",
            tags=["test"],
            input_model=Dummy,
        )
        assert td.name == "test_tool"
        assert td.description == "A test tool"
        assert td.tier == "bronze"
        assert td.tags == ["test"]
        assert td.input_model is Dummy

    def test_frozen(self):
        class Dummy(BaseModel):
            x: int = 1

        td = ToolDefinition(
            name="test_tool",
            description="desc",
            tier="bronze",
            tags=[],
            input_model=Dummy,
        )
        with pytest.raises(AttributeError):
            td.name = "changed"  # type: ignore[misc]


# ── ToolRegistry — Loading ──────────────────────────────────────────────


class TestToolRegistryLoading:
    def test_loads_from_yaml(self, yaml_path):
        registry = ToolRegistry(yaml_path)
        assert registry.tool_count > 0

    def test_loads_9_tools(self, yaml_path):
        registry = ToolRegistry(yaml_path)
        assert registry.tool_count == 9

    def test_tool_names_match(self, yaml_path):
        registry = ToolRegistry(yaml_path)
        assert registry.tool_names() == EXPECTED_TOOL_NAMES

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            ToolRegistry(Path("/nonexistent/path/tools.yaml"))

    def test_invalid_yaml_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("{{not: valid: yaml::")
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            ToolRegistry(p)

    def test_missing_tools_key_raises(self, tmp_path):
        p = tmp_path / "no_tools.yaml"
        p.write_text("other_key: true\n")
        with pytest.raises(ValueError, match="tools"):
            ToolRegistry(p)

    def test_empty_tools_list_raises(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("tools: []\n")
        with pytest.raises(ValueError, match="[Nn]o tools"):
            ToolRegistry(p)

    def test_duplicate_tool_name_raises(self, tmp_path):
        yaml = """\
tools:
  - name: semantic_search
    description: "First"
    tier: bronze
    tags: []
  - name: semantic_search
    description: "Dup"
    tier: gold
    tags: []
"""
        p = tmp_path / "dup.yaml"
        p.write_text(yaml)
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            ToolRegistry(p)

    def test_unknown_tool_name_raises(self, tmp_path):
        yaml = """\
tools:
  - name: nonexistent_tool
    description: "Unknown"
    tier: bronze
    tags: []
"""
        p = tmp_path / "unknown.yaml"
        p.write_text(yaml)
        with pytest.raises(ValueError, match="[Uu]nknown tool"):
            ToolRegistry(p)

    def test_missing_description_raises(self, tmp_path):
        yaml = """\
tools:
  - name: semantic_search
    tier: bronze
    tags: []
"""
        p = tmp_path / "nodesc.yaml"
        p.write_text(yaml)
        with pytest.raises((ValueError, KeyError)):
            ToolRegistry(p)

    def test_missing_tier_raises(self, tmp_path):
        yaml = """\
tools:
  - name: semantic_search
    description: "Search"
    tags: []
"""
        p = tmp_path / "notier.yaml"
        p.write_text(yaml)
        with pytest.raises((ValueError, KeyError)):
            ToolRegistry(p)


# ── ToolRegistry — Access ───────────────────────────────────────────────


class TestToolRegistryAccess:
    def test_get_existing_tool(self, registry):
        tool = registry.get("semantic_search")
        assert tool is not None
        assert isinstance(tool, ToolDefinition)
        assert tool.name == "semantic_search"

    def test_get_unknown_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_list_all_returns_9(self, registry):
        tools = registry.list_all()
        assert len(tools) == 9
        assert all(isinstance(t, ToolDefinition) for t in tools)

    def test_tool_count(self, registry):
        assert registry.tool_count == 9

    def test_every_tool_has_description(self, registry):
        for tool in registry.list_all():
            assert tool.description, f"{tool.name} has empty description"

    def test_every_tool_has_valid_tier(self, registry):
        for tool in registry.list_all():
            assert tool.tier in VALID_TIERS, f"{tool.name} has invalid tier '{tool.tier}'"

    def test_every_tool_has_tags_list(self, registry):
        for tool in registry.list_all():
            assert isinstance(tool.tags, list), f"{tool.name} tags not a list"


# ── ToolRegistry — Input Model Resolution ───────────────────────────────


class TestToolRegistryInputModels:
    def test_every_tool_has_input_model(self, registry):
        for tool in registry.list_all():
            assert tool.input_model is not None
            assert issubclass(tool.input_model, BaseModel), (
                f"{tool.name} input_model is not a Pydantic BaseModel"
            )

    @pytest.mark.parametrize(
        "tool_name,expected_model_name",
        [
            ("semantic_search", "SemanticSearchInput"),
            ("hybrid_search", "HybridSearchInput"),
            ("code_analyze", "CodeAnalyzeInput"),
            ("code_pattern_audit", "CodePatternAuditInput"),
            ("graph_query", "GraphQueryInput"),
            ("llm_complete", "LLMCompleteInput"),
            ("run_discussion", "RunDiscussionInput"),
            ("run_agent_function", "RunAgentFunctionInput"),
            ("agent_execute", "AgentExecuteInput"),
        ],
    )
    def test_input_model_name(self, registry, tool_name, expected_model_name):
        tool = registry.get(tool_name)
        assert tool is not None
        assert tool.input_model.__name__ == expected_model_name


# ── ToolRegistry — Real Config ──────────────────────────────────────────


class TestToolRegistryRealConfig:
    def test_loads_real_config(self):
        """Smoke test with the actual config/tools.yaml file."""
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "tools.yaml"
        if not config_path.exists():
            pytest.skip("config/tools.yaml not yet created")
        registry = ToolRegistry(config_path)
        assert registry.tool_count == 9
        assert registry.tool_names() == EXPECTED_TOOL_NAMES
