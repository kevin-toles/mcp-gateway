"""Tool registration and permission model — WBS-MCP8 (GREEN).

Loads tool definitions from a YAML config file and resolves each tool's
Pydantic input model from ``src.models.schemas``.

Reference: AC-8.4 (tool registry from config, not hardcoded)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from src.models.schemas import (
    A2ACancelTaskInput,
    A2AGetTaskInput,
    A2ASendMessageInput,
    AnalyzeTaxonomyCoverageInput,
    BatchExtractMetadataInput,
    CodeAnalyzeInput,
    CodePatternAuditInput,
    ConvertPDFInput,
    EnhanceGuidelineInput,
    EnrichBookMetadataInput,
    ExtractBookMetadataInput,
    GenerateTaxonomyInput,
    GraphQueryInput,
    HybridSearchInput,
    LLMCompleteInput,
    SemanticSearchInput,
)

# ── Input model mapping ────────────────────────────────────────────────

_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "semantic_search": SemanticSearchInput,
    "hybrid_search": HybridSearchInput,
    "code_analyze": CodeAnalyzeInput,
    "code_pattern_audit": CodePatternAuditInput,
    "graph_query": GraphQueryInput,
    "llm_complete": LLMCompleteInput,
    "a2a_send_message": A2ASendMessageInput,
    "a2a_get_task": A2AGetTaskInput,
    "a2a_cancel_task": A2ACancelTaskInput,
    # Workflow tools (WBS-WF6)
    "convert_pdf": ConvertPDFInput,
    "extract_book_metadata": ExtractBookMetadataInput,
    "batch_extract_metadata": BatchExtractMetadataInput,
    "generate_taxonomy": GenerateTaxonomyInput,
    "enrich_book_metadata": EnrichBookMetadataInput,
    "enhance_guideline": EnhanceGuidelineInput,
    # Taxonomy Analysis (WBS-TAP9)
    "analyze_taxonomy_coverage": AnalyzeTaxonomyCoverageInput,
}


# ── Data class ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolDefinition:
    """Immutable definition of a single MCP tool.

    Attributes:
        name:        Tool identifier (e.g. ``semantic_search``).
        description: Human-readable description shown in ``tools/list``.
        tier:        Minimum tenant tier required (bronze/silver/gold/enterprise).
        tags:        Classification tags for filtering.
        input_model: Pydantic ``BaseModel`` subclass for input validation.
    """

    name: str
    description: str
    tier: str
    tags: list[str] = field(default_factory=list)
    input_model: type[BaseModel] = field(default=BaseModel)


# ── Registry ────────────────────────────────────────────────────────────


class ToolRegistry:
    """Loads tool definitions from a YAML config file (AC-8.4).

    Args:
        config_path: Path to ``tools.yaml`` with the ``tools:`` list.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML is invalid, missing required keys,
                    contains unknown tool names, or has duplicates.
    """

    def __init__(self, config_path: str | Path) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._load(Path(config_path))

    # ── Loading ─────────────────────────────────────────────────────

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Tool config not found: {path}")

        raw = path.read_text(encoding="utf-8")
        try:
            data: Any = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(data, dict) or "tools" not in data:
            raise ValueError(f"YAML must contain a top-level 'tools' key in {path}")

        tools_list = data["tools"]
        if not tools_list:
            raise ValueError(f"No tools defined in {path}")

        for item in tools_list:
            self._register(item, path)

    def _register(self, item: dict[str, Any], path: Path) -> None:
        name = item.get("name")
        if not name:
            raise ValueError(f"Tool entry missing 'name' in {path}")

        if name in self._tools:
            raise ValueError(f"Duplicate tool name '{name}' in {path}")

        if name not in _INPUT_MODELS:
            raise ValueError(f"Unknown tool '{name}' in {path}. Valid tools: {sorted(_INPUT_MODELS.keys())}")

        description = item.get("description")
        if not description:
            raise ValueError(f"Tool '{name}' missing 'description' in {path}")

        tier = item.get("tier")
        if not tier:
            raise ValueError(f"Tool '{name}' missing 'tier' in {path}")

        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            tier=tier,
            tags=item.get("tags", []),
            input_model=_INPUT_MODELS[name],
        )

    # ── Access ──────────────────────────────────────────────────────

    def get(self, name: str) -> ToolDefinition | None:
        """Return the ``ToolDefinition`` for *name*, or ``None``."""
        return self._tools.get(name)

    def list_all(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def tool_names(self) -> set[str]:
        """Return the set of all registered tool names."""
        return set(self._tools.keys())
