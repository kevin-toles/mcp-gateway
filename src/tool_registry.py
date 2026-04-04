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
    AMVEAnalysisInput,
    AMVECommunicationInput,
    AMVEDetectDeadCodeInput,
    AMVEDetectDriftInput,
    AMVEEvaluateFitnessInput,
    AMVEExtractArchitectureInput,
    AMVEGenerateArchitectureLogInput,
    AnalyzeTaxonomyCoverageInput,
    AskInput,
    AuditCodeMetricsInput,
    AuditCorpusSearchInput,
    AuditDependencyAssessInput,
    AuditQualityScanInput,
    AuditResolveLookupInput,
    AuditSearchCVEsInput,
    AuditSearchExploitsInput,
    AuditSecurityScanInput,
    BatchEnrichMetadataInput,
    BatchExtractMetadataInput,
    CodeAnalyzeInput,
    CodePatternAuditInput,
    ConvertPDFInput,
    DiagramSearchInput,
    EnhanceGuidelineInput,
    EnrichBookMetadataInput,
    ExtractBookMetadataInput,
    FindCodePatternInput,
    FoundationSearchInput,
    GenerateTaxonomyInput,
    GraphQueryInput,
    GraphTraverseInput,
    HybridSearchInput,
    KnowledgeRefineInput,
    KnowledgeSearchInput,
    LLMCompleteInput,
    PatternSearchInput,
    PushToGithubInput,
    SearchInInput,
    SemanticSearchInput,
)

# ── Input model mapping ────────────────────────────────────────────────

_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "semantic_search": SemanticSearchInput,
    "hybrid_search": HybridSearchInput,
    "graph_traverse": GraphTraverseInput,
    # Issue #6: consolidated KB tools
    "knowledge_search": KnowledgeSearchInput,
    "knowledge_refine": KnowledgeRefineInput,
    "pattern_search": PatternSearchInput,
    # diagram_search: searches ascii_diagrams via CLIP text→image cross-modal retrieval
    "diagram_search": DiagramSearchInput,
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
    "batch_enrich_metadata": BatchEnrichMetadataInput,
    "enhance_guideline": EnhanceGuidelineInput,
    # Taxonomy Analysis (WBS-TAP9)
    "analyze_taxonomy_coverage": AnalyzeTaxonomyCoverageInput,
    # AMVE tools (AEI-7)
    "amve_detect_patterns": AMVEAnalysisInput,
    "amve_detect_boundaries": AMVEAnalysisInput,
    "amve_detect_communication": AMVECommunicationInput,
    "amve_build_call_graph": AMVEAnalysisInput,
    "amve_evaluate_fitness": AMVEEvaluateFitnessInput,
    "amve_generate_architecture_log": AMVEGenerateArchitectureLogInput,
    # AEI-17: Dead code detection
    "amve_detect_dead_code": AMVEDetectDeadCodeInput,
    # Phase 2: Content-Addressed Snapshot Store (G2.12 GREEN)
    "amve_extract_architecture": AMVEExtractArchitectureInput,
    "amve_detect_drift": AMVEDetectDriftInput,
    # Audit Service tools (WBS-AEI13)
    "audit_security_scan": AuditSecurityScanInput,
    "audit_code_metrics": AuditCodeMetricsInput,
    "audit_corpus_search": AuditCorpusSearchInput,
    # AEI-18: Dependency assessment
    "audit_dependency_assess": AuditDependencyAssessInput,
    # AEI-20: Resolution lookup
    "audit_resolve_lookup": AuditResolveLookupInput,
    # AEI-23: VRE quarantine tools
    "audit_search_exploits": AuditSearchExploitsInput,
    "audit_search_cves": AuditSearchCVEsInput,
    # Phase 7: Quality audit (pattern compliance + anti-patterns)
    "audit_quality_scan": AuditQualityScanInput,
    # WBS-F7: Foundation search (scientific / theoretical layer)
    "foundation_search": FoundationSearchInput,
    # MCP Facade tools (MCP-F)
    "ask": AskInput,
    "search_in": SearchInInput,
    "find_code_pattern": FindCodePatternInput,
    # GitHub push tool
    "push_to_github": PushToGithubInput,
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
