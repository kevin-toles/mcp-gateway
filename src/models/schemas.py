"""Tool input/output Pydantic models — WBS-MCP1.6, MCP4 (GREEN).

Shared response models and all 9 tool input schemas with field-level
constraints, null-byte stripping, and Unicode NFC normalization.

Reference: Strategy §4.3 (Input Validation), §8 APP_SEC.API_SECURITY.INPUT_VALIDATION
"""

from pydantic import BaseModel, Field, field_validator

from src.security.cypher_validator import CypherValidationError, validate_cypher
from src.security.input_validators import sanitize_string


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    service: str
    version: str
    status: str
    uptime_seconds: float


# ── Shared sanitization validator ───────────────────────────────────────


def _sanitize_str_field(v: str) -> str:
    """Pydantic field_validator wrapper around sanitize_string."""
    if isinstance(v, str):
        return sanitize_string(v)
    return v


# ── Tool input schemas (AC-4.6: all 9) ─────────────────────────────────


class SemanticSearchInput(BaseModel):
    """Input for semantic_search tool."""

    query: str = Field(..., min_length=1, max_length=2000)
    collection: str = Field(default="all", pattern=r"^(code|docs|textbooks|all)$")
    top_k: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return _sanitize_str_field(v)


class HybridSearchInput(BaseModel):
    """Input for hybrid_search tool."""

    query: str = Field(..., min_length=1, max_length=2000)
    collection: str = Field(default="all", pattern=r"^(code|docs|textbooks|all)$")
    top_k: int = Field(default=10, ge=1, le=100)
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return _sanitize_str_field(v)


class CodeAnalyzeInput(BaseModel):
    """Input for code_analyze tool."""

    code: str = Field(..., min_length=1, max_length=100000)
    language: str = Field(default="", max_length=20)
    analysis_type: str = Field(
        default="all",
        pattern=r"^(complexity|dependencies|patterns|quality|security|all)$",
    )

    @field_validator("code", mode="before")
    @classmethod
    def sanitize_code(cls, v: str) -> str:
        return _sanitize_str_field(v)


class CodePatternAuditInput(BaseModel):
    """Input for code_pattern_audit tool."""

    code: str = Field(..., min_length=1, max_length=100000)
    language: str = Field(default="", max_length=20)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator("code", mode="before")
    @classmethod
    def sanitize_code(cls, v: str) -> str:
        return _sanitize_str_field(v)


class GraphQueryInput(BaseModel):
    """Input for graph_query tool."""

    cypher: str = Field(..., max_length=5000)
    parameters: dict = Field(default_factory=dict)

    @field_validator("cypher", mode="before")
    @classmethod
    def validate_cypher_field(cls, v: str) -> str:
        """Reject write operations and admin commands (AC-5.6 / MCP5.16)."""
        if isinstance(v, str):
            try:
                validate_cypher(v)
            except CypherValidationError as exc:
                raise ValueError(str(exc)) from exc
        return v


class LLMCompleteInput(BaseModel):
    """Input for llm_complete tool."""

    prompt: str = Field(..., min_length=1, max_length=50000)
    system_prompt: str = Field(default="", max_length=10000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    model_preference: str = Field(default="auto", pattern=r"^(auto|local|cloud)$")

    @field_validator("prompt", "system_prompt", mode="before")
    @classmethod
    def sanitize_prompts(cls, v: str) -> str:
        return _sanitize_str_field(v)


# Agent tool schemas (RunDiscussionInput, RunAgentFunctionInput,
# AgentExecuteInput) removed 2026-02-08: ai-agents is not in the
# MCP tool path. MCP tools dispatch direct to backend services.


# ── A2A Protocol tool schemas (MCP ↔ A2A bridge) ───────────────────────


class A2ASendMessageInput(BaseModel):
    """Input for a2a_send_message tool — create a task via A2A protocol."""

    content: str = Field(..., min_length=1, max_length=50000, description="Message content to send to the agent")
    skill_id: str = Field(default="", max_length=100, description="Target skill/function ID (optional)")
    context_id: str = Field(default="", max_length=100, description="Context ID for task grouping (optional)")

    @field_validator("content", mode="before")
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        return _sanitize_str_field(v)


class A2AGetTaskInput(BaseModel):
    """Input for a2a_get_task tool — retrieve task status and results."""

    task_id: str = Field(..., min_length=1, max_length=100, description="The A2A task ID to retrieve")


class A2ACancelTaskInput(BaseModel):
    """Input for a2a_cancel_task tool — cancel a running task."""

    task_id: str = Field(..., min_length=1, max_length=100, description="The A2A task ID to cancel")


# ── Workflow tool schemas (WBS-WF6) ─────────────────────────────────────


class ConvertPDFInput(BaseModel):
    """Input for convert_pdf tool — convert PDF to structured JSON."""

    input_path: str = Field(..., min_length=1, max_length=1000, description="Path to PDF file")
    output_path: str | None = Field(
        default=None, max_length=1000, description="Output JSON path (auto-generated if omitted)"
    )
    enable_ocr: bool = Field(default=True, description="Enable OCR fallback for image-only pages")

    @field_validator("input_path", mode="before")
    @classmethod
    def sanitize_input_path(cls, v: str) -> str:
        return _sanitize_str_field(v)


class ExtractBookMetadataInput(BaseModel):
    """Input for extract_book_metadata tool — extract metadata from book JSON."""

    input_path: str = Field(..., min_length=1, max_length=1000, description="Path to book JSON file")
    output_path: str | None = Field(
        default=None, max_length=1000, description="Output path (auto-generated if omitted)"
    )
    chapters: list[dict] | None = Field(default=None, description="Pre-defined chapter definitions (skips auto-detect)")
    options: dict | None = Field(default=None, description="Metadata extraction options")

    @field_validator("input_path", mode="before")
    @classmethod
    def sanitize_input_path(cls, v: str) -> str:
        return _sanitize_str_field(v)


class BatchExtractMetadataInput(BaseModel):
    """Input for batch_extract_metadata tool — batch extraction with progress."""

    input_dir: str = Field(..., min_length=1, max_length=1000, description="Directory containing raw book JSON files")
    output_dir: str | None = Field(
        default=None, max_length=1000, description="Output directory for metadata (defaults to sibling 'metadata' dir)"
    )
    file_pattern: str = Field(default="*.json", max_length=100, description="Glob pattern for book files")
    skip_existing: bool = Field(default=True, description="Skip books that already have metadata output files")

    @field_validator("input_dir", mode="before")
    @classmethod
    def sanitize_input_dir(cls, v: str) -> str:
        return _sanitize_str_field(v)


class GenerateTaxonomyInput(BaseModel):
    """Input for generate_taxonomy tool — generate concept taxonomy."""

    tier_books: dict[str, list[str]] = Field(
        ...,
        min_length=1,
        description="Mapping of tier name to list of book JSON file paths",
    )
    output_path: str | None = Field(
        default=None, max_length=1000, description="Output path (auto-generated if omitted)"
    )
    concepts: list[str] | None = Field(default=None, description="Custom concept list (overrides defaults)")
    domain: str = Field(
        default="auto",
        pattern=r"^(python|architecture|data_science|auto)$",
        description="Domain for concept list selection",
    )


class EnrichBookMetadataInput(BaseModel):
    """Input for enrich_book_metadata tool — enrich metadata via MSEP."""

    input_path: str = Field(..., min_length=1, max_length=1000, description="Path to WF1 metadata JSON file")
    output_path: str | None = Field(
        default=None, max_length=1000, description="Path to write enriched output (auto-generated if omitted)"
    )
    taxonomy_path: str | None = Field(default=None, max_length=1000, description="Path to WF2 taxonomy JSON file")
    mode: str = Field(default="msep", pattern=r"^(msep|basic)$", description="Enrichment mode")

    @field_validator("input_path", mode="before")
    @classmethod
    def sanitize_input_path(cls, v: str) -> str:
        return _sanitize_str_field(v)


class EnhanceGuidelineInput(BaseModel):
    """Input for enhance_guideline tool — LLM-powered guideline enhancement."""

    aggregate_path: str = Field(..., min_length=1, max_length=1000, description="Path to aggregate package JSON")
    guideline_path: str = Field(..., min_length=1, max_length=1000, description="Path to guideline JSON")
    output_dir: str = Field(default="output", max_length=1000, description="Output directory for enhanced markdown")
    provider: str = Field(
        default="gateway",
        pattern=r"^(gateway|anthropic|local)$",
        description="LLM provider name",
    )
    max_tokens: int = Field(default=4096, ge=256, le=32768, description="Maximum tokens per LLM call")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")

    @field_validator("aggregate_path", "guideline_path", mode="before")
    @classmethod
    def sanitize_paths(cls, v: str) -> str:
        return _sanitize_str_field(v)


# ── Taxonomy Analysis tool schema (WBS-TAP9) ───────────────────────────


class AnalyzeTaxonomyCoverageInput(BaseModel):
    """Input for analyze_taxonomy_coverage tool — taxonomy coverage analysis."""

    taxonomy_path: str = Field(..., min_length=1, max_length=1000, description="Path to taxonomy JSON file")
    output_path: str | None = Field(
        default=None, max_length=1000, description="Output path for report JSON (auto-generated if omitted)"
    )
    collection: str = Field(default="all", description="Search collection to query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results per query")
    threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity threshold")
    max_leaf_nodes: int = Field(
        default=500, ge=1, le=5000, description="Maximum leaf nodes before requiring subtree_root"
    )
    subtree_root: str | None = Field(
        default=None, max_length=500, description="Limit analysis to a subtree rooted at this node"
    )
    concurrency: int = Field(default=10, ge=1, le=50, description="Max concurrent search queries")
    include_evidence: bool = Field(default=True, description="Include search evidence in report")
    scoring_weights: dict[str, float] | None = Field(
        default=None, description="Custom scoring weights {breadth, depth, spread}"
    )

    @field_validator("taxonomy_path", mode="before")
    @classmethod
    def sanitize_taxonomy_path(cls, v: str) -> str:
        return _sanitize_str_field(v)


# ── AMVE tool schemas (AEI-7) ──────────────────────────────────────────


class AMVEAnalysisInput(BaseModel):
    """Input for amve_detect_patterns, amve_detect_boundaries, amve_build_call_graph."""

    source_path: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Path to source directory or file to analyze",
    )
    include_confidence: bool = Field(
        default=False,
        description="Include confidence scores in results",
    )

    @field_validator("source_path", mode="before")
    @classmethod
    def sanitize_source_path(cls, v: str) -> str:
        return _sanitize_str_field(v)


class AMVECommunicationInput(BaseModel):
    """Input for amve_detect_communication -- consolidated events + messaging."""

    source_path: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Path to source directory or file to analyze",
    )
    scope: str = Field(
        default="all",
        pattern=r"^(all|events|messaging)$",
        description="Detection scope: events, messaging, or all",
    )
    include_confidence: bool = Field(
        default=False,
        description="Include confidence scores in results",
    )

    @field_validator("source_path", mode="before")
    @classmethod
    def sanitize_source_path(cls, v: str) -> str:
        return _sanitize_str_field(v)


class AMVEEvaluateFitnessInput(BaseModel):
    """Input for amve_evaluate_fitness -- evaluate fitness functions against a snapshot."""

    snapshot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Architecture snapshot ID to evaluate",
    )
    fitness_function_ids: list[str] | None = Field(
        default=None,
        description="Optional list of specific fitness function IDs to evaluate",
    )

    @field_validator("snapshot_id", mode="before")
    @classmethod
    def sanitize_snapshot_id(cls, v: str) -> str:
        return _sanitize_str_field(v)


class AMVEGenerateArchitectureLogInput(BaseModel):
    """Input for amve_generate_architecture_log -- batch scan with baseline comparison."""

    source_paths: list[str] = Field(
        ...,
        min_length=1,
        description="List of source paths to scan",
    )
    violations: list[dict] = Field(
        default_factory=list,
        description="Known violations to include in scan",
    )
    patterns: list[dict] = Field(
        default_factory=list,
        description="Known patterns to include in scan",
    )
    baseline_json: dict | None = Field(
        default=None,
        description="Optional baseline snapshot for delta comparison",
    )


# ── Audit Service schemas (WBS-AEI13) ──────────────────────────────────────


class AuditSecurityScanInput(BaseModel):
    """Input for audit_security_scan — scan source code for security vulnerabilities."""

    code: str = Field(..., description="Source code to scan")
    language: str = Field(default="python", description="Programming language")
    mode: str = Field(default="quick", description="Scan mode: quick or full")
    domains: list[str] | None = Field(
        default=None,
        description="Optional security domains to scan (e.g. injection, secrets)",
    )
    severity_threshold: str = Field(
        default="low",
        description="Minimum severity to report: low, medium, high, critical",
    )


class AuditCodeMetricsInput(BaseModel):
    """Input for audit_code_metrics — compute per-pillar code metrics."""

    code: str = Field(..., description="Source code to analyse")
    language: str = Field(default="python", description="Programming language")
    pillars: list[str] = Field(
        default_factory=lambda: ["structural", "architectural", "eloquence"],
        description="Pillars to compute: structural, architectural, eloquence",
    )


class AuditCorpusSearchInput(BaseModel):
    """Input for audit_corpus_search — search code corpus via Qdrant."""

    query: str = Field(..., description="Search query text")
    collections: list[str] = Field(
        default_factory=lambda: ["code_chunks", "chapters"],
        description="Qdrant collections to search",
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Max results per collection")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")


# ── AMVE Dead Code Detection (AEI-17) ───────────────────────────────────────


class AMVEDetectDeadCodeInput(BaseModel):
    """Input for amve_detect_dead_code — detect dead code in a source tree."""

    source_path: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Path to source directory or file to analyse for dead code",
    )
    include_unused_imports: bool = Field(
        default=True,
        description="Whether to include unused-import analysis (tree-sitter)",
    )


# -- Dependency Assessment (AEI-18) ----------------------------------------


class AuditDependencyAssessInput(BaseModel):
    """Input for audit_dependency_assess -- assess declared dependency health.

    Scans the project at *source_path*, parses the manifest, and computes
    Martin instability, import ratio, and dependency weight per package.
    Returns a health score, zone classification, and violation list.
    """

    source_path: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Absolute path to the Python source directory to scan for imports.",
    )
    manifest_path: str | None = Field(
        default=None,
        description=(
            "Absolute path to pyproject.toml or requirements.txt. If omitted the tool auto-detects from source_path."
        ),
    )
    include_transitive: bool = Field(
        default=True,
        description=(
            "Whether to include transitive dependency counting via pip show. "
            "Set False to skip transitive BFS for faster scans."
        ),
    )


# -- Resolution Lookup (AEI-20) ----------------------------------------


class AuditResolveLookupInput(BaseModel):
    """Input for audit_resolve_lookup — look up resolution evidence chain.

    Searches the resolution knowledge base (Qdrant :6335) to return the full
    evidence chain: violation → principle → resolution pattern → code examples.
    """

    violation_type: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Violation type identifier or description to look up. "
            "E.g. 'sql_injection', 'DEP_LOW_RATIO', 'circular_dependency'."
        ),
    )
    pillar: str | None = Field(
        default=None,
        description=("Optional pillar hint: structural, architectural, eloquence, security, or dependency."),
    )
    include_code_examples: bool = Field(
        default=True,
        description=(
            "When True (default), fetches matching code examples from the code_chunks corpus via semantic-search."
        ),
    )
