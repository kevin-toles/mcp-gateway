"""Tool input/output Pydantic models — WBS-MCP1.6, MCP4 (GREEN).

Shared response models and all 9 tool input schemas with field-level
constraints, null-byte stripping, and Unicode NFC normalization.

Reference: Strategy §4.3 (Input Validation), §8 APP_SEC.API_SECURITY.INPUT_VALIDATION
"""

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """Input for semantic_search tool.

    **Parameter compatibility:** Accepts BOTH `top_k` and `limit` interchangeably
    to prevent LLM validation errors. Canonical form is `top_k`.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    collection: str = Field(default="all", pattern=r"^(code|docs|textbooks|all)$")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Max results (canonical)")
    limit: int | None = Field(default=None, ge=1, le=100, description="Max results (alias for top_k)")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return _sanitize_str_field(v)

    @model_validator(mode="after")
    def normalize_result_count(self) -> "SemanticSearchInput":
        """Accept top_k OR limit, normalize to top_k (canonical form)."""
        if self.top_k is None and self.limit is None:
            self.top_k = 10  # default
        elif self.top_k is None:
            self.top_k = self.limit  # normalize limit → top_k
        elif self.limit is not None and self.limit != self.top_k:
            raise ValueError("Cannot specify both top_k and limit with different values")
        self.limit = None  # clear alias
        return self


class HybridSearchInput(BaseModel):
    """Input for hybrid_search tool — WBS-TXS5 (tier params added).

    **Parameter compatibility:** Accepts BOTH `top_k` and `limit` interchangeably
    to prevent LLM validation errors. Canonical form is `top_k`.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    collection: str = Field(default="all", pattern=r"^(code|docs|textbooks|all)$")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Max results (canonical)")
    limit: int | None = Field(default=None, ge=1, le=100, description="Max results (alias for top_k)")
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    # TXS5: Taxonomy-enhanced search parameters
    bloom_tier_filter: list[int] | None = Field(
        default=None,
        description="Bloom cognitive tier filter for chapters (int 0-6: T0=0 … T6=6)",
    )
    quality_tier_filter: list[int] | None = Field(
        default=None,
        description="CRE repo quality tier filter for code_chunks (1=flagship, 2=standard, 3=supplemental)",
    )
    bloom_tier_boost: bool = Field(
        default=True,
        description="Apply Bloom/CRE tier-based score boosting to results",
    )
    # ── Graph control ──────────────────────────────────────────────────────
    include_graph: bool = Field(
        default=True,
        description="Include Neo4j graph-based scoring in result fusion (set False for pure vector search)",
    )
    # ── MMR reranking (score/MMR traversal style) ──────────────────────────
    mmr_rerank: bool = Field(
        default=False,
        description="Apply Maximal Marginal Relevance (MMR) reranking for diversity (score/MMR traversal)",
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR lambda: 0.0=maximum diversity, 1.0=maximum relevance (only used when mmr_rerank=True)",
    )
    # ── Taxonomy query expansion (TXS6) ───────────────────────────────────
    expand_taxonomy: bool = Field(
        default=False,
        description="Expand query using Neo4j SIMILAR_TO relationships between TaxonomyConcept nodes",
    )
    # ── Custom traversal / domain focus ───────────────────────────────────
    focus_areas: list[str] | None = Field(
        default=None,
        description="Domain focus areas for relevance scoring (e.g. ['llm_rag', 'microservices_architecture'])",
    )
    focus_keywords: list[str] | None = Field(
        default=None,
        description="Custom keywords to boost relevance scoring in result fusion",
    )

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return _sanitize_str_field(v)

    @field_validator("bloom_tier_filter", mode="before")
    @classmethod
    def validate_bloom_tier_filter(cls, v: list[int] | None) -> list[int] | None:
        """Enforce bloom tier range 0-6."""
        if v is not None:
            for tier in v:
                if tier < 0 or tier > 6:
                    msg = f"bloom_tier_filter value {tier} out of range 0-6"
                    raise ValueError(msg)
        return v

    @field_validator("quality_tier_filter", mode="before")
    @classmethod
    def validate_quality_tier_filter(cls, v: list[int] | None) -> list[int] | None:
        """Enforce CRE quality tier range 1-3."""
        if v is not None:
            for tier in v:
                if tier < 1 or tier > 3:
                    msg = f"quality_tier_filter value {tier} out of range 1-3"
                    raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def normalize_result_count(self) -> "HybridSearchInput":
        """Accept top_k OR limit, normalize to top_k (canonical form)."""
        if self.top_k is None and self.limit is None:
            self.top_k = 10  # default
        elif self.top_k is None:
            self.top_k = self.limit  # normalize limit → top_k
        elif self.limit is not None and self.limit != self.top_k:
            raise ValueError("Cannot specify both top_k and limit with different values")
        self.limit = None  # clear alias
        return self


class GraphTraverseInput(BaseModel):
    """Input for graph_traverse tool — BFS + optional MMR graph traversal."""

    start_nodes: list[str] = Field(
        ...,
        min_length=1,
        description="Starting Neo4j node IDs for BFS graph traversal",
    )
    relationship_types: list[str] | None = Field(
        default=None,
        description="Relationship types to follow (e.g. ['SIMILAR_TO', 'REQUIRES']); None = all types",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum BFS traversal depth",
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of nodes to return",
    )
    mmr_enabled: bool = Field(
        default=False,
        description="Enable MMR diversity-aware traversal (true = MMR traversal, false = basic BFS)",
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR lambda: 0.0=maximum diversity, 1.0=maximum relevance (only used when mmr_enabled=True)",
    )

    @field_validator("start_nodes", mode="before")
    @classmethod
    def sanitize_start_nodes(cls, v: list[str]) -> list[str]:
        return [sanitize_string(n) if isinstance(n, str) else n for n in v]


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
    """Input for generate_taxonomy tool — build full taxonomy from enriched corpus.

    Reads all *_enriched.json files in enriched_dir, aggregates keywords/concepts,
    applies quality gates, and writes the full uber_taxonomy format.
    """

    enriched_dir: str = Field(
        default="/Users/kevintoles/POC/ai-platform-data/collections/software-engineering/enriched",
        min_length=1,
        max_length=1000,
        description="Directory containing *_enriched.json files",
    )
    output_path: str | None = Field(
        default=None, max_length=1000, description="Output taxonomy JSON path (auto-generated if omitted)"
    )

    @field_validator("enriched_dir", mode="before")
    @classmethod
    def sanitize_enriched_dir(cls, v: str) -> str:
        return _sanitize_str_field(v)


class EnrichBookMetadataInput(BaseModel):
    """Input for enrich_book_metadata tool — enrich metadata via CO pipeline (DMA-§2.3)."""

    input_path: str = Field(..., min_length=1, max_length=1000, description="Path to WF1 metadata JSON file")
    output_path: str | None = Field(
        default=None, max_length=1000, description="Path to write enriched output (auto-generated if omitted)"
    )
    taxonomy_path: str | None = Field(default=None, max_length=1000, description="Path to taxonomy JSON file")
    mode: str = Field(default="direct_co", description="Enrichment mode (informational)")

    @field_validator("input_path", mode="before")
    @classmethod
    def sanitize_input_path(cls, v: str) -> str:
        return _sanitize_str_field(v)


class BatchEnrichMetadataInput(BaseModel):
    """Input for batch_enrich_metadata tool — batch enrich all metadata files in a directory."""

    metadata_dir: str = Field(
        default="/Users/kevintoles/POC/ai-platform-data/collections/software-engineering/metadata",
        min_length=1,
        max_length=1000,
        description="Directory containing *_metadata.json files",
    )
    output_dir: str = Field(
        default="/Users/kevintoles/POC/ai-platform-data/collections/software-engineering/enriched",
        min_length=1,
        max_length=1000,
        description="Output directory for enriched JSON files",
    )
    taxonomy_path: str | None = Field(default=None, max_length=1000, description="Path to taxonomy JSON file")
    mode: str = Field(
        default="auto",
        pattern=r"^(auto|software|foundation)$",
        description=(
            "Enrichment mode: 'auto' (detect from metadata_dir path — "
            "ext_metadata → foundation, else software), 'software' (default KB), "
            "'foundation' (scientific collections, auto-sets vtf/seed/classifier/"
            "graphcodebert unless overridden)."
        ),
    )
    vtf_path: str | None = Field(
        default=None,
        max_length=1000,
        description=(
            "Path to validated term filter JSON (overrides CO default; "
            "pass ESC_validated_term_filter.json for scientific collections)"
        ),
    )
    seed_concepts: list[str] | None = Field(
        default=None,
        description=(
            "SBERT anchor concepts for ConceptValidator "
            "(overrides SEED_PROGRAMMING_CONCEPTS; pass SEED_SCIENTIFIC_CONCEPTS "
            "for scientific collections)"
        ),
    )
    classifier_enabled: bool = Field(
        default=True, description="Enable HTC LightGBM classifier (set False for scientific collections)"
    )
    graphcodebert_enabled: bool = Field(
        default=True, description="Enable GraphCodeBERT validation (set False for scientific collections)"
    )
    raw_content_dir: str | None = Field(
        default=None,
        max_length=1000,
        description=(
            "Directory containing raw book JSONs (with 'content' per chapter). "
            "Required for foundation collections (e.g. ext_raw/<domain>/). "
            "Omit for software KB."
        ),
    )
    resume: bool = Field(default=True, description="Skip books that already have an enriched output file")
    limit: int = Field(default=0, ge=0, description="Cap at N books (0 = no limit)")
    book: str = Field(default="", max_length=500, description="Process only books whose filename contains this string")

    @field_validator("metadata_dir", "output_dir", mode="before")
    @classmethod
    def sanitize_dir(cls, v: str) -> str:
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
    """Input for audit_corpus_search — search code corpus via Qdrant.

    **Parameter compatibility:** Accepts BOTH `top_k` and `limit` interchangeably
    to prevent LLM validation errors. Canonical form is `top_k`.
    """

    query: str = Field(..., description="Search query text")
    collections: list[str] = Field(
        default_factory=lambda: ["code_chunks", "chapters"],
        description="Qdrant collections to search",
    )
    top_k: int | None = Field(default=None, ge=1, le=100, description="Max results per collection (canonical)")
    limit: int | None = Field(default=None, ge=1, le=100, description="Max results per collection (alias for top_k)")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")

    @model_validator(mode="after")
    def normalize_result_count(self) -> "AuditCorpusSearchInput":
        """Accept top_k OR limit, normalize to top_k (canonical form)."""
        if self.top_k is None and self.limit is None:
            self.top_k = 10  # default
        elif self.top_k is None:
            self.top_k = self.limit  # normalize limit → top_k
        elif self.limit is not None and self.limit != self.top_k:
            raise ValueError("Cannot specify both top_k and limit with different values")
        self.limit = None  # clear alias
        return self


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


# -- Phase 2: Content-Addressed Snapshot Store (G2.2 GREEN) ---------------


class AMVEExtractArchitectureInput(BaseModel):
    """Input for amve_extract_architecture — extract architecture snapshot and compute SHA."""

    source_path: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Absolute path to the source directory to extract the architecture from.",
    )

    @field_validator("source_path", mode="before")
    @classmethod
    def sanitize_source_path(cls, v: str) -> str:
        return _sanitize_str_field(v)


class AMVEDetectDriftInput(BaseModel):
    """Input for amve_detect_drift — detect drift between two architecture snapshots.

    Accepts either:
    - SHA-pair mode: ``snapshot_a_sha`` + ``snapshot_b_sha`` (retrieves from Redis stream)
    - Passthrough mode: ``snapshot_a`` + ``snapshot_b`` dicts (calls AMVE drift directly)
    """

    snapshot_a_sha: str | None = Field(
        default=None,
        description="SHA-256 hex of the first (baseline) snapshot stored in the stream.",
    )
    snapshot_b_sha: str | None = Field(
        default=None,
        description="SHA-256 hex of the second (current) snapshot stored in the stream.",
    )
    snapshot_a: dict | None = Field(
        default=None,
        description="Raw baseline snapshot dict for passthrough mode (no Redis lookup).",
    )
    snapshot_b: dict | None = Field(
        default=None,
        description="Raw current snapshot dict for passthrough mode (no Redis lookup).",
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


# -- VRE Quarantine Tools (AEI-23) ----------------------------------------


class AuditToolBase(BaseModel):
    """Shared base for audit search tools with result-ranking controls.

    PDW3.9 REFACTOR: Extracts common ``top_k`` and ``min_similarity``
    pagination/ranking fields so VRE search schemas don't redeclare them.

    **Parameter compatibility:** Accepts BOTH `top_k` and `limit` interchangeably
    to prevent LLM validation errors. Canonical form is `top_k`.
    """

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of results to return (canonical).",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of results to return (alias for top_k).",
    )
    min_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold for returned results.",
    )

    @model_validator(mode="after")
    def normalize_result_count(self) -> "AuditToolBase":
        """Accept top_k OR limit, normalize to top_k (canonical form)."""
        if self.top_k is None and self.limit is None:
            self.top_k = 10  # default
        elif self.top_k is None:
            self.top_k = self.limit  # normalize limit → top_k
        elif self.limit is not None and self.limit != self.top_k:
            raise ValueError("Cannot specify both top_k and limit with different values")
        self.limit = None  # clear alias
        return self


class AuditSearchExploitsInput(AuditToolBase):
    """Input for audit_search_exploits — search quarantine Qdrant for exploit vectors.

    Searches the vuln_exploits collection on qdrant-quarantine (:6336) using
    CodeBERT vector embeddings. Returns ranked exploit matches with CVE
    cross-references.

    **Parameter compatibility:** Inherits dual `top_k`/`limit` support from AuditToolBase.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "Natural-language description of the vulnerability pattern to search "
            "for (e.g. 'sql injection unsanitised user input')."
        ),
    )
    cwe_ids: list[str] | None = Field(
        default=None,
        description=("Optional list of CWE IDs to restrict results (e.g. ['CWE-89', 'CWE-79'])."),
    )
    # top_k and min_similarity inherited from AuditToolBase (PDW3.9 REFACTOR)


class AuditSearchCVEsInput(BaseModel):
    """Input for audit_search_cves — query PostgreSQL for CVE records.

    Filters the vuln_cve_records table by CWE ID, severity, and/or ecosystem.
    Returns structured CVE records with CVSS scores and references.
    """

    cwe_id: str | None = Field(
        default=None,
        description="Filter by CWE identifier (e.g. 'CWE-89'). Optional.",
    )
    severity: str | None = Field(
        default=None,
        description=("Filter by severity level ('critical', 'high', 'medium', 'low'). Optional."),
    )
    ecosystem: str | None = Field(
        default=None,
        description=("Filter by affected ecosystem (e.g. 'python', 'npm', 'java'). Optional."),
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of CVE records to return (default 50).",
    )

    @model_validator(mode="after")
    def require_at_least_one_filter(self) -> "AuditSearchCVEsInput":
        """Reject queries with no filter criteria.

        AC-PDW3.4: At least one of ``cwe_id``, ``severity``, or ``ecosystem``
        must be provided.  A fully unfiltered CVE query would scan the entire
        ``vuln_cve_records`` table and is disallowed for both performance and
        security-signal clarity reasons.
        """
        if self.cwe_id is None and self.severity is None and self.ecosystem is None:
            raise ValueError("At least one filter is required: provide cwe_id, severity, or ecosystem.")
        return self


# -- Quality Audit (Phase 7) -----------------------------------------------


class AuditQualityScanInput(BaseModel):
    """Input for audit_quality_scan — pattern compliance and anti-pattern detection.

    Runs CP001-CP011 (coding patterns), SA001-SA008 (static analysis),
    SEC001-SEC008 (security rules), and structural anti-pattern detection
    (Blob, Lava Flow, Boat Anchor, Redundant Wrapper, Premature Abstraction).
    """

    code: str = Field(..., description="Source code to audit")
    language: str = Field(
        default="python",
        description="Language hint (currently Python AST only)",
    )
    rule_categories: list[str] | None = Field(
        default=None,
        description=(
            "Restrict to specific rule categories: function_design, naming, "
            "idiom, structure, test_quality, api_design, architecture, security. "
            "Null means all categories."
        ),
    )
    severity_threshold: str = Field(
        default="info",
        description="Minimum severity to include: info | warning | error",
    )
    include_antipatterns: bool = Field(
        default=True,
        description="Run structural anti-pattern detector (Blob, Lava Flow, etc.)",
    )


# ── Three consolidated KB tools ────────────────────────────────────────────


class KnowledgeSearchInput(BaseModel):
    """Input for knowledge_search — batteries-included KB search with taxonomy expansion.

    **Parameter compatibility:** Accepts BOTH `limit` and `top_k` interchangeably
    to prevent LLM validation errors. Canonical form is `limit`.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    limit: int | None = Field(default=None, ge=1, le=50, description="Max results (canonical)")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Max results (alias for limit)")
    expand_taxonomy: bool = Field(
        default=True,
        description="Expand query via Neo4j SIMILAR_TO edges (default ON for broad KB retrieval)",
    )
    mmr_rerank: bool = Field(
        default=False,
        description="Apply MMR diversity reranking across the fan-out results",
    )
    mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR lambda: 0.0=max diversity, 1.0=max relevance",
    )
    bloom_tier_filter: list[int] | None = Field(
        default=None,
        description="Filter chapters by Bloom cognitive tier (0=Foundational … 6=Innovation)",
    )

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return _sanitize_str_field(v)

    @field_validator("bloom_tier_filter", mode="before")
    @classmethod
    def validate_bloom_tier(cls, v: list[int] | None) -> list[int] | None:
        if v is not None:
            for tier in v:
                if tier < 0 or tier > 6:
                    raise ValueError(f"bloom_tier_filter value {tier} out of range 0-6")
        return v

    @model_validator(mode="after")
    def normalize_result_count(self) -> "KnowledgeSearchInput":
        """Accept limit OR top_k, normalize to limit (canonical form)."""
        if self.limit is None and self.top_k is None:
            self.limit = 10  # default
        elif self.limit is None:
            self.limit = self.top_k  # normalize top_k → limit
        elif self.top_k is not None and self.top_k != self.limit:
            raise ValueError("Cannot specify both limit and top_k with different values")
        self.top_k = None  # clear alias
        return self


class KnowledgeRefineInput(BaseModel):
    """Input for knowledge_refine — targeted single-collection KB search.

    **Parameter compatibility:** Accepts BOTH `limit` and `top_k` interchangeably
    to prevent LLM validation errors. Canonical form is `limit`.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    collection: str = Field(
        default="chapters",
        pattern=r"^(chapters|textbooks|code_chunks|code|pattern_instances|patterns|code_good_patterns|repo_concepts|concepts)$",
        description=(
            "Collection to search: chapters/textbooks, code_chunks/code, "
            "pattern_instances/patterns, code_good_patterns, repo_concepts/concepts"
        ),
    )
    limit: int | None = Field(default=None, ge=1, le=20, description="Max results (canonical)")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Max results (alias for limit)")
    bloom_tier_filter: list[int] | None = Field(
        default=None,
        description="Filter chapters by Bloom cognitive tier (0-6)",
    )
    quality_tier_filter: list[int] | None = Field(
        default=None,
        description="Filter code_chunks by CRE quality tier (1=flagship, 2=standard, 3=supplemental)",
    )
    mmr_rerank: bool = Field(
        default=True,
        description="Apply MMR reranking for diversity within the single collection",
    )

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return _sanitize_str_field(v)

    @field_validator("bloom_tier_filter", mode="before")
    @classmethod
    def validate_bloom_tier(cls, v: list[int] | None) -> list[int] | None:
        if v is not None:
            for tier in v:
                if tier < 0 or tier > 6:
                    raise ValueError(f"bloom_tier_filter value {tier} out of range 0-6")
        return v

    @field_validator("quality_tier_filter", mode="before")
    @classmethod
    def validate_quality_tier(cls, v: list[int] | None) -> list[int] | None:
        if v is not None:
            for tier in v:
                if tier < 1 or tier > 3:
                    raise ValueError(f"quality_tier_filter value {tier} out of range 1-3")
        return v

    @model_validator(mode="after")
    def normalize_result_count(self) -> "KnowledgeRefineInput":
        """Accept limit OR top_k, normalize to limit (canonical form)."""
        if self.limit is None and self.top_k is None:
            self.limit = 5  # default
        elif self.limit is None:
            self.limit = self.top_k  # normalize top_k → limit
        elif self.top_k is not None and self.top_k != self.limit:
            raise ValueError("Cannot specify both limit and top_k with different values")
        self.top_k = None  # clear alias
        return self


class PatternSearchInput(BaseModel):
    """Input for pattern_search — code pattern and anti-pattern retrieval.

    **Parameter compatibility:** Accepts BOTH `limit` and `top_k` interchangeably
    to prevent LLM validation errors. Canonical form is `limit`.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    pattern_type: str = Field(
        default="all",
        pattern=r"^(good|bad|all)$",
        description=(
            "Which pattern set to search: "
            "'good' = code_good_patterns only, "
            "'bad' = pattern_instances only, "
            "'all' = both (fan-out across all primary collections)"
        ),
    )
    limit: int | None = Field(default=None, ge=1, le=30, description="Max results (canonical)")
    top_k: int | None = Field(default=None, ge=1, le=30, description="Max results (alias for limit)")

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return _sanitize_str_field(v)

    @model_validator(mode="after")
    def normalize_result_count(self) -> "PatternSearchInput":
        """Accept limit OR top_k, normalize to limit (canonical form)."""
        if self.limit is None and self.top_k is None:
            self.limit = 10  # default
        elif self.limit is None:
            self.limit = self.top_k  # normalize top_k → limit
        elif self.top_k is not None and self.top_k != self.limit:
            raise ValueError("Cannot specify both limit and top_k with different values")
        self.top_k = None  # clear alias
        return self


class DiagramSearchInput(BaseModel):
    """Input for diagram_search — semantic search over ASCII / sequence diagrams.

    Searches the ``ascii_diagrams`` Qdrant collection using CLIP text encoding
    so query semantics align with the 512-dim CLIP image vectors stored there.

    **Parameter compatibility:** Accepts BOTH `limit` and `top_k` interchangeably
    to prevent LLM validation errors. Canonical form is `limit`.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    diagram_type: str | None = Field(
        default=None,
        pattern=r"^(ascii|sequence|box_flow)$",
        description=(
            "Optional filter on diagram_type payload field: "
            "'ascii' = plain ASCII art, "
            "'sequence' = UML-style sequence diagrams, "
            "'box_flow' = box-and-arrow flow diagrams. "
            "Omit (null) to search all diagram types."
        ),
    )
    limit: int | None = Field(default=None, ge=1, le=30, description="Max results (canonical)")
    top_k: int | None = Field(default=None, ge=1, le=30, description="Max results (alias for limit)")

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return _sanitize_str_field(v)

    @model_validator(mode="after")
    def normalize_result_count(self) -> "DiagramSearchInput":
        """Accept limit OR top_k, normalize to limit (canonical form)."""
        if self.limit is None and self.top_k is None:
            self.limit = 10  # default
        elif self.limit is None:
            self.limit = self.top_k  # normalize top_k → limit
        elif self.top_k is not None and self.top_k != self.limit:
            raise ValueError("Cannot specify both limit and top_k with different values")
        # Clear the alias after normalization (optional - keeps payload clean)
        self.top_k = None
        return self


# ── MCP Facade Input Models (MCP-F) ────────────────────────────────────


class AskInput(BaseModel):
    """Input for the `ask` facade tool — intent-level KB question.

    Wraps ``hybrid_search`` with a friendly difficulty hint instead of
    raw ``bloom_tier_filter`` integers.
    """

    query: str
    max_results: int = Field(default=10, ge=1, le=50)
    difficulty: str | None = None


class SearchInInput(BaseModel):
    """Input for the `search_in` facade tool — shelf-targeted KB search.

    Wraps ``knowledge_refine`` with human-readable shelf names instead of
    raw collection identifiers.
    """

    query: str
    source: str = "textbooks"
    max_results: int = Field(default=5, ge=1, le=30)


class FindCodePatternInput(BaseModel):
    """Input for the `find_code_pattern` facade tool — code example search.

    Wraps ``pattern_search`` with the friendly ``examples`` parameter instead
    of the internal ``pattern_type`` field.
    """

    query: str
    examples: str = "both"


class FoundationSearchInput(BaseModel):
    """Input for the ``foundation_search`` tool — WBS-F7.

    Routes to USS ``/v1/search/foundation`` for mathematical, statistical,
    or theoretical underpinnings of software concepts.

    **Parameter compatibility:** Accepts BOTH ``limit`` and ``top_k``
    interchangeably to prevent LLM validation errors. Canonical form is
    ``limit``.
    """

    query: str = Field(..., min_length=1, max_length=2000)
    domains: list[str] | None = Field(default=None)
    include_graph_neighbors: bool = Field(default=False)
    limit: int | None = Field(default=None, ge=1, le=50, description="Max results (canonical)")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Max results (alias for limit)")

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return _sanitize_str_field(v)

    @model_validator(mode="after")
    def normalize_result_count(self) -> "FoundationSearchInput":
        """Accept limit OR top_k, normalize to limit (canonical form)."""
        if self.limit is None and self.top_k is None:
            self.limit = 5  # default
        elif self.limit is None:
            self.limit = self.top_k  # normalize top_k → limit
        elif self.top_k is not None and self.top_k != self.limit:
            raise ValueError("Cannot specify both limit and top_k with different values")
        self.top_k = None
        return self
