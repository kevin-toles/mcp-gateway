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
