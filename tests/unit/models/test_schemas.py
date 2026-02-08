"""Tests for tool input schemas — WBS-MCP4 (RED).

Covers all 6 Acceptance Criteria:
- AC-4.1: Every tool has a Pydantic input model with field-level constraints
- AC-4.2: query fields: min 1 char, max 2000, null bytes stripped, Unicode NFC
- AC-4.3: collection fields: allow-list (code|docs|textbooks|all)
- AC-4.4: Numeric fields bounded (top_k 1-100, threshold 0.0-1.0, etc.)
- AC-4.5: Invalid inputs return 422 with field-level messages (no stack traces)
- AC-4.6: All 9 tool input schemas defined and validated

Technical debt: None — input validation is standalone.
"""

import unicodedata

import pytest
from pydantic import ValidationError


# ═══════════════════════════════════════════════════════════════════════
# AC-4.1 / AC-4.6: All 9 tool schemas exist
# ═══════════════════════════════════════════════════════════════════════


class TestAllSchemasExist:
    """AC-4.6: All 9 tool input schemas must be importable."""

    @pytest.mark.parametrize("name", [
        "SemanticSearchInput",
        "HybridSearchInput",
        "CodeAnalyzeInput",
        "CodePatternAuditInput",
        "GraphQueryInput",
        "LLMCompleteInput",
        "RunAgentFunctionInput",
        "RunDiscussionInput",
        "AgentExecuteInput",
    ])
    def test_schema_importable(self, name):
        from src.models import schemas
        assert hasattr(schemas, name), f"{name} not found in schemas module"

    def test_exactly_nine_tool_schemas(self):
        """There should be exactly 9 tool input models."""
        from src.models import schemas
        tool_models = [
            n for n in dir(schemas)
            if n.endswith("Input") and not n.startswith("_")
        ]
        assert len(tool_models) == 9


# ═══════════════════════════════════════════════════════════════════════
# AC-4.2: SemanticSearchInput — query field validation
# ═══════════════════════════════════════════════════════════════════════


class TestSemanticSearchInput:
    """AC-4.2: query min 1 / max 2000, null bytes stripped, Unicode NFC."""

    def test_valid_input(self):
        from src.models.schemas import SemanticSearchInput
        m = SemanticSearchInput(query="find error handling patterns")
        assert m.query == "find error handling patterns"
        assert m.collection == "all"
        assert m.top_k == 10
        assert m.threshold == 0.5

    def test_empty_query_rejected(self):
        from src.models.schemas import SemanticSearchInput
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchInput(query="")
        assert "query" in str(exc_info.value).lower()

    def test_query_too_long_rejected(self):
        from src.models.schemas import SemanticSearchInput
        with pytest.raises(ValidationError):
            SemanticSearchInput(query="x" * 2001)

    def test_query_max_length_accepted(self):
        from src.models.schemas import SemanticSearchInput
        m = SemanticSearchInput(query="x" * 2000)
        assert len(m.query) == 2000

    def test_null_bytes_stripped(self):
        from src.models.schemas import SemanticSearchInput
        m = SemanticSearchInput(query="hello\x00world")
        assert "\x00" not in m.query
        assert m.query == "helloworld"

    def test_unicode_normalized_nfc(self):
        """Combining characters should be normalized to NFC."""
        from src.models.schemas import SemanticSearchInput
        # é as e + combining acute (NFD) → é as single char (NFC)
        nfd = "e\u0301"
        m = SemanticSearchInput(query=nfd)
        assert m.query == unicodedata.normalize("NFC", nfd)

    def test_query_required(self):
        from src.models.schemas import SemanticSearchInput
        with pytest.raises(ValidationError):
            SemanticSearchInput()


# ═══════════════════════════════════════════════════════════════════════
# AC-4.3: Collection allow-list
# ═══════════════════════════════════════════════════════════════════════


class TestCollectionField:
    """AC-4.3: collection restricted to (code|docs|textbooks|all)."""

    @pytest.mark.parametrize("value", ["code", "docs", "textbooks", "all"])
    def test_valid_collection_accepted(self, value):
        from src.models.schemas import SemanticSearchInput
        m = SemanticSearchInput(query="test", collection=value)
        assert m.collection == value

    @pytest.mark.parametrize("value", [
        "../../etc",
        "secrets",
        "CODE",
        "All",
        "",
        "code; DROP TABLE",
    ])
    def test_invalid_collection_rejected(self, value):
        from src.models.schemas import SemanticSearchInput
        with pytest.raises(ValidationError):
            SemanticSearchInput(query="test", collection=value)


# ═══════════════════════════════════════════════════════════════════════
# AC-4.4: Numeric field bounds
# ═══════════════════════════════════════════════════════════════════════


class TestNumericBounds:
    """AC-4.4: top_k 1-100, threshold 0.0-1.0, etc."""

    def test_top_k_zero_rejected(self):
        from src.models.schemas import SemanticSearchInput
        with pytest.raises(ValidationError):
            SemanticSearchInput(query="test", top_k=0)

    def test_top_k_101_rejected(self):
        from src.models.schemas import SemanticSearchInput
        with pytest.raises(ValidationError):
            SemanticSearchInput(query="test", top_k=101)

    def test_top_k_boundaries_accepted(self):
        from src.models.schemas import SemanticSearchInput
        m1 = SemanticSearchInput(query="test", top_k=1)
        m100 = SemanticSearchInput(query="test", top_k=100)
        assert m1.top_k == 1
        assert m100.top_k == 100

    def test_threshold_negative_rejected(self):
        from src.models.schemas import SemanticSearchInput
        with pytest.raises(ValidationError):
            SemanticSearchInput(query="test", threshold=-0.1)

    def test_threshold_over_one_rejected(self):
        from src.models.schemas import SemanticSearchInput
        with pytest.raises(ValidationError):
            SemanticSearchInput(query="test", threshold=1.1)

    def test_threshold_boundaries_accepted(self):
        from src.models.schemas import SemanticSearchInput
        m0 = SemanticSearchInput(query="test", threshold=0.0)
        m1 = SemanticSearchInput(query="test", threshold=1.0)
        assert m0.threshold == 0.0
        assert m1.threshold == 1.0


# ═══════════════════════════════════════════════════════════════════════
# AC-4.6: All 9 schemas — valid and invalid inputs
# ═══════════════════════════════════════════════════════════════════════


class TestHybridSearchInput:
    """HybridSearchInput schema validation."""

    def test_valid_input(self):
        from src.models.schemas import HybridSearchInput
        m = HybridSearchInput(query="test")
        assert m.semantic_weight == 0.7
        assert m.keyword_weight == 0.3

    def test_empty_query_rejected(self):
        from src.models.schemas import HybridSearchInput
        with pytest.raises(ValidationError):
            HybridSearchInput(query="")

    def test_invalid_collection_rejected(self):
        from src.models.schemas import HybridSearchInput
        with pytest.raises(ValidationError):
            HybridSearchInput(query="test", collection="invalid")

    def test_weight_over_one_rejected(self):
        from src.models.schemas import HybridSearchInput
        with pytest.raises(ValidationError):
            HybridSearchInput(query="test", semantic_weight=1.5)

    def test_null_bytes_stripped_from_query(self):
        from src.models.schemas import HybridSearchInput
        m = HybridSearchInput(query="hello\x00world")
        assert "\x00" not in m.query


class TestCodeAnalyzeInput:
    """CodeAnalyzeInput schema validation."""

    def test_valid_input(self):
        from src.models.schemas import CodeAnalyzeInput
        m = CodeAnalyzeInput(code="def foo(): pass")
        assert m.analysis_type == "all"
        assert m.language == ""

    def test_empty_code_rejected(self):
        from src.models.schemas import CodeAnalyzeInput
        with pytest.raises(ValidationError):
            CodeAnalyzeInput(code="")

    def test_code_too_long_rejected(self):
        from src.models.schemas import CodeAnalyzeInput
        with pytest.raises(ValidationError):
            CodeAnalyzeInput(code="x" * 100001)

    def test_invalid_analysis_type_rejected(self):
        from src.models.schemas import CodeAnalyzeInput
        with pytest.raises(ValidationError):
            CodeAnalyzeInput(code="x", analysis_type="hack")

    @pytest.mark.parametrize("atype", [
        "complexity", "dependencies", "patterns", "quality", "security", "all",
    ])
    def test_valid_analysis_types(self, atype):
        from src.models.schemas import CodeAnalyzeInput
        m = CodeAnalyzeInput(code="x", analysis_type=atype)
        assert m.analysis_type == atype


class TestCodePatternAuditInput:
    """CodePatternAuditInput schema validation."""

    def test_valid_input(self):
        from src.models.schemas import CodePatternAuditInput
        m = CodePatternAuditInput(code="def foo(): pass")
        assert m.confidence_threshold == 0.3

    def test_empty_code_rejected(self):
        from src.models.schemas import CodePatternAuditInput
        with pytest.raises(ValidationError):
            CodePatternAuditInput(code="")

    def test_confidence_over_one_rejected(self):
        from src.models.schemas import CodePatternAuditInput
        with pytest.raises(ValidationError):
            CodePatternAuditInput(code="x", confidence_threshold=1.5)

    def test_confidence_negative_rejected(self):
        from src.models.schemas import CodePatternAuditInput
        with pytest.raises(ValidationError):
            CodePatternAuditInput(code="x", confidence_threshold=-0.1)


class TestGraphQueryInput:
    """GraphQueryInput schema validation."""

    def test_valid_input(self):
        from src.models.schemas import GraphQueryInput
        m = GraphQueryInput(cypher="MATCH (n) RETURN n LIMIT 10")
        assert m.parameters == {}

    def test_cypher_required(self):
        from src.models.schemas import GraphQueryInput
        with pytest.raises(ValidationError):
            GraphQueryInput()

    def test_cypher_too_long_rejected(self):
        from src.models.schemas import GraphQueryInput
        with pytest.raises(ValidationError):
            GraphQueryInput(cypher="x" * 5001)

    def test_parameters_dict_accepted(self):
        from src.models.schemas import GraphQueryInput
        m = GraphQueryInput(cypher="MATCH (n) WHERE n.id = $id", parameters={"id": 42})
        assert m.parameters == {"id": 42}

    def test_write_cypher_rejected_via_schema(self):
        """AC-5.6 / MCP5.16: write operations blocked at schema level."""
        from src.models.schemas import GraphQueryInput
        with pytest.raises(ValidationError, match="forbidden"):
            GraphQueryInput(cypher="CREATE (n:Evil)")

    def test_admin_cypher_rejected_via_schema(self):
        from src.models.schemas import GraphQueryInput
        with pytest.raises(ValidationError, match="admin"):
            GraphQueryInput(cypher="CALL dbms.security.listUsers()")


class TestLLMCompleteInput:
    """LLMCompleteInput schema validation."""

    def test_valid_input(self):
        from src.models.schemas import LLMCompleteInput
        m = LLMCompleteInput(prompt="Explain recursion")
        assert m.temperature == 0.7
        assert m.max_tokens == 4096
        assert m.model_preference == "auto"

    def test_empty_prompt_rejected(self):
        from src.models.schemas import LLMCompleteInput
        with pytest.raises(ValidationError):
            LLMCompleteInput(prompt="")

    def test_prompt_too_long_rejected(self):
        from src.models.schemas import LLMCompleteInput
        with pytest.raises(ValidationError):
            LLMCompleteInput(prompt="x" * 50001)

    def test_temperature_over_two_rejected(self):
        from src.models.schemas import LLMCompleteInput
        with pytest.raises(ValidationError):
            LLMCompleteInput(prompt="test", temperature=2.5)

    def test_max_tokens_zero_rejected(self):
        from src.models.schemas import LLMCompleteInput
        with pytest.raises(ValidationError):
            LLMCompleteInput(prompt="test", max_tokens=0)

    def test_max_tokens_over_limit_rejected(self):
        from src.models.schemas import LLMCompleteInput
        with pytest.raises(ValidationError):
            LLMCompleteInput(prompt="test", max_tokens=32769)

    def test_invalid_model_preference_rejected(self):
        from src.models.schemas import LLMCompleteInput
        with pytest.raises(ValidationError):
            LLMCompleteInput(prompt="test", model_preference="gpt-4")

    @pytest.mark.parametrize("pref", ["auto", "local", "cloud"])
    def test_valid_model_preferences(self, pref):
        from src.models.schemas import LLMCompleteInput
        m = LLMCompleteInput(prompt="test", model_preference=pref)
        assert m.model_preference == pref

    def test_null_bytes_stripped_from_prompt(self):
        from src.models.schemas import LLMCompleteInput
        m = LLMCompleteInput(prompt="hello\x00world")
        assert "\x00" not in m.prompt


class TestRunDiscussionInput:
    """RunDiscussionInput schema validation."""

    def test_valid_input(self):
        from src.models.schemas import RunDiscussionInput
        m = RunDiscussionInput(protocol_id="ROUNDTABLE_DISCUSSION", topic="Architecture")
        assert m.context == ""

    def test_empty_topic_rejected(self):
        from src.models.schemas import RunDiscussionInput
        with pytest.raises(ValidationError):
            RunDiscussionInput(protocol_id="DEBATE_PROTOCOL", topic="")

    def test_invalid_protocol_id_rejected(self):
        from src.models.schemas import RunDiscussionInput
        with pytest.raises(ValidationError):
            RunDiscussionInput(protocol_id="lower-case", topic="test")

    def test_topic_too_long_rejected(self):
        from src.models.schemas import RunDiscussionInput
        with pytest.raises(ValidationError):
            RunDiscussionInput(protocol_id="TEST_PROTO", topic="x" * 5001)

    @pytest.mark.parametrize("pid", [
        "ROUNDTABLE_DISCUSSION",
        "DEBATE_PROTOCOL",
        "WBS_GENERATION",
        "ARCHITECTURE_RECONCILIATION",
        "RELEVANCE_VALIDATION",
    ])
    def test_valid_protocol_ids(self, pid):
        from src.models.schemas import RunDiscussionInput
        m = RunDiscussionInput(protocol_id=pid, topic="test")
        assert m.protocol_id == pid


class TestRunAgentFunctionInput:
    """RunAgentFunctionInput schema validation."""

    def test_valid_input(self):
        from src.models.schemas import RunAgentFunctionInput
        m = RunAgentFunctionInput(function_name="summarize-content")
        assert m.input == {}

    def test_invalid_function_name_rejected(self):
        from src.models.schemas import RunAgentFunctionInput
        with pytest.raises(ValidationError):
            RunAgentFunctionInput(function_name="UPPER_CASE")

    def test_function_name_too_short_rejected(self):
        from src.models.schemas import RunAgentFunctionInput
        with pytest.raises(ValidationError):
            RunAgentFunctionInput(function_name="ab")

    def test_function_name_must_start_with_letter(self):
        from src.models.schemas import RunAgentFunctionInput
        with pytest.raises(ValidationError):
            RunAgentFunctionInput(function_name="1-invalid")

    def test_function_name_with_dict_input(self):
        from src.models.schemas import RunAgentFunctionInput
        m = RunAgentFunctionInput(
            function_name="cross-reference",
            input={"query": "test", "sources": ["code"]},
        )
        assert m.input["query"] == "test"


class TestAgentExecuteInput:
    """AgentExecuteInput schema validation."""

    def test_valid_input(self):
        from src.models.schemas import AgentExecuteInput
        m = AgentExecuteInput(task="Analyze the codebase")
        assert m.max_steps == 15

    def test_empty_task_rejected(self):
        from src.models.schemas import AgentExecuteInput
        with pytest.raises(ValidationError):
            AgentExecuteInput(task="")

    def test_task_too_long_rejected(self):
        from src.models.schemas import AgentExecuteInput
        with pytest.raises(ValidationError):
            AgentExecuteInput(task="x" * 10001)

    def test_max_steps_zero_rejected(self):
        from src.models.schemas import AgentExecuteInput
        with pytest.raises(ValidationError):
            AgentExecuteInput(task="test", max_steps=0)

    def test_max_steps_over_50_rejected(self):
        from src.models.schemas import AgentExecuteInput
        with pytest.raises(ValidationError):
            AgentExecuteInput(task="test", max_steps=51)

    def test_max_steps_boundaries_accepted(self):
        from src.models.schemas import AgentExecuteInput
        m1 = AgentExecuteInput(task="test", max_steps=1)
        m50 = AgentExecuteInput(task="test", max_steps=50)
        assert m1.max_steps == 1
        assert m50.max_steps == 50


# ═══════════════════════════════════════════════════════════════════════
# AC-4.5: Validation error format — 422, field names, no stack traces
# ═══════════════════════════════════════════════════════════════════════


class TestValidationErrorFormat:
    """AC-4.5: ValidationError contains field info, no stack traces."""

    def test_validation_error_contains_field_name(self):
        from src.models.schemas import SemanticSearchInput
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchInput(query="", top_k=-1)
        error_str = str(exc_info.value)
        # Must mention the failing field(s)
        assert "query" in error_str.lower() or "top_k" in error_str.lower()

    def test_validation_error_has_structured_errors(self):
        from src.models.schemas import SemanticSearchInput
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchInput(query="")
        errors = exc_info.value.errors()
        assert len(errors) >= 1
        assert "loc" in errors[0]
        assert "msg" in errors[0]

    def test_validation_error_no_traceback_in_message(self):
        from src.models.schemas import SemanticSearchInput
        with pytest.raises(ValidationError) as exc_info:
            SemanticSearchInput(query="")
        error_str = str(exc_info.value)
        assert "traceback" not in error_str.lower()
        assert "file " not in error_str.lower()


# ═══════════════════════════════════════════════════════════════════════
# Cross-schema sanitization: null bytes + Unicode NFC on all string models
# ═══════════════════════════════════════════════════════════════════════


class TestCrossModelSanitization:
    """AC-4.2: Null byte stripping and Unicode NFC across all schemas."""

    @pytest.mark.parametrize("schema_name,field,value", [
        ("SemanticSearchInput", "query", "test\x00query"),
        ("HybridSearchInput", "query", "test\x00query"),
        ("LLMCompleteInput", "prompt", "test\x00prompt"),
        ("LLMCompleteInput", "system_prompt", "test\x00system"),
        ("CodeAnalyzeInput", "code", "def\x00foo(): pass"),
        ("CodePatternAuditInput", "code", "class\x00Bar: pass"),
        ("RunDiscussionInput", "topic", "test\x00topic"),
        ("RunDiscussionInput", "context", "test\x00context"),
        ("AgentExecuteInput", "task", "do\x00something"),
    ])
    def test_null_bytes_stripped(self, schema_name, field, value):
        from src.models import schemas
        cls = getattr(schemas, schema_name)
        # Build minimal valid kwargs
        kwargs = self._minimal_kwargs(schema_name)
        kwargs[field] = value
        m = cls(**kwargs)
        assert "\x00" not in getattr(m, field)

    @pytest.mark.parametrize("schema_name,field", [
        ("SemanticSearchInput", "query"),
        ("HybridSearchInput", "query"),
        ("LLMCompleteInput", "prompt"),
        ("CodeAnalyzeInput", "code"),
        ("RunDiscussionInput", "topic"),
        ("AgentExecuteInput", "task"),
    ])
    def test_unicode_nfc_normalized(self, schema_name, field):
        from src.models import schemas
        cls = getattr(schemas, schema_name)
        nfd = "e\u0301"  # NFD: e + combining acute
        nfc = unicodedata.normalize("NFC", nfd)
        kwargs = self._minimal_kwargs(schema_name)
        kwargs[field] = nfd
        m = cls(**kwargs)
        assert getattr(m, field) == nfc

    @staticmethod
    def _minimal_kwargs(schema_name: str) -> dict:
        """Return minimal valid kwargs for each schema."""
        return {
            "SemanticSearchInput": {"query": "x"},
            "HybridSearchInput": {"query": "x"},
            "CodeAnalyzeInput": {"code": "x"},
            "CodePatternAuditInput": {"code": "x"},
            "GraphQueryInput": {"cypher": "MATCH (n) RETURN n"},
            "LLMCompleteInput": {"prompt": "x"},
            "RunDiscussionInput": {"protocol_id": "TEST_PROTO", "topic": "x"},
            "RunAgentFunctionInput": {"function_name": "test-func"},
            "AgentExecuteInput": {"task": "x"},
        }[schema_name]
