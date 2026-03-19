"""Test parameter standardization — limit/top_k compatibility.

Tests that all search tools accept BOTH `limit` and `top_k` interchangeably
to prevent LLM validation errors when calling MCP tools.
"""

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    DiagramSearchInput,
    HybridSearchInput,
    KnowledgeRefineInput,
    KnowledgeSearchInput,
    PatternSearchInput,
    SemanticSearchInput,
)


class TestSemanticSearchInput:
    """Tests for SemanticSearchInput parameter compatibility."""

    def test_accepts_top_k_canonical(self) -> None:
        """Canonical parameter top_k works."""
        input_data = SemanticSearchInput(query="test", top_k=5)
        assert input_data.top_k == 5
        assert input_data.limit is None  # alias cleared

    def test_accepts_limit_alias(self) -> None:
        """Alias parameter limit is normalized to top_k."""
        input_data = SemanticSearchInput(query="test", limit=5)
        assert input_data.top_k == 5
        assert input_data.limit is None  # alias cleared

    def test_accepts_neither_uses_default(self) -> None:
        """Neither parameter provided → default 10."""
        input_data = SemanticSearchInput(query="test")
        assert input_data.top_k == 10

    def test_rejects_conflicting_values(self) -> None:
        """Both parameters with different values raises error."""
        with pytest.raises(ValidationError, match="Cannot specify both"):
            SemanticSearchInput(query="test", top_k=5, limit=10)

    def test_accepts_both_when_equal(self) -> None:
        """Both parameters with same value is accepted."""
        input_data = SemanticSearchInput(query="test", top_k=5, limit=5)
        assert input_data.top_k == 5


class TestHybridSearchInput:
    """Tests for HybridSearchInput parameter compatibility."""

    def test_accepts_top_k_canonical(self) -> None:
        """Canonical parameter top_k works."""
        input_data = HybridSearchInput(query="test", top_k=5)
        assert input_data.top_k == 5
        assert input_data.limit is None

    def test_accepts_limit_alias(self) -> None:
        """Alias parameter limit is normalized to top_k."""
        input_data = HybridSearchInput(query="test", limit=5)
        assert input_data.top_k == 5
        assert input_data.limit is None


class TestKnowledgeSearchInput:
    """Tests for KnowledgeSearchInput parameter compatibility."""

    def test_accepts_limit_canonical(self) -> None:
        """Canonical parameter limit works."""
        input_data = KnowledgeSearchInput(query="test", limit=5)
        assert input_data.limit == 5
        assert input_data.top_k is None

    def test_accepts_top_k_alias(self) -> None:
        """Alias parameter top_k is normalized to limit."""
        input_data = KnowledgeSearchInput(query="test", top_k=5)
        assert input_data.limit == 5
        assert input_data.top_k is None

    def test_accepts_neither_uses_default(self) -> None:
        """Neither parameter provided → default 10."""
        input_data = KnowledgeSearchInput(query="test")
        assert input_data.limit == 10

    def test_rejects_conflicting_values(self) -> None:
        """Both parameters with different values raises error."""
        with pytest.raises(ValidationError, match="Cannot specify both"):
            KnowledgeSearchInput(query="test", limit=5, top_k=10)


class TestKnowledgeRefineInput:
    """Tests for KnowledgeRefineInput parameter compatibility."""

    def test_accepts_limit_canonical(self) -> None:
        """Canonical parameter limit works."""
        input_data = KnowledgeRefineInput(query="test", limit=3)
        assert input_data.limit == 3
        assert input_data.top_k is None

    def test_accepts_top_k_alias(self) -> None:
        """Alias parameter top_k is normalized to limit."""
        input_data = KnowledgeRefineInput(query="test", top_k=3)
        assert input_data.limit == 3
        assert input_data.top_k is None

    def test_accepts_neither_uses_default(self) -> None:
        """Neither parameter provided → default 5."""
        input_data = KnowledgeRefineInput(query="test")
        assert input_data.limit == 5


class TestPatternSearchInput:
    """Tests for PatternSearchInput parameter compatibility."""

    def test_accepts_limit_canonical(self) -> None:
        """Canonical parameter limit works."""
        input_data = PatternSearchInput(query="test", limit=7)
        assert input_data.limit == 7
        assert input_data.top_k is None

    def test_accepts_top_k_alias(self) -> None:
        """Alias parameter top_k is normalized to limit."""
        input_data = PatternSearchInput(query="test", top_k=7)
        assert input_data.limit == 7
        assert input_data.top_k is None


class TestDiagramSearchInput:
    """Tests for DiagramSearchInput parameter compatibility."""

    def test_accepts_limit_canonical(self) -> None:
        """Canonical parameter limit works."""
        input_data = DiagramSearchInput(query="test", limit=3)
        assert input_data.limit == 3
        assert input_data.top_k is None

    def test_accepts_top_k_alias(self) -> None:
        """Alias parameter top_k is normalized to limit."""
        input_data = DiagramSearchInput(query="test", top_k=3)
        assert input_data.limit == 3
        assert input_data.top_k is None

    def test_accepts_neither_uses_default(self) -> None:
        """Neither parameter provided → default 10."""
        input_data = DiagramSearchInput(query="test")
        assert input_data.limit == 10

    def test_rejects_conflicting_values(self) -> None:
        """Both parameters with different values raises error."""
        with pytest.raises(ValidationError, match="Cannot specify both"):
            DiagramSearchInput(query="test", limit=5, top_k=10)


class TestRealWorldLLMUsage:
    """Tests simulating real LLM tool calls with unpredictable parameter names."""

    def test_llm_uses_top_k_on_diagram_search_works(self) -> None:
        """LLM guesses 'top_k' for diagram_search → normalized to limit."""
        input_data = DiagramSearchInput(
            query="greedy algorithm flowchart set cover maximum coverage",
            top_k=3,  # LLM guessed wrong parameter name
        )
        assert input_data.limit == 3  # normalized
        assert input_data.top_k is None

    def test_llm_uses_limit_on_semantic_search_works(self) -> None:
        """LLM guesses 'limit' for semantic_search → normalized to top_k."""
        input_data = SemanticSearchInput(
            query="maximum coverage greedy algorithm Python",
            limit=5,  # LLM guessed wrong parameter name
        )
        assert input_data.top_k == 5  # normalized
        assert input_data.limit is None

    def test_llm_uses_no_parameter_gets_default(self) -> None:
        """LLM omits result count → defaults applied."""
        semantic = SemanticSearchInput(query="test")
        assert semantic.top_k == 10

        diagram = DiagramSearchInput(query="test")
        assert diagram.limit == 10

        knowledge = KnowledgeSearchInput(query="test")
        assert knowledge.limit == 10
