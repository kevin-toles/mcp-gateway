"""GAP-10 RED phase: Tests for underscore→hyphen service key normalization.

All 6 tests MUST fail when run against the current codebase.
The GREEN phase will make them pass.
"""

import pytest

from src.core.config import ServiceKey, Settings
from src.middleware.health_proxy import HealthAwareProxy


def test_tool_to_service_returns_hyphen_keys():
    """`_tool_to_service("semantic_search")` returns `ServiceKey("semantic-search")`."""
    proxy = HealthAwareProxy()
    result = proxy._tool_to_service("semantic_search")
    assert result == ServiceKey("semantic-search")


def test_tool_to_service_all_values_are_hyphen():
    """All values returned by `_tool_to_service()` contain no underscores."""
    proxy = HealthAwareProxy()
    tool_names = [
        "semantic_search",
        "hybrid_search",
        "knowledge_search",
        "knowledge_refine",
        "pattern_search",
        "diagram_search",
        "graph_query",
        "graph_traverse",
        "code_analyze",
        "code_pattern_audit",
        "llm_complete",
        "a2a_send_message",
        "a2a_get_task",
        "a2a_cancel_task",
        "enhance_guideline",
        "audit_security_scan",
        "audit_code_metrics",
        "audit_corpus_search",
        "audit_dependency_assess",
        "audit_resolve_lookup",
        "audit_search_exploits",
        "audit_search_cves",
        "audit_quality_scan",
        "generate_taxonomy",
        "extract_book_metadata",
        "enrich_book_metadata",
        "batch_extract_metadata",
        "batch_enrich_metadata",
        "analyze_taxonomy_coverage",
        "run_agent_function",
        "run_discussion",
        "agent_execute",
        "context_management",
        "amve_evaluate_fitness",
        "foundation_search",
    ]
    for tool in tool_names:
        value = proxy._tool_to_service(tool)
        assert value is not None, f"{tool} mapped to None"
        assert "_" not in value, f"{tool} → {value!r} still contains underscore"


def test_health_proxy_config_keys_use_hyphens():
    """All keys in HEALTH_PROXY_SERVICE_CONFIG use hyphens not underscores."""
    config = Settings().HEALTH_PROXY_SERVICE_CONFIG
    for key in config:
        assert "_" not in key, f"Config key {key!r} contains underscore (should use hyphen)"
        assert "-" in key or key.isalpha(), (
            f"Config key {key!r} should use hyphens as separators"
        )


def test_normalize_key_is_module_level():
    """`from src.core.config import normalize_service_key` succeeds."""
    from src.core.config import normalize_service_key
    assert callable(normalize_service_key)


def test_normalize_key_converts_underscore():
    """`normalize_service_key("semantic_search")` returns `"semantic-search"`."""
    from src.core.config import normalize_service_key
    assert normalize_service_key("semantic_search") == "semantic-search"


def test_normalize_key_idempotent():
    """`normalize_service_key("semantic-search")` returns `"semantic-search"` unchanged."""
    from src.core.config import normalize_service_key
    assert normalize_service_key("semantic-search") == "semantic-search"
