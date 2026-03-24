"""Schema enforcement — prevent parameter inconsistency across search tools.

This meta-test ensures ALL search-related input models accept both `limit` and `top_k`
to prevent future developers from introducing the validation error that external LLMs
cannot predict parameter names.

If this test fails, a developer added a new search tool without dual-parameter support.
"""

import inspect

import pytest
from pydantic import BaseModel

from src.models import schemas

# Facade models use max_results as their external parameter name by design.
# They do not participate in the limit/top_k dual-parameter pattern.
_FACADE_MODEL_EXCLUSIONS = {"AskInput", "SearchInInput", "FindCodePatternInput"}


def _get_all_search_input_models() -> list[tuple[str, type[BaseModel]]]:
    """Discover all *SearchInput and *Input models that have query + result limit fields."""
    search_models = []

    for name in dir(schemas):
        if not name.endswith("Input"):
            continue

        obj = getattr(schemas, name)

        if not (inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel):
            continue

        # Check if it has a 'query' field (indicates it's a search-like tool)
        fields = obj.model_fields
        if "query" not in fields:
            continue

        # Facade models use max_results as their external parameter name by design;
        # they are excluded from the dual-parameter enforcement rule.
        if name in _FACADE_MODEL_EXCLUSIONS:
            continue

        # Check if it has ANY of: limit, top_k, max_results, max_depth
        # (these indicate result limiting)
        result_limit_fields = {"limit", "top_k", "max_results", "max_depth"}
        if not any(f in fields for f in result_limit_fields):
            continue

        search_models.append((name, obj))

    return search_models


class TestSearchSchemaEnforcement:
    """Enforce dual-parameter pattern across ALL search tools."""

    @pytest.fixture
    def search_models(self) -> list[tuple[str, type[BaseModel]]]:
        """Get all search-related input models."""
        return _get_all_search_input_models()

    def test_all_search_models_discovered(self, search_models) -> None:
        """Sanity check: we found the expected search models."""
        model_names = {name for name, _ in search_models}

        # Known search models that MUST be present
        expected = {
            "SemanticSearchInput",
            "HybridSearchInput",
            "KnowledgeSearchInput",
            "KnowledgeRefineInput",
            "PatternSearchInput",
            "DiagramSearchInput",
        }

        assert expected.issubset(model_names), f"Missing expected models: {expected - model_names}"

    def test_all_search_models_accept_both_limit_and_top_k(self, search_models) -> None:
        """CRITICAL: All search models MUST accept both limit and top_k.

        This prevents the Pydantic validation error external LLMs cannot predict.
        If this test fails, a developer added a new search tool without dual support.
        """
        failures = []

        for name, model_class in search_models:
            fields = model_class.model_fields

            # Skip models that use different naming (e.g., max_depth for traversal)
            if "max_depth" in fields:
                continue  # GraphTraverseInput uses max_depth + limit, not top_k

            # Core requirement: MUST have both limit and top_k fields
            has_limit = "limit" in fields
            has_top_k = "top_k" in fields

            if not (has_limit and has_top_k):
                failures.append(
                    f"{name}: Missing dual-parameter support. "
                    f"Has limit={has_limit}, has top_k={has_top_k}. "
                    f"MUST have BOTH to prevent LLM validation errors."
                )

        assert not failures, (
            "\n\n❌ SCHEMA VIOLATION: The following models lack dual-parameter support:\n\n"
            + "\n".join(failures)
            + "\n\n"
            "FIX: Add both `limit` and `top_k` fields (one canonical, one alias) "
            "with a @model_validator to normalize.\n"
            "See DiagramSearchInput or SemanticSearchInput for the pattern.\n"
        )

    def test_all_search_models_have_normalization_validator(self, search_models) -> None:
        """All search models with limit/top_k MUST have a normalization validator.

        Without the validator, both parameters would be required, causing confusion.
        The validator ensures EITHER parameter works and normalizes to canonical form.
        """
        failures = []

        for name, model_class in search_models:
            fields = model_class.model_fields

            # Skip if not a limit/top_k model
            if "max_depth" in fields or ("limit" not in fields and "top_k" not in fields):
                continue

            # Check if the class has a model_validator that mentions "normalize" or "result_count"
            validators = []
            for attr_name in dir(model_class):
                if "normalize" in attr_name.lower() or "result" in attr_name.lower():
                    attr = getattr(model_class, attr_name)
                    if callable(attr):
                        validators.append(attr_name)

            if not validators:
                failures.append(
                    f"{name}: Has limit/top_k fields but NO normalization validator. "
                    f"MUST have @model_validator(mode='after') named 'normalize_result_count' "
                    f"to handle EITHER parameter."
                )

        assert not failures, (
            "\n\n❌ SCHEMA VIOLATION: The following models lack normalization validators:\n\n"
            + "\n".join(failures)
            + "\n\n"
            "FIX: Add @model_validator(mode='after') method that:\n"
            "  1. Checks if both limit and top_k are None → apply default\n"
            "  2. Normalizes non-None value to canonical form\n"
            "  3. Rejects conflicting values (limit=5, top_k=10)\n"
            "  4. Clears the alias field\n"
        )

    def test_limit_and_top_k_fields_are_optional(self, search_models) -> None:
        """Both limit and top_k MUST be Optional (int | None) for the pattern to work.

        If either is required (int without None), the dual-parameter pattern breaks.
        """
        failures = []

        for name, model_class in search_models:
            fields = model_class.model_fields

            if "max_depth" in fields:
                continue

            for field_name in ["limit", "top_k"]:
                if field_name not in fields:
                    continue

                field_info = fields[field_name]

                # Check annotation
                # Pydantic 2.x: field_info.annotation
                annotation = field_info.annotation

                # Must be Union[int, None] or int | None
                if not (hasattr(annotation, "__origin__") and annotation.__origin__ is type(int | None).__origin__):
                    # Try string check for older Python
                    type_str = str(annotation)
                    if "None" not in type_str and "Optional" not in type_str:
                        failures.append(
                            f"{name}.{field_name}: Type is {annotation}, but MUST be 'int | None' "
                            f"for dual-parameter pattern to work."
                        )

        assert not failures, (
            "\n\n❌ SCHEMA VIOLATION: The following fields are not optional:\n\n" + "\n".join(failures) + "\n\n"
            "FIX: Change field declaration from:\n"
            "  limit: int = Field(default=10, ...)\n"
            "to:\n"
            "  limit: int | None = Field(default=None, ...)\n"
        )

    def test_new_search_tools_follow_pattern(self, search_models) -> None:
        """Integration check: validate actual instantiation works with both parameters.

        This is the ultimate test — can we actually create instances with both param names?
        """
        failures = []

        for name, model_class in search_models:
            fields = model_class.model_fields

            if "max_depth" in fields:
                continue

            if "limit" not in fields or "top_k" not in fields:
                continue

            # Try instantiating with limit
            try:
                instance_limit = model_class(query="test", limit=5)
                canonical_field = "limit" if instance_limit.limit == 5 else "top_k"
            except Exception as e:
                failures.append(f"{name}: Failed with limit=5: {e}")
                continue

            # Try instantiating with top_k
            try:
                instance_top_k = model_class(query="test", top_k=5)
                # Check normalization happened
                if canonical_field == "limit" and instance_top_k.limit != 5:
                    failures.append(f"{name}: top_k=5 not normalized to limit=5 (got limit={instance_top_k.limit})")
                elif canonical_field == "top_k" and instance_top_k.top_k != 5:
                    failures.append(f"{name}: limit=5 not normalized to top_k=5 (got top_k={instance_top_k.top_k})")
            except Exception as e:
                failures.append(f"{name}: Failed with top_k=5: {e}")
                continue

            # Try with neither (should use default)
            try:
                instance_default = model_class(query="test")
                if (
                    canonical_field == "limit"
                    and instance_default.limit is None
                    or canonical_field == "top_k"
                    and instance_default.top_k is None
                ):
                    failures.append(f"{name}: No default applied when both params omitted")
            except Exception as e:
                failures.append(f"{name}: Failed with neither param: {e}")

        assert not failures, (
            "\n\n❌ RUNTIME VALIDATION FAILED:\n\n" + "\n".join(failures) + "\n\n"
            "The dual-parameter pattern is declared but doesn't work at runtime.\n"
        )


class TestFutureProofing:
    """Tests to catch common mistakes when adding new search tools."""

    def test_no_search_tool_uses_only_limit(self) -> None:
        """No search tool should use ONLY limit without top_k support."""
        search_models = _get_all_search_input_models()
        violations = []

        for name, model_class in search_models:
            fields = model_class.model_fields

            if "max_depth" in fields:
                continue

            if "limit" in fields and "top_k" not in fields:
                violations.append(f"{name} uses 'limit' without 'top_k' alias")

        assert not violations, (
            "\n\n❌ ANTI-PATTERN DETECTED:\n\n"
            + "\n".join(violations)
            + "\n\nAll search tools MUST support both parameter names.\n"
        )

    def test_no_search_tool_uses_only_top_k(self) -> None:
        """No search tool should use ONLY top_k without limit support."""
        search_models = _get_all_search_input_models()
        violations = []

        for name, model_class in search_models:
            fields = model_class.model_fields

            if "max_depth" in fields:
                continue

            if "top_k" in fields and "limit" not in fields:
                violations.append(f"{name} uses 'top_k' without 'limit' alias")

        assert not violations, (
            "\n\n❌ ANTI-PATTERN DETECTED:\n\n"
            + "\n".join(violations)
            + "\n\nAll search tools MUST support both parameter names.\n"
        )

    def test_schema_documentation_mentions_compatibility(self) -> None:
        """All search input docstrings should mention parameter compatibility."""
        search_models = _get_all_search_input_models()
        missing_docs = []

        for name, model_class in search_models:
            fields = model_class.model_fields

            if "max_depth" in fields:
                continue

            if "limit" not in fields or "top_k" not in fields:
                continue

            docstring = model_class.__doc__ or ""

            if "parameter compatibility" not in docstring.lower():
                missing_docs.append(f"{name}: Docstring should mention parameter compatibility pattern")

        assert not missing_docs, (
            "\n\n⚠️  DOCUMENTATION WARNING:\n\n" + "\n".join(missing_docs) + "\n\n"
            "Add to docstring:\n"
            "**Parameter compatibility:** Accepts BOTH `limit` and `top_k` interchangeably\\n"
            "to prevent LLM validation errors. Canonical form is `<limit|top_k>`.\n"
        )
