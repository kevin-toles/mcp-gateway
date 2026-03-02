"""PDW-3: MCP VRE Tool Wiring Verification — input model validation.

Verifies that VRE tool Pydantic models enforce their validation contracts
before payloads reach the dispatcher.

PDW3.5 — AC-PDW3.3: AuditSearchExploitsInput raises ValidationError for empty query string
PDW3.6 — AC-PDW3.3: Validation is enforced via min_length=1 Field constraint (source check)
PDW3.7 — AC-PDW3.4: AuditSearchCVEsInput raises ValidationError when all filter fields are None
PDW3.8 — AC-PDW3.4: @model_validator enforces at least one of cwe_id/severity/ecosystem
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

# =============================================================================
# TestAuditSearchExploitsInputValidation
# =============================================================================


class TestAuditSearchExploitsInputValidation:
    """AC-PDW3.3: AuditSearchExploitsInput rejects empty or None query before dispatch."""

    def test_empty_string_query_raises_validation_error(self) -> None:
        """Empty string must be rejected — dispatching a zero-length query to CodeBERT is nonsensical."""
        from src.models.schemas import AuditSearchExploitsInput

        with pytest.raises(ValidationError):
            AuditSearchExploitsInput(query="")

    def test_missing_query_raises_validation_error(self) -> None:
        """query is required; omitting it must raise ValidationError (not silently default)."""
        from src.models.schemas import AuditSearchExploitsInput

        with pytest.raises(ValidationError):
            AuditSearchExploitsInput()  # type: ignore[call-arg]

    def test_valid_query_does_not_raise(self) -> None:
        """A non-empty query string must be accepted without error."""
        from src.models.schemas import AuditSearchExploitsInput

        obj = AuditSearchExploitsInput(query="sql injection pattern")
        assert obj.query == "sql injection pattern"

    def test_query_field_has_min_length_constraint(self) -> None:
        """Source of AuditSearchExploitsInput must have min_length on the query field."""
        from src.models.schemas import AuditSearchExploitsInput

        query_field = AuditSearchExploitsInput.model_fields["query"]
        # Pydantic v2: metadata contains annotated constraints
        metadata_str = str(query_field)
        assert "min_length" in metadata_str or "1" in metadata_str, (
            "query field must declare min_length=1 to reject empty strings"
        )

    def test_source_does_not_hardcode_query_rejection(self) -> None:
        """Rejection must come from Pydantic Field constraint, not a hand-rolled if/else."""
        from src.models import schemas as schemas_mod

        src = inspect.getsource(schemas_mod.AuditSearchExploitsInput)
        # No manual 'if not query' guard — constraint declared via Field or validator
        assert "if not" not in src or "min_length" in src, (
            "Empty-query rejection should use Field(min_length=1), not manual if-checks"
        )


# =============================================================================
# TestAuditSearchCVEsInputValidation
# =============================================================================


class TestAuditSearchCVEsInputValidation:
    """AC-PDW3.4: AuditSearchCVEsInput enforces at least one filter before dispatch."""

    def test_all_filters_none_raises_validation_error(self) -> None:
        """When cwe_id, severity, and ecosystem are all None, ValidationError must be raised.

        Without at least one filter the query becomes a full-table scan on vuln_cve_records
        which is unsafe and meaningless for a security lookup tool.
        """
        from src.models.schemas import AuditSearchCVEsInput

        with pytest.raises(ValidationError):
            AuditSearchCVEsInput(cwe_id=None, severity=None, ecosystem=None)

    def test_cwe_id_alone_is_valid(self) -> None:
        """cwe_id alone satisfies the at-least-one-filter requirement."""
        from src.models.schemas import AuditSearchCVEsInput

        obj = AuditSearchCVEsInput(cwe_id="CWE-89")
        assert obj.cwe_id == "CWE-89"

    def test_severity_alone_is_valid(self) -> None:
        """severity alone satisfies the at-least-one-filter requirement."""
        from src.models.schemas import AuditSearchCVEsInput

        obj = AuditSearchCVEsInput(severity="critical")
        assert obj.severity == "critical"

    def test_ecosystem_alone_is_valid(self) -> None:
        """ecosystem alone satisfies the at-least-one-filter requirement."""
        from src.models.schemas import AuditSearchCVEsInput

        obj = AuditSearchCVEsInput(ecosystem="python")
        assert obj.ecosystem == "python"

    def test_all_filters_provided_is_valid(self) -> None:
        """All three filters together is the most specific (and clearly valid) case."""
        from src.models.schemas import AuditSearchCVEsInput

        obj = AuditSearchCVEsInput(cwe_id="CWE-89", severity="critical", ecosystem="python")
        assert obj.cwe_id == "CWE-89"
        assert obj.severity == "critical"
        assert obj.ecosystem == "python"

    def test_model_validator_present_in_source(self) -> None:
        """Source of AuditSearchCVEsInput must include a model_validator enforcing the constraint."""
        from src.models import schemas as schemas_mod

        src = inspect.getsource(schemas_mod.AuditSearchCVEsInput)
        assert "model_validator" in src or "@model_validator" in src, (
            "AuditSearchCVEsInput must have a @model_validator enforcing at-least-one-filter"
        )


# =============================================================================
# TestAuditToolBaseInheritance
# =============================================================================


class TestAuditToolBaseInheritance:
    """AC-PDW3.3/3.4 (PDW3.9 REFACTOR): AuditToolBase provides shared top_k field."""

    def test_audit_tool_base_exists_in_schemas(self) -> None:
        """AuditToolBase must be importable from src.models.schemas after PDW3.9 REFACTOR."""
        from src.models.schemas import AuditToolBase  # noqa: F401

    def test_audit_search_exploits_input_is_subclass_of_audit_tool_base(self) -> None:
        """AuditSearchExploitsInput must inherit from AuditToolBase."""
        from src.models.schemas import AuditSearchExploitsInput, AuditToolBase

        assert issubclass(AuditSearchExploitsInput, AuditToolBase)

    def test_audit_tool_base_has_top_k_field(self) -> None:
        """AuditToolBase must define top_k so subclasses don't redeclare it."""
        from src.models.schemas import AuditToolBase

        assert "top_k" in AuditToolBase.model_fields
