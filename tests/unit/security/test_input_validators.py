"""Tests for input sanitization utilities — WBS-MCP4 (RED).

Covers AC-4.2 sanitization and AC-4.5 error formatting.
"""

import unicodedata

import pytest


class TestSanitizeString:
    """sanitize_string() strips null bytes and normalizes Unicode NFC."""

    def test_strips_null_bytes(self):
        from src.security.input_validators import sanitize_string
        assert sanitize_string("hello\x00world") == "helloworld"

    def test_normalizes_unicode_nfc(self):
        from src.security.input_validators import sanitize_string
        nfd = "e\u0301"
        assert sanitize_string(nfd) == unicodedata.normalize("NFC", nfd)

    def test_strips_null_and_normalizes(self):
        from src.security.input_validators import sanitize_string
        assert sanitize_string("e\u0301\x00x") == unicodedata.normalize("NFC", "e\u0301") + "x"

    def test_passthrough_clean_string(self):
        from src.security.input_validators import sanitize_string
        assert sanitize_string("clean string") == "clean string"

    def test_empty_string_passthrough(self):
        from src.security.input_validators import sanitize_string
        assert sanitize_string("") == ""

    def test_strips_multiple_null_bytes(self):
        from src.security.input_validators import sanitize_string
        assert sanitize_string("\x00a\x00b\x00") == "ab"


class TestFormatValidationError:
    """format_validation_errors() returns field-level structured errors."""

    def test_returns_list_of_field_errors(self):
        from pydantic import ValidationError
        from src.models.schemas import SemanticSearchInput
        from src.security.input_validators import format_validation_errors

        try:
            SemanticSearchInput(query="", top_k=-1)
        except ValidationError as exc:
            result = format_validation_errors(exc)
            assert isinstance(result, list)
            assert len(result) >= 1
            assert "field" in result[0]
            assert "message" in result[0]

    def test_no_stack_trace_in_output(self):
        from pydantic import ValidationError
        from src.models.schemas import SemanticSearchInput
        from src.security.input_validators import format_validation_errors

        try:
            SemanticSearchInput(query="")
        except ValidationError as exc:
            result = format_validation_errors(exc)
            result_str = str(result).lower()
            assert "traceback" not in result_str
            assert "file " not in result_str

    def test_includes_field_name(self):
        from pydantic import ValidationError
        from src.models.schemas import SemanticSearchInput
        from src.security.input_validators import format_validation_errors

        try:
            SemanticSearchInput(query="")
        except ValidationError as exc:
            result = format_validation_errors(exc)
            fields = [e["field"] for e in result]
            assert "query" in fields
