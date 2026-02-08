"""Tests for OutputSanitizer — WBS-MCP8 (RED).

AC-8.3 (output sanitization step in tools/call pipeline).
Phase 1: passthrough only.  Phase 2 will add active filtering.
"""

import pytest

from src.security.output_sanitizer import OutputSanitizer


class TestOutputSanitizerDefaults:
    def test_active_defaults_false(self):
        sanitizer = OutputSanitizer()
        assert sanitizer.active is False

    def test_can_set_active_true(self):
        sanitizer = OutputSanitizer(active=True)
        assert sanitizer.active is True


class TestOutputSanitizerPassthrough:
    @pytest.fixture()
    def sanitizer(self):
        return OutputSanitizer()

    def test_dict_passthrough(self, sanitizer):
        data = {"results": [{"text": "hello"}]}
        assert sanitizer.sanitize(data) == data

    def test_list_passthrough(self, sanitizer):
        data = [1, 2, 3]
        assert sanitizer.sanitize(data) == data

    def test_string_passthrough(self, sanitizer):
        assert sanitizer.sanitize("hello world") == "hello world"

    def test_none_passthrough(self, sanitizer):
        assert sanitizer.sanitize(None) is None

    def test_empty_dict_passthrough(self, sanitizer):
        assert sanitizer.sanitize({}) == {}

    def test_nested_dict_passthrough(self, sanitizer):
        data = {"outer": {"inner": [1, 2, {"deep": True}]}}
        assert sanitizer.sanitize(data) == data

    def test_int_passthrough(self, sanitizer):
        assert sanitizer.sanitize(42) == 42

    def test_bool_passthrough(self, sanitizer):
        assert sanitizer.sanitize(True) is True


class TestOutputSanitizerActiveMode:
    """Phase 1: even with active=True, sanitize still passes through."""

    def test_active_mode_still_passthrough_phase1(self):
        sanitizer = OutputSanitizer(active=True)
        data = {"results": [1, 2, 3]}
        assert sanitizer.sanitize(data) == data
