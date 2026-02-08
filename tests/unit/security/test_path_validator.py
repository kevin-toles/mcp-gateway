"""Path traversal prevention tests — WBS-MCP5 (RED).

Covers AC-5.1 (symlink + ALLOWED_ROOTS), AC-5.2 (URL-encoded traversal),
AC-5.3 (null bytes in paths), AC-5.7 (security event logging).
"""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.security.path_validator import PathValidationError, validate_file_path


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def allowed_roots(tmp_path: Path) -> list[Path]:
    """Create a temporary allowed root directory."""
    root = tmp_path / "safe"
    root.mkdir()
    return [root]


@pytest.fixture()
def safe_file(allowed_roots: list[Path]) -> Path:
    """Create a safe file inside the allowed root."""
    f = allowed_roots[0] / "data.txt"
    f.write_text("safe content")
    return f


# ── AC-5.1: validate_file_path resolves symlinks and rejects outside roots ──


class TestBasicTraversal:
    """Path traversal with dot-dot sequences."""

    def test_simple_dot_dot_rejected(self, allowed_roots: list[Path]) -> None:
        with pytest.raises(PathValidationError, match="traversal"):
            validate_file_path("../../../etc/passwd", allowed_roots)

    def test_dot_dot_in_middle_rejected(self, allowed_roots: list[Path]) -> None:
        root = allowed_roots[0]
        with pytest.raises(PathValidationError):
            validate_file_path(str(root / "sub" / ".." / ".." / ".." / "etc" / "passwd"), allowed_roots)

    def test_valid_file_accepted(self, allowed_roots: list[Path], safe_file: Path) -> None:
        result = validate_file_path(str(safe_file), allowed_roots)
        assert result == safe_file.resolve()

    def test_valid_subdir_accepted(self, allowed_roots: list[Path]) -> None:
        sub = allowed_roots[0] / "subdir"
        sub.mkdir()
        f = sub / "nested.txt"
        f.write_text("ok")
        result = validate_file_path(str(f), allowed_roots)
        assert result == f.resolve()

    def test_path_outside_all_roots_rejected(self, tmp_path: Path) -> None:
        root1 = tmp_path / "root1"
        root1.mkdir()
        outside = tmp_path / "outside" / "file.txt"
        outside.parent.mkdir()
        outside.write_text("nope")
        with pytest.raises(PathValidationError, match="outside"):
            validate_file_path(str(outside), [root1])

    def test_multiple_roots_first_matches(self, tmp_path: Path) -> None:
        r1 = tmp_path / "r1"
        r2 = tmp_path / "r2"
        r1.mkdir()
        r2.mkdir()
        f = r2 / "ok.txt"
        f.write_text("in r2")
        result = validate_file_path(str(f), [r1, r2])
        assert result == f.resolve()

    def test_empty_allowed_roots_rejects_everything(self, safe_file: Path) -> None:
        with pytest.raises(PathValidationError):
            validate_file_path(str(safe_file), [])


class TestSymlinkResolution:
    """AC-5.1: symlinks resolved before root check."""

    def test_symlink_inside_root_accepted(self, allowed_roots: list[Path], safe_file: Path) -> None:
        link = allowed_roots[0] / "link.txt"
        link.symlink_to(safe_file)
        result = validate_file_path(str(link), allowed_roots)
        assert result == safe_file.resolve()

    def test_symlink_pointing_outside_root_rejected(self, allowed_roots: list[Path], tmp_path: Path) -> None:
        outside = tmp_path / "evil.txt"
        outside.write_text("evil")
        link = allowed_roots[0] / "escape_link.txt"
        link.symlink_to(outside)
        with pytest.raises(PathValidationError, match="outside"):
            validate_file_path(str(link), allowed_roots)

    def test_chained_symlinks_resolved(self, allowed_roots: list[Path], tmp_path: Path) -> None:
        outside = tmp_path / "secret.txt"
        outside.write_text("secret")
        link1 = tmp_path / "link1"
        link1.symlink_to(outside)
        link2 = allowed_roots[0] / "link2"
        link2.symlink_to(link1)
        with pytest.raises(PathValidationError, match="outside"):
            validate_file_path(str(link2), allowed_roots)


# ── AC-5.2: URL-encoded traversal sequences ─────────────────────────────


class TestURLEncodedTraversal:
    """Single and double URL-encoded traversal attacks."""

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "%2e%2e/%2e%2e/etc/passwd",
            "..%2f..%2fetc%2fpasswd",
            "%2e%2e%5c%2e%2e%5cetc%5cpasswd",  # backslash encoded
        ],
        ids=["full-encode", "mixed-encode", "slash-only", "backslash-encode"],
    )
    def test_single_encoded_traversal_blocked(self, malicious_path: str, allowed_roots: list[Path]) -> None:
        with pytest.raises(PathValidationError, match="traversal"):
            validate_file_path(malicious_path, allowed_roots)

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "%252e%252e%252f%252e%252e%252fetc%252fpasswd",
            "%252e%252e/%252e%252e/etc/passwd",
            "..%252f..%252fetc/passwd",
        ],
        ids=["full-double", "mixed-double", "slash-double"],
    )
    def test_double_encoded_traversal_blocked(self, malicious_path: str, allowed_roots: list[Path]) -> None:
        with pytest.raises(PathValidationError, match="traversal"):
            validate_file_path(malicious_path, allowed_roots)


# ── AC-5.3: Null bytes in file paths ────────────────────────────────────


class TestNullBytePaths:
    """Null byte injection in file paths."""

    def test_null_byte_in_filename_rejected(self, allowed_roots: list[Path]) -> None:
        with pytest.raises(PathValidationError, match="null"):
            validate_file_path("file\x00.txt", allowed_roots)

    def test_null_byte_in_directory_rejected(self, allowed_roots: list[Path]) -> None:
        with pytest.raises(PathValidationError, match="null"):
            validate_file_path("dir\x00name/file.txt", allowed_roots)

    def test_url_encoded_null_byte_rejected(self, allowed_roots: list[Path]) -> None:
        with pytest.raises(PathValidationError, match="null"):
            validate_file_path("file%00.txt", allowed_roots)


# ── AC-5.7: Security event logging ─────────────────────────────────────


class TestPathSecurityLogging:
    """Blocked path attempts emit security events."""

    def test_traversal_logs_security_event(self, allowed_roots: list[Path], caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="mcp_gateway.security"):
            with pytest.raises(PathValidationError):
                validate_file_path("../../../etc/passwd", allowed_roots)
        assert any("SECURITY" in r.message and "path_traversal" in r.message for r in caplog.records)

    def test_null_byte_logs_security_event(self, allowed_roots: list[Path], caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="mcp_gateway.security"):
            with pytest.raises(PathValidationError):
                validate_file_path("file\x00.txt", allowed_roots)
        assert any("SECURITY" in r.message and "null_byte" in r.message for r in caplog.records)

    def test_outside_root_logs_security_event(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        root = tmp_path / "safe"
        root.mkdir()
        outside = tmp_path / "evil.txt"
        outside.write_text("evil")
        with caplog.at_level(logging.WARNING, logger="mcp_gateway.security"):
            with pytest.raises(PathValidationError):
                validate_file_path(str(outside), [root])
        assert any("SECURITY" in r.message for r in caplog.records)


# ── Edge cases ──────────────────────────────────────────────────────────


class TestPathEdgeCases:
    """Additional edge-case coverage."""

    def test_empty_path_rejected(self, allowed_roots: list[Path]) -> None:
        with pytest.raises(PathValidationError):
            validate_file_path("", allowed_roots)

    def test_relative_safe_path_with_root_prefix(self, allowed_roots: list[Path], safe_file: Path) -> None:
        result = validate_file_path(str(safe_file), allowed_roots)
        assert result.is_absolute()

    def test_nonexistent_file_rejected(self, allowed_roots: list[Path]) -> None:
        with pytest.raises(PathValidationError):
            validate_file_path(str(allowed_roots[0] / "no_such_file.txt"), allowed_roots)

    def test_backslash_traversal_rejected(self, allowed_roots: list[Path]) -> None:
        with pytest.raises(PathValidationError, match="traversal"):
            validate_file_path("..\\..\\..\\etc\\passwd", allowed_roots)

    def test_mixed_separator_traversal(self, allowed_roots: list[Path]) -> None:
        with pytest.raises(PathValidationError, match="traversal"):
            validate_file_path("..\\../..\\etc/passwd", allowed_roots)
