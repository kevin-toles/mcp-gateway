"""Unit tests for rename_normalization tool handler.

Tests the full handler lifecycle using mocked subprocess.run so no real
PDFs or rename_copilot.py invocation is needed.

ACs:
  RN-1: source_dir not found → {"status": "error"}
  RN-2: source_dir is a file, not a dir → {"status": "error"}
  RN-3: rename_copilot.py not found → {"status": "error"}
  RN-4: subprocess exits non-zero → {"status": "error", "returncode": N}
  RN-5: subprocess success → {"status": "ready", "context": <stdout>}
  RN-6: file_count + flagged parsed from stderr summary line
  RN-7: --limit flag passed when limit > 0
  RN-8: --limit flag NOT passed when limit == 0 (default)
  RN-9: handler is registered in server HANDLER_FACTORIES
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Settings
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher
from src.tools import rename_normalization

# ── Shared fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def handler():
    """Return a rename_normalization handler with real dispatcher/sanitizer."""
    dispatcher = ToolDispatcher(Settings())
    sanitizer = OutputSanitizer()
    return rename_normalization.create_handler(dispatcher, sanitizer)


@pytest.fixture()
def fake_source_dir(tmp_path: Path) -> Path:
    """Temp directory that exists (validator passes)."""
    return tmp_path


@pytest.fixture()
def fake_output(tmp_path: Path) -> str:
    return str(tmp_path / "out.json")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# RN-1: source_dir does not exist
# ═══════════════════════════════════════════════════════════════════════════════


class TestInputValidation:
    def test_source_dir_not_found_returns_error(self, handler, fake_output, tmp_path):
        """RN-1: Non-existent source_dir → error without calling subprocess."""
        missing = str(tmp_path / "does_not_exist")
        result = asyncio.get_event_loop().run_until_complete(handler(source_dir=missing, output=fake_output))
        assert result["status"] == "error"
        assert "does not exist" in result["error"]

    def test_source_dir_is_file_returns_error(self, handler, fake_output, tmp_path):
        """RN-2: source_dir pointing to a file → error."""
        f = tmp_path / "a_file.pdf"
        f.write_bytes(b"")
        result = asyncio.get_event_loop().run_until_complete(handler(source_dir=str(f), output=fake_output))
        assert result["status"] == "error"
        assert "not a directory" in result["error"]

    def test_script_not_found_returns_error(self, handler, fake_source_dir, fake_output):
        """RN-3: rename_copilot.py missing → error reported before subprocess call."""
        with patch.object(rename_normalization, "_RENAME_SCRIPT", "/nonexistent/rename_copilot.py"):
            result = asyncio.get_event_loop().run_until_complete(
                handler(source_dir=str(fake_source_dir), output=fake_output)
            )
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# RN-4 / RN-5: subprocess exit codes
# ═══════════════════════════════════════════════════════════════════════════════


class TestSubprocessExitCodes:
    def test_nonzero_returncode_returns_error(self, handler, fake_source_dir, fake_output):
        """RN-4: subprocess exits non-zero → status=error with returncode."""
        proc = _make_proc(returncode=1, stderr="something went wrong")
        with patch("subprocess.run", return_value=proc) as mock_run:
            # Also need the script to appear to exist
            with patch("os.path.exists", return_value=True):
                result = asyncio.get_event_loop().run_until_complete(
                    handler(source_dir=str(fake_source_dir), output=fake_output)
                )
        assert result["status"] == "error"
        assert result["returncode"] == 1
        assert mock_run.called

    def test_zero_returncode_returns_ready(self, handler, fake_source_dir, fake_output):
        """RN-5: subprocess exits 0 → status=ready with context from stdout."""
        context_doc = "# Rename Protocol\n\nRound 1: Scanner\n..."
        proc = _make_proc(returncode=0, stdout=context_doc, stderr="8 files — 3 flagged")
        with patch("subprocess.run", return_value=proc), patch("os.path.exists", return_value=True):
            result = asyncio.get_event_loop().run_until_complete(
                handler(source_dir=str(fake_source_dir), output=fake_output)
            )
        assert result["status"] == "ready"
        assert result["context"] == context_doc


# ═══════════════════════════════════════════════════════════════════════════════
# RN-6: Count parsing from stderr
# ═══════════════════════════════════════════════════════════════════════════════


class TestCountParsing:
    @pytest.mark.parametrize(
        "stderr, expected_file_count, expected_flagged",
        [
            ("Found 8 files — 3 flagged, 5 clean\n", 8, 3),
            ("Found 85 files — 72 flagged, 13 clean", 85, 72),
            ("Found 1 files — 0 flagged, 1 clean", 1, 0),
            # No summary line → zeroes returned
            ("verbose log line\nanother line\n", 0, 0),
        ],
    )
    def test_count_parsing(
        self,
        handler,
        fake_source_dir,
        fake_output,
        stderr,
        expected_file_count,
        expected_flagged,
    ):
        """RN-6: file_count and flagged are parsed from stderr summary line."""
        proc = _make_proc(returncode=0, stdout="context", stderr=stderr)
        with patch("subprocess.run", return_value=proc), patch("os.path.exists", return_value=True):
            result = asyncio.get_event_loop().run_until_complete(
                handler(source_dir=str(fake_source_dir), output=fake_output)
            )
        assert result["file_count"] == expected_file_count
        assert result["flagged"] == expected_flagged


# ═══════════════════════════════════════════════════════════════════════════════
# RN-7 / RN-8: --limit flag
# ═══════════════════════════════════════════════════════════════════════════════


class TestLimitFlag:
    def test_limit_flag_included_when_nonzero(self, handler, fake_source_dir, fake_output):
        """RN-7: limit=5 → ['--limit', '5'] in subprocess cmd."""
        proc = _make_proc(returncode=0, stdout="ctx", stderr="5 files — 2 flagged")
        with patch("subprocess.run", return_value=proc) as mock_run, patch("os.path.exists", return_value=True):
            asyncio.get_event_loop().run_until_complete(
                handler(source_dir=str(fake_source_dir), output=fake_output, limit=5)
            )
        cmd = mock_run.call_args[0][0]
        assert "--limit" in cmd
        assert "5" in cmd

    def test_limit_flag_absent_when_zero(self, handler, fake_source_dir, fake_output):
        """RN-8: limit=0 (default) → --limit NOT in subprocess cmd."""
        proc = _make_proc(returncode=0, stdout="ctx", stderr="5 files — 2 flagged")
        with patch("subprocess.run", return_value=proc) as mock_run, patch("os.path.exists", return_value=True):
            asyncio.get_event_loop().run_until_complete(
                handler(source_dir=str(fake_source_dir), output=fake_output, limit=0)
            )
        cmd = mock_run.call_args[0][0]
        assert "--limit" not in cmd

    def test_limit_defaults_to_zero(self, handler, fake_source_dir, fake_output):
        """RN-8 (default): calling without limit → --limit absent."""
        proc = _make_proc(returncode=0, stdout="ctx", stderr="")
        with patch("subprocess.run", return_value=proc) as mock_run, patch("os.path.exists", return_value=True):
            asyncio.get_event_loop().run_until_complete(handler(source_dir=str(fake_source_dir), output=fake_output))
        cmd = mock_run.call_args[0][0]
        assert "--limit" not in cmd


# ═══════════════════════════════════════════════════════════════════════════════
# RN-9: handler registered in server HANDLER_FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════


class TestServerRegistration:
    def test_rename_normalization_in_handler_factories(self):
        """RN-9: rename_normalization is in server._HANDLER_FACTORIES."""
        import src.server as server_module

        assert "rename_normalization" in server_module._HANDLER_FACTORIES, (
            "rename_normalization must be registered in _HANDLER_FACTORIES to be callable via MCP"
        )
