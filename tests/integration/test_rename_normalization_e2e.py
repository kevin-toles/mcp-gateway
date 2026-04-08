"""Integration test for rename_normalization — real subprocess invocation.

Runs rename_copilot.py with --limit 3 against actual Batch 2 PDFs. Requires
ai-agents venv and pymupdf to be installed.

Run with: INTEGRATION=1 pytest tests/integration/test_rename_normalization_e2e.py -m integration -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

BATCH2_DIR = "/Users/kevintoles/POC/textbooks/Books/Batch 2"
BATCH2_PATH = Path(BATCH2_DIR)


@pytest.mark.integration
class TestRenameNormalizationE2E:
    """End-to-end tests that actually invoke rename_copilot.py as a subprocess."""

    @pytest.fixture()
    def handler(self):
        from src.core.config import Settings
        from src.security.output_sanitizer import OutputSanitizer
        from src.tool_dispatcher import ToolDispatcher
        from src.tools import rename_normalization

        dispatcher = ToolDispatcher(Settings())
        sanitizer = OutputSanitizer()
        return rename_normalization.create_handler(dispatcher, sanitizer)

    @pytest.fixture()
    def output_path(self, tmp_path: Path) -> str:
        return str(tmp_path / "rename_suggestions_test.json")

    # ── Preconditions ────────────────────────────────────────────────────────

    def test_batch2_dir_exists(self):
        """Batch 2 directory must exist for integration tests to run."""
        assert BATCH2_PATH.exists(), f"Batch 2 dir not found: {BATCH2_DIR}"
        assert BATCH2_PATH.is_dir()
        pdfs = list(BATCH2_PATH.glob("*.pdf"))
        assert len(pdfs) >= 3, f"Expected ≥3 PDFs in Batch 2, found {len(pdfs)}"

    # ── Handler tests ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_tool_returns_ready_status(self, handler, output_path):
        """Tool returns status=ready when rename_copilot.py exits 0."""

        result = await handler(source_dir=BATCH2_DIR, output=output_path, limit=3)
        assert result["status"] == "ready", (
            f"Expected status=ready, got {result.get('status')!r}. Error: {result.get('error', '<none>')}"
        )

    @pytest.mark.asyncio
    async def test_context_doc_is_non_empty_string(self, handler, output_path):
        """context field contains the printed protocol document."""
        result = await handler(source_dir=BATCH2_DIR, output=output_path, limit=3)
        assert result["status"] == "ready"
        assert isinstance(result["context"], str)
        assert len(result["context"]) > 200, "Context doc unexpectedly short"

    @pytest.mark.asyncio
    async def test_context_contains_round_headers(self, handler, output_path):
        """Context doc contains round headers produced by rename_copilot.py."""
        result = await handler(source_dir=BATCH2_DIR, output=output_path, limit=3)
        assert result["status"] == "ready"
        ctx = result["context"]
        # rename_copilot.py prints Round 1 and Round 2 headers
        assert "Round 1" in ctx or "SCANNER" in ctx, "Expected Round 1/SCANNER header in context doc"

    @pytest.mark.asyncio
    async def test_file_count_is_three(self, handler, output_path):
        """With limit=3, file_count should be 3."""
        result = await handler(source_dir=BATCH2_DIR, output=output_path, limit=3)
        assert result["status"] == "ready"
        assert result["file_count"] == 3, f"Expected file_count=3 with --limit 3, got {result['file_count']}"

    @pytest.mark.asyncio
    async def test_source_dir_and_output_echoed_back(self, handler, output_path):
        """source_dir and output are returned in the response."""
        result = await handler(source_dir=BATCH2_DIR, output=output_path, limit=3)
        assert result["status"] == "ready"
        assert str(BATCH2_PATH) == result["source_dir"]
        assert output_path == result["output"]

    @pytest.mark.asyncio
    async def test_arxiv_slugs_flagged(self, handler, output_path):
        """Arxiv slug filenames must appear flagged in context doc (they exist in Batch 2)."""
        result = await handler(source_dir=BATCH2_DIR, output=output_path, limit=10)
        assert result["status"] == "ready"
        ctx = result["context"]
        # Batch 2 contains files like 1606.06565v2.pdf, 1802.04730v3.pdf
        assert "ARXIV_SLUG" in ctx, "Expected ARXIV_SLUG violation type in context doc — arxiv files exist in Batch 2"
