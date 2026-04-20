"""rename_normalization tool handler — 5-round PDF/EPUB rename protocol.

Runs rename_copilot.py synchronously, captures the full context document from stdout,
and returns it so Copilot can read it directly and execute all 5 protocol rounds.

**CRITICAL CHANGE (v3.0 — 2026-04-17)**:
- Protocol now uses VISION (`view_image`) instead of pytesseract OCR
- Vision extraction: 95%+ accuracy vs pytesseract: 20% accuracy
- Batch processing (5-8 covers per batch) to manage context budget
- Autonomous execution without user approval between batches

Standalone script: ai-agents/src/protocols/rename_copilot.py

Usage (via MCP):
    rename_normalization(
        source_dir="/path/to/Batch 2",
        output="/tmp/suggestions.json",
        limit=0,
    )

Copilot receives the full context document in the `context` field and executes:
  1. SCAN — Classify filenames by violation type
  2. EXTRACT — Render PDF covers as PNGs (~3 min)
  3. VISION — Extract titles via view_image in batches of 5-8 (~8 min)
  4. MERGE — Combine all batch corrections
  5. RENAME — Apply all file renames in one operation

Output: All renames applied directly to filesystem. No chat table, no file writing.
"""

import os
import subprocess
import sys
from pathlib import Path

from fastmcp import Context

from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "rename_normalization"

# Path to rename_copilot.py — resolve relative to this file's location
_RENAME_SCRIPT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "ai-agents",
        "src",
        "protocols",
        "rename_copilot.py",
    )
)

# Prefer the ai-agents venv python; fall back to system python3
_AI_AGENTS_PYTHON = os.path.abspath(
    os.path.join(os.path.dirname(_RENAME_SCRIPT), "..", "..", "..", ".venv", "bin", "python")
)


def _get_python() -> str:
    return _AI_AGENTS_PYTHON if os.path.exists(_AI_AGENTS_PYTHON) else sys.executable


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler for rename_normalization."""

    async def rename_normalization(
        source_dir: str,
        output: str,
        limit: int = 0,
        ctx: Context | None = None,
    ) -> dict:
        """Run the 5-round PDF/EPUB rename normalization protocol on a directory.

        Scans every PDF and EPUB in ``source_dir``, extracts cover pages as PNG images,
        uses Claude vision (`view_image`) to extract titles from covers in batches of 5-8,
        classifies each filename by violation type (ARXIV_SLUG, PUBLISHER_PREFIX,
        SLUG_OPAQUE, etc.), and returns a full context document for Copilot to execute
        all 5 protocol rounds and apply file renames directly.

        **CRITICAL**: Uses vision extraction (95%+ accuracy) instead of pytesseract OCR
        (20% accuracy). Batched processing to manage context budget. No chat output —
        renames applied directly to filesystem.

        Args:
            source_dir: Directory containing PDF/EPUB files to analyze.
            output:     Path for context document output (informational only).
            limit:      Cap analysis at N files (0 = no limit, useful for testing).

        Returns:
            dict with keys:
              status       — "ready" on success, "error" on failure
              context      — Full protocol context document (Copilot reads + executes this)
              file_count   — Total files found
              flagged      — Files with a non-CLEAN violation
              source_dir   — Resolved source directory
        """
        source_path = Path(source_dir).resolve()
        output_path = Path(output).resolve()

        if not source_path.exists():
            return {
                "status": "error",
                "error": f"source_dir does not exist: {source_path}",
            }

        if not source_path.is_dir():
            return {
                "status": "error",
                "error": f"source_dir is not a directory: {source_path}",
            }

        if not os.path.exists(_RENAME_SCRIPT):
            return {
                "status": "error",
                "error": f"rename_copilot.py not found at: {_RENAME_SCRIPT}",
            }

        python = _get_python()
        cmd = [
            python,
            _RENAME_SCRIPT,
            "--source-dir",
            str(source_path),
            "--output",
            str(output_path),
        ]
        if limit > 0:
            cmd += ["--limit", str(limit)]

        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            return {
                "status": "error",
                "error": result.stderr.strip() or "rename_copilot.py exited non-zero",
                "returncode": result.returncode,
            }

        context_doc = result.stdout

        # Extract counts from stderr summary line:
        #   "Found N files — M flagged, K clean"
        # Use regex so word position doesn't matter.
        import re as _re

        file_count = 0
        flagged = 0
        _summary_re = _re.compile(r"(\d+)\s+files\s+\S+\s+(\d+)\s+flagged")
        for line in result.stderr.splitlines():
            m = _summary_re.search(line)
            if m:
                file_count = int(m.group(1))
                flagged = int(m.group(2))
                break

        return {
            "status": "ready",
            "context": context_doc,
            "file_count": file_count,
            "flagged": flagged,
            "source_dir": str(source_path),
            "output": str(output_path),
        }

    return rename_normalization
