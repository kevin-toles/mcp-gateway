"""batch_extract_metadata tool handler — 1:1 mirror of llm-document-enhancer batch script.

This is the MCP tool equivalent of:
    llm-document-enhancer/scripts/batch_extract_metadata.py --use-orchestrator

Original script flow:
    1. Discover book JSON files in input directory
    2. For each book: subprocess → generate_metadata_universal.py → CO extract
    3. CO handles chapter detection, extraction, aggregation, file writing
    4. Report per-book progress + final summary

This tool mirrors that flow exactly:
    1. Discover book JSON files in input directory
    2. For each book: dispatch → CO /api/v1/workflows/extract-book
    3. CO handles chapter detection, extraction, aggregation, file writing
    4. Report per-book progress via ctx.info() + _log_progress()

Key parity with original:
    - Timeout: None (no timeout) — matches DEFAULT_ORCHESTRATOR_TIMEOUT = None
    - CO handles ALL internal logic (chapter detection, processing, file writing)
    - Tool is a simple dispatch loop with progress reporting
    - No chapter detection or aggregation duplicated in the tool
"""

import glob
import logging
import os
import subprocess
import time
from datetime import UTC, datetime

import httpx
from fastmcp import Context

from src.models.schemas import BatchExtractMetadataInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "extract_book_metadata"  # dispatches to CO /api/v1/workflows/extract-book
PROGRESS_LOG = "/tmp/extraction_progress.log"  # noqa: S108
_STANDALONE_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "batch_extract_standalone.py")

logger = logging.getLogger(__name__)


def _log_progress(msg: str) -> None:
    """Write a timestamped progress line to /tmp/extraction_progress.log.

    This file can be monitored live with: tail -f /tmp/extraction_progress.log
    """
    ts = datetime.now(UTC).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(PROGRESS_LOG, "a") as f:
        f.write(line)
        f.flush()


def _check_co_health(co_url: str = "http://localhost:8083") -> str | None:
    """Return None if healthy, or an error string if unreachable."""
    try:
        r = httpx.get(f"{co_url}/health", timeout=5.0)
        r.raise_for_status()
        return None
    except Exception as e:
        return (
            f"code-orchestrator health check failed: {e}. "
            "Start it first: cd /Users/kevintoles/POC/Code-Orchestrator-Service && "
            "source .venv/bin/activate && uvicorn src.main:app --host 0.0.0.0 --port 8083"
        )


def _launch_terminal(
    input_dir: str,
    out_dir: str,
    file_pattern: str,
    skip_existing: bool,
    enable_summary: bool,
    co_url: str,
    vtf_path: str | None = None,
) -> dict:
    """Open a new Terminal.app window running the extraction script.

    Mirrors seed.sh exactly:
      1. Write a temp .sh script that pipes through tee → log file
      2. Use osascript to open it in a fresh Terminal.app window
    """
    import stat
    import tempfile

    standalone = os.path.abspath(_STANDALONE_SCRIPT)
    python = os.path.abspath(os.path.join(os.path.dirname(standalone), "..", ".venv", "bin", "python"))
    if not os.path.exists(python):
        python = "python3"

    skip_flag = "--skip-existing" if skip_existing else ""
    summary_flag = "--enable-summary" if enable_summary else ""
    vtf_flag = f"--vtf-path '{vtf_path}'" if vtf_path else ""

    from datetime import datetime as _dt

    timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"/tmp/batch_extract_{timestamp}.log"  # noqa: S108

    # Write a temp .sh launcher — same approach as seed.sh
    fd, tmp_script = tempfile.mkstemp(suffix=".sh", prefix="batch_extract_run_")
    os.close(fd)
    with open(tmp_script, "w") as f:
        f.write(f"""#!/usr/bin/env bash
printf '\\n\\033[1;36m══ Batch Metadata Extraction ══  Log: {log_file}\\033[0m\\n\\n'
'{python}' -u '{standalone}' \\
    --input-dir '{input_dir}' \\
    --output-dir '{out_dir}' \\
    --file-pattern '{file_pattern}' \\
    --co-url '{co_url}' \\
    {skip_flag} {summary_flag} {vtf_flag} 2>&1 | tee '{log_file}'
EXIT_CODE=${{PIPESTATUS[0]}}
ln -sf '{log_file}' '{PROGRESS_LOG}'
echo ''
if [[ $EXIT_CODE -eq 0 ]]; then
  printf '\\033[1;32m✅  Extraction complete.\\033[0m\\n'
else
  printf '\\033[1;33m⚠️   Finished with some failures. Check log above.\\033[0m\\n'
fi
echo ''
read -rp 'Press Enter to close...'
""")
    os.chmod(tmp_script, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)  # noqa: S103

    # Open in a fresh Terminal.app window — same osascript pattern as seed.sh
    applescript = f"""
tell application "Terminal"
    do script "{tmp_script}"
    set bounds of front window to {{80, 80, 1300, 900}}
    activate
end tell
"""
    subprocess.Popen(["osascript", "-e", applescript])  # noqa: S603, S607

    return {
        "status": "started",
        "message": f"Extraction launched in Terminal.app. Monitor: tail -f {PROGRESS_LOG}",
        "log": log_file,
        "monitor_command": f"tail -f {PROGRESS_LOG}",
        "input_dir": input_dir,
        "output_dir": out_dir,
    }


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler — simple dispatch loop with progress."""

    async def batch_extract_metadata(
        input_dir: str,
        output_dir: str | None = None,
        file_pattern: str = "*.json",
        skip_existing: bool = True,
        enable_summary: bool = False,
        vtf_path: str | None = None,
        ctx: Context | None = None,
    ) -> dict:
        """Extract metadata from all books in a directory.

        Always launches in a new Terminal.app window and returns immediately.
        Monitor live progress with: tail -f /tmp/extraction_progress.log

        Args:
            input_dir: Directory containing raw book JSON files.
            output_dir: Directory for metadata output (defaults to sibling 'metadata' dir).
            file_pattern: Glob pattern for book files (default: *.json).
            skip_existing: Skip books that already have metadata output files.
            enable_summary: Whether to generate LLM summaries (default: False for speed).
            vtf_path: Optional path to a custom validated_term_filter.json for this batch.
        """
        validated = BatchExtractMetadataInput(
            input_dir=input_dir,
            output_dir=output_dir,
            file_pattern=file_pattern,
            skip_existing=skip_existing,
        )

        # Resolve output directory
        out_dir = validated.output_dir
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(validated.input_dir.rstrip("/")), "metadata")

        os.makedirs(out_dir, exist_ok=True)

        # ── Pre-flight: health check code-orchestrator ──────────────────
        co_url = getattr(dispatcher, "_settings", None)
        co_url = co_url.CODE_ORCHESTRATOR_URL if co_url else "http://localhost:8083"
        health_err = _check_co_health(co_url)
        if health_err:
            return {"status": "error", "message": health_err}

        # ── Launch in a new Terminal.app window and return immediately ──
        return _launch_terminal(
            input_dir=validated.input_dir,
            out_dir=out_dir,
            file_pattern=validated.file_pattern,
            skip_existing=validated.skip_existing,
            enable_summary=enable_summary,
            co_url=co_url,
            vtf_path=vtf_path,
        )

        # ── Discover books (mirrors original discover_books()) ──────────
        pattern = os.path.join(validated.input_dir, validated.file_pattern)
        all_files = sorted(glob.glob(pattern))

        if not all_files:
            return {"status": "no_files", "message": f"No files matching {pattern}"}

        # ── Filter existing (mirrors original skip / resume logic) ──────
        books_to_process = []
        skipped = []
        for fpath in all_files:
            basename = os.path.basename(fpath)
            stem = os.path.splitext(basename)[0]
            out_path = os.path.join(out_dir, f"{stem}_metadata.json")
            if validated.skip_existing and os.path.exists(out_path):
                skipped.append(basename)
            else:
                books_to_process.append(fpath)

        total_books = len(books_to_process)
        if total_books == 0:
            return {
                "status": "all_skipped",
                "message": f"All {len(skipped)} books already have metadata",
                "skipped": len(skipped),
            }

        # ── Batch header ────────────────────────────────────────────────
        batch_start = time.time()
        header = f"\U0001f4da Batch extraction: {total_books} books to process, {len(skipped)} skipped (existing)"
        if ctx:
            await ctx.info(header)
        _log_progress("=" * 60)
        _log_progress(header)
        _log_progress("=" * 60)

        # ── Extraction loop (mirrors original _run_extraction_loop()) ──
        book_results = []
        succeeded = 0
        failed = 0

        for book_idx, book_path in enumerate(books_to_process):
            book_name = os.path.splitext(os.path.basename(book_path))[0]
            book_start = time.time()
            out_path = os.path.join(out_dir, f"{book_name}_metadata.json")

            # Progress: starting book
            progress_msg = f"[{book_idx + 1:3d}/{total_books}] Processing: {book_name}"
            if ctx:
                await ctx.report_progress(book_idx, total_books, progress_msg)
                await ctx.info(f"\n{'=' * 60}\n\U0001f4d6 {progress_msg}\n{'=' * 60}")
            _log_progress("=" * 60)
            _log_progress(f"\U0001f4d6 {progress_msg}")

            try:
                # ── Dispatch to CO (mirrors original subprocess call) ───
                # CO handles: read file → detect chapters → extract metadata
                #             → aggregate → write output file
                payload = {
                    "input_path": book_path,
                    "output_path": out_path,
                    "chapters": None,  # let CO auto-detect
                    "options": {"enable_summary": enable_summary},
                }
                result = await dispatcher.dispatch(TOOL_NAME, payload)
                body = sanitizer.sanitize(result.body)

                book_elapsed = time.time() - book_start

                # ── Parse CO response (mirrors _parse_extraction_output) ──
                if isinstance(body, dict):
                    total_chapters = body.get("total_chapters", 0)
                    unique_kw = body.get("unique_keywords", 0)
                    unique_concepts = body.get("unique_concepts", 0)
                    total_code = body.get("total_code_blocks", 0)
                    total_diagrams = body.get("total_ascii_diagrams", 0)
                    co_output_path = body.get("output_path", out_path)
                    chapter_results = body.get("chapter_results", [])

                    succeeded += 1
                    book_results.append(
                        {
                            "book": book_name,
                            "status": "success",
                            "total_chapters": total_chapters,
                            "unique_keywords": unique_kw,
                            "unique_concepts": unique_concepts,
                            "total_code_blocks": total_code,
                            "total_ascii_diagrams": total_diagrams,
                            "elapsed_s": round(book_elapsed, 1),
                            "output_path": co_output_path,
                        }
                    )

                    # ── Per-chapter detail from CO response ─────────────
                    for cr in chapter_results:
                        ch_num = cr.get("chapter_number", "?")
                        ch_title = cr.get("title", "?")
                        ch_kw = cr.get("keywords_count", 0)
                        ch_concepts = cr.get("concepts_count", 0)
                        ch_summary_len = cr.get("summary_length", 0)
                        ch_code = cr.get("code_blocks_count", 0)
                        ch_status = cr.get("status", "?")

                        ch_msg = (
                            f"   {'✅' if ch_status == 'success' else '⚠️'} "
                            f"Ch {ch_num}: {ch_title} — "
                            f"{ch_summary_len} char summary, "
                            f"{ch_kw} kw, {ch_concepts} concepts, "
                            f"{ch_code} code"
                        )
                        if ctx:
                            await ctx.info(ch_msg)
                        _log_progress(ch_msg)

                    # ── Book completion summary ─────────────────────────
                    done_msg = (
                        f"   \U0001f3c1 Complete: {total_chapters} chapters | "
                        f"{unique_kw:,} keywords | "
                        f"{unique_concepts:,} concepts | "
                        f"{total_code} code blocks — "
                        f"{book_elapsed:.1f}s\n"
                        f"   \U0001f4c4 Written: {co_output_path}"
                    )
                    if ctx:
                        await ctx.info(done_msg)
                    _log_progress(done_msg)

                else:
                    # Unexpected response shape
                    failed += 1
                    book_elapsed = time.time() - book_start
                    err_msg = f"Unexpected CO response type: {type(body).__name__}"
                    book_results.append(
                        {
                            "book": book_name,
                            "status": "failed",
                            "error": err_msg,
                            "elapsed_s": round(book_elapsed, 1),
                        }
                    )
                    if ctx:
                        await ctx.info(f"   \u274c {err_msg}")
                    _log_progress(f"   ❌ {err_msg}")

            except Exception as book_err:
                # ── Error handling (mirrors _handle_failed_extraction) ──
                failed += 1
                book_elapsed = time.time() - book_start
                error_msg = str(book_err)[:200]
                book_results.append(
                    {
                        "book": book_name,
                        "status": "failed",
                        "error": error_msg,
                        "elapsed_s": round(book_elapsed, 1),
                    }
                )
                fail_msg = f"   \u274c Failed: {error_msg} ({book_elapsed:.1f}s)"
                if ctx:
                    await ctx.info(fail_msg)
                _log_progress(fail_msg)
                logger.error("Book extraction failed: %s: %s", book_name, book_err)

        # ── Final summary (mirrors _print_extraction_summary) ──────────
        batch_elapsed = time.time() - batch_start
        if ctx:
            await ctx.report_progress(total_books, total_books, "Batch complete")
            await ctx.info(
                f"\n{'=' * 60}\n"
                f"\U0001f3c1 BATCH COMPLETE\n"
                f"   Duration:   {batch_elapsed:.0f}s\n"
                f"   Successful: {succeeded} books\n"
                f"   Failed:     {failed} books\n"
                f"   Skipped:    {len(skipped)} books\n"
                f"{'=' * 60}"
            )
        _log_progress("=" * 60)
        _log_progress(
            f"\U0001f3c1 BATCH COMPLETE: {succeeded} succeeded, "
            f"{failed} failed, {len(skipped)} skipped — {batch_elapsed:.0f}s"
        )
        _log_progress("=" * 60)

        if failed:
            _log_progress("❌ FAILED BOOKS:")
            for r in book_results:
                if r["status"] == "failed":
                    _log_progress(f"  - {r['book']}: {r.get('error', '?')[:100]}")

        return {
            "status": "complete",
            "total_processed": total_books,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": len(skipped),
            "duration_s": round(batch_elapsed, 1),
            "results": book_results,
        }

    return batch_extract_metadata
