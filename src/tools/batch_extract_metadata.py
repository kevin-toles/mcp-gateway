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
import time
from datetime import UTC, datetime

from fastmcp import Context

from src.models.schemas import BatchExtractMetadataInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "extract_book_metadata"  # dispatches to CO /api/v1/workflows/extract-book
PROGRESS_LOG = "/tmp/extraction_progress.log"

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


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler — simple dispatch loop with progress."""

    async def batch_extract_metadata(
        input_dir: str,
        output_dir: str | None = None,
        file_pattern: str = "*.json",
        skip_existing: bool = True,
        enable_summary: bool = False,
        ctx: Context | None = None,
    ) -> dict:
        """Extract metadata from all books in a directory.

        Mirrors llm-document-enhancer/scripts/batch_extract_metadata.py:
        - Discovers books, dispatches each to CO extract-book endpoint
        - CO handles chapter detection, extraction, aggregation, file writing
        - Reports per-book progress

        Monitor live: tail -f /tmp/extraction_progress.log

        Args:
            input_dir: Directory containing raw book JSON files.
            output_dir: Directory for metadata output (defaults to sibling 'metadata' dir).
            file_pattern: Glob pattern for book files (default: *.json).
            skip_existing: Skip books that already have metadata output files.
            enable_summary: Whether to generate LLM summaries (default: False for speed).
        """
        validated = BatchExtractMetadataInput(
            input_dir=input_dir,
            output_dir=output_dir,
            file_pattern=file_pattern,
            skip_existing=skip_existing,
        )

        # Store enable_summary for use in dispatch loop
        _enable_summary = enable_summary

        # Resolve output directory (mirrors original METADATA_EXTRACTION_CONFIG)
        out_dir = validated.output_dir
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(validated.input_dir.rstrip("/")), "metadata")

        os.makedirs(out_dir, exist_ok=True)

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
                    "options": {"enable_summary": _enable_summary},
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
