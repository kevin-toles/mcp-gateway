#!/usr/bin/env python3
"""Standalone batch metadata extraction script.

Launched by the batch_extract_metadata MCP tool when background=True.
Calls code-orchestrator /api/v1/workflows/extract-book for each book.

Usage:
    python scripts/batch_extract_standalone.py \\
        --input-dir /path/to/raw \\
        --output-dir /path/to/metadata \\
        [--file-pattern *.json] \\
        [--skip-existing] \\
        [--enable-summary] \\
        [--co-url http://localhost:8083]

Monitor live from any terminal:
    tail -f /tmp/extraction_progress.log
"""

import argparse
import glob
import os
import sys
import time
from datetime import UTC, datetime

import httpx

PROGRESS_LOG = "/tmp/extraction_progress.log"  # noqa: S108
CO_HEALTH_TIMEOUT = 30.0
CO_REQUEST_TIMEOUT = None  # no timeout — mirrors original behaviour


# ── ANSI colours (disabled when not a TTY) ──────────────────────────────────
def _c(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def green(t: str) -> str:
    return _c("1;32", t)


def red(t: str) -> str:
    return _c("1;31", t)


def cyan(t: str) -> str:
    return _c("1;36", t)


def yellow(t: str) -> str:
    return _c("1;33", t)


def bold(t: str) -> str:
    return _c("1", t)


def _log(msg: str) -> None:
    """Print to stdout AND append to progress log."""
    print(msg, flush=True)
    ts = datetime.now(UTC).strftime("%H:%M:%S")
    with open(PROGRESS_LOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")
        f.flush()


def _health_check(co_url: str) -> None:
    """Fail fast if code-orchestrator is not reachable."""
    try:
        r = httpx.get(f"{co_url}/health", timeout=CO_HEALTH_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        _log(red(f"❌  code-orchestrator health check failed: {e}"))
        _log(red("    Start it first:  cd /Users/kevintoles/POC/Code-Orchestrator-Service"))
        _log(red("    source .venv/bin/activate && uvicorn src.main:app --host 0.0.0.0 --port 8083"))
        sys.exit(1)
    _log(green("✅  code-orchestrator healthy"))


def _extract_book(
    client: httpx.Client,
    co_url: str,
    book_path: str,
    out_path: str,
    enable_summary: bool,
    vtf_path: str | None = None,
) -> dict:
    options: dict = {"enable_summary": enable_summary}
    if vtf_path:
        options["vtf_path"] = vtf_path
    payload = {
        "input_path": book_path,
        "output_path": out_path,
        "chapters": None,
        "options": options,
    }
    resp = client.post(
        f"{co_url}/api/v1/workflows/extract-book",
        json=payload,
        timeout=CO_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch metadata extraction via code-orchestrator")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--file-pattern", default="*.json")
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--enable-summary", action="store_true", default=False)
    parser.add_argument("--vtf-path", default=None, help="Path to a custom validated_term_filter.json")
    parser.add_argument("--co-url", default=os.environ.get("CODE_ORCHESTRATOR_URL", "http://localhost:8083"))
    args = parser.parse_args()

    # Resolve output dir
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.input_dir.rstrip("/")), "metadata")
    os.makedirs(out_dir, exist_ok=True)

    # Header
    _log("=" * 60)
    _log(cyan("📚 Batch Metadata Extraction"))
    _log(f"   Input:   {args.input_dir}")
    _log(f"   Output:  {out_dir}")
    _log(f"   Pattern: {args.file_pattern}")
    _log(f"   Skip existing: {args.skip_existing}")
    if args.vtf_path:
        _log(f"   VTF:    {args.vtf_path}")
    _log("=" * 60)

    # Health check — fail fast before touching any files
    _health_check(args.co_url)

    # Discover books — support recursive glob when '**' is in pattern
    pattern = os.path.join(args.input_dir, args.file_pattern)
    is_recursive = "**" in args.file_pattern
    all_files = sorted(glob.glob(pattern, recursive=is_recursive))
    if not all_files:
        _log(yellow(f"⚠️  No files matching: {pattern}"))
        sys.exit(0)

    # Build (book_path, out_path) pairs — preserve subdir structure relative to input_dir
    books_to_process = []  # list of (fpath, out_path)
    skipped = []
    for fpath in all_files:
        rel = os.path.relpath(fpath, args.input_dir)
        rel_dir = os.path.dirname(rel)
        stem = os.path.splitext(os.path.basename(fpath))[0]
        out_subdir = os.path.join(out_dir, rel_dir)
        out_path = os.path.join(out_subdir, f"{stem}_metadata.json")
        if args.skip_existing and os.path.exists(out_path):
            skipped.append(os.path.basename(fpath))
        else:
            books_to_process.append((fpath, out_path))

    total = len(books_to_process)
    _log(f"   Books to process: {bold(str(total))}  |  Skipped (existing): {len(skipped)}")
    _log("=" * 60)

    if total == 0:
        _log(green("✅  All books already have metadata. Nothing to do."))
        sys.exit(0)

    # Extraction loop
    succeeded = 0
    failed = 0
    batch_start = time.time()

    with httpx.Client() as client:
        for idx, (book_path, out_path) in enumerate(books_to_process, 1):
            book_name = os.path.splitext(os.path.basename(book_path))[0]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            book_start = time.time()

            _log("")
            _log(f"[{idx:3d}/{total}] 📖 {book_name}")

            try:
                body = _extract_book(client, args.co_url, book_path, out_path, args.enable_summary, args.vtf_path)
                elapsed = time.time() - book_start

                ch = body.get("total_chapters", 0)
                kw = body.get("unique_keywords", 0)
                concepts = body.get("unique_concepts", 0)
                code_blocks = body.get("total_code_blocks", 0)

                _log(
                    green(
                        f"   ✅  {ch} chapters | {kw:,} keywords | {concepts:,} concepts | "
                        f"{code_blocks} code blocks — {elapsed:.1f}s"
                    )
                )
                _log(f"   📄  {out_path}")
                succeeded += 1

            except Exception as e:
                elapsed = time.time() - book_start
                _log(red(f"   ❌  Failed ({elapsed:.1f}s): {str(e)[:200]}"))
                failed += 1

    # Summary
    batch_elapsed = time.time() - batch_start
    _log("")
    _log("=" * 60)
    if failed == 0:
        _log(green(f"🏁  COMPLETE — {succeeded} succeeded, {len(skipped)} skipped — {batch_elapsed:.0f}s"))
    else:
        _log(
            yellow(
                f"🏁  COMPLETE — {succeeded} succeeded, {failed} failed, {len(skipped)} skipped — {batch_elapsed:.0f}s"
            )
        )
    _log("=" * 60)

    sys.exit(1 if failed and succeeded == 0 else 0)


if __name__ == "__main__":
    main()
