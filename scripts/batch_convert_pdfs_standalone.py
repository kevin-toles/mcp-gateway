#!/usr/bin/env python3
"""Standalone batch PDF conversion script.

Launched by the convert_pdf MCP tool. Discovers all PDF files in input_dir
and calls code-orchestrator /api/v1/workflows/convert-pdf for each one.

Usage:
    python scripts/batch_convert_pdfs_standalone.py \\
        --input-dir /path/to/pdfs \\
        --output-dir /path/to/json \\
        [--file-pattern *.pdf] \\
        [--skip-existing] \\
        [--enable-ocr] \\
        [--co-url http://localhost:8083]

Monitor live from any terminal:
    tail -f /tmp/conversion_progress.log
"""

import argparse
import glob
import json as _json
import os
import sys
import threading
import time
from datetime import UTC, datetime

import httpx

PROGRESS_LOG = "/tmp/conversion_progress.log"  # noqa: S108
PROGRESS_FILE = "/tmp/pdf_progress.json"  # noqa: S108
CO_HEALTH_TIMEOUT = 5.0
CO_REQUEST_TIMEOUT = None  # no timeout — large PDFs can take minutes


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


def _convert_pdf(
    client: httpx.Client,
    co_url: str,
    input_path: str,
    output_path: str,
    enable_ocr: bool,
) -> dict:
    payload = {
        "input_path": input_path,
        "output_path": output_path,
        "enable_ocr": enable_ocr,
    }
    resp = client.post(
        f"{co_url}/api/v1/workflows/convert-pdf",
        json=payload,
        timeout=CO_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _watch_progress(stop_event: threading.Event, pdf_name: str) -> None:
    """Poll CO's progress file; print a live page counter to stdout only (not to log)."""
    last_page = -1
    prefix = pdf_name[:25]
    while not stop_event.wait(0.4):
        try:
            with open(PROGRESS_FILE) as _pf:
                data = _json.load(_pf)
            pdf_file = data.get("pdf", "").replace(".pdf", "")
            if pdf_file.startswith(prefix):
                page = data.get("page", 0)
                total = data.get("total", "?")
                ocr = data.get("ocr", 0)
                method = data.get("method", "")
                if page != last_page:
                    ocr_tag = "  \033[33m🖼  OCR\033[0m" if method == "OCR" else "       "
                    line = f"   ⏳  page {page}/{total}{ocr_tag}  ({ocr} OCR pages)"
                    print(f"\r{line:<80}", end="", flush=True)
                    last_page = page
        except Exception:  # noqa: BLE001, S110
            pass  # progress file may not exist yet or be mid-write; silently skip
    # Clear the progress line on exit
    print(f"\r{' ' * 80}\r", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch PDF conversion via code-orchestrator")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--file-pattern", default="*.pdf")
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--enable-ocr", action="store_true", default=False)
    parser.add_argument("--co-url", default=os.environ.get("CODE_ORCHESTRATOR_URL", "http://localhost:8083"))
    args = parser.parse_args()

    # Resolve output dir
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.input_dir.rstrip("/")), "converted")
    os.makedirs(out_dir, exist_ok=True)

    # Header
    _log("=" * 60)
    _log(cyan("📄 Batch PDF Conversion"))
    _log(f"   Input:   {args.input_dir}")
    _log(f"   Output:  {out_dir}")
    _log(f"   Pattern: {args.file_pattern}")
    _log(f"   Skip existing: {args.skip_existing}")
    _log(f"   OCR: {args.enable_ocr}")
    _log("=" * 60)

    # Health check — fail fast before touching any files
    _health_check(args.co_url)

    # Discover PDFs
    pattern = os.path.join(args.input_dir, args.file_pattern)
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        _log(yellow(f"⚠️  No files matching: {pattern}"))
        sys.exit(0)

    to_process = []
    skipped = []
    for fpath in all_files:
        stem = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(out_dir, f"{stem}.json")
        if args.skip_existing and os.path.exists(out_path):
            skipped.append(os.path.basename(fpath))
        else:
            to_process.append((fpath, out_path))

    total = len(to_process)
    _log(f"   PDFs to convert: {bold(str(total))}  |  Skipped (existing): {len(skipped)}")
    _log("=" * 60)

    if total == 0:
        _log(green("✅  All PDFs already converted. Nothing to do."))
        sys.exit(0)

    # Conversion loop
    succeeded = 0
    failed = 0
    failed_names: list[str] = []
    batch_start = time.time()

    with httpx.Client() as client:
        for idx, (pdf_path, out_path) in enumerate(to_process, 1):
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            book_start = time.time()

            _log("")
            _log(f"[{idx:3d}/{total}] 📄 {pdf_name}")
            _log(f"         → {out_path}")

            stop_evt = threading.Event()
            watcher = threading.Thread(target=_watch_progress, args=(stop_evt, pdf_name), daemon=True)
            watcher.start()
            try:
                body = _convert_pdf(client, args.co_url, pdf_path, out_path, args.enable_ocr)
                elapsed = time.time() - book_start
                stop_evt.set()
                watcher.join(timeout=1.0)

                pages = body.get("pages_converted", body.get("total_pages", "?"))
                ocr_pages = body.get("ocr_pages", 0)
                out = body.get("output_path", out_path)
                ocr_note = f"  ({ocr_pages} OCR)" if ocr_pages else ""
                _log(green(f"   ✅  {pages} pages{ocr_note} — {elapsed:.1f}s"))
                _log(f"   📄  {out}")
                succeeded += 1

            except Exception as e:
                elapsed = time.time() - book_start
                stop_evt.set()
                watcher.join(timeout=1.0)
                _log(red(f"   ❌  Failed ({elapsed:.1f}s): {str(e)[:200]}"))
                failed += 1
                failed_names.append(pdf_name)

    # Summary
    batch_elapsed = time.time() - batch_start
    _log("")
    _log("=" * 60)
    if failed == 0:
        _log(green(f"🏁  COMPLETE — {succeeded} converted, {len(skipped)} skipped — {batch_elapsed:.0f}s"))
    else:
        _log(
            yellow(
                f"🏁  COMPLETE — {succeeded} succeeded, {failed} failed, {len(skipped)} skipped — {batch_elapsed:.0f}s"
            )
        )
        _log(yellow("   Failed PDFs:"))
        for name in failed_names:
            _log(yellow(f"     • {name}"))
    _log("=" * 60)

    sys.exit(1 if failed and succeeded == 0 else 0)


if __name__ == "__main__":
    main()
