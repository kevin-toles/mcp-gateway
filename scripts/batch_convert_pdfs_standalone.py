#!/usr/bin/env python3
"""Standalone batch document conversion script.

Launched by the convert_pdf MCP tool. Discovers PDF and Markdown files in
input_dir and calls code-orchestrator /api/v1/workflows/convert-pdf for each.

  .pdf files  — converted via PyMuPDF + optional OCR.
               If extracted author/subject are blank, metadata_overrides are
               auto-populated from known_papers_registry.json.
  .md files   — converted via the markdown bridge in the convert-pdf endpoint
               (no PDF processing, builds raw JSON directly from header fields).

Usage:
    python scripts/batch_convert_pdfs_standalone.py \\
        --input-dir /path/to/docs \\
        --output-dir /path/to/json \\
        [--skip-existing] \\
        [--enable-ocr] \\
        [--registry /path/known_papers_registry.json] \\
        [--co-url http://localhost:8083]

Monitor live from any terminal:
    tail -f /tmp/conversion_progress.log
"""

import argparse
import json as _json
import os
import re
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

PROGRESS_LOG = "/tmp/conversion_progress.log"  # noqa: S108
PROGRESS_FILE = "/tmp/pdf_progress.json"  # noqa: S108
CO_HEALTH_TIMEOUT = 5.0
CO_REQUEST_TIMEOUT = None  # no timeout — large PDFs can take minutes

# Default registry path — built by scripts/_build_papers_registry.py
_DEFAULT_REGISTRY = os.path.join(
    os.path.dirname(__file__), "..", "..", "ai-platform-data", "data", "known_papers_registry.json"
)

SUPPORTED_EXTENSIONS = {".pdf", ".md"}


def _exists_case_sensitive(path: str) -> bool:
    """Case-sensitive file existence check.

    On macOS (case-insensitive HFS+) ``os.path.exists`` treats
    ``How to Share a Secret.json`` and ``How To Share A Secret.json``
    as the same file, silently preserving wrong-cased files that were
    created by older pipelines.  Comparing against ``os.listdir`` of
    the parent directory enforces exact case matching.
    """
    p = os.path.abspath(path)
    parent, name = os.path.split(p)
    return os.path.isdir(parent) and name in os.listdir(parent)


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
    metadata_overrides: dict | None = None,
) -> dict:
    payload: dict = {
        "input_path": input_path,
        "output_path": output_path,
        "enable_ocr": enable_ocr,
    }
    if metadata_overrides:
        payload["metadata_overrides"] = metadata_overrides
    resp = client.post(
        f"{co_url}/api/v1/workflows/convert-pdf",
        json=payload,
        timeout=CO_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ── Known-papers registry ————————————————————————————————————————————————————
_registry_cache: dict | None = None


def _load_registry(registry_path: str) -> dict:
    """Load (and cache) known_papers_registry.json."""
    global _registry_cache
    if _registry_cache is None:
        resolved = os.path.realpath(registry_path)
        if os.path.exists(resolved):
            try:
                with open(resolved) as f:
                    _registry_cache = _json.load(f)
            except Exception:
                _registry_cache = {}
        else:
            _registry_cache = {}
    return _registry_cache


def _normalize_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[\u2019\u2018'`]", "", t)
    t = re.sub(r"[:\\.?!]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _get_overrides(stem: str, registry: dict) -> dict | None:
    """Return metadata_overrides for a PDF stem if registry has an entry.

    The stem is the filename without extension; PDF files from the academic
    collection are named after the paper title (e.g.,
    'time-clocks-and-the-ordering-of-events.pdf').
    Only returns overrides when at least one useful field is present.
    """
    norm = _normalize_title(stem.replace("-", " ").replace("_", " "))
    entry = registry.get(norm)
    if not entry:
        return None
    overrides = {k: v for k, v in entry.items() if v}
    return overrides or None


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
    parser = argparse.ArgumentParser(description="Batch document conversion (PDF + Markdown) via code-orchestrator")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--enable-ocr", action="store_true", default=False)
    parser.add_argument(
        "--registry",
        default=_DEFAULT_REGISTRY,
        help="Path to known_papers_registry.json for metadata overrides",
    )
    parser.add_argument("--co-url", default=os.environ.get("CODE_ORCHESTRATOR_URL", "http://localhost:8083"))
    args = parser.parse_args()

    # Resolve output dir
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.input_dir.rstrip("/")), "converted")
    os.makedirs(out_dir, exist_ok=True)

    # Load metadata override registry (silent if missing)
    registry = _load_registry(args.registry)
    registry_note = f"{len(registry)} entries" if registry else "not found — no overrides"

    # Header
    _log("=" * 60)
    _log(cyan("📄 Batch Document Conversion (PDF + Markdown)"))
    _log(f"   Input:         {args.input_dir}")
    _log(f"   Output:        {out_dir}")
    _log(f"   Skip existing: {args.skip_existing}")
    _log(f"   OCR:           {args.enable_ocr}")
    _log(f"   Registry:      {registry_note}")
    _log("=" * 60)

    # Health check — fail fast before touching any files
    _health_check(args.co_url)

    # Discover supported files — PDFs and Markdowns (recursive)
    input_root = Path(args.input_dir)
    all_files: list[Path] = sorted(
        p for p in input_root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
    )
    if not all_files:
        _log(yellow(f"⚠️  No .pdf or .md files found (recursively) in: {args.input_dir}"))
        sys.exit(0)

    to_process: list[tuple[str, str]] = []
    skipped: list[str] = []
    for fpath in all_files:
        # Mirror the relative sub-path under output dir so nested structure is preserved
        rel = fpath.relative_to(input_root)
        out_path = str(Path(out_dir) / rel.parent / (fpath.stem + ".json"))
        if args.skip_existing and _exists_case_sensitive(out_path):
            skipped.append(str(rel))
        else:
            to_process.append((str(fpath), out_path))

    pdf_count = sum(1 for f, _ in to_process if f.endswith(".pdf"))
    md_count = sum(1 for f, _ in to_process if f.endswith(".md"))
    total = len(to_process)

    _log(
        f"   To convert: {bold(str(total))}  "
        f"(PDF: {pdf_count}, Markdown: {md_count})  |  "
        f"Skipped (existing): {len(skipped)}"
    )
    _log("=" * 60)

    if total == 0:
        _log(green("✅  All documents already converted. Nothing to do."))
        sys.exit(0)

    # Conversion loop
    succeeded = 0
    failed = 0
    failed_names: list[str] = []
    batch_start = time.time()

    with httpx.Client() as client:
        for idx, (doc_path, out_path) in enumerate(to_process, 1):
            doc_name = os.path.splitext(os.path.basename(doc_path))[0]
            ext = Path(doc_path).suffix.lower()
            book_start = time.time()

            icon = "📄" if ext == ".pdf" else "📝"
            _log("")
            _log(f"[{idx:3d}/{total}] {icon} {doc_name}  [{ext}]")
            _log(f"         → {out_path}")

            try:
                if ext == ".pdf":
                    # Auto-look up overrides for PDFs with potentially blank metadata
                    overrides = _get_overrides(doc_name, registry)
                    if overrides:
                        _log(f"   🔍  registry match — overrides: {list(overrides.keys())}")

                    stop_evt = threading.Event()
                    watcher = threading.Thread(target=_watch_progress, args=(stop_evt, doc_name), daemon=True)
                    watcher.start()
                    body = _convert_pdf(client, args.co_url, doc_path, out_path, args.enable_ocr, overrides)
                    stop_evt.set()
                    watcher.join(timeout=1.0)

                    pages = body.get("total_pages", "?")
                    ocr_pages = body.get("ocr_pages", 0)
                    ocr_note = f"  ({ocr_pages} OCR)" if ocr_pages else ""
                    _log(green(f"   ✅  {pages} pages{ocr_note} — {time.time() - book_start:.1f}s"))

                else:
                    # Markdown: send to the same endpoint — it routes internally
                    body = _convert_pdf(client, args.co_url, doc_path, out_path, False, None)
                    _log(green(f"   ✅  markdown converted — {time.time() - book_start:.1f}s"))

                _log(f"   📄  {body.get('output_path', out_path)}")
                succeeded += 1

            except Exception as e:
                elapsed = time.time() - book_start
                if ext == ".pdf":
                    try:
                        stop_evt.set()
                        watcher.join(timeout=1.0)
                    except Exception:
                        pass
                _log(red(f"   ❌  Failed ({elapsed:.1f}s): {str(e)[:200]}"))
                failed += 1
                failed_names.append(doc_name)

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
        _log(yellow("   Failed files:"))
        for name in failed_names:
            _log(yellow(f"     • {name}"))
    _log("=" * 60)

    sys.exit(1 if failed and succeeded == 0 else 0)


if __name__ == "__main__":
    main()
