#!/usr/bin/env python3
"""Triage PDFs that are missing a JSON output — classify each failure locally.

Finds every PDF in input_dir that has no matching JSON in output_dir, then
classifies it using PyMuPDF only — no CO calls, no HTTP. Fast local analysis.

Categories
----------
broken_page_tree    xref table has Page objects but catalog page_count == 0.
                    → Fixable with: qpdf --rebuild-from-scratch in.pdf out.pdf

password_protected  PDF is encrypted and requires a password.
                    → Skip or attempt password removal with qpdf/ghostscript.

wrong_file_type     File header is not %PDF (HTML, XML, zip, etc. with .pdf ext).
                    → Delete or re-acquire.

empty               File is zero bytes or under 512 bytes.
                    → Delete or re-acquire.

corrupt             fitz raises an unrecoverable exception on open.
                    → Try ghostscript re-render; otherwise unrecoverable.

already_converted   JSON exists — skip_existing missed it (edge case, logged for info).

unknown             Opens with fitz, page_count == 0, no Page xrefs found.
                    → Try ghostscript re-render as last resort.

Usage
-----
    python scripts/triage_broken_pdfs.py \\
        --input-dir "/Volumes/TDS-Primary-SSD/Platform Data Sets [RAW]/Fresh PDFs" \\
        --output-dir "/Volumes/TDS-Primary-SSD/Platform Data Sets [RAW]/Fresh JSONs" \\
        [--file-pattern "*.pdf"] \\
        [--workers 8] \\
        [--report triage_report.json] \\
        [--lists-dir /tmp/triage_lists]

Output
------
  triage_report.json   — summary counts + per-category file lists
  /tmp/triage_lists/   — one .txt per category (full paths, one per line)
                          feed directly into repair scripts
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

CATEGORIES = [
    "broken_page_tree",
    "password_protected",
    "wrong_file_type",
    "empty",
    "corrupt",
    "unknown",
    "already_converted",
]

_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def _check_pdf_header(path: str) -> bool:
    """Return True if file starts with %PDF."""
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def _count_page_xrefs(doc: fitz.Document) -> int:
    """Count xref entries whose /Type is /Page."""
    count = 0
    for i in range(doc.xref_length()):
        try:
            t = doc.xref_get_key(i, "Type")
            if t and "Page" in str(t) and "Pages" not in str(t):
                count += 1
        except Exception:  # noqa: BLE001
            pass
    return count


def classify(pdf_path: str) -> tuple[str, str]:
    """Classify one PDF. Returns (category, detail_note)."""
    try:
        size = os.path.getsize(pdf_path)
    except OSError as e:
        return "corrupt", f"stat failed: {e}"

    if size < 512:
        return "empty", f"{size} bytes"

    if not _check_pdf_header(pdf_path):
        try:
            with open(pdf_path, "rb") as f:
                header = f.read(16).decode("latin-1", errors="replace")
        except OSError:
            header = "unreadable"
        return "wrong_file_type", f"header: {header!r}"

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        err = str(e).lower()
        if "password" in err or "encrypted" in err:
            return "password_protected", str(e)
        return "corrupt", str(e)[:200]

    try:
        if doc.needs_pass or (doc.is_encrypted and doc.page_count == 0):
            return "password_protected", "encrypted, needs password"

        if doc.page_count > 0:
            # Has pages — CO's 422 was for a different reason (rare)
            return "unknown", f"page_count={doc.page_count} but CO returned 422"

        # page_count == 0 — check xref for orphaned Page objects
        page_xrefs = _count_page_xrefs(doc)
        if page_xrefs > 0:
            return "broken_page_tree", f"{page_xrefs} orphaned Page xrefs, xref_length={doc.xref_length()}"

        return "unknown", f"page_count=0, no Page xrefs, xref_length={doc.xref_length()}, size={size}"
    finally:
        doc.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage PDFs missing JSON output")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--file-pattern", default="*.pdf")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", default="/tmp/triage_report.json")  # noqa: S108
    parser.add_argument("--lists-dir", default="/tmp/triage_lists")  # noqa: S108
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    lists_dir = Path(args.lists_dir)
    lists_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover all PDFs ──────────────────────────────────────────────────────
    print(f"Scanning {args.input_dir} ...")
    if "**" in args.file_pattern:
        import glob as _glob
        all_pdfs = sorted(_glob.glob(os.path.join(args.input_dir, args.file_pattern), recursive=True))
    else:
        all_pdfs = sorted(str(p) for p in input_path.rglob(args.file_pattern))
    print(f"Found {len(all_pdfs):,} PDFs total")

    # ── Determine mirror structure ─────────────────────────────────────────────
    has_subdirs = any(Path(f).parent != input_path for f in all_pdfs)

    # ── Find PDFs missing JSON ─────────────────────────────────────────────────
    print("Building existing JSON set ...")
    existing_jsons: set[str] = set()
    for dirpath, _dirs, filenames in os.walk(output_path):
        for fn in filenames:
            if fn.endswith(".json"):
                existing_jsons.add(os.path.join(dirpath, fn))
    print(f"Found {len(existing_jsons):,} existing JSONs")

    missing: list[str] = []
    already_converted: list[str] = []
    for fpath in all_pdfs:
        fobj = Path(fpath)
        if has_subdirs:
            rel = fobj.relative_to(input_path)
            out = str(output_path / rel.with_suffix(".json"))
        else:
            out = str(output_path / (fobj.stem + ".json"))

        if out in existing_jsons:
            already_converted.append(fpath)
        else:
            missing.append(fpath)

    print(f"Missing JSON: {len(missing):,}  |  Already converted: {len(already_converted):,}")
    print(f"Classifying {len(missing):,} files with {args.workers} workers ...\n")

    # ── Classify in parallel ───────────────────────────────────────────────────
    results: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    details: dict[str, str] = {}
    done = 0
    total = len(missing)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {executor.submit(classify, p): p for p in missing}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            done += 1
            try:
                category, note = future.result()
            except Exception as e:  # noqa: BLE001
                category, note = "corrupt", str(e)[:200]

            results[category].append(path)
            details[path] = note

            if done % 100 == 0 or done == total:
                pct = done / total * 100
                with _PRINT_LOCK:
                    print(f"\r  {done:,}/{total:,} ({pct:.1f}%)", end="", flush=True)

    results["already_converted"] = already_converted
    print("\n")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("TRIAGE RESULTS")
    print("=" * 60)

    fixes = {
        "broken_page_tree":   "→ qpdf --rebuild-from-scratch",
        "password_protected":  "→ skip or attempt password removal",
        "wrong_file_type":    "→ delete or re-acquire",
        "empty":              "→ delete or re-acquire",
        "corrupt":            "→ try ghostscript re-render, else unrecoverable",
        "unknown":            "→ try ghostscript re-render as last resort",
        "already_converted":  "→ no action needed",
    }

    report_categories = []
    for cat in CATEGORIES:
        count = len(results[cat])
        if count == 0:
            continue
        fix = fixes[cat]
        print(f"  {cat:<22} {count:>6,}  {fix}")
        report_categories.append({"category": cat, "count": count, "fix": fix})

    print("=" * 60)
    fixable = len(results["broken_page_tree"])
    unrecoverable = len(results["wrong_file_type"]) + len(results["empty"])
    uncertain = len(results["corrupt"]) + len(results["unknown"])
    print(f"  Fixable (qpdf):      {fixable:,}")
    print(f"  Try GS re-render:    {uncertain:,}")
    print(f"  Unrecoverable:       {unrecoverable:,}")
    print("=" * 60)

    # ── Write per-category lists ───────────────────────────────────────────────
    for cat in CATEGORIES:
        if not results[cat]:
            continue
        list_path = lists_dir / f"{cat}.txt"
        with open(list_path, "w") as f:
            f.write("\n".join(results[cat]) + "\n")
        print(f"  Written: {list_path}  ({len(results[cat]):,} paths)")

    # ── Write JSON report ──────────────────────────────────────────────────────
    report = {
        "generated": datetime.now(UTC).isoformat(),
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "total_pdfs": len(all_pdfs),
        "missing_json": len(missing),
        "categories": report_categories,
        "files": {cat: results[cat] for cat in CATEGORIES if results[cat]},
        "details": {p: details[p] for p in missing},
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report: {args.report}")


if __name__ == "__main__":
    main()
