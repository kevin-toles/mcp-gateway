#!/usr/bin/env python3
"""Audit JSON files: verify metadata.source_file and metadata.title match filename.

Checks every JSON under --json-dir:
  - metadata.source_file stem  == JSON filename stem
  - metadata.title              is non-empty

Usage:
    python scripts/audit_json_metadata.py \
        --json-dir "/Volumes/TDS-Primary-SSD/Platform Data Sets [RAW]/Fresh JSONs" \
        [--workers 12] \
        [--report /tmp/json_metadata_audit.json]
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def audit_file(json_path: Path) -> dict | None:
    """Return a mismatch record if anything is wrong, else None."""
    stem = json_path.stem

    try:
        with open(json_path, "rb") as f:
            data = json.load(f)
    except Exception as e:
        return {
            "file": str(json_path),
            "issue": "parse_error",
            "detail": str(e)[:200],
        }

    meta = data.get("metadata", {})
    if not isinstance(meta, dict):
        return {
            "file": str(json_path),
            "issue": "missing_metadata",
            "detail": f"metadata field is {type(meta).__name__}",
        }

    issues = []

    # ── source_file check ─────────────────────────────────────────────────────
    source_file = meta.get("source_file", "")
    if not source_file:
        issues.append("source_file_empty")
    else:
        # source_file should be "<stem>.pdf" — compare stems
        source_stem = Path(source_file).stem
        if source_stem.lower() != stem.lower():
            issues.append(f"source_file_mismatch: '{source_stem}' != '{stem}'")

    # ── title check ───────────────────────────────────────────────────────────
    title = meta.get("title", "")
    if not title or not title.strip():
        issues.append("title_empty")

    if issues:
        return {
            "file": str(json_path),
            "stem": stem,
            "source_file": source_file,
            "title": (title or "")[:120],
            "issues": issues,
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit JSON metadata fields vs filename")
    parser.add_argument("--json-dir", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--report", default="/tmp/json_metadata_audit.json")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    print(f"Discovering JSON files under {json_dir} ...")
    all_jsons = list(json_dir.rglob("*.json"))
    total = len(all_jsons)
    print(f"Found {total:,} JSON files. Auditing with {args.workers} workers ...\n")

    mismatches: list[dict] = []
    ok = 0
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_file, p): p for p in all_jsons}
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is None:
                ok += 1
            else:
                mismatches.append(result)

            if done % 5000 == 0 or done == total:
                pct = done / total * 100
                with _PRINT_LOCK:
                    print(f"\r  {done:,}/{total:,} ({pct:.1f}%)  issues found: {len(mismatches):,}",
                          end="", flush=True)

    print("\n")

    # ── Tally by issue type ────────────────────────────────────────────────────
    from collections import Counter
    issue_counts: Counter = Counter()
    for m in mismatches:
        for issue in m.get("issues", [m.get("issue", "unknown")]):
            # Normalize source_file_mismatch variants to one bucket
            key = issue if "source_file_mismatch" not in issue else "source_file_mismatch"
            issue_counts[key] += 1

    print("=" * 60)
    print("AUDIT RESULTS")
    print("=" * 60)
    print(f"  Total JSONs:       {total:,}")
    print(f"  Clean (no issues): {ok:,}")
    print(f"  Has issues:        {len(mismatches):,}")
    print()
    print("  Issue breakdown:")
    for issue, count in issue_counts.most_common():
        print(f"    {issue:<35} {count:>8,}")
    print("=" * 60)

    # ── Write report ───────────────────────────────────────────────────────────
    report = {
        "json_dir": str(json_dir),
        "total": total,
        "clean": ok,
        "has_issues": len(mismatches),
        "issue_counts": dict(issue_counts),
        "mismatches": mismatches,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report: {args.report}")

    sys.exit(0 if not mismatches else 1)


if __name__ == "__main__":
    main()
