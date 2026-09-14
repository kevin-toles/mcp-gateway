#!/usr/bin/env python3
"""Validate YouTube conversion output against the extract-book pipeline contract.

Checks every JSON in the output directory for:
  1. Schema compliance — required top-level keys (metadata, pages, chapters), field types
  2. Content integrity — no empty chapters, no missing content
  3. extract-book compatibility — pages array, page-chapter alignment, page_map assembly
  4. Source fidelity — chapter count vs source, chunk coverage

Usage:
    python scripts/validate_youtube_conversion.py \\
        --output-dir '/Volumes/TDS-Primary-SSD/Platform Data Sets [RAW]/Youtube_Finalized' \\
        [--source-dir '/Volumes/TDS-Primary-SSD/Platform Data Sets [RAW]/Youtube'] \\
        [--sample 0]  # 0 = all files (default), N = random sample of N files
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Schema contract (PDFConversionResult.to_dict() + enrichment pipeline) ───

REQUIRED_TOP_LEVEL = {"metadata", "pages", "chapters"}

REQUIRED_METADATA = {
    "title", "author", "creator", "total_pages", "source_file",
}

REQUIRED_CHAPTER = {
    "title", "chapter_id", "start_page", "end_page", "content", "page_count",
}

VALID_METHODS = {"youtube_chapters", "topic_shift_sbert"}


@dataclass
class ValidationError:
    file: str
    field: str
    message: str


@dataclass
class FileResult:
    path: str
    valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)
    chapter_count: int = 0
    method: str = ""
    empty_content_chapters: int = 0
    total_content_chars: int = 0


@dataclass
class ValidationReport:
    total_files: int = 0
    valid_files: int = 0
    invalid_files: int = 0
    total_chapters: int = 0
    empty_content_chapters: int = 0
    total_content_chars: int = 0
    methods: dict[str, int] = field(default_factory=dict)
    errors_by_type: dict[str, int] = field(default_factory=dict)
    invalid_file_details: list[FileResult] = field(default_factory=list)
    empty_content_details: list[tuple[str, int, str]] = field(default_factory=list)


def _validate_file(filepath: str, source_dir: str | None = None) -> FileResult:
    result = FileResult(path=filepath)
    rel = os.path.basename(filepath)

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result.valid = False
        result.errors.append(ValidationError(rel, "json", f"Invalid JSON: {e}"))
        return result
    except Exception as e:
        result.valid = False
        result.errors.append(ValidationError(rel, "io", f"Read error: {e}"))
        return result

    # 1. Top-level keys
    if not isinstance(data, dict):
        result.valid = False
        result.errors.append(ValidationError(rel, "schema", f"Expected dict, got {type(data).__name__}"))
        return result

    missing_top = REQUIRED_TOP_LEVEL - set(data.keys())
    if missing_top:
        result.valid = False
        result.errors.append(ValidationError(rel, "schema", f"Missing top-level keys: {missing_top}"))

    # 2. Metadata
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        result.valid = False
        result.errors.append(ValidationError(rel, "metadata", "metadata is not a dict"))
    else:
        missing_meta = REQUIRED_METADATA - set(metadata.keys())
        if missing_meta:
            result.valid = False
            result.errors.append(ValidationError(rel, "metadata", f"Missing metadata fields: {missing_meta}"))

        method = metadata.get("chapter_detection_method", "")
        result.method = method
        if method and method not in VALID_METHODS:
            result.valid = False
            result.errors.append(ValidationError(rel, "metadata", f"Unknown method: {method}"))

        if not metadata.get("title", "").strip():
            result.errors.append(ValidationError(rel, "metadata", "Empty title"))

        total_pages = metadata.get("total_pages", 0)
        if not isinstance(total_pages, int) or total_pages < 1:
            result.errors.append(ValidationError(rel, "metadata", f"Invalid total_pages: {total_pages}"))

    # 3. Pages (extract-book contract: pages must be a list with page_number and text)
    pages = data.get("pages", [])
    if not isinstance(pages, list):
        result.valid = False
        result.errors.append(ValidationError(rel, "pages", "pages is not a list"))
    elif len(pages) == 0:
        result.valid = False
        result.errors.append(ValidationError(rel, "pages", "No pages"))
    else:
        page_numbers = set()
        for pi, pg in enumerate(pages):
            if not isinstance(pg, dict):
                result.valid = False
                result.errors.append(ValidationError(rel, f"page[{pi}]", "page is not a dict"))
                continue
            pn = pg.get("page_number")
            if pn is None or not isinstance(pn, int):
                result.valid = False
                result.errors.append(ValidationError(rel, f"page[{pi}]", f"Invalid page_number: {pn}"))
            else:
                page_numbers.add(pn)
            if not pg.get("text", "").strip() and not pg.get("content", "").strip():
                pass  # empty content tracked via chapters below

    # 4. Chapters
    chapters = data.get("chapters", [])
    if not isinstance(chapters, list):
        result.valid = False
        result.errors.append(ValidationError(rel, "chapters", "chapters is not a list"))
        return result

    if len(chapters) == 0:
        result.valid = False
        result.errors.append(ValidationError(rel, "chapters", "No chapters"))
        return result

    result.chapter_count = len(chapters)

    # Check total_pages matches chapter count and page count
    if isinstance(metadata, dict):
        tp = metadata.get("total_pages", 0)
        if tp != len(chapters):
            result.errors.append(ValidationError(
                rel, "consistency",
                f"total_pages ({tp}) != chapter count ({len(chapters)})"
            ))
        if isinstance(pages, list) and tp != len(pages):
            result.errors.append(ValidationError(
                rel, "consistency",
                f"total_pages ({tp}) != page count ({len(pages)})"
            ))

    # Check page-chapter alignment: every chapter's start_page..end_page must exist in pages
    if isinstance(pages, list) and pages:
        for i, ch in enumerate(chapters):
            sp = ch.get("start_page")
            ep = ch.get("end_page")
            if isinstance(sp, int) and isinstance(ep, int):
                for pn in range(sp, ep + 1):
                    if pn not in page_numbers:
                        result.valid = False
                        result.errors.append(ValidationError(
                            rel, f"chapter[{i}]",
                            f"start_page/end_page references page {pn} not in pages array"
                        ))

    seen_ids = set()
    for i, ch in enumerate(chapters):
        ch_label = f"chapter[{i}]"

        if not isinstance(ch, dict):
            result.valid = False
            result.errors.append(ValidationError(rel, ch_label, "chapter is not a dict"))
            continue

        # Required fields
        missing_ch = REQUIRED_CHAPTER - set(ch.keys())
        if missing_ch:
            result.valid = False
            result.errors.append(ValidationError(rel, ch_label, f"Missing fields: {missing_ch}"))

        # chapter_id uniqueness
        cid = ch.get("chapter_id")
        if cid is not None:
            if cid in seen_ids:
                result.errors.append(ValidationError(rel, ch_label, f"Duplicate chapter_id: {cid}"))
            seen_ids.add(cid)

        # Content
        content = ch.get("content", "")
        if not isinstance(content, str):
            result.valid = False
            result.errors.append(ValidationError(rel, ch_label, f"content is {type(content).__name__}, not str"))
        elif not content.strip():
            result.empty_content_chapters += 1
        else:
            result.total_content_chars += len(content)

        # Title
        title = ch.get("title", "")
        if not isinstance(title, str) or not title.strip():
            result.errors.append(ValidationError(rel, ch_label, "Empty or missing title"))

        # start_page / end_page
        sp = ch.get("start_page")
        ep = ch.get("end_page")
        if sp is not None and ep is not None:
            if not isinstance(sp, int) or not isinstance(ep, int):
                result.errors.append(ValidationError(rel, ch_label, f"start_page/end_page not int"))
            elif ep < sp:
                result.errors.append(ValidationError(rel, ch_label, f"end_page ({ep}) < start_page ({sp})"))

        # YouTube chapter timestamps (if present)
        if method == "youtube_chapters":
            st = ch.get("start_time")
            et = ch.get("end_time")
            if st is None or et is None:
                result.errors.append(ValidationError(rel, ch_label, "YouTube chapter missing start_time/end_time"))

    # 4. Source fidelity (optional)
    if source_dir and isinstance(metadata, dict):
        source_file = metadata.get("source_file", "")
        if source_file:
            # Try to find source in mirrored structure
            rel_path = os.path.relpath(filepath, os.path.dirname(filepath))
            parent_dir = Path(filepath).parent.name
            source_path = os.path.join(source_dir, parent_dir, source_file)
            if os.path.exists(source_path):
                try:
                    with open(source_path, encoding="utf-8") as sf:
                        src = json.load(sf)
                    src_chunks = src.get("chunks", [])
                    if not src_chunks:
                        result.errors.append(ValidationError(rel, "source", "Source has no chunks"))
                    src_chapters = src.get("chapters", [])
                    if method == "youtube_chapters" and src_chapters:
                        if len(chapters) != len(src_chapters):
                            result.errors.append(ValidationError(
                                rel, "source",
                                f"Chapter count mismatch: output {len(chapters)} vs source {len(src_chapters)}"
                            ))
                except Exception:
                    pass

    return result


def validate(output_dir: str, source_dir: str | None = None, sample: int = 0) -> ValidationReport:
    report = ValidationReport()

    all_files = sorted(str(p) for p in Path(output_dir).rglob("*.json"))
    if sample > 0 and sample < len(all_files):
        all_files = random.sample(all_files, sample)

    report.total_files = len(all_files)

    for filepath in all_files:
        result = _validate_file(filepath, source_dir)

        if result.valid and result.empty_content_chapters == 0:
            report.valid_files += 1
        elif not result.valid:
            report.invalid_files += 1
            report.invalid_file_details.append(result)
        else:
            report.valid_files += 1  # structurally valid, just empty content

        report.total_chapters += result.chapter_count
        report.empty_content_chapters += result.empty_content_chapters
        report.total_content_chars += result.total_content_chars

        if result.method:
            report.methods[result.method] = report.methods.get(result.method, 0) + 1

        for err in result.errors:
            report.errors_by_type[err.field] = report.errors_by_type.get(err.field, 0) + 1

        if result.empty_content_chapters > 0:
            report.empty_content_details.append((
                filepath, result.empty_content_chapters, result.method
            ))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YouTube conversion output")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--sample", type=int, default=0,
                        help="Random sample size (0 = validate all)")
    args = parser.parse_args()

    print("=" * 70)
    print("🔍 YouTube Conversion Validator")
    print(f"   Output: {args.output_dir}")
    if args.source_dir:
        print(f"   Source: {args.source_dir}")
    if args.sample:
        print(f"   Sample: {args.sample}")
    print("=" * 70)

    start = time.time()
    report = validate(args.output_dir, args.source_dir, args.sample)
    elapsed = time.time() - start

    print()
    print(f"   Files validated:     {report.total_files:,}")
    print(f"   Valid:               {report.valid_files:,}")
    print(f"   Invalid (schema):    {report.invalid_files:,}")
    print(f"   Total chapters:      {report.total_chapters:,}")
    print(f"   Empty-content chaps: {report.empty_content_chapters:,}")
    print(f"   Total content:       {report.total_content_chars:,} chars")
    print()
    print(f"   Methods:")
    for method, count in sorted(report.methods.items()):
        print(f"     {method}: {count:,}")
    print()

    if report.errors_by_type:
        print(f"   Errors by type:")
        for etype, count in sorted(report.errors_by_type.items(), key=lambda x: -x[1]):
            print(f"     {etype}: {count:,}")
        print()

    if report.invalid_file_details:
        print(f"   ❌ Invalid files ({len(report.invalid_file_details)}):")
        for fr in report.invalid_file_details[:20]:
            print(f"     {os.path.basename(fr.path)}:")
            for err in fr.errors:
                print(f"       [{err.field}] {err.message}")
        if len(report.invalid_file_details) > 20:
            print(f"     ... and {len(report.invalid_file_details) - 20} more")
        print()

    if report.empty_content_details:
        print(f"   ⚠️  Files with empty-content chapters ({len(report.empty_content_details)}):")
        for path, count, method in report.empty_content_details[:20]:
            print(f"     {os.path.basename(path)}: {count} empty ({method})")
        if len(report.empty_content_details) > 20:
            print(f"     ... and {len(report.empty_content_details) - 20} more")
        print()

    print(f"   Elapsed: {elapsed:.1f}s")
    print("=" * 70)

    if report.invalid_files == 0:
        print("✅  ALL FILES PASS SCHEMA VALIDATION")
    else:
        print(f"❌  {report.invalid_files} FILES FAILED SCHEMA VALIDATION")

    print("=" * 70)
    sys.exit(1 if report.invalid_files else 0)


if __name__ == "__main__":
    main()
