#!/usr/bin/env python3
"""Patch title_empty JSONs by extracting title from page 1 text or filename.

Reads /tmp/json_metadata_audit.json for the list of title_empty files, then for
each:
  1. Strips leading page-number artifacts from page 1 text.
  2. If text begins with known non-title patterns, falls back to filename stem.
  3. Handles 'Article Title:' prefix pattern.
  4. Collects title lines until author/institution/abstract content begins.
  5. Normalises: collapses whitespace, title-cases ALL-CAPS strings.
  6. Writes patched title back into metadata.title in-place.

Usage:
    python scripts/patch_empty_titles.py [--apply] [--report /tmp/json_metadata_audit.json]

Default is DRY-RUN (shows proposed titles, writes nothing).
Pass --apply to commit changes.
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ── Stop-condition patterns ────────────────────────────────────────────────────

# Hard-stop: entire line is one of these → don't extract title from this page
PAGE_NONTITLE_START_RE = re.compile(
    r"^\s*abstract[\s:—–-]|"           # starts with Abstract
    r"^\s*submitted\s+to\b|"            # "Submitted to IEEE…"
    r"^\s*citation\s*:|"                # "CITATION: Connor…"
    r"^\s*accepted\s+|"                 # "Accepted for…"
    r"^\s*copyright\s*|"               # copyright notice
    r"^\s*©\s*\d|"                      # © 2020
    r"^\s*this\s+is\s+the\s+pre\-|"    # "This is the pre-peer…"
    r"^\s*\>\s*replace\s+|"            # "> Replace This Line…"
    r"^\s*ieee\s+\w+\s+journal|"       # "IEEE SENSORS JOURNAL…"
    r"^\s*ieee\s+transactions|"        # "IEEE TRANSACTIONS ON…"
    r"^\s*acm\s+\w|"                   # ACM journal header
    r"^\s*proceedings\s+of\b",         # "Proceedings of the…"
    re.IGNORECASE,
)

# Per-line stops during extraction (stop collecting title at this line)
LINE_STOP_RE = re.compile(
    r"@|"                                       # email address
    r"\buniversity\b|\binstitute\b|\bschool\b|"
    r"\bdepartment\b|\blaboratory\b|\bcollege\b|"
    r"\bcenter\b|\bcentre\b|\borgid\b|"
    r"\borcid\b|\baffiliation\b|"
    r"^\s*authors?\s*:|"                        # "Authors:" label
    r"^\s*abstract[\s:—–-]|"                   # Abstract
    r"^\s*submitted\s+to\b|"                    # journal submission
    r"^\s*citation\s*:|"
    r"^\s*©\s*\d|"
    r"^\s*\>\s*|"                               # template line starting with >
    r"[，、]",                                   # Chinese/CJK comma (author lists)
    re.IGNORECASE,
)

# Author-name patterns (to stop extraction)
SUPERSCRIPT_NUM_RE = re.compile(r"\*?\s*\d+\s*[,\*]?\s*$")  # "Wu 1", "Wang 1*"
INLINE_SUPERSCRIPT_RE = re.compile(r"\(\d+\)")              # "(1)" mid-line in author names
ALLCAPS_NAME_RE = re.compile(r"^[A-Z][A-Z\s\-]{4,}[A-Z]$")  # "BIRGIT VOGEL-HEUSER"
DIGIT_AUTHOR_RE = re.compile(r"^\d+[A-Z]|;\s*\d+[A-Z]")    # "1Ye, Han; 1Lijun, Zhang*"
# Author name list — handles "First [Mid] Last*, First Last, and First Last"
NAME_LIST_RE = re.compile(
    r"^[A-Z][a-zé\-]+\.?(\s+[A-Z][a-zé\-]+\.?){1,2}\s*\*?"
    r"([,\s]+(and\s+)?[A-Z][a-zé\-]+\.?(\s+[A-Z][a-zé\-]+\.?){1,2}\s*\*?)*"
    r"\s*$"
)

MAX_TITLE_LINES = 4   # Never take more than this many lines as a title


# ── Normalization ──────────────────────────────────────────────────────────────

def _is_allcaps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 6:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) >= 0.65


def _to_titlecase(s: str) -> str:
    result = s.title()
    # Fix possessives/contractions mangled by title(): "It'S" → "It's"
    result = re.sub(r"(\w)'(\w)", lambda m: m.group(1) + "'" + m.group(2).lower(), result)
    return result


def _normalise(title: str) -> str:
    title = re.sub(r"[\r\n]+", " ", title)
    title = re.sub(r"  +", " ", title)
    title = title.strip().rstrip(":")   # trailing colon from split titles
    if _is_allcaps(title):
        title = _to_titlecase(title)
    return title


def _strip_leading_artifacts(text: str) -> str:
    """Remove leading page numbers / stray digits / blank lines."""
    text = re.sub(r"^\s*\d{1,4}\s*\n", "", text)
    text = text.lstrip("\n\r \t")
    return text


# ── Extraction ─────────────────────────────────────────────────────────────────

def _extract_from_text(text: str) -> str | None:
    """Try to pull the paper title from page-1 text. Returns None on failure."""
    text = _strip_leading_artifacts(text)
    if not text:
        return None

    first_block = text[:400]  # only look at the top of the page

    # Non-title page starts → fall back to filename immediately
    if PAGE_NONTITLE_START_RE.match(text):
        return None

    # First substantial line is an author/institution line → fall back
    first_nonblank = next((l.strip() for l in text.splitlines() if l.strip()), "")
    if LINE_STOP_RE.search(first_nonblank):
        return None
    if NAME_LIST_RE.match(first_nonblank) and len(first_nonblank) < 60:
        return None

    # Pattern: "Article Title:\nActual Title\nAuthors:\n…"
    m = re.search(r"article\s+title\s*:\s*", first_block, re.IGNORECASE)
    if m:
        after = text[m.end():]
        lines = after.splitlines()
        title_lines: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r"^authors?\s*:", line, re.IGNORECASE):
                break
            if LINE_STOP_RE.search(line):
                break
            title_lines.append(line)
            if len(title_lines) >= MAX_TITLE_LINES:
                break
        if title_lines:
            return _normalise(" ".join(title_lines))

    # General extraction
    lines = text.splitlines()
    title_lines = []
    seen_blank = False

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()

        if not line:
            if title_lines:
                seen_blank = True
            continue

        # Handle hyphen line-break FIRST — always join regardless of other patterns.
        # Keep the hyphen: "DECISION-" + "MAKING" → "DECISION-MAKING" (compound word).
        if title_lines and title_lines[-1].endswith("-"):
            title_lines[-1] = title_lines[-1] + line
            continue

        # Hard per-line stops
        if LINE_STOP_RE.search(line):
            break

        # Inline author superscripts: "Thi(1), Dinh(2)"
        if title_lines and INLINE_SUPERSCRIPT_RE.search(line):
            break

        # ALL-CAPS short line after initial title lines → author name
        if title_lines and ALLCAPS_NAME_RE.match(line):
            break

        # Author with affiliation number: "Zeyang Wu 1", "Wang 1*"
        if title_lines and SUPERSCRIPT_NUM_RE.search(line) and len(line) < 60:
            break

        # Digit-prefixed author lines: "1Ye, Han; 1Lijun, Zhang*"
        if title_lines and DIGIT_AUTHOR_RE.search(line):
            break

        # Comma-separated name list — require "and"/"," so plain 2-word phrases
        # like "Software Applications" aren't mistaken for author names
        if title_lines and NAME_LIST_RE.match(line) and ("and " in line or "," in line):
            break

        # Trailing comma on author name: "Er.Akshay Bhardwaj,"
        if title_lines and line.endswith(",") and len(line) < 50:
            break

        # After a blank line, a short line (< 55 chars) that's not a clear
        # continuation of a long title is likely an author line
        if seen_blank and len(line) < 55 and not line.endswith(":"):
            break

        title_lines.append(line)

        if len(title_lines) >= MAX_TITLE_LINES:
            break

    if not title_lines:
        return None

    candidate = _normalise(" ".join(title_lines))

    # Sanity checks
    if len(candidate) < 8:
        return None
    if re.match(r"^\d+$", candidate):
        return None
    # If it still looks like an author/institution line, reject
    if LINE_STOP_RE.search(candidate[:80]):
        return None

    return candidate


def _title_from_stem(stem: str) -> str:
    """Derive a readable title from the JSON filename stem.

    Preserves meaningful hyphens (e.g. 'Tri-axial', 'Self-Attention').
    Replaces underscores with spaces only.
    """
    t = stem.replace("_", " ")
    t = re.sub(r"  +", " ", t).strip()
    if _is_allcaps(t):
        t = _to_titlecase(t)
    return t


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Patch title_empty JSONs")
    parser.add_argument("--report", default="/tmp/json_metadata_audit.json")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    args = parser.parse_args()

    with open(args.report) as f:
        audit = json.load(f)

    title_empty_files = [
        m["file"]
        for m in audit.get("mismatches", [])
        if "title_empty" in m.get("issues", [])
    ]

    if not title_empty_files:
        print("No title_empty entries found in report.")
        sys.exit(0)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{'=' * 60}")
    print(f"Patch empty titles — {mode}")
    print(f"{'=' * 60}")
    print(f"Files to patch: {len(title_empty_files)}")
    print()

    patched = 0
    fallback = 0
    failed = 0
    seen_paths: set[str] = set()

    for fp in title_empty_files:
        if fp in seen_paths:
            continue
        seen_paths.add(fp)

        path = Path(fp)
        try:
            with open(path, "rb") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ❌  PARSE ERROR  {path.name}: {e}")
            failed += 1
            continue

        pages = data.get("pages", [])
        page1_text = pages[0].get("text", "") if pages else ""
        stem = path.stem

        extracted = _extract_from_text(page1_text) if page1_text else None
        if extracted:
            new_title = extracted
            source = "extracted"
            patched += 1
        else:
            new_title = _title_from_stem(stem)
            source = "filename"
            fallback += 1

        marker = "✅" if source == "extracted" else "📄"
        print(f"  {marker} [{source:<9}] {path.name[:55]}")
        print(f"             → {new_title}")
        print()

        if args.apply:
            data["metadata"]["title"] = new_title
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"  Extracted from content : {patched}")
    print(f"  Derived from filename  : {fallback}")
    print(f"  Parse errors (skipped) : {failed}")
    print("=" * 60)
    if not args.apply:
        print("\n  Dry-run complete. Pass --apply to commit changes.")


if __name__ == "__main__":
    main()
