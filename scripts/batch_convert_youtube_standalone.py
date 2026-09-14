#!/usr/bin/env python3
"""Standalone batch YouTube transcript conversion script.

Launched by the convert_youtube_to_json MCP tool. Discovers all YouTube
transcript JSON files in input_dir, reassembles subtitle chunks into
chapter-scoped prose, and writes raw JSONs compatible with the
COS enrich-book pipeline.

Videos WITH YouTube chapter timestamps use those directly (no COS needed).
Videos WITHOUT chapters use SBERT cosine-similarity topic-shift detection
via COS /api/v1/similarity/matrix (PGC w=2, o=1, threshold=mean-0.3*std,
min 3-paragraph merge floor).

Usage:
    python scripts/batch_convert_youtube_standalone.py \\
        --input-dir '/Volumes/TDS-Primary-SSD/Platform Data Sets [RAW]/Youtube' \\
        --output-dir /path/to/output \\
        [--skip-existing] \\
        [--workers 6] \\
        [--min-paragraphs 3] \\
        [--co-url http://localhost:8083]

Monitor live from any terminal:
    tail -f /tmp/youtube_conversion_progress.log
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import httpx

PROGRESS_LOG = "/tmp/youtube_conversion_progress.log"  # noqa: S108
CO_HEALTH_TIMEOUT = 5.0
CO_SIMILARITY_TIMEOUT = 60.0  # SBERT matrix for ~100 paragraph groups

_LOG_LOCK = threading.Lock()

# PGC parameters — match SpanDetector
PGC_WINDOW_SIZE = 2
PGC_OVERLAP = 1
PGC_THRESHOLD_FACTOR = 0.3
MIN_PARAGRAPH_MERGE = 3
SENTENCES_PER_PARAGRAPH = 3


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
    """Print to stdout AND append to progress log (thread-safe)."""
    ts = datetime.now(UTC).strftime("%H:%M:%S")
    with _LOG_LOCK:
        print(msg, flush=True)
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


# ── Transcript reassembly ───────────────────────────────────────────────────

def _reassemble_chunks(chunks: list[dict]) -> str:
    """Join subtitle chunks into flowing prose."""
    raw = " ".join(c.get("text", "") for c in chunks)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _chunks_in_range(chunks: list[dict], start_time: float, end_time: float) -> list[dict]:
    """Filter chunks whose start falls within [start_time, end_time)."""
    return [c for c in chunks if start_time <= c.get("start", 0) < end_time]


def _text_to_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuation boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _sentences_to_paragraphs(sentences: list[str], per_para: int = SENTENCES_PER_PARAGRAPH) -> list[str]:
    """Group sentences into paragraphs of `per_para` sentences each."""
    paragraphs = []
    for i in range(0, len(sentences), per_para):
        para = " ".join(sentences[i:i + per_para])
        if para:
            paragraphs.append(para)
    return paragraphs


# ── Topic-shift detection (SBERT via COS, PGC algorithm) ───────────────────

def _sbert_similarity_matrix(client: httpx.Client, co_url: str, texts: list[str]) -> list[list[float]]:
    """Call COS /api/v1/similarity/matrix to get SBERT pairwise cosine similarity."""
    resp = client.post(
        f"{co_url}/api/v1/similarity/matrix",
        json={"texts": texts},
        timeout=CO_SIMILARITY_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["similarity_matrix"]


def _detect_topic_shifts(
    client: httpx.Client,
    co_url: str,
    paragraphs: list[str],
    min_merge: int = MIN_PARAGRAPH_MERGE,
) -> list[tuple[int, int]]:
    """Detect topic-shift boundaries using SBERT cosine similarity via COS.

    Uses PGC (Paragraph Group Chunking) with window_size=2, overlap=1.
    Splits at similarity valleys (threshold = mean - 0.3*std).
    Returns list of (start_idx, end_idx) paragraph ranges for each detected chapter.
    """
    if len(paragraphs) < 2:
        return [(0, len(paragraphs))]

    # Build paragraph groups (window_size=2, overlap=1)
    groups = []
    step = PGC_WINDOW_SIZE - PGC_OVERLAP
    for i in range(0, len(paragraphs), step):
        end = min(i + PGC_WINDOW_SIZE, len(paragraphs))
        group_text = " ".join(paragraphs[i:end])
        groups.append(group_text)
        if end >= len(paragraphs):
            break

    if len(groups) < 2:
        return [(0, len(paragraphs))]

    # SBERT similarity matrix via COS
    matrix = _sbert_similarity_matrix(client, co_url, groups)

    # Extract adjacent-pair similarities from the matrix
    similarities = [float(matrix[i][i + 1]) for i in range(len(groups) - 1)]

    if not similarities:
        return [(0, len(paragraphs))]

    # Find valleys: threshold = mean - 0.3 * std
    mean_sim = sum(similarities) / len(similarities)
    variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
    std_sim = math.sqrt(variance) if variance > 0 else 0.0
    threshold = mean_sim - PGC_THRESHOLD_FACTOR * std_sim

    # Identify split points (group indices where similarity drops below threshold)
    split_group_indices = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

    # Convert group indices to paragraph indices
    step = PGC_WINDOW_SIZE - PGC_OVERLAP
    split_para_indices = [idx * step for idx in split_group_indices]
    split_para_indices = [s for s in split_para_indices if s < len(paragraphs)]

    # Build segments
    boundaries = [0] + sorted(set(split_para_indices)) + [len(paragraphs)]
    segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # Merge segments below minimum paragraph count
    merged = []
    for seg in segments:
        seg_len = seg[1] - seg[0]
        if merged and seg_len < min_merge:
            prev = merged[-1]
            merged[-1] = (prev[0], seg[1])
        else:
            merged.append(seg)

    # Final pass: if last segment is too short, merge into previous
    if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_merge:
        prev = merged[-2]
        merged[-2] = (prev[0], merged[-1][1])
        merged.pop()

    return merged


# ── Conversion logic ────────────────────────────────────────────────────────

def _convert_youtube_json(
    input_path: str,
    output_path: str,
    co_url: str,
    client: httpx.Client,
    min_paragraphs: int = MIN_PARAGRAPH_MERGE,
) -> dict:
    """Convert a single YouTube transcript JSON to the COS raw JSON format.

    Returns a stats dict with chapter_count, chunk_count, method.
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if not chunks:
        raise ValueError("No transcript chunks found")

    yt_chapters = data.get("chapters", [])
    video_title = data.get("title", Path(input_path).stem)
    channel = data.get("channel", "")
    duration = data.get("duration", 0)

    chapters_out = []

    if yt_chapters:
        # ── Strategy A: YouTube chapter timestamps (no COS needed) ──────
        method = "youtube_chapters"
        for i, ch in enumerate(yt_chapters):
            start_time = ch.get("start_time", 0)
            end_time = ch.get("end_time", duration or float("inf"))
            chapter_chunks = _chunks_in_range(chunks, start_time, end_time)
            content = _reassemble_chunks(chapter_chunks)

            chapters_out.append({
                "title": ch.get("title", f"Chapter {i + 1}"),
                "chapter_id": i + 1,
                "start_page": i + 1,
                "end_page": i + 1,
                "content": content,
                "page_count": 1,
                "start_time": start_time,
                "end_time": end_time,
            })
    else:
        # ── Strategy B: SBERT topic-shift detection via COS ─────────────
        method = "topic_shift_sbert"
        full_text = _reassemble_chunks(chunks)
        sentences = _text_to_sentences(full_text)
        paragraphs = _sentences_to_paragraphs(sentences)

        if not paragraphs:
            paragraphs = [full_text]

        segments = _detect_topic_shifts(client, co_url, paragraphs, min_merge=min_paragraphs)

        for i, (start_idx, end_idx) in enumerate(segments):
            segment_paras = paragraphs[start_idx:end_idx]
            content = "\n\n".join(segment_paras)

            first_sentence = _text_to_sentences(segment_paras[0])[:1]
            title = (first_sentence[0][:80] + "...") if first_sentence and len(first_sentence[0]) > 80 else (first_sentence[0] if first_sentence else f"Segment {i + 1}")

            chapters_out.append({
                "title": title,
                "chapter_id": i + 1,
                "start_page": i + 1,
                "end_page": i + 1,
                "content": content,
                "page_count": 1,
            })

    # Synthetic pages — one page per chapter, 1:1 mapping.
    # ChapterDetector sees predefined chapters and uses them directly.
    # BookProcessor builds page_map from pages and assembles text via start_page/end_page.
    pages_out = []
    for ch in chapters_out:
        pages_out.append({
            "page_number": ch["start_page"],
            "text": ch["content"],
            "content": ch["content"],
            "extraction_method": method,
        })

    # Build output matching PDFConversionResult.to_dict() shape
    output = {
        "metadata": {
            "title": video_title,
            "author": channel,
            "subject": "",
            "creator": "youtube-transcript-converter",
            "total_pages": len(chapters_out),
            "source_file": Path(input_path).name,
            "source_type": "youtube_transcript",
            "video_id": data.get("video_id", ""),
            "channel": channel,
            "upload_date": data.get("upload_date", ""),
            "duration": duration,
            "duration_string": data.get("duration_string", ""),
            "view_count": data.get("view_count", 0),
            "like_count": data.get("like_count", 0),
            "categories": data.get("categories", []),
            "tags": data.get("tags", []),
            "language": data.get("language", ""),
            "transcript_source": data.get("transcript_source", ""),
            "chapter_detection_method": method,
        },
        "pages": pages_out,
        "chapters": chapters_out,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return {
        "chapter_count": len(chapters_out),
        "chunk_count": len(chunks),
        "method": method,
    }


def _convert_one(
    idx: int,
    total: int,
    input_path: str,
    out_path: str,
    co_url: str,
    min_paragraphs: int,
) -> tuple[str, str | None]:
    """Convert a single video. Returns (video_name, error_msg | None).

    Each call creates its own httpx.Client so workers don't share connection state.
    """
    video_name = Path(input_path).stem
    start = time.time()
    _log(f"[{idx:4d}/{total}] 🎬 {video_name}")

    try:
        with httpx.Client() as client:
            stats = _convert_youtube_json(input_path, out_path, co_url, client, min_paragraphs)
        elapsed = time.time() - start
        method_tag = "📌 YT chapters" if stats["method"] == "youtube_chapters" else "🔍 SBERT topic-shift"
        _log(green(
            f"[{idx:4d}/{total}] ✅  {video_name} — "
            f"{stats['chapter_count']} chapters ({method_tag}) — {elapsed:.1f}s"
        ))
        return video_name, None
    except Exception as e:
        elapsed = time.time() - start
        _log(red(f"[{idx:4d}/{total}] ❌  {video_name} ({elapsed:.1f}s): {str(e)[:200]}"))
        return video_name, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch YouTube transcript conversion")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--mirror-structure", action="store_true", default=False,
                        help="Mirror input directory structure under output dir (auto-enabled when subdirs detected)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Concurrent conversions (default: 6). Use 1 for sequential.")
    parser.add_argument("--min-paragraphs", type=int, default=MIN_PARAGRAPH_MERGE,
                        help="Minimum paragraphs per detected chapter segment (default: 3)")
    parser.add_argument("--co-url", default=os.environ.get("CODE_ORCHESTRATOR_URL", "http://localhost:8083"))
    args = parser.parse_args()

    # Resolve output dir
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.input_dir.rstrip("/")), "youtube-converted")
    os.makedirs(out_dir, exist_ok=True)

    # Discover YouTube transcript JSONs
    input_path = Path(args.input_dir)
    all_files = sorted(str(p) for p in input_path.rglob("*.json"))

    if not all_files:
        _log(yellow(f"⚠️  No JSON files found under: {args.input_dir}"))
        sys.exit(0)

    # Determine whether to mirror directory structure
    has_subdirs = any(Path(f).parent != input_path for f in all_files)
    mirror = args.mirror_structure or has_subdirs

    # Header
    _log("=" * 60)
    _log(cyan("🎬 Batch YouTube Transcript Conversion"))
    _log(f"   Input:   {args.input_dir}")
    _log(f"   Output:  {out_dir}")
    _log(f"   COS:     {args.co_url}")
    _log(f"   Skip existing: {args.skip_existing}")
    _log(f"   Mirror structure: {mirror}")
    _log(f"   Workers: {args.workers}")
    _log(f"   Min paragraphs: {args.min_paragraphs}")
    _log("=" * 60)

    # Health check — fail fast if COS unreachable (needed for SBERT topic-shift detection)
    _health_check(args.co_url)

    # Build existing-JSON set once via scandir
    if args.skip_existing:
        _log("   Scanning output dir for existing JSONs...")
        existing_jsons: set[str] = set()
        try:
            if mirror:
                for dirpath, _dirs, filenames in os.walk(out_dir):
                    for fn in filenames:
                        if fn.endswith(".json"):
                            existing_jsons.add(os.path.join(dirpath, fn))
            else:
                with os.scandir(out_dir) as it:
                    for entry in it:
                        if entry.name.endswith(".json"):
                            existing_jsons.add(entry.path)
        except FileNotFoundError:
            pass
        _log(f"   Found {len(existing_jsons):,} existing JSONs")
    else:
        existing_jsons = set()

    to_process = []
    skipped = []
    for fpath in all_files:
        fpath_obj = Path(fpath)
        if mirror:
            rel = fpath_obj.relative_to(input_path)
            out_path = os.path.join(out_dir, str(rel))
        else:
            out_path = os.path.join(out_dir, fpath_obj.name)

        if args.skip_existing and out_path in existing_jsons:
            skipped.append(fpath_obj.name)
        else:
            to_process.append((fpath, out_path))

    total = len(to_process)
    _log(f"   Videos to convert: {bold(str(total))}  |  Skipped (existing): {len(skipped)}")
    _log("=" * 60)

    if total == 0:
        _log(green("✅  All videos already converted. Nothing to do."))
        sys.exit(0)

    # Ensure output subdirectories exist when mirroring
    if mirror:
        for _, out_path in to_process:
            out_parent = os.path.dirname(out_path)
            if out_parent and not os.path.exists(out_parent):
                os.makedirs(out_parent, exist_ok=True)

    succeeded = 0
    failed = 0
    failed_names: list[str] = []
    batch_start = time.time()

    if args.workers == 1 or total == 1:
        # ── Sequential path ─────────────────────────────────────────────
        for idx, (input_file, out_path) in enumerate(to_process, 1):
            video_name, error = _convert_one(
                idx, total, input_file, out_path, args.co_url, args.min_paragraphs,
            )
            if error is None:
                succeeded += 1
            else:
                failed += 1
                failed_names.append(video_name)
    else:
        # ── Parallel path ───────────────────────────────────────────────
        _log(cyan(f"   Running {args.workers} workers in parallel"))
        _log("")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _convert_one, idx, total, input_file, out_path,
                    args.co_url, args.min_paragraphs,
                ): input_file
                for idx, (input_file, out_path) in enumerate(to_process, 1)
            }
            try:
                for future in as_completed(futures):
                    try:
                        video_name, error = future.result()
                        if error is None:
                            succeeded += 1
                        else:
                            failed += 1
                            failed_names.append(video_name)
                    except Exception as e:
                        failed += 1
                        failed_names.append(str(futures[future]))
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                _log(yellow("\n⚠️  Interrupted — partial results saved"))
                raise

    # ── Summary ──────────────────────────────────────────────────────────────
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
        _log(yellow("   Failed videos:"))
        for name in failed_names:
            _log(yellow(f"     • {name}"))
    _log("=" * 60)

    sys.exit(1 if failed and succeeded == 0 else 0)


if __name__ == "__main__":
    main()
