#!/usr/bin/env python3
"""Rename YouTube converted JSONs to Channel_Playlist_VideoTitle.json format
and move them into the collections/software-engineering/raw directory.

Updates metadata.source_file to match the new filename.

Usage:
    python scripts/rename_and_move_youtube.py \
        --input-dir '/Volumes/TDS-Primary-SSD/Platform Data Sets [RAW]/Youtube_Finalized' \
        --output-dir '/Volumes/TDS-Primary-SSD/Platform Data Sets [RAW]/collections/software-engineering/raw'
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def _sanitize(name: str, max_len: int = 120) -> str:
    """Sanitize a string for use in a filename."""
    name = name.replace("/", "-").replace("\\", "-")
    name = re.sub(r'[<>:"|?*]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.rstrip(".")
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def _build_filename(channel: str, playlist: str, title: str) -> str:
    parts = [_sanitize(channel)]
    if playlist and playlist.strip():
        parts.append(_sanitize(playlist))
    parts.append(_sanitize(title))
    return "_".join(parts) + ".json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename and move YouTube JSONs")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print renames without writing")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(str(p) for p in Path(input_dir).rglob("*.json"))
    total = len(all_files)
    print(f"Files to process: {total:,}")
    print(f"Destination: {output_dir}")
    print()

    moved = 0
    skipped = 0
    collisions = 0
    errors = 0
    seen_names: dict[str, str] = {}
    start = time.time()

    for i, fpath in enumerate(all_files, 1):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)

            meta = data.get("metadata", {})
            channel = meta.get("channel", "") or meta.get("author", "")
            title = meta.get("title", Path(fpath).stem)
            playlist = ""

            # Playlist isn't in converted metadata — derive from directory structure
            # Structure: Youtube_Finalized/Channel/Playlist/video.json (3 levels)
            # or: Youtube_Finalized/Channel/video.json (2 levels)
            rel = Path(fpath).relative_to(input_dir)
            parts = rel.parts
            if len(parts) >= 3:
                # Channel/Playlist/video.json
                playlist = parts[-2]
                # Don't use playlist if it's the same as channel folder
                if playlist == parts[0]:
                    playlist = ""

            new_name = _build_filename(channel, playlist, title)

            # Handle collisions by appending a counter
            if new_name in seen_names:
                collisions += 1
                base, ext = os.path.splitext(new_name)
                counter = 2
                while f"{base}_{counter}{ext}" in seen_names:
                    counter += 1
                new_name = f"{base}_{counter}{ext}"

            seen_names[new_name] = fpath
            dest = os.path.join(output_dir, new_name)

            if args.dry_run:
                print(f"  {Path(fpath).name}")
                print(f"    -> {new_name}")
                moved += 1
                continue

            # Update source_file in metadata
            meta["source_file"] = new_name
            data["metadata"] = meta

            with open(dest, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            moved += 1
            if i % 5000 == 0 or i == total:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  [{i:,}/{total:,}] {rate:.0f} files/s — {new_name[:80]}")

        except Exception as e:
            errors += 1
            print(f"  ERROR: {Path(fpath).name}: {e}", file=sys.stderr)

    elapsed = time.time() - start
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Moved:      {moved:,}")
    print(f"  Collisions: {collisions:,} (resolved with counter suffix)")
    print(f"  Errors:     {errors:,}")


if __name__ == "__main__":
    main()
