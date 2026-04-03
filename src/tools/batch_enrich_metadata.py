"""batch_enrich_metadata — thin passthrough that opens a Terminal window and returns.

The tool:
  1. Writes a Python runner script to a tmpfile
  2. Opens it in a new Terminal.app window via osascript
  3. Returns immediately — Terminal IS the live status display

Monitor: tail -f /tmp/batch_enrich_latest.log
"""

import glob
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fastmcp import Context

from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

CO_ENRICH_URL = "http://localhost:8083/api/v1/workflows/enrich-book"
LATEST_LINK = "/tmp/batch_enrich_latest.log"  # noqa: S108


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):

    async def batch_enrich_metadata(
        metadata_dir: str = "/Users/kevintoles/POC/ai-platform-data/collections/software-engineering/metadata",
        output_dir: str = "/Users/kevintoles/POC/ai-platform-data/collections/software-engineering/enriched",
        taxonomy_path: str | None = None,
        mode: str = "auto",
        vtf_path: str | None = None,
        seed_concepts: list[str] | None = None,
        classifier_enabled: bool = True,
        graphcodebert_enabled: bool = True,
        raw_content_dir: str | None = None,
        resume: bool = True,
        limit: int = 0,
        book: str = "",
        ctx: Context | None = None,
    ) -> dict:
        """Batch-enrich *_metadata.json files to DMA-§2.3 format.

        Opens a fresh Terminal.app window that runs the enrichment with live
        per-book status output (mirrors seeding_v2/seed.sh pattern).
        Returns immediately — the Terminal window IS the progress display.

        Args:
            metadata_dir: Directory containing *_metadata.json files.
            output_dir: Where to write *_enriched.json output files.
            taxonomy_path: Taxonomy JSON path (auto-discovers latest uber_taxonomy_v*.json if omitted).
            mode: Enrichment mode — "software" (default KB), "foundation" (scientific collections),
                  or "auto" (detect from metadata_dir path: ext_metadata → foundation, else software).
                  "foundation" automatically sets vtf_path, seed_concepts, classifier_enabled=False,
                  graphcodebert_enabled=False unless the caller explicitly overrides them.
            vtf_path: Validated term filter JSON path. Explicit override; mode default applies if omitted.
            seed_concepts: SBERT anchor concepts list. Explicit override; mode default applies if omitted.
            classifier_enabled: Enable HTC LightGBM classifier. Explicit override; mode default applies if omitted.
            graphcodebert_enabled: Enable GraphCodeBERT validation. Explicit override; mode default applies if omitted.
            resume: Skip books that already have an enriched output file (default: True).
            limit: Cap at N books — 0 means all (default: 0).
            book: Only process books whose filename contains this string (default: all).

        Monitor from any terminal:
            tail -f /tmp/batch_enrich_latest.log
        """
        metadata_dir = str(Path(metadata_dir).expanduser().resolve())
        output_dir = str(Path(output_dir).expanduser().resolve())

        if not Path(metadata_dir).exists():
            return {"status": "error", "message": f"metadata_dir not found: {metadata_dir}"}

        # Resolve mode: "auto" detects from path, otherwise use explicit value.
        _esc_root = "/Users/kevintoles/POC/ai-platform-data"
        _esc_vtf = f"{_esc_root}/vtfs/foundations/ESC_validated_term_filter.json"
        _foundation_seed: list[str] = []
        try:
            import importlib.util as _ilu

            _spec = _ilu.spec_from_file_location("_fsc", f"{_esc_root}/data/foundation_seed_concepts.py")
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
                _foundation_seed = list(getattr(_mod, "SEED_SCIENTIFIC_CONCEPTS", []))
        except Exception:  # noqa: S110
            pass

        _resolved_mode = mode
        if _resolved_mode == "auto":
            _resolved_mode = "foundation" if "ext_metadata" in metadata_dir else "software"

        if _resolved_mode == "foundation":
            # Apply foundation defaults for any param not explicitly overridden by caller.
            # Caller-supplied non-None values take precedence.
            if vtf_path is None:
                vtf_path = _esc_vtf
            if seed_concepts is None:
                seed_concepts = _foundation_seed or None
            # classifier_enabled and graphcodebert_enabled default True in the signature;
            # only override them to False if the caller left them at their defaults (True).
            # There is no way to distinguish "caller passed True" from "default True" in Python,
            # so foundation mode always forces these off — callers who need them on must use
            # mode="software" or mode="auto" with a non-ext_metadata path.
            classifier_enabled = False
            graphcodebert_enabled = False
            # ADR-011 §5: derive raw_content_dir from metadata_dir if caller didn't supply it.
            # Callers own layout knowledge — CO no longer guesses this.
            if raw_content_dir is None:
                raw_content_dir = metadata_dir.replace("ext_metadata", "ext_raw")

        # Resolve taxonomy path
        if not taxonomy_path:
            candidates = sorted(Path(metadata_dir).parent.glob("taxonomies/se/uber_taxonomy_v*.json"))
            taxonomy_path = str(candidates[-1]) if candidates else ""

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        log_file = f"/tmp/batch_enrich_{timestamp}.log"  # noqa: S108

        # Build resume/limit/book filter flags as shell variables
        resume_flag = "true" if resume else "false"
        limit_val = str(limit)
        book_filter = book.lower()
        taxonomy_arg = taxonomy_path or ""
        vtf_arg = vtf_path or ""
        seed_concepts_json = __import__("json").dumps(seed_concepts) if seed_concepts else ""
        classifier_flag = "true" if classifier_enabled else "false"
        graphcodebert_flag = "true" if graphcodebert_enabled else "false"
        raw_content_dir_arg = raw_content_dir or ""

        # Write the temp runner script (same pattern as seed.sh inner script)
        tmpscript_fd, tmpscript_path = tempfile.mkstemp(suffix=".sh", prefix="batch_enrich_run_")
        os.close(tmpscript_fd)

        script_body = f"""#!/usr/bin/env bash
set -euo pipefail
METADATA_DIR='{metadata_dir}'
OUTPUT_DIR='{output_dir}'
TAXONOMY_PATH='{taxonomy_arg}'
VTF_PATH='{vtf_arg}'
SEED_CONCEPTS_JSON='{seed_concepts_json}'
CLASSIFIER_ENABLED={classifier_flag}
GRAPHCODEBERT_ENABLED={graphcodebert_flag}
RAW_CONTENT_DIR='{raw_content_dir_arg}'
RESUME={resume_flag}
LIMIT={limit_val}
BOOK_FILTER='{book_filter}'
LOG_FILE='{log_file}'
LATEST_LINK='{LATEST_LINK}'
CO_URL='{CO_ENRICH_URL}'

mkdir -p "$OUTPUT_DIR"

printf '\\n\\033[1;36m══ Batch Enrich ══  Log: '"$LOG_FILE"'\\033[0m\\n\\n'

# Collect files
ALL_FILES=()
while IFS= read -r _f; do ALL_FILES+=("$_f"); done < <(ls "$METADATA_DIR"/*_metadata.json 2>/dev/null | sort)
TOTAL=${{#ALL_FILES[@]}}
if [[ $TOTAL -eq 0 ]]; then
  echo "ERROR: No *_metadata.json files in $METADATA_DIR"
  exit 1
fi

# Apply book filter
if [[ -n "$BOOK_FILTER" ]]; then
  FILTERED=()
  for f in "${{ALL_FILES[@]}}"; do
    fname=$(basename "$f")
    fname_lower=$(echo "$fname" | tr '[:upper:]' '[:lower:]')
    if [[ "$fname_lower" == *"$BOOK_FILTER"* ]]; then
      FILTERED+=("$f")
    fi
  done
  ALL_FILES=("${{FILTERED[@]}}")
fi

# Apply resume filter
if [[ "$RESUME" == "true" ]]; then
  TO_PROCESS=()
  SKIPPED=0
  for f in "${{ALL_FILES[@]}}"; do
    stem=$(basename "$f" _metadata.json)
    if [[ -f "$OUTPUT_DIR/${{stem}}_enriched.json" ]]; then
      (( SKIPPED++ )) || true
    else
      TO_PROCESS+=("$f")
    fi
  done
  echo "Resume: skipping $SKIPPED already-enriched books"
else
  TO_PROCESS=("${{ALL_FILES[@]}}")
  SKIPPED=0
fi

# Apply limit
if [[ "$LIMIT" -gt 0 ]]; then
  TO_PROCESS=("${{TO_PROCESS[@]:0:$LIMIT}}")
fi

TOTAL=${{#TO_PROCESS[@]}}
if [[ $TOTAL -eq 0 ]]; then
  echo "Nothing to enrich (all books already enriched or limit=0)."
  exit 0
fi

echo "Starting enrichment: $TOTAL books  ($SKIPPED skipped)"
echo "Output: $OUTPUT_DIR"
[[ -n "$TAXONOMY_PATH" ]] && echo "Taxonomy: $TAXONOMY_PATH"
echo ""

SUCCEEDED=0
FAILED=0
T_START=$SECONDS

for i in "${{!TO_PROCESS[@]}}"; do
  IDX=$(( i + 1 ))
  FPATH="${{TO_PROCESS[$i]}}"
  STEM=$(basename "$FPATH" _metadata.json)
  OUT_PATH="$OUTPUT_DIR/${{STEM}}_enriched.json"

  T_BOOK=$SECONDS
  printf "\\033[0;33m[%d/%d]\\033[0m %s ... " "$IDX" "$TOTAL" "$STEM"

  # Build JSON payload
  PAYLOAD=$(python3 -c "
import json, sys
p = {{'input_path': sys.argv[1], 'output_path': sys.argv[2]}}
if sys.argv[3]: p['taxonomy_path'] = sys.argv[3]
if sys.argv[4]: p['vtf_path'] = sys.argv[4]
if sys.argv[5]: p['seed_concepts'] = json.loads(sys.argv[5])
p['classifier_enabled'] = sys.argv[6] == 'true'
p['graphcodebert_enabled'] = sys.argv[7] == 'true'
if sys.argv[8]: p['raw_content_dir'] = sys.argv[8]
print(json.dumps(p))
" "$FPATH" "$OUT_PATH" "$TAXONOMY_PATH" "$VTF_PATH" \\
  "$SEED_CONCEPTS_JSON" "$CLASSIFIER_ENABLED" "$GRAPHCODEBERT_ENABLED" "$RAW_CONTENT_DIR")

  HTTP_STATUS=$(curl -sf -o /tmp/_enrich_resp.json -w "%{{http_code}}" \\
    -X POST "$CO_URL" \\
    -H 'Content-Type: application/json' \\
    -d "$PAYLOAD" 2>/dev/null || echo "000")

  ELAPSED=$(( SECONDS - T_BOOK ))
  ELAPSED_TOTAL=$(( SECONDS - T_START ))
  RATE=$(python3 -c "r=$IDX/max(${{ELAPSED_TOTAL:-0}},1); print(f'{{r:.2f}}')")
  ETA=$(python3 -c "
r=$IDX/max($ELAPSED_TOTAL,1)
rem=$TOTAL-$IDX
eta=int(rem/r) if r>0 else 0
print(f'{{eta//60}}m{{eta%60:02d}}s')
")

  if [[ "$HTTP_STATUS" == "200" ]]; then
    (( SUCCEEDED++ )) || true
    printf "\\033[0;32m✓\\033[0m  (%ds) ETA ~%s\\n" "$ELAPSED" "$ETA"
  else
    (( FAILED++ )) || true
    ERR=$(python3 -c "import json; d=json.load(open('/tmp/_enrich_resp.json')); \\
  print(d.get('detail','?'))" 2>/dev/null || echo "HTTP $HTTP_STATUS")
    printf "\\033[0;31m✗\\033[0m  (%ds) ERROR: %s\\n" "$ELAPSED" "$ERR"
  fi
done

ELAPSED_TOTAL=$(( SECONDS - T_START ))
MINS=$(( ELAPSED_TOTAL / 60 ))
SECS=$(( ELAPSED_TOTAL % 60 ))

echo ""
if [[ $FAILED -eq 0 ]]; then
  printf "\\033[1;32m✅  Done — %d/%d succeeded in %dm%02ds\\033[0m\\n" "$SUCCEEDED" "$TOTAL" "$MINS" "$SECS"
else
  printf "\\033[1;33m⚠️   Done — %d succeeded, %d failed in %dm%02ds\\033[0m\\n" "$SUCCEEDED" "$FAILED" "$MINS" "$SECS"
fi
echo ""
ln -sf "$LOG_FILE" "$LATEST_LINK"
read -rp 'Press Enter to close...'
"""

        with open(tmpscript_path, "w") as f:
            f.write(script_body)
        os.chmod(tmpscript_path, 0o755)  # noqa: S103

        # Count books to process for the summary (quick estimate)
        all_files = sorted(glob.glob(os.path.join(metadata_dir, "*_metadata.json")))
        if book:
            all_files = [f for f in all_files if book.lower() in os.path.basename(f).lower()]
        if resume:
            all_files = [
                f
                for f in all_files
                if not Path(
                    os.path.join(output_dir, Path(f).stem.removesuffix("_metadata").strip() + "_enriched.json")
                ).exists()
            ]
        if limit > 0:
            all_files = all_files[:limit]
        total = len(all_files)

        # osascript — confirmed working one-liner form; opens new Terminal window
        applescript = f'tell application "Terminal" to do script "bash {tmpscript_path}"'

        try:
            result = subprocess.run(  # noqa: S603
                ["osascript", "-e", applescript],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=10,
            )
            terminal_opened = result.returncode == 0
            if not terminal_opened and ctx:
                await ctx.info(f"osascript error: {result.stderr.strip()} — run manually: bash {tmpscript_path}")
        except Exception as exc:
            terminal_opened = False
            if ctx:
                await ctx.info(f"Could not open Terminal window: {exc} — run manually: bash {tmpscript_path}")

        msg = (
            f"Batch enrichment launched in a new Terminal window\n"
            f"  Books  : {total}\n"
            f"  Input  : {metadata_dir}\n"
            f"  Output : {output_dir}\n"
            f"  Log    : {log_file}\n"
            f"\nMonitor from any terminal:\n"
            f"  tail -f {LATEST_LINK}"
        )
        if not terminal_opened:
            msg += f"\n\nRun manually:\n  bash {tmpscript_path} 2>&1 | tee {log_file}"

        if ctx:
            await ctx.info(msg)

        return {
            "status": "launched",
            "total": total,
            "input_dir": metadata_dir,
            "output_dir": output_dir,
            "log_file": log_file,
            "latest_log": LATEST_LINK,
            "script": tmpscript_path,
            "terminal_opened": terminal_opened,
        }

    return batch_enrich_metadata
