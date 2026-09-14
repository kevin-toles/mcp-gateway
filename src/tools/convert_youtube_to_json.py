"""convert_youtube_to_json tool handler.

Mirrors convert_pdf_to_json — health-checks COS, launches Terminal.app window,
and returns immediately. Discovers YouTube transcript JSONs in input_dir and
converts each one to raw JSON compatible with the COS enrich-book pipeline.

COS dependency: SBERT similarity matrix for topic-shift detection on videos
without YouTube chapter timestamps (1,643 of 4,543 videos).

Monitor live progress with:
    tail -f /tmp/youtube_conversion_progress.log
"""

import os
import stat
import subprocess
import tempfile
from datetime import datetime

import httpx

from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "convert_youtube_to_json"
PROGRESS_LOG = "/tmp/youtube_conversion_progress.log"  # noqa: S108
_STANDALONE_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "batch_convert_youtube_standalone.py")


def _launch_terminal(
    input_dir: str,
    out_dir: str,
    skip_existing: bool,
    co_url: str,
    workers: int = 6,
    min_paragraphs: int = 3,
) -> dict:
    """Open a new Terminal.app window running the conversion script."""
    standalone = os.path.abspath(_STANDALONE_SCRIPT)
    python = os.path.abspath(os.path.join(os.path.dirname(standalone), "..", ".venv", "bin", "python"))
    if not os.path.exists(python):
        python = "python3"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    log_file = f"/tmp/batch_youtube_convert_{timestamp}.log"  # noqa: S108

    skip_flag = "--skip-existing" if skip_existing else ""

    fd, tmp_script = tempfile.mkstemp(suffix=".sh", prefix="batch_youtube_run_")
    os.close(fd)
    with open(tmp_script, "w") as f:
        f.write(f"""#!/usr/bin/env bash
printf '\\n\\033[1;36m══ Batch YouTube Transcript Conversion ══  Log: {log_file}\\033[0m\\n\\n'
'{python}' -u '{standalone}' \\
    --input-dir '{input_dir}' \\
    --output-dir '{out_dir}' \\
    --co-url '{co_url}' \\
    --workers '{workers}' \\
    --min-paragraphs '{min_paragraphs}' \\
    {skip_flag} 2>&1 | tee '{log_file}'
EXIT_CODE=${{PIPESTATUS[0]}}
ln -sf '{log_file}' '{PROGRESS_LOG}'
echo ''
if [[ $EXIT_CODE -eq 0 ]]; then
  printf '\\033[1;32m✅  Conversion complete.\\033[0m\\n'
else
  printf '\\033[1;33m⚠️   Finished with some failures. Check log above.\\033[0m\\n'
fi
echo ''
read -rp 'Press Enter to close...'
""")
    os.chmod(tmp_script, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)  # noqa: S103

    applescript = f"""
tell application "Terminal"
    do script "{tmp_script}"
    set bounds of front window to {{80, 80, 1300, 900}}
    activate
end tell
"""
    subprocess.Popen(["osascript", "-e", applescript])  # noqa: S603, S607

    return {
        "status": "started",
        "message": f"YouTube transcript conversion launched in Terminal.app. Monitor: tail -f {PROGRESS_LOG}",
        "log": log_file,
        "monitor_command": f"tail -f {PROGRESS_LOG}",
        "input_dir": input_dir,
        "output_dir": out_dir,
    }


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler — mirrors convert_pdf_to_json interface."""

    async def convert_youtube_to_json(
        input_path: str,
        output_path: str | None = None,
        skip_existing: bool = True,
        workers: int = 6,
        min_paragraphs: int = 3,
    ) -> dict:
        """Convert YouTube transcript JSONs to pipeline-compatible raw JSONs.

        Always launches in a new Terminal.app window and returns immediately.
        Monitor live progress with: tail -f /tmp/youtube_conversion_progress.log

        Videos WITH YouTube chapter timestamps use those directly.
        Videos WITHOUT chapters use SBERT topic-shift detection via COS
        /api/v1/similarity/matrix (same PGC algorithm as SpanDetector).

        Output JSONs match PDFConversionResult shape — feed directly into
        batch_enrich_metadata for enrichment + HTC classification.

        Args:
            input_path: Path to a YouTube JSON file or directory of them.
            output_path: Output directory. Auto-generated if omitted.
            skip_existing: Skip videos that already have a converted JSON.
            workers: Concurrent conversions (default: 6). Use 1 for sequential.
            min_paragraphs: Minimum paragraphs per detected chapter (default: 3).
        """
        settings = getattr(dispatcher, "_settings", None)
        if settings is None:
            from src.core.config import Settings
            settings = Settings()
        co_url = settings.CODE_ORCHESTRATOR_URL

        try:
            async with httpx.AsyncClient(timeout=3.0) as _c:
                _health = await _c.get(f"{co_url}/health")
        except Exception:
            _health = None
        if _health is None or _health.status_code != 200:
            if not await dispatcher._try_auto_start("code-orchestrator", co_url):
                return {"status": "error", "message": f"code-orchestrator did not become healthy at {co_url}"}

        if os.path.isdir(input_path):
            input_dir = input_path
            out_dir = output_path or os.path.join(os.path.dirname(input_dir.rstrip("/")), "youtube-converted")
        else:
            input_dir = os.path.dirname(os.path.abspath(input_path))
            if output_path:
                out_dir = os.path.dirname(os.path.abspath(output_path))
            else:
                out_dir = os.path.join(os.path.dirname(input_dir.rstrip("/")), "youtube-converted")

        os.makedirs(out_dir, exist_ok=True)

        return _launch_terminal(
            input_dir=input_dir,
            out_dir=out_dir,
            skip_existing=skip_existing,
            co_url=co_url,
            workers=workers,
            min_paragraphs=min_paragraphs,
        )

    return convert_youtube_to_json
