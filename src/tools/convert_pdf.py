"""convert_pdf tool handler — WBS-WF6.

1:1 mirror of batch_extract_metadata — launches Terminal.app window and returns
immediately. Discovers all PDFs in input_dir and converts each one.

Monitor live progress with:
    tail -f /tmp/conversion_progress.log
"""

import os
import stat
import subprocess
import tempfile
from datetime import datetime

import httpx

from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "convert_pdf"
PROGRESS_LOG = "/tmp/conversion_progress.log"  # noqa: S108
_STANDALONE_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "batch_convert_pdfs_standalone.py")


def _check_co_health(co_url: str = "http://localhost:8083") -> str | None:
    """Return None if healthy, or an error string if unreachable."""
    try:
        r = httpx.get(f"{co_url}/health", timeout=5.0)
        r.raise_for_status()
        return None
    except Exception as e:
        return (
            f"code-orchestrator health check failed: {e}. "
            "Start it first: cd /Users/kevintoles/POC/Code-Orchestrator-Service && "
            "source .venv/bin/activate && uvicorn src.main:app --host 0.0.0.0 --port 8083"
        )


def _launch_terminal(
    input_dir: str,
    out_dir: str,
    file_pattern: str,
    skip_existing: bool,
    enable_ocr: bool,
    co_url: str,
) -> dict:
    """Open a new Terminal.app window running the conversion script.

    Mirrors batch_extract_metadata._launch_terminal exactly:
      1. Write a temp .sh script that pipes through tee → log file
      2. Use osascript to open it in a fresh Terminal.app window
    """
    standalone = os.path.abspath(_STANDALONE_SCRIPT)
    python = os.path.abspath(os.path.join(os.path.dirname(standalone), "..", ".venv", "bin", "python"))
    if not os.path.exists(python):
        python = "python3"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    log_file = f"/tmp/batch_convert_{timestamp}.log"  # noqa: S108

    skip_flag = "--skip-existing" if skip_existing else ""
    ocr_flag = "--enable-ocr" if enable_ocr else ""

    fd, tmp_script = tempfile.mkstemp(suffix=".sh", prefix="batch_convert_run_")
    os.close(fd)
    with open(tmp_script, "w") as f:
        f.write(f"""#!/usr/bin/env bash
printf '\\n\\033[1;36m══ Batch PDF Conversion ══  Log: {log_file}\\033[0m\\n\\n'
'{python}' -u '{standalone}' \\
    --input-dir '{input_dir}' \\
    --output-dir '{out_dir}' \\
    --file-pattern '{file_pattern}' \\
    --co-url '{co_url}' \\
    {skip_flag} {ocr_flag} 2>&1 | tee '{log_file}'
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
        "message": f"PDF conversion launched in Terminal.app. Monitor: tail -f {PROGRESS_LOG}",
        "log": log_file,
        "monitor_command": f"tail -f {PROGRESS_LOG}",
        "input_dir": input_dir,
        "output_dir": out_dir,
    }


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler — mirrors batch_extract_metadata interface."""

    async def convert_pdf(
        input_path: str,
        output_path: str | None = None,
        file_pattern: str = "*.pdf",
        skip_existing: bool = True,
        enable_ocr: bool = True,
    ) -> dict:
        """Convert PDFs to structured JSON.

        Always launches in a new Terminal.app window and returns immediately.
        Monitor live progress with: tail -f /tmp/conversion_progress.log

        input_path can be:
          - A directory: converts all PDFs matching file_pattern inside it.
          - A single .pdf file: converts just that file.

        Args:
            input_path: Path to a PDF file or a directory of PDFs.
            output_path: Output JSON file (single mode) or output directory (batch mode).
                         Auto-generated if omitted.
            file_pattern: Glob pattern when input_path is a directory (default: *.pdf).
            skip_existing: Skip PDFs that already have a JSON output file.
            enable_ocr: Enable OCR fallback for image-only pages.
        """
        # Resolve CO URL from dispatcher settings
        settings = getattr(dispatcher, "_settings", None)
        co_url = settings.CODE_ORCHESTRATOR_URL if settings else "http://localhost:8083"

        # No pre-flight health check — the Terminal.app launcher handles CO
        # startup with nohup+disown so it survives VS Code terminal signals.

        if os.path.isdir(input_path):
            # ── Batch mode: directory ──────────────────────────────────────
            input_dir = input_path
            out_dir = output_path or os.path.join(os.path.dirname(input_dir.rstrip("/")), "converted")
            pattern = file_pattern
        else:
            # ── Single-file mode: use parent dir + exact filename as pattern
            input_dir = os.path.dirname(os.path.abspath(input_path))
            pattern = os.path.basename(input_path)
            # Derive output dir from output_path (file → its parent, None → sibling converted/)
            if output_path:
                out_dir = os.path.dirname(os.path.abspath(output_path))
            else:
                out_dir = os.path.join(os.path.dirname(input_dir.rstrip("/")), "converted")

        os.makedirs(out_dir, exist_ok=True)

        return _launch_terminal(
            input_dir=input_dir,
            out_dir=out_dir,
            file_pattern=pattern,
            skip_existing=skip_existing,
            enable_ocr=enable_ocr,
            co_url=co_url,
        )

    return convert_pdf
