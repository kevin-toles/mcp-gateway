"""push_to_github tool handler — push files to GitHub via git push.

Mirrors the convert_pdf / batch_extract_metadata pattern exactly:
  1. Accept file paths + repo + dest as parameters
  2. Launch a Terminal.app window running push_large_files.py
  3. Return immediately with log path to monitor

Standalone script: ai-platform-data/scripts/github_upload.py

Routing is automatic — github_upload.py picks the right flow based on file size:
  < 30 MB  → GitHub Git Data API (blob upload, zero git)
  30-100 MB → GitHub Git Data API (same path, heavier files)
  > 100 MB  → git-lfs push (sparse checkout, GIT_LFS_SKIP_SMUDGE=1)

Monitor live progress with:
    tail -f /tmp/push_github_latest.log
"""

import os
import shlex
import stat
import subprocess
import tempfile
from datetime import datetime

from fastmcp import Context

from src.models.schemas import PushToGithubInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "push_to_github"
PROGRESS_LOG = "/tmp/push_github_latest.log"  # noqa: S108
_PUSH_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "ai-platform-data", "scripts", "github_upload.py")
)


def _launch_terminal(files: list[str], repo: str, dest: str) -> dict:
    """Open a new Terminal.app window running push_large_files.py.

    Follows the same osascript pattern as convert_pdf and batch_extract_metadata:
      1. Write a temp .sh script with tee → log file
      2. Use osascript to open it in a fresh Terminal.app window
      3. Return immediately
    """
    python = os.path.abspath(os.path.join(os.path.dirname(_PUSH_SCRIPT), "..", ".venv", "bin", "python"))
    if not os.path.exists(python):
        python = "python3"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    log_file = f"/tmp/push_github_{timestamp}.log"  # noqa: S108

    file_args = " ".join(shlex.quote(f) for f in files)

    fd, tmp_script = tempfile.mkstemp(suffix=".sh", prefix="push_github_run_")
    os.close(fd)
    with open(tmp_script, "w") as f:
        f.write(f"""#!/usr/bin/env bash
printf '\\n\\033[1;36m\u2550\u2550 Push to GitHub \u2192 {repo}/{dest}  Log: {log_file}\\033[0m\\n\\n'
'{python}' -u '{_PUSH_SCRIPT}' --files {file_args} --repo '{repo}' --dest '{dest}' 2>&1 | tee '{log_file}'
EXIT_CODE=${{PIPESTATUS[0]}}
ln -sf '{log_file}' '{PROGRESS_LOG}'
echo ''
if [[ $EXIT_CODE -eq 0 ]]; then
  printf '\\033[1;32m\u2705  Push complete.\\033[0m\\n'
else
  printf '\\033[1;31m\u274c  Push failed (exit '$EXIT_CODE'). Check log above.\\033[0m\\n'
fi
echo ''
read -rp 'Press Enter to close...'
""")
    os.chmod(tmp_script, os.stat(tmp_script).st_mode | stat.S_IEXEC)

    osascript = """on run argv
    set scriptPath to item 1 of argv
    tell application "Terminal"
        do script scriptPath
        set bounds of front window to {80, 80, 1200, 700}
        activate
    end tell
end run
"""
    fd2, osa_file = tempfile.mkstemp(suffix=".applescript")
    os.close(fd2)
    with open(osa_file, "w") as f:
        f.write(osascript)

    subprocess.Popen(["/usr/bin/osascript", osa_file, tmp_script])  # noqa: S603

    return {
        "status": "launched",
        "log_file": log_file,
        "monitor_cmd": f"tail -f {log_file}",
        "files": files,
        "repo": repo,
        "dest": dest,
    }


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler for push_to_github."""

    async def push_to_github(
        files: list[str],
        repo: str = "kevin-toles/pdf-text-repo",
        dest: str = "Textbooks",
        skip_existing: bool = True,
        ctx: Context | None = None,
    ) -> dict:
        """Push one or more files to a GitHub repo. Routing is automatic by size:

        < 30 MB  → GitHub Git Data API (no git required)
        30-100 MB → GitHub Git Data API
        > 100 MB  → git-lfs push

        Always launches in a new Terminal.app window and returns immediately.
        Monitor live progress with: tail -f /tmp/push_github_latest.log

        Args:
            files: List of absolute local file paths to upload.
            repo: GitHub repo as owner/name (default: kevin-toles/pdf-text-repo).
            dest: Destination directory inside the repo (default: Textbooks).
            skip_existing: Skip files already present in the repo (default: True).
        """
        validated = PushToGithubInput(
            files=files,
            repo=repo,
            dest=dest,
            skip_existing=skip_existing,
        )

        missing = [f for f in validated.files if not os.path.exists(f)]
        if missing:
            return {
                "status": "error",
                "message": f"Files not found locally: {missing}",
                "hint": "Provide absolute paths to existing files.",
            }

        result = _launch_terminal(
            files=validated.files,
            repo=validated.repo,
            dest=validated.dest,
        )

        if ctx:
            await ctx.info(
                f"Push launched: {len(validated.files)} file(s) → {validated.repo}/{validated.dest}\n"
                f"Monitor: tail -f {result['log_file']}"
            )

        return result

    return push_to_github
