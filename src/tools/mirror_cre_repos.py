"""mirror_cre_repos tool handler — mirror repos from repo_registry.json into CRE.

Wraps mirror_all_repos.sh exactly as push_to_github wraps github_upload.py:
  1. Accept repo IDs (or none = all new repos) and optional flags
  2. Launch a Terminal.app window running mirror_all_repos.sh
  3. Return immediately with log path to monitor

Standalone script: ai-platform-data/scripts/seeding/mirror_all_repos.sh

Behavior:
  - By default skips repos already present in CRE (skip-existing is the default)
  - Pass repo_ids=[...] to mirror specific repos only (--only flag)
  - Pass force=True to re-mirror repos that already exist (--force flag)
  - Pass auto_continue=True to skip failure prompts (--auto-continue flag)

Monitor live progress with:
    tail -f /tmp/mirror_latest.log
"""

import os
import shlex
import stat
import subprocess
import tempfile
from datetime import datetime

from fastmcp import Context

from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "mirror_cre_repos"
PROGRESS_LOG = "/tmp/mirror_latest.log"  # noqa: S108

_MIRROR_SCRIPT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "ai-platform-data",
        "scripts",
        "seeding",
        "mirror_all_repos.sh",
    )
)


def _build_args(repo_ids: list[str], force: bool, auto_continue: bool, dry_run: bool) -> str:
    """Build shell argument string for mirror_all_repos.sh."""
    args = []
    if repo_ids:
        args += ["--only", ",".join(repo_ids)]
    if force:
        args.append("--force")
    if auto_continue:
        args.append("--auto-continue")
    if dry_run:
        args.append("--dry-run")
    return " ".join(shlex.quote(a) for a in args)


def _launch_terminal(repo_ids: list[str], force: bool, auto_continue: bool, dry_run: bool) -> dict:
    """Open a new Terminal.app window running mirror_all_repos.sh."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
    log_file = f"/tmp/mirror_{timestamp}.log"  # noqa: S108

    args_str = _build_args(repo_ids, force, auto_continue, dry_run)
    label = f"--only {','.join(repo_ids)}" if repo_ids else "all new repos"

    fd, tmp_script = tempfile.mkstemp(suffix=".sh", prefix="mirror_cre_run_")
    os.close(fd)
    with open(tmp_script, "w") as f:
        f.write(f"""#!/usr/bin/env bash
export _MIRROR_INNER=1
printf '\\n\\033[1;36m\u2550\u2550 CRE Mirror \u2192 {label}  Log: {log_file}\\033[0m\\n\\n'
cd '{os.path.dirname(os.path.dirname(_MIRROR_SCRIPT))}'
'{_MIRROR_SCRIPT}' {args_str} 2>&1 | tee '{log_file}'
EXIT_CODE=${{PIPESTATUS[0]}}
ln -sf '{log_file}' '{PROGRESS_LOG}'
echo ''
if [[ $EXIT_CODE -eq 0 ]]; then
  printf '\\033[1;32m\u2705  Mirror complete.\\033[0m\\n'
else
  printf '\\033[1;31m\u274c  Mirror completed with failures (exit '$EXIT_CODE'). Check log above.\\033[0m\\n'
fi
echo ''
read -rp 'Press Enter to close...'
""")
    os.chmod(tmp_script, os.stat(tmp_script).st_mode | stat.S_IEXEC)

    osascript = """on run argv
    set scriptPath to item 1 of argv
    tell application "Terminal"
        do script scriptPath
        set bounds of front window to {80, 80, 1300, 900}
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
        "repo_ids": repo_ids if repo_ids else "all-new",
        "force": force,
        "dry_run": dry_run,
    }


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler for mirror_cre_repos."""

    async def mirror_cre_repos(
        repo_ids: list[str] | None = None,
        force: bool = False,
        auto_continue: bool = False,
        dry_run: bool = False,
        ctx: Context | None = None,
    ) -> dict:
        """Mirror one or more repositories into the Code Reference Engine (CRE).

        Wraps mirror_all_repos.sh and launches a Terminal.app window for live
        progress. Returns immediately — monitor with: tail -f /tmp/mirror_latest.log

        Repos are identified by their registry ID (the ``id`` field in
        repo_registry.json, e.g. ``"aws-lambda-power-tuning"``, ``"openfaas"``).

        By default, repos already present in the CRE are skipped automatically.
        Pass force=True only when you need to re-mirror an existing repo.

        Args:
            repo_ids:      List of registry repo IDs to mirror. Omit (or pass [])
                           to mirror all new repos not yet in the CRE.
            force:         Re-mirror even if already present in CRE (default False).
            auto_continue: Skip failure prompts and continue automatically (default False).
            dry_run:       Preview what would be mirrored without actually doing it.
        """
        ids = repo_ids or []

        if not os.path.exists(_MIRROR_SCRIPT):
            return {
                "status": "error",
                "error": f"mirror_all_repos.sh not found at {_MIRROR_SCRIPT}",
            }

        result = _launch_terminal(ids, force, auto_continue, dry_run)
        return sanitizer.sanitize(result)

    return mirror_cre_repos
