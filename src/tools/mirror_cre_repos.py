"""mirror_cre_repos tool handler — mirror repos from repo_registry.json into CRE.

Wraps mirror_all_repos.sh exactly as push_to_github wraps github_upload.py:
  1. Check repo IDs against registry; auto-register any unknown IDs when source_url provided
  2. Accept repo IDs (or none = all new repos) and optional flags
  3. Launch a Terminal.app window running mirror_all_repos.sh
  4. Return immediately with log path to monitor

Standalone script: ai-platform-data/scripts/seeding/mirror_all_repos.sh

Behavior:
  - By default skips repos already present in CRE (skip-existing is the default)
  - Pass repo_ids=[...] to mirror specific repos only (--only flag)
  - Pass force=True to re-mirror repos that already exist (--force flag)
  - Pass auto_continue=True to skip failure prompts (--auto-continue flag)
  - Pass source_url + domain to auto-register an unknown repo_id before mirroring

Monitor live progress with:
    tail -f /tmp/mirror_latest.log
"""

import json
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

_REGISTRY_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "ai-platform-data",
        "repos",
        "repo_registry.json",
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


def _all_registry_ids() -> set[str]:
    """Return the set of all repo IDs currently in repo_registry.json."""
    if not os.path.exists(_REGISTRY_PATH):
        return set()
    with open(_REGISTRY_PATH) as f:
        reg = json.load(f)
    ids: set[str] = set()
    for domain in reg.get("domains", []):
        for repo in domain.get("repos", []):
            ids.add(repo["id"])
    return ids


def _infer_repo_entry(repo_id: str, source_url: str, domain_id: str) -> dict:
    """Build a minimal registry entry from a GitHub URL.

    Infers target_path, language, and concepts from the URL path and repo_id.
    """
    return {
        "id": repo_id,
        "source_url": source_url,
        "target_path": f"{domain_id}/{repo_id}",
        "priority": 1,
        "languages": [],  # caller can enrich later
        "concepts": [],
        "_auto_registered": True,  # flag for auditing; not used by mirror script
    }


def _register_repo(
    repo_id: str,
    source_url: str,
    domain_id: str,
) -> dict:
    """Add a new repo entry to repo_registry.json.

    Returns a dict describing what was done: {"registered": bool, "domain": str, ...}
    """
    if not os.path.exists(_REGISTRY_PATH):
        return {"registered": False, "error": f"Registry not found at {_REGISTRY_PATH}"}

    with open(_REGISTRY_PATH) as f:
        reg = json.load(f)

    # Find the target domain; fall back to "specialized" if domain_id not found
    target_domain = None
    fallback_domain = None
    for d in reg.get("domains", []):
        if d["id"] == domain_id:
            target_domain = d
        if d["id"] == "specialized":
            fallback_domain = d

    if target_domain is None:
        target_domain = fallback_domain
        domain_id = "specialized"

    if target_domain is None:
        return {"registered": False, "error": f"Domain '{domain_id}' not found in registry"}

    # Double-check ID is not already present (race safety)
    existing_ids = {r["id"] for r in target_domain.get("repos", [])}
    if repo_id in existing_ids:
        return {"registered": False, "already_existed": True, "domain": domain_id}

    entry = _infer_repo_entry(repo_id, source_url, domain_id)
    target_domain.setdefault("repos", []).append(entry)
    target_domain["repo_count"] = len(target_domain["repos"])
    reg["total_repos"] = reg.get("total_repos", 0) + 1

    with open(_REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)

    return {
        "registered": True,
        "repo_id": repo_id,
        "source_url": source_url,
        "domain": domain_id,
        "target_path": entry["target_path"],
    }


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
        source_url: str | None = None,
        domain: str | None = None,
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

        **Auto-registration:** if a repo_id is not in the registry AND source_url
        is provided, the repo is automatically registered in repo_registry.json
        before mirroring. Supply ``domain`` to control which domain it lands in
        (defaults to ``"specialized"`` if omitted or unknown).

        By default, repos already present in the CRE are skipped automatically.
        Pass force=True only when you need to re-mirror an existing repo.

        Args:
            repo_ids:      List of registry repo IDs to mirror. Omit (or pass [])
                           to mirror all new repos not yet in the CRE.
            source_url:    GitHub URL of the repo to register if the repo_id is
                           unknown (e.g. 'https://github.com/schedmd/slurm').
                           Only used when repo_ids has exactly one unknown entry.
            domain:        Registry domain to register the new repo under
                           (e.g. 'infrastructure'). Defaults to 'specialized'.
            force:         Re-mirror even if already present in CRE (default False).
            auto_continue: Skip failure prompts and continue automatically (default False).
            dry_run:       Preview what would be mirrored without actually doing it.
        """
        ids = repo_ids or []

        if not os.path.exists(_MIRROR_SCRIPT):
            return sanitizer.sanitize(
                {
                    "status": "error",
                    "error": f"mirror_all_repos.sh not found at {_MIRROR_SCRIPT}",
                }
            )

        # --- Registry check + auto-register step ---
        registration_result: dict | None = None
        if ids:
            known_ids = _all_registry_ids()
            unknown = [rid for rid in ids if rid not in known_ids]

            if unknown and not source_url:
                return sanitizer.sanitize(
                    {
                        "status": "error",
                        "error": (
                            f"Repo ID(s) {unknown} not found in registry. "
                            "Provide source_url to auto-register, or add them to "
                            "repo_registry.json manually first."
                        ),
                        "registry_path": _REGISTRY_PATH,
                        "known_id_count": len(known_ids),
                    }
                )

            if unknown and source_url:
                if len(unknown) > 1:
                    return sanitizer.sanitize(
                        {
                            "status": "error",
                            "error": (
                                f"source_url supports registering one repo at a time, "
                                f"but {len(unknown)} unknown IDs were given: {unknown}. "
                                "Register them individually or add them to the registry manually."
                            ),
                        }
                    )
                reg_domain = domain or "specialized"
                registration_result = _register_repo(unknown[0], source_url, reg_domain)
                if not registration_result.get("registered") and not registration_result.get("already_existed"):
                    return sanitizer.sanitize(
                        {
                            "status": "error",
                            "error": "Auto-registration failed",
                            "details": registration_result,
                        }
                    )

        result = _launch_terminal(ids, force, auto_continue, dry_run)
        if registration_result:
            result["auto_registered"] = registration_result
        return sanitizer.sanitize(result)

    return mirror_cre_repos
