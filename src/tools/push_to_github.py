"""push_to_github — blind pass-through proxy to github_upload.py.

Direct "Pipe" Language — no validation, no transformation.
Passes the raw --json argument exactly as received to the script.
"""

import os
import stat
import subprocess
import tempfile
from datetime import datetime

from fastmcp import Context

from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "push_to_github"
PROGRESS_LOG = "/tmp/push_github_latest.log"  # noqa: S108
_PUSH_SCRIPT = "/Users/kevintoles/POC/ai-platform-data/scripts/github_upload.py"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler for push_to_github — blind pass-through."""

    async def push_to_github(
        json: str | None = None,
        json_file: str | None = None,
        ctx: Context | None = None,
    ) -> dict:
        """Blind pass-through to github_upload.py --json.

        For small payloads pass json directly. For large payloads (>10KB)
        write the JSON to a file and pass json_file instead.

        Args:
            json: Raw JSON string passed directly to --json flag.
            json_file: Path to file containing the JSON payload (for large batches).
        """
        python = "python3"
        if os.path.exists(os.path.join(os.path.dirname(_PUSH_SCRIPT), "..", ".venv", "bin", "python")):
            python = os.path.join(os.path.dirname(_PUSH_SCRIPT), "..", ".venv", "bin", "python")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
        log_file = f"/tmp/push_github_{timestamp}.log"  # noqa: S108

        # Use json_file if provided, otherwise write json string to temp file
        if json_file:
            payload_file = json_file
        else:
            import json as _json_module
            fd, payload_file = tempfile.mkstemp(suffix=".json", prefix="push_github_payload_")
            os.close(fd)
            with open(payload_file, "w") as f:
                f.write(json if json else "{}")

        fd, tmp_script = tempfile.mkstemp(suffix=".sh", prefix="push_github_run_")
        os.close(fd)

        script_content = (
            "#!/usr/bin/env bash\n"
            "printf '\\n\\033[1;36m══ Push to GitHub\\033[0m\\n'\n"
            "printf '\\033[0;90mLog: " + log_file + "\\033[0m\\n\\n'\n"
            "'" + python + "' -u '" + _PUSH_SCRIPT + "' --json \"$(cat '" + payload_file + "')\" 2>&1 | tee '" + log_file + "'\n"
            "EXIT_CODE=${PIPESTATUS[0]}\n"
            "ln -sf '" + log_file + "' '" + PROGRESS_LOG + "'\n"
            "echo ''\n"
            "if [[ $EXIT_CODE -eq 0 ]]; then\n"
            "  printf '\\033[1;32m✅  Push complete.\\033[0m\\n'\n"
            "else\n"
            "  printf '\\033[1;31m❌  Push failed (exit '$EXIT_CODE'). Check log above.\\033[0m\\n'\n"
            "fi\n"
            "echo ''\n"
            "read -rp 'Press Enter to close...'\n"
        )
        with open(tmp_script, "w") as f:
            f.write(script_content)
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

        if ctx:
            await ctx.info(f"Push launched. Monitor: tail -f {log_file}")

        return {
            "status": "launched",
            "log_file": log_file,
            "monitor_cmd": f"tail -f {log_file}",
        }

    return push_to_github
