#!/usr/bin/env python3
"""
Validate (and optionally fix) the ai-platform MCP server entry in ~/.claude.json.

Claude Code stores MCP server registrations in ~/.claude.json under:
  projects.<project_path>.mcpServers.<server_name>

The mcp-gateway port in services.toml is the single source of truth.
If the registered URL points to a stale port all MCP tools silently disappear.

Usage:
    python3 scripts/validate_mcp_json.py          # check only, exit 1 on drift
    python3 scripts/validate_mcp_json.py --fix    # re-register via `claude mcp add`
"""

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "config" / "services.toml"
CLAUDE_JSON = Path.home() / ".claude.json"
SERVER_KEY = "ai-platform"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_canonical_port() -> int:
    with open(MANIFEST, "rb") as f:
        manifest = tomllib.load(f)
    services = manifest.get("services", {})
    if "mcp-gateway" not in services:
        sys.exit("ERROR: 'mcp-gateway' not found in services.toml")
    return int(services["mcp-gateway"]["port"])


def get_registered_url() -> str | None:
    """Read ~/.claude.json and return the registered URL for ai-platform, or None."""
    if not CLAUDE_JSON.is_file():
        return None
    with open(CLAUDE_JSON) as f:
        data = json.load(f)
    projects = data.get("projects", {})
    project = projects.get(str(PROJECT_ROOT), {})
    servers = project.get("mcpServers", {})
    return servers.get(SERVER_KEY, {}).get("url")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="Re-register via claude mcp add if URL is wrong")
    args = parser.parse_args()

    port = load_canonical_port()
    expected_url = f"http://localhost:{port}/mcp/sse"
    actual_url = get_registered_url()

    if actual_url == expected_url:
        print(f"OK  {CLAUDE_JSON}")
        print(f"    {SERVER_KEY}.url = {actual_url}  ✓")
        sys.exit(0)

    print(f"DRIFT  {CLAUDE_JSON}")
    print(f"    {SERVER_KEY}.url (registered) = {actual_url or '<not registered>'}")
    print(f"    expected                      = {expected_url}")
    print(f"    (manifest: config/services.toml  mcp-gateway.port = {port})")

    if not args.fix:
        print("\nRun with --fix to re-register automatically.")
        sys.exit(1)

    # Re-register using the documented CLI command
    claude = os.environ.get("CLAUDE_BIN", "claude")
    try:
        subprocess.run([claude, "mcp", "remove", SERVER_KEY], check=False, capture_output=True)
        subprocess.run(
            [claude, "mcp", "add", "--transport", "sse", SERVER_KEY, expected_url],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        sys.exit(f"ERROR: '{claude}' not found — run: claude mcp add --transport sse {SERVER_KEY} {expected_url}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"ERROR: claude mcp add failed: {e.stderr.decode()}")

    print(f"\nFIXED  registered {SERVER_KEY} → {expected_url}")
    print("Restart Claude Code for the change to take effect.")


if __name__ == "__main__":
    main()
