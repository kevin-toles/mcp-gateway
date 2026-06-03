#!/usr/bin/env python3
"""Validate tools.yaml against _INPUT_MODELS in tool_registry.py.

Catches config drift: tools added to YAML without a corresponding input model.
Run as part of CI or pre-commit.

Usage:
    python scripts/validate_tools_config.py
    python scripts/validate_tools_config.py --fix  # remove stale entries
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_YAML = REPO_ROOT / "config" / "tools.yaml"
TOOL_REGISTRY = REPO_ROOT / "src" / "tool_registry.py"


def get_yaml_tool_names(path: Path) -> set[str]:
    """Extract tool names from tools.yaml."""
    names: set[str] = set()
    for line in path.read_text().splitlines():
        m = re.match(r"  - name: (\S+)", line)
        if m:
            names.add(m.group(1))
    return names


def get_registered_tool_names(path: Path) -> set[str]:
    """Extract registered tool names from _INPUT_MODELS dict in tool_registry.py."""
    text = path.read_text()
    # Find the _INPUT_MODELS dict
    m = re.search(r"_INPUT_MODELS[:\s]*dict\[.*?\][:\s]*=\s*\{", text, re.DOTALL)
    if not m:
        print("ERROR: Could not find _INPUT_MODELS dict", file=sys.stderr)
        sys.exit(1)

    # Extract all string keys from the dict
    start = m.end()
    brace_depth = 1
    i = start
    names: set[str] = set()
    while i < len(text) and brace_depth > 0:
        if text[i] == '{':
            brace_depth += 1
        elif text[i] == '}':
            brace_depth -= 1
        elif text[i] == '"' or text[i] == "'":
            quote = text[i]
            j = i + 1
            while j < len(text) and text[j] != quote:
                j += 1
            if j > i + 1:
                # Check if this is a dict key (followed by :)
                rest = text[j+1:j+2].strip()
                if rest == ':':
                    names.add(text[i+1:j])
            i = j
        i += 1

    return names


def main():
    yaml_names = get_yaml_tool_names(TOOLS_YAML)
    registered_names = get_registered_tool_names(TOOL_REGISTRY)

    only_in_yaml = yaml_names - registered_names
    only_in_code = registered_names - yaml_names

    has_errors = False

    if only_in_yaml:
        print(f"ERROR: {len(only_in_yaml)} tool(s) in tools.yaml have no _INPUT_MODELS:")
        for name in sorted(only_in_yaml):
            print(f"  - {name}")
        print()
        print("Fix: either add an input model to _INPUT_MODELS in tool_registry.py,")
        print("or remove the entry from config/tools.yaml")
        has_errors = True

    if only_in_code:
        print(f"INFO: {len(only_in_code)} tool(s) registered in code but not in tools.yaml:")
        for name in sorted(only_in_code):
            print(f"  - {name}")
        print("(Not an error — tools can exist in code before being exposed in config)")

    if has_errors:
        sys.exit(1)

    print(f"✅ All {len(yaml_names)} tools in tools.yaml have matching input models.")
    sys.exit(0)


if __name__ == "__main__":
    main()
