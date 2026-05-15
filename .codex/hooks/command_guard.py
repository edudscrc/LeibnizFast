#!/usr/bin/env python3
"""Block destructive Bash commands before Codex runs them."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DANGEROUS_PATTERNS = [
    r"rm -rf",
    r"rm -fr",
    r"rm -r \.",
    r"rm -r \*",
    r"mkfs",
    r"> /dev/null",
]


def load_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def path_is_inside_repo(path: str | None) -> bool:
    try:
        cwd = Path(path).resolve() if path else Path.cwd().resolve()
    except OSError:
        return False
    return cwd == REPO_ROOT or REPO_ROOT in cwd.parents


def extract_command(payload: dict[str, Any]) -> str:
    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict):
        command = tool_input.get("command") or tool_input.get("cmd")
        return command if isinstance(command, str) else ""
    return tool_input if isinstance(tool_input, str) else ""


def block(reason: str) -> None:
    print(json.dumps({"decision": "block", "reason": reason}))


def main() -> int:
    payload = load_payload()
    if not path_is_inside_repo(payload.get("cwd")):
        return 0

    command = extract_command(payload)
    if not command:
        return 0

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            block(
                "SECURITY BLOCK: attempted to run a destructive command.\n"
                f"Blocked command: {command}\n"
                "Ask the user to run this manually if it is absolutely necessary."
            )
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
