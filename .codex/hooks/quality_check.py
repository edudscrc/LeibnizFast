#!/usr/bin/env python3
"""Run LeibnizFast quality checks after Codex edits files."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_FILE_RE = re.compile(
    r"^\*\*\* (?:Add|Update|Delete) File: (?P<path>.+)$", re.MULTILINE
)
PATCH_MOVE_RE = re.compile(r"^\*\*\* Move to: (?P<path>.+)$", re.MULTILINE)
MAX_REASON_CHARS = 6000


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


def normalize_repo_file(path: str) -> str | None:
    candidate = path.strip()
    if not candidate or candidate == "/dev/null":
        return None

    raw_path = Path(candidate)
    if raw_path.is_absolute():
        try:
            resolved = raw_path.resolve()
        except OSError:
            return None
        if resolved == REPO_ROOT:
            return None
        if REPO_ROOT not in resolved.parents:
            return None
        return resolved.relative_to(REPO_ROOT).as_posix()

    normalized = Path(candidate)
    if any(part == ".." for part in normalized.parts):
        return None
    return normalized.as_posix()


def parse_patch_files(patch_text: str) -> set[str]:
    files: set[str] = set()
    for regex in (PATCH_FILE_RE, PATCH_MOVE_RE):
        for match in regex.finditer(patch_text):
            normalized = normalize_repo_file(match.group("path"))
            if normalized:
                files.add(normalized)
    return files


def strings_from_tool_input(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key in ("command", "patch", "input", "text", "content"):
            item = value.get(key)
            if isinstance(item, str):
                yield item
        for key in ("file_path", "path"):
            item = value.get(key)
            if isinstance(item, str):
                yield f"*** Update File: {item}"
    elif isinstance(value, list):
        for item in value:
            yield from strings_from_tool_input(item)


def files_from_payload(payload: dict[str, Any]) -> set[str]:
    files: set[str] = set()
    for text in strings_from_tool_input(payload.get("tool_input")):
        files.update(parse_patch_files(text))

    tool_response = payload.get("tool_response")
    if isinstance(tool_response, dict):
        for text in strings_from_tool_input(tool_response):
            files.update(parse_patch_files(text))

    return files


def git_changed_files() -> set[str]:
    files: set[str] = set()
    for args in (
        ["git", "diff", "--name-only"],
        ["git", "diff", "--name-only", "--cached"],
    ):
        result = subprocess.run(
            args,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            files.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    return files


def run_command(args: list[str]) -> tuple[int, str]:
    print(f"--> {' '.join(args)}", file=sys.stderr)
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = ""
    if result.stdout:
        print(result.stdout, file=sys.stderr, end="")
        combined += result.stdout
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
        combined += result.stderr
    return result.returncode, combined


def fail(command: list[str], output: str) -> int:
    tail = output[-MAX_REASON_CHARS:]
    reason = (
        "LeibnizFast quality check failed while running:\n"
        f"{' '.join(command)}\n\n"
        f"{tail}"
    ).strip()
    print(json.dumps({"decision": "block", "reason": reason}))
    return 0


def run_or_fail(args: list[str]) -> bool:
    code, output = run_command(args)
    if code != 0:
        fail(args, output)
        return False
    return True


def existing(files: Iterable[str]) -> list[str]:
    return sorted({file for file in files if (REPO_ROOT / file).exists()})


def main() -> int:
    payload = load_payload()
    if not path_is_inside_repo(payload.get("cwd")):
        return 0

    os.chdir(REPO_ROOT)
    changed = files_from_payload(payload)
    if not changed:
        changed = git_changed_files()

    if os.environ.get("LEIBNIZ_FAST_HOOK_LIST_FILES") == "1":
        print(json.dumps({"changed_files": sorted(changed)}))
        return 0

    rust_files = existing(file for file in changed if file.endswith(".rs"))
    ts_files = existing(file for file in changed if file.endswith((".ts", ".tsx")))
    wgsl_files = existing(file for file in changed if file.endswith(".wgsl"))

    if not rust_files and not ts_files and not wgsl_files:
        return 0

    if rust_files:
        for args in (
            ["cargo", "fmt"],
            ["cargo", "clippy", "--", "-D", "warnings"],
            ["cargo", "test"],
            ["wasm-pack", "build", "--target", "web", "--release"],
        ):
            if not run_or_fail(args):
                return 0

    if ts_files:
        for args in (
            ["npx", "prettier", "--write", *ts_files],
            ["npx", "eslint", *ts_files],
            ["npx", "tsc", "--noEmit"],
            ["npm", "run", "test"],
        ):
            if not run_or_fail(args):
                return 0

    if wgsl_files:
        args = ["npx", "prettier", "--plugin=prettier-plugin-wgsl", "--write", *wgsl_files]
        if not run_or_fail(args):
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
