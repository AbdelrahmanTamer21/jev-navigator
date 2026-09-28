"""Thin wrappers over the command-line tools the index runs: ast-grep, ripgrep and git."""

from __future__ import annotations

import io
import json
import logging
import subprocess
import tarfile
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

AST_GREP = "ast-grep"
RIPGREP = "rg"
COMMAND_TIMEOUT_SECONDS = 30
_NO_MATCHES_EXIT = 1


class ToolFailedError(RuntimeError):
    """A command-line tool failed for a reason other than finding nothing."""


def run_command(arguments: Sequence[str], cwd: Path, *, no_match_exit: int | None = None) -> str:
    """The command's output; ``no_match_exit`` is the exit code a search tool uses for "nothing found"."""
    completed = subprocess.run(
        list(arguments), cwd=cwd, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS
    )
    if completed.returncode not in (0, no_match_exit):
        detail = completed.stderr.strip()[:300]
        raise ToolFailedError(f"{arguments[0]} exited {completed.returncode}: {detail}")
    return completed.stdout


def ast_grep_rules(rules_yaml: str, files: Sequence[str], cwd: Path) -> list[dict]:
    if not files:
        return []
    output = run_command(
        [AST_GREP, "scan", "--inline-rules", rules_yaml, "--json=compact", *files],
        cwd,
        no_match_exit=_NO_MATCHES_EXIT,
    )
    return _json_list(output)


def ripgrep_fixed(text: str, files: Sequence[str], cwd: Path, max_hits: int) -> list[dict]:
    if not files:
        return []
    output = run_command(
        [RIPGREP, "--json", "--fixed-strings", "--max-count", str(max_hits), "--", text, *files],
        cwd,
        no_match_exit=_NO_MATCHES_EXIT,
    )
    events = (json.loads(line) for line in output.splitlines() if line.strip())
    return [event["data"] for event in events if event.get("type") == "match"]


def git(arguments: Sequence[str], cwd: Path) -> str:
    return run_command(["git", *arguments], cwd)


def _json_list(output: str) -> list[dict]:
    return json.loads(output) if output.strip() else []


def export_tree(repository: Path, commit: str, prefixes: Sequence[str], destination: Path) -> None:
    """Writes the files under ``prefixes`` at ``commit`` into ``destination`` via ``git archive``."""
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit, "--", *prefixes],
        cwd=repository,
        capture_output=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    if archive.returncode != 0:
        raise ToolFailedError(f"git archive exited {archive.returncode}: {archive.stderr.decode()[:300]}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(destination, filter=_regular_members)


def _regular_members(member: tarfile.TarInfo, destination: str) -> tarfile.TarInfo | None:
    """Links are skipped, so a link pointing outside the tree cannot fail or escape the export."""
    if member.issym() or member.islnk():
        return None
    return tarfile.data_filter(member, destination)
