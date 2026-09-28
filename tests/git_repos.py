"""Small git repositories for tests: files are written, then committed as the first commit."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path


def commit_all(root: Path) -> None:
    """Commits everything under ``root`` as the first commit of a new repository."""
    for command in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"],
    ):
        subprocess.run(["git", *command], cwd=root, check=True, capture_output=True)


def commit_files(root: Path, files: Mapping[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    commit_all(root)
