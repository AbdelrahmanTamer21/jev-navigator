from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest
from git_repos import git

from jev_navigator.index import tools

MISSING_OBJECT = "0" * 40


def stored_blob(repository: Path, text: str) -> str:
    """Writes ``text`` into the repository's object store only and returns its object id."""
    git(repository, "init", "-q")
    return git(repository, "hash-object", "-w", "--stdin", stdin=text).strip()


def test_export_writes_each_blob_at_its_path(tmp_path: Path) -> None:
    # Arrange
    repository, destination = tmp_path / "repo", tmp_path / "export"
    repository.mkdir()
    object_id = stored_blob(repository, "def kept():\n    return 1\n")

    # Act
    tools.export_blobs(repository, {"app/kept.py": object_id, "copy.py": object_id}, destination)

    # Assert
    assert (destination / "app/kept.py").read_text() == "def kept():\n    return 1\n"
    assert (destination / "copy.py").read_text() == "def kept():\n    return 1\n"


def test_export_refuses_a_path_that_leaves_the_destination(tmp_path: Path) -> None:
    # Arrange
    repository, destination = tmp_path / "repo", tmp_path / "export"
    repository.mkdir()
    object_id = stored_blob(repository, "escaped = True\n")

    # Act
    with pytest.raises(tools.ToolFailedError, match="outside"):
        tools.export_blobs(repository, {"../escaped.py": object_id}, destination)

    # Assert
    assert not (tmp_path / "escaped.py").exists()


def test_export_fails_loudly_when_git_lacks_an_object(tmp_path: Path) -> None:
    # Arrange
    repository = tmp_path / "repo"
    repository.mkdir()
    stored_blob(repository, "present = True\n")

    # Act and assert
    with pytest.raises(tools.ToolFailedError, match=MISSING_OBJECT):
        tools.export_blobs(repository, {"gone.py": MISSING_OBJECT}, tmp_path / "export")


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX preprocessor script")
@pytest.mark.parametrize(
    "search",
    [
        lambda repository: tools.ripgrep_files("needle", ("a.py",), repository),
        lambda repository: tuple(hit.file for hit in tools.ripgrep_fixed("needle", ("a.py",), repository, 5)),
    ],
    ids=["ripgrep_files", "ripgrep_fixed"],
)
def test_ripgrep_ignores_a_configured_preprocessor(tmp_path: Path, monkeypatch, search) -> None:
    # A ripgrep config in the environment (RIPGREP_CONFIG_PATH) can name `--pre=<program>`, which
    # ripgrep runs for each searched file. Over an untrusted repository that is code execution, so
    # jvn's searches must ignore the config entirely.
    marker = tmp_path / "preprocessor-ran"
    preprocessor = tmp_path / "pre.sh"
    preprocessor.write_text(f'#!/bin/sh\n: > "{marker}"\ncat "$1"\n')
    preprocessor.chmod(preprocessor.stat().st_mode | stat.S_IXUSR)
    config = tmp_path / "rg.conf"
    config.write_text(f"--pre={preprocessor}\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))

    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "a.py").write_text("needle = 1\n")

    found = search(repository)

    assert found == ("a.py",)  # the search still works
    assert not marker.exists()  # but the configured preprocessor never ran


def test_listing_outside_git_ignores_a_configured_ripgrep_filter(tmp_path: Path, monkeypatch) -> None:
    # Outside a Git worktree the file inventory comes from `rg --files`; a ripgrep config must not
    # change which files the index sees there either.
    config = tmp_path / "rg.conf"
    config.write_text("--glob=!a.py\n")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))
    directory = tmp_path / "plain"
    directory.mkdir()
    (directory / "a.py").write_text("needle = 1\n")

    assert tools.listed_files(directory) == ("a.py",)
