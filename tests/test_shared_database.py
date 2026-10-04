from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from jev_navigator import shared_database
from jev_navigator.judgments.store import SqliteAnswerStore
from jev_navigator.shared_database import UnusableDatabaseError


def test_a_store_path_that_is_not_a_sqlite_file_is_refused_naming_the_path(tmp_path: Path) -> None:
    # Arrange
    notes = tmp_path / "notes.txt"
    notes.write_text("these are notes, not a database\n" * 10)

    # Act
    with pytest.raises(UnusableDatabaseError) as refused:
        SqliteAnswerStore(notes)

    # Assert
    assert str(notes) in str(refused.value)


def test_a_file_sqlite_cannot_switch_to_wal_mode_is_refused_naming_the_path(tmp_path: Path) -> None:
    # Arrange: SQLite's lock-free dotfile mode, like a file system without POSIX locks, answers a WAL
    # request with the rollback mode instead of an error
    path = tmp_path / "answers.sqlite"
    database = sqlite3.connect(f"file:{tmp_path / 'probe.sqlite'}?vfs=unix-dotfile", uri=True)

    # Act
    with pytest.raises(UnusableDatabaseError) as refused:
        shared_database.switch_to_wal(database, path)

    # Assert
    assert str(path) in str(refused.value)
    assert "delete" in str(refused.value)
