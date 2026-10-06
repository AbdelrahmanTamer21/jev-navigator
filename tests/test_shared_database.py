from __future__ import annotations

import gc
import os
import sqlite3
import time
import warnings
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest

from jev_navigator import shared_database
from jev_navigator.index import name_table
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
    probe = f"file:{tmp_path / 'probe.sqlite'}?vfs=unix-dotfile"

    # Act
    with (
        closing(sqlite3.connect(probe, uri=True)) as database,
        pytest.raises(UnusableDatabaseError) as refused,
    ):
        shared_database.switch_to_wal(database, path)

    # Assert
    assert str(path) in str(refused.value)
    assert "delete" in str(refused.value)


SCHEMA = "create table rows (text text not null);"


def test_a_shared_database_returns_the_space_of_deleted_rows_to_the_disk(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "store.sqlite"
    with closing(shared_database.open_shared_database(path, SCHEMA)) as database:
        with database:
            database.executemany("insert into rows values (?)", [("x" * 500,)] * 5_000)
        database.execute("pragma wal_checkpoint(truncate)")
        before = path.stat().st_size
        with database:
            database.execute("delete from rows")

        # Act
        shared_database.release_free_pages(database)

    # Assert
    assert path.stat().st_size < before / 10


def test_opening_a_shared_database_stamps_when_a_process_last_used_it(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "store.sqlite"
    shared_database.open_shared_database(path, SCHEMA).close()
    ten_days_ago = time.time() - 10 * 86_400
    os.utime(path, (ten_days_ago, ten_days_ago))

    # Act
    shared_database.open_shared_database(path, SCHEMA).close()

    # Assert
    assert time.time() - path.stat().st_mtime < 60


@pytest.mark.parametrize("name", ["store.sqlite", "store.sqlite-wal", "store.sqlite-shm"])
def test_a_side_file_belongs_to_its_database(tmp_path: Path, name: str) -> None:
    # Act
    base = shared_database.database_base(tmp_path / name)

    # Assert
    assert base == tmp_path / "store.sqlite"


@pytest.mark.parametrize(
    "opened",
    [
        pytest.param(lambda root: name_table.NameTable(root), id="name table"),
        pytest.param(lambda root: SqliteAnswerStore(root / "answers.sqlite"), id="answer store"),
    ],
)
def test_a_shared_database_owner_closes_its_connection_when_it_is_collected(
    tmp_path: Path, opened: Callable[[Path], object]
) -> None:
    # Arrange
    gc.collect()
    owner = opened(tmp_path)

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        del owner
        gc.collect()

    # Assert
    assert [str(warning.message) for warning in caught if warning.category is ResourceWarning] == []
