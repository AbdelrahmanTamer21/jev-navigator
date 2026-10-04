"""SQLite files that several JVN processes read and write at once."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing, suppress
from pathlib import Path


def open_shared_database(path: Path, schema: str, version: int = 0) -> sqlite3.Connection:
    """A connection to the file at ``path``, created first when it is missing. A new file is built
    whole under a temporary name, in WAL mode with ``schema`` and ``version``, then linked into
    place. Linking fails when another process got there first, so every process opens one finished
    file and none has to change its journal mode, which needs the file to itself."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _create(path, schema, version)
    return sqlite3.connect(path, timeout=30, check_same_thread=False)


def _create(path: Path, schema: str, version: int) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    try:
        with closing(sqlite3.connect(temporary)) as database:
            database.execute("pragma journal_mode=wal")
            database.executescript(schema)
            database.execute(f"pragma user_version = {version}")
            database.commit()
        with suppress(FileExistsError):
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
