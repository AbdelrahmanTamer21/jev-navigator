"""Every name a file's facts hold, and where it sits, kept on disk per file content.

A row ties a name to a definition, a call or a reference in one file content, identified by its git
blob id, so a new index maps its files to rows without reading them, and answers a name lookup with
no text search and no parse. Rows hold names and line numbers, never code: a receiver is a plain
chain of names or ``scope_scan.OPAQUE_RECEIVER``, as the facts hold it.

One table file serves one ``table_identity``: a change to the parser, to any language's rules or to
the code that turns facts into rows starts a new table. Rows are only ever inserted, and a file's
rows are written in one transaction with its entry in ``files``, so two processes writing at once
leave the table whole and a reader never sees half a file.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from itertools import islice
from pathlib import Path

from ..cache_root import cache_root
from ..shared_database import open_shared_database
from .fact_cache import facts_identity
from .scope_scan import FileFacts

SYMBOL = "symbol"
DECLARATION = "declaration"
CALL = "call"
REFERENCE = "reference"
DEFINITION_KINDS = (SYMBOL, DECLARATION)
_ANONYMOUS = "<anonymous>"
_QUERY_CHUNK = 500


@dataclass(frozen=True)
class FileEntry:
    """What the table knows about one file content beyond its names: whether the parser could only
    partly read it, and the lines its ERROR nodes span."""

    incomplete: bool
    unparsed_lines: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class NameRow:
    """One place a name sits in one file content. ``position`` is the place's index among the file's
    facts of its kind, so rows come back in the facts' order."""

    blob: str
    kind: str
    position: int
    start: int
    end: int
    role: str | None = None
    receiver: str | None = None


class NameTable:
    def __init__(self, root: Path | None = None) -> None:
        self.path = (root or cache_root() / "names") / f"{table_identity()}.sqlite"
        self._lock = threading.Lock()
        self._db = open_shared_database(self.path, _SCHEMA)

    def entries(self, blobs: Collection[str]) -> dict[str, FileEntry]:
        """The entries of those ``blobs`` whose rows the table holds."""
        found: dict[str, FileEntry] = {}
        with self._lock:
            for chunk in _chunks(sorted(blobs)):
                marks = ",".join("?" * len(chunk))
                query = f"select blob, incomplete, unparsed_lines from files where blob in ({marks})"
                for blob, incomplete, stretches in self._db.execute(query, chunk):
                    found[blob] = FileEntry(bool(incomplete), tuple(map(tuple, json.loads(stretches))))
        return found

    def add(self, facts_by_blob: Mapping[str, FileFacts]) -> None:
        """Writes the rows of each file content the table does not hold yet, one transaction per
        content. Contents already held take no write lock, so warm runs never wait on each other;
        a content another process wrote meanwhile is left as it is."""
        held = self.entries(facts_by_blob.keys())
        with self._lock:
            for blob, facts in facts_by_blob.items():
                if blob not in held:
                    self._write(blob, facts)

    def _write(self, blob: str, facts: FileFacts) -> None:
        with self._db:
            self._db.execute("begin immediate")
            if self._add_entry(blob, facts):
                self._db.executemany("insert into names values (?, ?, ?, ?, ?, ?, ?, ?)", _rows(blob, facts))

    def rows(self, name: str) -> tuple[NameRow, ...]:
        with self._lock:
            found = self._db.execute(
                "select blob, kind, position, start, end, role, receiver from names where name = ?",
                (name,),
            ).fetchall()
        return tuple(NameRow(*row) for row in found)

    def _add_entry(self, blob: str, facts: FileFacts) -> bool:
        stretches = json.dumps([list(stretch) for stretch in facts.unparsed_lines])
        added = self._db.execute(
            "insert or ignore into files values (?, ?, ?)", (blob, int(facts.incomplete), stretches)
        )
        return added.rowcount == 1


@cache
def table_identity() -> str:
    """The facts' identity and the source of this module, which decides what a row holds."""
    source = Path(__file__).read_bytes()
    return hashlib.sha256(facts_identity().encode() + b"\0" + source).hexdigest()


def git_blob_id(content: bytes) -> str:
    """The id git gives ``content`` as a blob, so a clean tracked file is identified from the index
    listing alone."""
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def _rows(blob: str, facts: FileFacts) -> Iterator[tuple]:
    structure = facts.structure
    for kind, spans in ((SYMBOL, structure.symbols), (DECLARATION, structure.declarations)):
        for position, span in enumerate(spans):
            if span.name != _ANONYMOUS:
                yield span.name, blob, kind, position, span.start, span.end, None, None
    for position, call in enumerate(facts.calls):
        yield call.name, blob, CALL, position, call.line, call.line, None, call.receiver
    for position, reference in enumerate(facts.references):
        line, role = reference.line, reference.role
        yield reference.name, blob, REFERENCE, position, line, line, role, reference.receiver


def _chunks(values: list[str]) -> Iterator[list[str]]:
    iterator = iter(values)
    while chunk := list(islice(iterator, _QUERY_CHUNK)):
        yield chunk


_SCHEMA = """
create table files (
    blob text primary key,
    incomplete integer not null,
    unparsed_lines text not null
) without rowid;
create table names (
    name text not null,
    blob text not null,
    kind text not null,
    position integer not null,
    start integer not null,
    end integer not null,
    role text,
    receiver text,
    primary key (name, blob, kind, position)
) without rowid;
"""
