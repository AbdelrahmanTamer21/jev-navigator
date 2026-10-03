"""Every Jev answer is kept: by request hash for replay, and by item content for reuse on the next scan.

A record holds hashes, question ids (each with its wording hash), raw answers, the served model, the
thresholds in force, and the source (file, line range, commit) of every code item it judged, so the
request can be rebuilt from the repository at that commit. It holds no request text unless
``keep_requests`` is set, because a request carries code the library cannot know the owner of; set
it only for your own or open-source code.

With ``keep_requests`` a record keeps the request twice: ``request``, written with sorted keys for
reading, and the body as it was sent (``sent_body_base64``; ``sent_exact`` when the client's transport
captured the wire bytes, else the body as the library handed it over). Jev can answer the two orders
differently, so ask again only from ``record.sent_request()``, with a judge that has no store (one
with this store answers from it).

``SqliteAnswerStore`` is one store shared by every run on a machine, so a repeated run on unchanged
code asks nothing again; ``LayeredAnswerStore`` puts a run's own pack in front of it, so the pack
still holds every answer the run used.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .answers import Answer, JevResponse, answer_from_json

INPUT_BUDGET_REFUSAL = "input_budget_refusal"
"""A store line recording that a route refused one exact request for its input size."""


@dataclass(frozen=True)
class AnswerRecord:
    request_sha256: str
    question_ids: tuple[str, ...]
    answers: Mapping[str, dict]
    model: str
    input_tokens: int | None
    thresholds: Mapping[str, float]
    item_keys: Mapping[str, str] = field(default_factory=dict)
    sources: Mapping[str, Mapping] = field(default_factory=dict)
    skeleton: Mapping = field(default_factory=dict)
    request: Mapping | None = None
    recorded_at: str = ""
    sent_body_base64: str | None = None
    sent_exact: bool = False

    def response(self) -> JevResponse:
        answers = {question_id: answer_from_json(raw) for question_id, raw in self.answers.items()}
        return JevResponse(answers, self.model, 0, self.request_sha256, from_store=True)

    def sent_request(self) -> tuple[dict, dict]:
        """The state and questions as they were sent, every key in its sent order."""
        if self.sent_body_base64 is None:
            raise ValueError("this record keeps no request; store answers with keep_requests=True")
        body = json.loads(base64.b64decode(self.sent_body_base64))
        return body["state"], body["questions"]


@dataclass(frozen=True)
class StoredItemAnswer:
    """One item's stored answer and the request that produced it."""

    answer: Answer
    request_sha256: str
    model: str


class AnswerStore(Protocol):
    def by_request(self, request_sha256: str) -> AnswerRecord | None: ...

    def by_item(self, item_key: str, served_model: str | None) -> StoredItemAnswer | None: ...

    def put(self, record: AnswerRecord) -> None: ...

    def refused(self, request_sha256: str, route: str) -> bool: ...

    def put_refusal(self, request_sha256: str, route: str) -> None: ...


class JsonlAnswerStore:
    """Append-only JSON lines. ``item_keys`` maps an item key (item content hash, shared-state hash,
    question id with its wording hash, batch membership hash) to the question id that answered it;
    lookups also match the served model recorded with the answer. Input-size refusals are kept as
    ``input_budget_refusal`` lines keyed by request hash and route."""

    def __init__(self, path: Path, *, keep_requests: bool = False) -> None:
        self.path = Path(path)
        self.keep_requests = keep_requests
        self._records: dict[str, AnswerRecord] = {}
        self._items: dict[str, dict[str, StoredItemAnswer]] = {}
        self._refusals: set[tuple[str, str]] = set()
        self._write_lock = threading.Lock()
        self._load()

    def by_request(self, request_sha256: str) -> AnswerRecord | None:
        return self._records.get(request_sha256)

    def records(self) -> tuple[AnswerRecord, ...]:
        return tuple(self._records.values())

    def by_item(self, item_key: str, served_model: str | None) -> StoredItemAnswer | None:
        """``served_model`` None accepts an answer from any model (replay)."""
        answers = self._items.get(item_key, {})
        if served_model is None:
            return next(reversed(answers.values()), None)
        return answers.get(served_model)

    def refused(self, request_sha256: str, route: str) -> bool:
        """Whether ``route`` refused this exact request for its input size before."""
        return (request_sha256, route) in self._refusals

    def put_refusal(self, request_sha256: str, route: str) -> None:
        with self._write_lock:
            self._append({"kind": INPUT_BUDGET_REFUSAL, "request_sha256": request_sha256, "route": route})
            self._refusals.add((request_sha256, route))

    def put(self, record: AnswerRecord) -> None:
        stored = record if self.keep_requests else _without_request(record)
        stored = _stamped(stored)
        with self._write_lock:
            self._append(asdict(stored))
            self._index(stored)

    def _append(self, line: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as lines:
            lines.write(json.dumps(line, sort_keys=True) + "\n")

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            kind = raw.get("kind")
            if kind == INPUT_BUDGET_REFUSAL:
                self._refusals.add((raw["request_sha256"], raw["route"]))
            elif kind != "llm_step":
                self._index(_record_from_json(raw))

    def _index(self, record: AnswerRecord) -> None:
        self._records[record.request_sha256] = record
        for item_key, question_id in record.item_keys.items():
            by_model = self._items.setdefault(item_key, {})
            by_model[record.model] = StoredItemAnswer(
                answer_from_json(record.answers[question_id]), record.request_sha256, record.model
            )


_SCHEMA = """
create table if not exists answers (request_sha256 text not null, model text not null, record text not null);
create index if not exists answers_by_request on answers (request_sha256, model);
create table if not exists item_answers (
    item_key text not null, model text not null, request_sha256 text not null, answer text not null
);
create index if not exists item_answers_by_key on item_answers (item_key, model);
create table if not exists refusals (request_sha256 text not null, route text not null);
create index if not exists refusals_by_request on refusals (request_sha256, route);
"""


class SqliteAnswerStore:
    """One SQLite file shared by every run on a machine. Rows are only ever inserted, never updated or
    deleted, so no answer is lost; the newest answer for a key wins a lookup. WAL mode lets several
    runs read and write the file at once. Request text follows ``keep_requests`` as in
    ``JsonlAnswerStore``."""

    def __init__(self, path: Path, *, keep_requests: bool = False) -> None:
        self.path = Path(path)
        self.keep_requests = keep_requests
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._db.execute("pragma journal_mode=wal")
        self._db.executescript(_SCHEMA)

    def by_request(self, request_sha256: str) -> AnswerRecord | None:
        row = self._one(
            "select record from answers where request_sha256 = ? order by rowid desc limit 1",
            (request_sha256,),
        )
        return _record_from_json(json.loads(row[0])) if row else None

    def record_for(self, request_sha256: str, model: str) -> AnswerRecord | None:
        row = self._one(
            "select record from answers where request_sha256 = ? and model = ? order by rowid desc limit 1",
            (request_sha256, model),
        )
        return _record_from_json(json.loads(row[0])) if row else None

    def records(self) -> tuple[AnswerRecord, ...]:
        with self._lock:
            rows = self._db.execute("select record from answers order by rowid").fetchall()
        return tuple(_record_from_json(json.loads(record)) for (record,) in rows)

    def by_item(self, item_key: str, served_model: str | None) -> StoredItemAnswer | None:
        """``served_model`` None accepts an answer from any model (replay)."""
        if served_model is None:
            query, parameters = (
                "select answer, request_sha256, model from item_answers where item_key = ?",
                (item_key,),
            )
        else:
            query = "select answer, request_sha256, model from item_answers where item_key = ? and model = ?"
            parameters = (item_key, served_model)
        row = self._one(f"{query} order by rowid desc limit 1", parameters)
        return StoredItemAnswer(answer_from_json(json.loads(row[0])), row[1], row[2]) if row else None

    def put(self, record: AnswerRecord) -> None:
        stored = _stamped(record if self.keep_requests else _without_request(record))
        items = [
            (item_key, stored.model, stored.request_sha256, json.dumps(stored.answers[question_id]))
            for item_key, question_id in stored.item_keys.items()
        ]
        with self._lock, self._db:
            self._db.execute(
                "insert into answers values (?, ?, ?)",
                (stored.request_sha256, stored.model, json.dumps(asdict(stored), sort_keys=True)),
            )
            self._db.executemany("insert into item_answers values (?, ?, ?, ?)", items)

    def refused(self, request_sha256: str, route: str) -> bool:
        return (
            self._one(
                "select 1 from refusals where request_sha256 = ? and route = ?", (request_sha256, route)
            )
            is not None
        )

    def put_refusal(self, request_sha256: str, route: str) -> None:
        with self._lock, self._db:
            self._db.execute("insert into refusals values (?, ?)", (request_sha256, route))

    def _one(self, query: str, parameters: tuple) -> tuple | None:
        with self._lock:
            return self._db.execute(query, parameters).fetchone()


class LayeredAnswerStore:
    """A run's own pack in front of the shared store. Lookups try the pack first; an answer found only
    in the shared store is copied into the pack, so the pack alone replays every answer the run used.
    Every new answer and refusal goes to both."""

    def __init__(self, run: JsonlAnswerStore, shared: SqliteAnswerStore) -> None:
        self.run = run
        self.shared = shared

    def by_request(self, request_sha256: str) -> AnswerRecord | None:
        found = self.run.by_request(request_sha256)
        if found is not None:
            return found
        shared = self.shared.by_request(request_sha256)
        if shared is not None:
            self.run.put(shared)
        return shared

    def by_item(self, item_key: str, served_model: str | None) -> StoredItemAnswer | None:
        found = self.run.by_item(item_key, served_model)
        if found is not None:
            return found
        shared = self.shared.by_item(item_key, served_model)
        if shared is not None and self.run.by_request(shared.request_sha256) is None:
            self.run.put(self.shared.record_for(shared.request_sha256, shared.model))
        return shared

    def records(self) -> tuple[AnswerRecord, ...]:
        return self.run.records()

    def put(self, record: AnswerRecord) -> None:
        self.run.put(record)
        self.shared.put(record)

    def refused(self, request_sha256: str, route: str) -> bool:
        return self.run.refused(request_sha256, route) or self.shared.refused(request_sha256, route)

    def put_refusal(self, request_sha256: str, route: str) -> None:
        self.run.put_refusal(request_sha256, route)
        self.shared.put_refusal(request_sha256, route)


def _record_from_json(raw: dict) -> AnswerRecord:
    raw["question_ids"] = tuple(raw["question_ids"])
    return AnswerRecord(**raw)


def _without_request(record: AnswerRecord) -> AnswerRecord:
    return AnswerRecord(
        **{
            **asdict(record),
            "request": None,
            "sent_body_base64": None,
            "sent_exact": False,
            "question_ids": record.question_ids,
        }
    )


def _stamped(record: AnswerRecord) -> AnswerRecord:
    stamp = record.recorded_at or datetime.now(UTC).isoformat(timespec="seconds")
    return AnswerRecord(**{**asdict(record), "recorded_at": stamp, "question_ids": record.question_ids})
