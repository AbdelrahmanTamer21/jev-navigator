"""One answer store shared across runs: answers replay in a later run and land in that run's pack."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from conftest import BudgetedClient

from jev_navigator.judgments.client import ReplayOnlyClient
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.store import (
    AnswerRecord,
    JsonlAnswerStore,
    LayeredAnswerStore,
    SqliteAnswerStore,
)
from jev_navigator.testing import ScriptedJevClient

DESCRIBES = Check(
    name="describes",
    instructions="Is `{item}.code` the implementation that `doc.sentence` describes?",
    yes=Criterion("The code performs what the sentence says."),
    no=Criterion("The code does something else, or only calls it."),
)
SHARED = {"doc": {"sentence": "s"}}
ITEMS = [{"code": "x = 1"}, {"code": "y = 2"}, {"code": "z = 3"}]


def _run_store(run: Path, shared: Path) -> LayeredAnswerStore:
    return LayeredAnswerStore(JsonlAnswerStore(run / "answers.jsonl"), SqliteAnswerStore(shared))


def test_a_later_run_replays_every_answer_from_the_shared_store(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    Judge(ScriptedJevClient(default_noul=0.9), store=_run_store(tmp_path / "run1", shared)).check_each(
        DESCRIBES, ITEMS, SHARED
    )
    client = ScriptedJevClient(default_noul=0.1)

    # Act
    results = Judge(
        client, store=_run_store(tmp_path / "run2", shared), served_model="jev-scripted"
    ).check_each(DESCRIBES, ITEMS, SHARED)

    # Assert
    assert client.requests == []
    assert [(result.probability, result.from_store) for result in results] == [(0.9, True)] * 3


def test_answers_replayed_from_the_shared_store_land_in_the_runs_own_pack(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    Judge(ScriptedJevClient(default_noul=0.9), store=_run_store(tmp_path / "run1", shared)).check_each(
        DESCRIBES, ITEMS, SHARED
    )
    Judge(
        ScriptedJevClient(), store=_run_store(tmp_path / "run2", shared), served_model="jev-scripted"
    ).check_each(DESCRIBES, ITEMS, SHARED)

    # Act: the second run's pack alone, without the shared store, answers offline
    pack_only = Judge(ReplayOnlyClient(), store=JsonlAnswerStore(tmp_path / "run2" / "answers.jsonl"))
    results = pack_only.check_each(DESCRIBES, ITEMS, SHARED)

    # Assert
    assert [result.probability for result in results] == [0.9] * 3


def test_answers_from_another_served_model_are_kept_and_not_reused(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    old = ScriptedJevClient(default_noul=0.9, model="jev-1.12.0")
    Judge(old, store=SqliteAnswerStore(shared)).check_each(DESCRIBES, ITEMS, SHARED)
    upgraded = ScriptedJevClient(default_noul=0.2, model="jev-1.13.0")

    # Act
    Judge(upgraded, store=SqliteAnswerStore(shared), served_model="jev-1.13.0").check_each(
        DESCRIBES, ITEMS, SHARED
    )
    replay_old = Judge(
        ScriptedJevClient(), store=SqliteAnswerStore(shared), served_model="jev-1.12.0"
    ).check_each(DESCRIBES, ITEMS, SHARED)

    # Assert
    assert len(upgraded.requests) == 1
    assert [result.probability for result in replay_old] == [0.9] * 3


def test_the_shared_store_keeps_no_request_text_unless_asked(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    secretless_code = [{"code": "def unique_marker_function(): ..."}]

    # Act
    Judge(ScriptedJevClient(), store=SqliteAnswerStore(shared)).check_each(DESCRIBES, secretless_code, SHARED)
    Judge(
        ScriptedJevClient(), store=SqliteAnswerStore(tmp_path / "kept.sqlite", keep_requests=True)
    ).check_each(DESCRIBES, secretless_code, SHARED)

    # Assert
    stored = sqlite3.connect(shared).execute("select record from answers").fetchall()
    assert stored and not any("unique_marker_function" in record for (record,) in stored)
    record = SqliteAnswerStore(tmp_path / "kept.sqlite").records()[0]
    assert record.sent_request()[0]["items"][0]["code"] == "def unique_marker_function(): ..."


def test_two_writers_on_one_shared_file_lose_no_record(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    writers = [SqliteAnswerStore(shared), SqliteAnswerStore(shared)]

    def write(store: SqliteAnswerStore, prefix: str) -> None:
        for index in range(50):
            store.put(AnswerRecord(f"{prefix}{index}", ("q",), {"q": {"type": "noul", "p": 0.5}}, "m", 1, {}))

    threads = [
        threading.Thread(target=write, args=(store, prefix))
        for store, prefix in zip(writers, "ab", strict=True)
    ]

    # Act
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Assert
    rows = sqlite3.connect(shared).execute("select count(*) from answers").fetchone()[0]
    assert rows == 100


def test_a_size_refusal_is_remembered_across_runs(tmp_path: Path) -> None:
    # Arrange
    shared = tmp_path / "answers.sqlite"
    items = [{"file": f"p{index}.py", "code": "y" * 12_000} for index in range(4)]
    Judge(BudgetedClient(34_000), store=SqliteAnswerStore(shared), served_model="jev-scripted").check_each(
        DESCRIBES, items, SHARED
    )
    replay = BudgetedClient(34_000)

    # Act
    Judge(replay, store=SqliteAnswerStore(shared), served_model="jev-scripted").check_each(
        DESCRIBES, items, SHARED
    )

    # Assert
    assert replay.requests == [] and replay.refusals == 0
