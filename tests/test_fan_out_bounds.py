"""Each round of a search and each set of history checks records at most ``judge.max_concurrency``
requests as started and not yet answered, whatever the beam width or the number of history sections.
The judge's send slots bound the requests in flight; these bounds keep the requests waiting for a
slot, and in the sync forms their threads, within the same number. ``test_send_bound`` proves the
thread bound at scale. Ctrl-C meets the places queued behind the bound, and Resume asks them."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import pytest
from send_bound_search import TARGET, AsyncHeldClient, HeldClient, function_starts, many_functions_index

from jev_navigator.directives.find_code import Outcome, SearchBudget, find_code, find_code_async
from jev_navigator.history import History, HistoryCheck, judge_sections, judge_sections_async
from jev_navigator.judgments.journal import JournalRequest, JsonlJournal, RawResponse
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion, request_sha256
from jev_navigator.testing import ScriptedJevClient

BOUND = 4
HOLD_SECONDS = 0.05
ARRIVAL_SECONDS = 5


class OutstandingJournal(JsonlJournal):
    """A real journal that also counts the most requests it held as started and not yet answered."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.outstanding = 0
        self.peak = 0
        self._counting = threading.Lock()

    def record_request(self, request: JournalRequest) -> str:
        with self._counting:
            self.outstanding += 1
            self.peak = max(self.peak, self.outstanding)
        return super().record_request(request)

    def record_response(self, request_id: str, response: RawResponse) -> None:
        super().record_response(request_id, response)
        self._settled()

    def record_failure(
        self, request_id: str, error: BaseException, response: RawResponse | None = None
    ) -> None:
        super().record_failure(request_id, error, response)
        self._settled()

    def _settled(self) -> None:
        with self._counting:
            self.outstanding -= 1


class InterruptedAtOnePlace:
    """Answers like ``ScriptedJevClient`` and records every request it receives. The request that shows
    ``interrupted`` (a slice's file and lines) raises ``KeyboardInterrupt``, as a Ctrl-C would. Every
    other request first waits up to ``ARRIVAL_SECONDS`` for that one to arrive, then is held
    ``HOLD_SECONDS``, so which request is interrupted never depends on which thread sends first, and
    the places queued behind the bound have not started when the interrupt lands."""

    model = "jev-scripted"

    def __init__(self, interrupted: tuple[str, str]) -> None:
        self.script = ScriptedJevClient(default_noul=0.05)
        self.interrupted_place = interrupted
        self.interrupted: tuple[Mapping, Mapping] | None = None
        self.answered: list[tuple[Mapping, Mapping]] = []
        self._arrived = threading.Event()
        self._lock = threading.Lock()

    def ask(self, state: Mapping, questions: Mapping):
        if _slice_of(state) == self.interrupted_place and not self._arrived.is_set():
            self.interrupted = (state, questions)
            self._arrived.set()
            raise KeyboardInterrupt
        self._arrived.wait(timeout=ARRIVAL_SECONDS)
        with self._lock:
            self.answered.append((state, questions))
        time.sleep(HOLD_SECONDS)
        return self.script.ask(state, questions)

    @property
    def received(self) -> list[tuple[Mapping, Mapping]]:
        return [self.interrupted, *self.answered] if self.interrupted is not None else self.answered


def _slice_of(state: Mapping) -> tuple[str, str]:
    return state["slice"]["file"], state["slice"]["lines"]


def _bounded_judge(mode: str, journal: OutstandingJournal) -> Judge:
    """A judge of ``BOUND`` whose client holds each send, so requests that may run together do."""
    held = HeldClient()
    return Judge(held if mode == "sync" else AsyncHeldClient(held), journal=journal, max_concurrency=BOUND)


def _hashes(requests: list[tuple[Mapping, Mapping]]) -> list[str]:
    return [request_sha256(state, questions) for state, questions in requests]


def _section_checks(names: list[str]) -> dict[str, HistoryCheck]:
    """One check per section, so each is a group and a request of its own."""
    return {
        name: HistoryCheck(
            Check(
                f"mentions_{name}",
                f"Does `{name}` mention a limit?",
                Criterion(f"`{name}` names a limit."),
                Criterion(f"`{name}` names no limit."),
            ),
            (name,),
        )
        for name in names
    }


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_round_of_forty_records_at_most_the_bound_of_requests_as_started_at_once(
    tmp_path: Path, mode: str
) -> None:
    # Arrange
    index = many_functions_index(tmp_path, 40)
    starts = function_starts(index, 40)
    budget = SearchBudget(beam_width=40, max_steps=40, neighbours_per_kind=2)
    journal = OutstandingJournal(tmp_path / "journal.jsonl")
    judge = _bounded_judge(mode, journal)

    # Act
    if mode == "sync":
        result = find_code(index, judge, TARGET, starts, budget=budget)
    else:
        result = asyncio.run(find_code_async(index, judge, TARGET, starts, budget=budget))

    # Assert
    assert result.steps == 40
    assert journal.peak == BOUND


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_forty_history_section_groups_record_at_most_the_bound_of_requests_as_started_at_once(
    tmp_path: Path, mode: str
) -> None:
    # Arrange
    names = [f"notes_{number}" for number in range(40)]
    history = History(sections={name: f"note {name}" for name in names})
    journal = OutstandingJournal(tmp_path / "journal.jsonl")
    judge = _bounded_judge(mode, journal)

    # Act
    if mode == "sync":
        judged = judge_sections(judge, history, _section_checks(names))
    else:
        judged = asyncio.run(judge_sections_async(judge, history, _section_checks(names)))

    # Assert
    assert set(judged) == set(names)
    assert journal.peak == BOUND


@pytest.mark.parametrize(
    "interrupted_start", [0, 1], ids=["first start interrupted", "second start interrupted"]
)
def test_ctrl_c_in_a_round_wider_than_the_bound_resumes_to_the_uninterrupted_result(
    tmp_path: Path, interrupted_start: int
) -> None:
    # Arrange: six starts asked two at a time, so four wait in the pool when one of the two requests
    # in flight is interrupted. No step limit: steps are a fresh allowance per run, so every run goes
    # to its end.
    index = many_functions_index(tmp_path, 6)
    starts = function_starts(index, 6)
    budget = SearchBudget(beam_width=6, neighbours_per_kind=2)
    whole = ScriptedJevClient(default_noul=0.05)
    uninterrupted = find_code(index, Judge(whole, max_concurrency=2), TARGET, starts, budget=budget)
    opened = starts[interrupted_start].open()
    stopping = InterruptedAtOnePlace((opened.span.file, f"{opened.span.start}-{opened.span.end}"))
    resuming = ScriptedJevClient(default_noul=0.05)

    # Act
    stopped = find_code(index, Judge(stopping, max_concurrency=2), TARGET, starts, budget=budget)
    resumed = find_code(
        index, Judge(resuming, max_concurrency=2), TARGET, starts, budget=budget, resume=stopped
    )

    # Assert: a round fills its beam with low-scored neighbours too, and an interrupt changes which ones
    # share a round, so the resume may open other low-scored ones; each it leaves is listed as such.
    # The interrupted place is asked again, though not always in the same bytes: a neighbour an
    # answered request already queued is not offered beside it twice.
    starts_left = {start.key for start in starts} - stopped.visited
    assert stopped.outcome == Outcome.CANCELLED
    assert len(stopping.received) < 6
    assert {entry.place_key for entry in stopped.not_inspected if entry.reason == "cancelled"} >= starts_left
    assert resumed.outcome == uninterrupted.outcome
    assert {start.key for start in starts} <= resumed.visited
    left_unopened = {entry.place_key for entry in resumed.not_inspected if entry.reason == "deprioritized"}
    assert uninterrupted.visited - resumed.visited <= left_unopened
    assert stopping.interrupted is not None
    assert not set(_hashes(stopping.answered)) & set(_hashes(resuming.requests))
    assert stopping.interrupted_place in [_slice_of(state) for state, _questions in resuming.requests]
