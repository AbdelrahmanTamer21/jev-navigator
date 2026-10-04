"""Every pool that asks Jev in parallel creates at most ``judge.max_concurrency`` workers, whatever the
beam width or the number of history sections, so nested asks stay bounded at that number squared."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.find_code import SearchBudget, find_code, find_code_async
from jev_navigator.directives.places import function_place
from jev_navigator.history import History, HistoryCheck, judge_sections, judge_sections_async
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.client import InputBudgetExceededError
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.testing import ScriptedJevClient

TARGET = "the check that limits how many items an order may have"
BOUND = 4
HOLD_SECONDS = 0.05
DEADLINE_SECONDS = 60
HELPERS = "\n\n".join(f"def h{number}(order):\n    return {number}" for number in range(4)) + "\n"


class CountingClient:
    """Answers like ``ScriptedJevClient``, holding each request ``HOLD_SECONDS`` so concurrent ones
    overlap, and records the most requests in flight and the most threads alive at once. With
    ``refuse_lists`` it refuses for its size any request carrying more than one candidate, which splits
    every opening into its found question and one batch per neighbour."""

    model = "jev-scripted"

    def __init__(self, refuse_lists: bool = False) -> None:
        self.script = ScriptedJevClient(default_noul=0.05)
        self.refuse_lists = refuse_lists
        self.in_flight = 0
        self.peak_in_flight = 0
        self.peak_threads = 0
        self._lock = threading.Lock()

    def ask(self, state: Mapping, questions: Mapping):
        self._arrive()
        try:
            time.sleep(HOLD_SECONDS)
            return self._answer(state, questions)
        finally:
            self._leave()

    def _arrive(self) -> None:
        with self._lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.peak_threads = max(self.peak_threads, threading.active_count())

    def _leave(self) -> None:
        with self._lock:
            self.in_flight -= 1

    def _answer(self, state: Mapping, questions: Mapping):
        if self.refuse_lists and len(state.get("candidates", ())) > 1:
            raise InputBudgetExceededError("TypeSafeBadRequestError: 400 max_tokens_exceeded")
        return self.script.ask(state, questions)


class AsyncCountingClient:
    """``CountingClient`` with an awaitable ask that holds each request on the event loop."""

    model = "jev-scripted"

    def __init__(self, counting: CountingClient) -> None:
        self.counting = counting

    async def ask(self, state: Mapping, questions: Mapping):
        self.counting._arrive()
        try:
            await asyncio.sleep(HOLD_SECONDS)
            return self.counting._answer(state, questions)
        finally:
            self.counting._leave()


def _finished_within_the_deadline(call: Callable[[], object]) -> object:
    """``call()`` on a thread of its own, failing the test if it has not finished by the deadline."""
    outcome: list[object] = []
    worker = threading.Thread(target=lambda: outcome.append(call()), daemon=True)
    worker.start()
    worker.join(DEADLINE_SECONDS)
    assert not worker.is_alive(), f"the search did not finish within {DEADLINE_SECONDS} s"
    return outcome[0]


def _functions(tmp_path: Path, count: int) -> CodeIndex:
    """``count`` functions, one per file, each calling two shared helpers, all parsed up front."""
    files = {"app/__init__.py": "", "app/helpers.py": HELPERS}
    for number in range(count):
        first, second = number % 4, (number + 1) % 4
        files[f"app/f{number}.py"] = (
            f"from app.helpers import h{first}, h{second}\n\n\n"
            f"def f{number}(order):\n    return h{first}(order) + h{second}(order) + {number}\n"
        )
    commit_files(tmp_path / "repo", files)
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=tmp_path / "facts")
    assert not index.unparsed_files
    return index


def _beam_of(index: CodeIndex, width: int):
    starts = [
        function_place(index, index.enclosing_symbol(f"app/f{number}.py", 5)) for number in range(width)
    ]
    budget = SearchBudget(beam_width=width, max_steps=width, neighbours_per_kind=2)
    return starts, budget


def test_a_beam_of_two_hundred_with_nested_batches_keeps_its_threads_bounded_and_finishes(
    tmp_path: Path,
) -> None:
    # Arrange
    index = _functions(tmp_path, 200)
    starts, budget = _beam_of(index, 200)
    client = CountingClient(refuse_lists=True)
    judge = Judge(client, items_per_request=1, max_concurrency=BOUND)
    threads_before = threading.active_count()

    # Act
    result = _finished_within_the_deadline(lambda: find_code(index, judge, TARGET, starts, budget=budget))

    # Assert
    assert result.steps == 200
    assert client.peak_threads - threads_before <= 1 + BOUND + BOUND * BOUND
    assert client.peak_in_flight <= BOUND * BOUND


def test_an_async_beam_of_two_hundred_with_nested_batches_stays_bounded_and_finishes(tmp_path: Path) -> None:
    # Arrange
    index = _functions(tmp_path, 200)
    starts, budget = _beam_of(index, 200)
    client = CountingClient(refuse_lists=True)
    judge = Judge(AsyncCountingClient(client), items_per_request=1, max_concurrency=BOUND)

    # Act
    result = _finished_within_the_deadline(
        lambda: asyncio.run(find_code_async(index, judge, TARGET, starts, budget=budget))
    )

    # Assert
    assert result.steps == 200
    assert client.peak_in_flight <= BOUND * BOUND


@pytest.mark.parametrize("entry", ["sync", "async"])
def test_forty_history_section_groups_are_asked_at_most_the_bound_at_once(entry: str) -> None:
    # Arrange: forty checks, each over a section of its own, so each is its own group and request.
    names = [f"notes_{number}" for number in range(40)]
    history = History(sections={name: f"note {name}" for name in names})
    checks = {
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
    client = CountingClient()

    # Act
    if entry == "sync":
        judged = judge_sections(Judge(client, max_concurrency=BOUND), history, checks)
    else:
        judge = Judge(AsyncCountingClient(client), max_concurrency=BOUND)
        judged = asyncio.run(judge_sections_async(judge, history, checks))

    # Assert
    assert set(judged) == set(names)
    assert client.peak_in_flight == BOUND
