from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import BudgetedClient

from jev_navigator.directives.find_all import (
    ITEMS,
    NOT_REACHED,
    SKIPPED,
    TARGETS,
    TOO_LARGE,
    NameHits,
    find_all,
    find_all_async,
    match_check,
)
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import UNSUPPORTED_LANGUAGE, RangeAnchor
from jev_navigator.judgments.client import JEV_INPUT_LIMITS, InputBudgetExceededError, InputLimits
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import serialized_chars
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient

LIMIT = {"limit": "the check that limits the items of an order"}
ORDERS = {
    "rules.py": "LIMIT = 4\n\n\ndef accept(order):\n    return len(order.items) <= LIMIT\n",
    "api.py": (
        "from rules import accept\n\n\ndef submit(order):\n    audit(order)\n    return accept(order)\n"
    ),
    "web.ts": "export function capacity(order: {items: unknown[]}) { return order.items.length <= 4; }\n",
    "notes.md": "accept orders of up to four items\n",
}


def repository(root: Path, files: Mapping[str, str] = ORDERS) -> CodeIndex:
    for name, source in files.items():
        (root / name).write_text(source)
    return CodeIndex(root, files)


def labelled(labels: Mapping[tuple[str, str], float], default: float = 0.05) -> ScriptedJevClient:
    """A provider fixture: P(yes) by target and a marker in the unit's code. The real listing,
    resolution, batching and composition run; this proves execution, not Jev's accuracy."""

    def answer(question_id: str, question: Mapping, state: Mapping) -> float:
        code = state[ITEMS][int(question_id.rsplit("#", 1)[1])]["code"]
        asked = question_id.split("@", 1)[0]
        matches = (
            p
            for (target, marker), p in labels.items()
            if match_check(target).name == asked and marker in code
        )
        return next(matches, default)

    return ScriptedJevClient(nouls=answer)


def sent_code(provider: ScriptedJevClient) -> list[str]:
    return [item["code"] for state, _ in provider.requests for item in state[ITEMS]]


def test_every_unit_is_asked_every_target_in_one_request_with_only_its_file_and_code(tmp_path: Path) -> None:
    # Arrange
    index = repository(tmp_path)
    targets = {**LIMIT, "audit": "the call that records an audit entry"}
    provider = labelled(
        {("limit", "<= LIMIT"): 0.95, ("limit", "length <= 4"): 0.93, ("audit", "audit(order)"): 0.92}
    )

    # Act
    result = find_all(index, Judge(provider), targets, files=index.files)

    # Assert
    [(state, questions)] = provider.requests
    assert state[TARGETS] == targets
    assert all(sorted(item) == ["code", "file"] for item in state[ITEMS])
    assert len(questions) == 2 * len(state[ITEMS]) == 2 * len(result.units)
    assert {score.unit.symbol for score in result.scores("limit") if score.probability > 0.9} == {
        "accept",
        "capacity",
    }
    assert {score.unit.symbol for score in result.scores("audit") if score.probability > 0.9} == {"submit"}
    assert result.unlisted == {"notes.md": UNSUPPORTED_LANGUAGE}
    assert result.coverage == "scope_incomplete"


def test_at_one_batch_per_wave_the_units_anchors_name_are_judged_before_the_units_of_the_files(
    tmp_path: Path,
) -> None:
    # Arrange
    index = repository(tmp_path)
    provider = labelled({})

    # Act
    result = find_all(
        index,
        Judge(provider, max_calls=1, items_per_request=1),
        LIMIT,
        files=index.files,
        anchors=[RangeAnchor("web.ts", 1, 1)],
        batches_per_wave=1,
    )

    # Assert
    assert [score.unit.path for score in result.scores("limit")] == ["web.ts"]
    assert result.stopped_by == "budget"
    assert result.coverage == "partial"
    assert sorted(result.not_judged.values()) == [NOT_REACHED] * 3


WAVE_ORDER = {
    "a.py": "def f1(x):\n    return x\n\n\ndef f2(x):\n    return x\n",
    "b.py": "def g(x):\n    return x\n",
    "z.py": "def h(x):\n    return x\n",
}


def test_a_wave_holds_the_next_places_in_population_order_whatever_their_files(tmp_path: Path) -> None:
    # Arrange: one item per request, two requests per wave, and a cap of one wave
    index = repository(tmp_path, WAVE_ORDER)

    # Act
    result = find_all(
        index,
        Judge(labelled({}), max_calls=2, items_per_request=1),
        LIMIT,
        files=["a.py", "b.py"],
        anchors=[RangeAnchor("z.py", 1, 2)],
        batches_per_wave=2,
    )

    # Assert: the anchored h and the first file's f1 make the first wave; f2 and g wait for the next
    assert {score.unit.symbol for score in result.scores("limit")} == {"h", "f1"}
    assert sorted(result.not_judged.values()) == [NOT_REACHED] * 2
    assert result.batches_per_wave == 2


RARE_AND_COMMON = {
    "a.py": "def first(x):\n    return common(x)\n\n\ndef second(x):\n    return common(common(x))\n",
    "b.py": "def holder(x):\n    rare_token(x)\n    return common(x)\n",
    "c.md": "rare_token is documented here\n",
}


def test_at_one_batch_per_wave_the_hits_of_the_rarest_name_are_judged_first(tmp_path: Path) -> None:
    # Arrange
    index = repository(tmp_path, RARE_AND_COMMON)

    # Act
    result = find_all(
        index,
        Judge(labelled({}), max_calls=1, items_per_request=1),
        LIMIT,
        names=["common", "rare_token"],
        batches_per_wave=1,
    )

    # Assert
    assert [score.unit.symbol for score in result.scores("limit")] == ["holder"]
    assert result.names == {"rare_token": NameHits(2, 2, 1), "common": NameHits(3, 1, 0)}
    assert result.stopped_by == "budget"


MANY_HITS = {
    "a.py": "def holder(x):\n    tok(x)\n    tok(x)\n    tok(x)\n\n\ndef other(x):\n    return tok(x)\n"
}


def test_a_unit_many_hits_reach_is_judged_once_even_when_its_hits_fall_in_different_chunks(
    tmp_path: Path,
) -> None:
    # Arrange: two hits per chunk, so holder's third hit comes in the second chunk before any request
    index = repository(tmp_path, MANY_HITS)
    provider = labelled({})

    # Act
    result = find_all(index, Judge(provider, items_per_request=2), LIMIT, names=["tok"])

    # Assert
    assert len(sent_code(provider)) == len(set(sent_code(provider))) == 2
    assert {score.unit.symbol for score in result.scores("limit")} == {"holder", "other"}
    assert result.names == {"tok": NameHits(4, 4, 0)}
    assert result.coverage == "units_examined"


def test_a_hit_inside_a_nested_function_names_the_listed_function_holding_it(tmp_path: Path) -> None:
    # Arrange
    nested = {"outer.py": "def outer(x):\n    def inner(y):\n        return marker(y)\n    return inner(x)\n"}
    index = repository(tmp_path, nested)

    # Act
    result = find_all(index, Judge(labelled({})), LIMIT, names=["marker"])

    # Assert
    assert [score.unit.symbol for score in result.scores("limit")] == ["outer"]


def test_a_unit_the_caller_skips_is_named_and_never_sent(tmp_path: Path) -> None:
    # Arrange
    index = repository(tmp_path)
    provider = labelled({})

    # Act
    result = find_all(
        index, Judge(provider), LIMIT, files=index.files, skip=lambda unit: unit.symbol == "accept"
    )

    # Assert
    [accept] = [unit for unit in result.units if unit.symbol == "accept"]
    assert result.not_judged == {accept.id: SKIPPED}
    assert not any("<= LIMIT" in code for code in sent_code(provider))


@dataclass
class SmallBoxClient(ScriptedJevClient):
    input_limits: InputLimits = InputLimits(1500, None)


def test_a_unit_larger_than_its_room_is_judged_by_its_pieces_and_scored_by_the_best(tmp_path: Path) -> None:
    # Arrange: a function of 130 lines; line 101 sets x99
    body = "".join(f"    x{number} = {number}\n" for number in range(129))
    index = repository(tmp_path, {"big.py": f"def big():\n{body}"})
    provider = SmallBoxClient(nouls=labelled({("limit", "x99 = 99"): 0.95}).nouls)

    # Act
    result = find_all(index, Judge(provider), LIMIT, files=index.files)

    # Assert
    [score] = result.scores("limit")
    assert len(score.unit.pieces) == 3
    assert (score.piece.start, score.piece.end, score.probability) == (61, 120, 0.95)
    assert result.coverage == "units_examined"


def sized_function(chars: int) -> str:
    """A function whose text, as a request spells it, is ``chars`` long."""
    empty = 'def big():\n    return ""'
    return empty.replace('""', '"' + "x" * (chars - serialized_chars(empty)) + '"') + "\n"


def test_a_unit_of_exactly_its_room_is_judged_whole_and_one_character_more_is_too_large(
    tmp_path: Path,
) -> None:
    # Arrange: the room depends on the paths and targets only, so a first search measures it
    room = find_all(
        repository(tmp_path, {"big.py": "def big():\n    return 1\n"}), Judge(labelled({})), LIMIT
    ).room
    whole = BudgetedClient(JEV_INPUT_LIMITS.request_chars, input_box=JEV_INPUT_LIMITS.box_chars)
    over = BudgetedClient(JEV_INPUT_LIMITS.request_chars, input_box=JEV_INPUT_LIMITS.box_chars)
    at_room = repository(tmp_path, {"big.py": sized_function(room)})

    # Act
    fits = find_all(at_room, Judge(whole), LIMIT, files=at_room.files)
    past_room = repository(tmp_path, {"big.py": sized_function(room + 1)})
    too_large = find_all(past_room, Judge(over), LIMIT, files=past_room.files)

    # Assert
    [score] = fits.scores("limit")
    assert (score.piece, whole.refusals, len(whole.requests)) == (None, 0, 1)
    [unit] = too_large.units
    assert too_large.not_judged == {unit.piece_id(unit.pieces[0]): TOO_LARGE}
    assert over.requests == []
    entry = {"file": "big.py", "code": sized_function(room + 1).removesuffix("\n")}
    with pytest.raises(InputBudgetExceededError):
        Judge(over).check_each(match_check("limit"), [entry], {TARGETS: LIMIT})


def test_a_budget_stop_resumes_from_its_answers_without_asking_them_again(tmp_path: Path) -> None:
    # Arrange
    index = repository(tmp_path)
    stopped = find_all(index, Judge(labelled({}), max_calls=1, items_per_request=2), LIMIT, files=index.files)
    provider = labelled({})

    # Act
    resumed = find_all(
        index, Judge(provider, items_per_request=2), LIMIT, files=index.files, completed=stopped.judged
    )

    # Assert
    assert (stopped.stopped_by, len(stopped.judged["limit"])) == ("budget", 2)
    assert not set(sent_code(provider)) & {answer.item["code"] for answer in stopped.judged["limit"]}
    assert (resumed.stopped_by, len(resumed.judged["limit"]), len(resumed.scores("limit"))) == (
        "scope_examined",
        4,
        4,
    )


def test_cancelled_before_the_search_parses_nothing_and_sends_nothing(tmp_path: Path) -> None:
    # Arrange
    index = repository(tmp_path)
    provider = labelled({})

    # Act
    result = find_all(
        index,
        Judge(provider),
        LIMIT,
        files=index.files,
        anchors=[RangeAnchor("rules.py", 1, 5)],
        names=["accept"],
        cancelled=lambda: True,
    )

    # Assert
    assert (result.stopped_by, result.coverage, result.units) == ("cancelled", "partial", ())
    assert index.parser_scans_pending == ("facts",)
    assert provider.requests == []


def test_a_target_name_must_be_an_identifier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identifiers"):
        find_all(repository(tmp_path), Judge(labelled({})), {"item limit": "the item limit"})


class ReleasesInOrder:
    """A provider that holds a wave's requests until all of them arrived, then answers them one at a
    time, by the number of the first function in the request, ascending or descending, so the judge
    sees them complete in a chosen order."""

    def __init__(self, expected: int, *, descending: bool) -> None:
        self.script = ScriptedJevClient(default_noul=0.05)
        self.expected = expected
        self.descending = descending
        self.arrived: list[int] = []
        self.answered: list[int] = []
        self.turns = threading.Condition()

    @property
    def model(self) -> str:
        return self.script.model

    def send(self, state, questions):
        first = int(state[ITEMS][0]["code"].split("check_", 1)[1].split("(", 1)[0])
        with self.turns:
            self.arrived.append(first)
            self.turns.notify_all()
            self.turns.wait_for(lambda: len(self.arrived) >= self.expected, timeout=10)
            order = sorted(self.arrived, reverse=self.descending)
            self.turns.wait_for(lambda: order[len(self.answered)] == first, timeout=10)
            self.answered.append(first)
            self.turns.notify_all()
        return self.script.send(state, questions)

    def parse(self, raw):
        return self.script.parse(raw)


def test_answers_are_listed_in_file_and_line_order_however_they_arrive(tmp_path: Path) -> None:
    # Arrange: forty functions in ten batches, answered in line order in one run and reversed in the other
    source = "".join(
        f"def check_{number}(items):\n    return len(items) <= {number}\n\n" for number in range(40)
    )
    index = repository(tmp_path, {"checks.py": source})
    runs = []

    # Act
    for descending in (False, True):
        provider = ReleasesInOrder(10, descending=descending)
        result = find_all(index, Judge(provider, items_per_request=4), LIMIT, files=index.files)
        runs.append(([answer.place.id for answer in result.judged["limit"]], provider.answered))

    # Assert: the providers answered in opposite orders, and both results list the same order
    (forward, forward_answered), (backward, backward_answered) = runs
    assert forward_answered == sorted(forward_answered)
    assert backward_answered == sorted(backward_answered, reverse=True)
    assert forward == backward == [unit.id for unit in result.units]
    assert [unit.start for unit in result.units] == sorted(unit.start for unit in result.units)


class CancelsAsItFails:
    """Answers its first request, then fails the second while the host asks to cancel, as a host does
    when it stops on the first error it sees."""

    def __init__(self, error: Exception) -> None:
        self.script = ScriptedJevClient(default_noul=0.96)
        self.model = self.script.model
        self.error = error
        self.cancel_requested = False
        self.requests = 0

    def ask(self, state, questions):
        self.requests += 1
        if self.requests == 2:
            self.cancel_requested = True
            raise self.error
        return self.script.ask(state, questions)


def test_a_failed_request_stays_failed_when_the_host_cancels_at_the_same_time(tmp_path: Path) -> None:
    # Arrange
    index = repository(tmp_path)
    error = RuntimeError("Jev answered 503")
    provider = CancelsAsItFails(error)

    # Act
    result = find_all(
        index,
        Judge(provider, items_per_request=1),
        LIMIT,
        files=index.files,
        cancelled=lambda: provider.cancel_requested,
    )

    # Assert
    assert result.stopped_by == "failed"
    assert result.failure is error
    assert len(result.judged["limit"]) == provider.requests - 1


def judged_shape(result) -> tuple:
    """Everything a search decided: each answer's place and probability, what it left unjudged and
    why, each name's hits, the units and the stop."""
    answers = [(answer.place.id, answer.probability) for answer in result.judged["limit"]]
    return answers, result.not_judged, result.names, [unit.id for unit in result.units], result.stopped_by


def test_find_all_async_judges_exactly_what_find_all_judges(tmp_path: Path) -> None:
    # Arrange
    index = repository(tmp_path, {**ORDERS, **RARE_AND_COMMON})
    labels = {("limit", "<= LIMIT"): 0.95, ("limit", "common(x)"): 0.6}
    population = {
        "files": ["rules.py", "api.py", "web.ts", "notes.md"],
        "anchors": [RangeAnchor("web.ts", 1, 1)],
        "names": ["common", "rare_token"],
        "batches_per_wave": 2,
    }

    # Act
    sync = find_all(index, Judge(labelled(labels), items_per_request=2), LIMIT, **population)
    concurrent = asyncio.run(
        find_all_async(
            index, Judge(AsyncScriptedJevClient(labelled(labels)), items_per_request=2), LIMIT, **population
        )
    )

    # Assert
    assert judged_shape(concurrent) == judged_shape(sync)
    assert len(sync.judged["limit"]) == 7


class FailsSecondRequest:
    """An async provider that answers its first request and fails its second."""

    def __init__(self, error: Exception) -> None:
        self.script = ScriptedJevClient(default_noul=0.5)
        self.error = error
        self.requests = 0

    @property
    def model(self) -> str:
        return self.script.model

    async def ask(self, state, questions):
        await asyncio.sleep(0)
        self.requests += 1
        if self.requests == 2:
            raise self.error
        return self.script.ask(state, questions)


def test_find_all_async_keeps_the_answers_a_wave_got_before_its_failed_request(tmp_path: Path) -> None:
    # Arrange: one wave of four requests, sent one at a time, the second of them failing
    index = repository(tmp_path)
    error = RuntimeError("Jev answered 503")

    # Act
    result = asyncio.run(
        find_all_async(
            index,
            Judge(FailsSecondRequest(error), items_per_request=1, max_concurrency=1),
            LIMIT,
            files=index.files,
            batches_per_wave=4,
        )
    )

    # Assert
    assert (result.stopped_by, result.failure) == ("failed", error)
    assert len(result.judged["limit"]) == 1


def test_find_all_async_reads_cancellation_before_it_parses_anything(tmp_path: Path) -> None:
    # Arrange
    index = repository(tmp_path)
    provider = AsyncScriptedJevClient(labelled({}))

    # Act
    result = asyncio.run(
        find_all_async(index, Judge(provider), LIMIT, files=index.files, cancelled=lambda: True)
    )

    # Assert
    assert (result.stopped_by, result.units) == ("cancelled", ())
    assert index.parser_scans_pending == ("facts",)
    assert provider.requests == []


class CancelsDuringFirstRequest:
    """An async provider that answers every request, and asks the host to cancel during its first."""

    def __init__(self) -> None:
        self.script = ScriptedJevClient(default_noul=0.5)
        self.cancel_requested = False

    @property
    def model(self) -> str:
        return self.script.model

    @property
    def requests(self) -> list:
        return self.script.requests

    async def ask(self, state, questions):
        await asyncio.sleep(0)
        self.cancel_requested = True
        return self.script.ask(state, questions)


def test_find_all_async_sends_no_further_wave_once_the_host_cancels(tmp_path: Path) -> None:
    # Arrange
    index = repository(tmp_path)
    provider = CancelsDuringFirstRequest()

    # Act
    result = asyncio.run(
        find_all_async(
            index,
            Judge(provider, items_per_request=1),
            LIMIT,
            files=index.files,
            cancelled=lambda: provider.cancel_requested,
            batches_per_wave=1,
        )
    )

    # Assert
    assert (result.stopped_by, len(provider.requests), len(result.judged["limit"])) == ("cancelled", 1, 1)
