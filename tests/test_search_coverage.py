from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from conftest import BudgetedClient

from jev_navigator.directives.find_all import (
    DELIVERED,
    ITEMS,
    NOT_REACHED,
    REFUSED,
    TOO_LARGE,
    find_all,
    match_check,
)
from jev_navigator.directives.frontier import Source
from jev_navigator.directives.search_coverage import Outcome, PointCoverage, Round, point_results
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import LineAnchor, RangeAnchor
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import ScriptedJevClient

SHOP = {
    "rules.py": (
        "def accept(order):\n    return len(order.items) <= 4\n\n\ndef reject(order):\n    return order\n"
    ),
    "api.py": "def submit(order):\n    return order\n\n\ndef cancel(order):\n    return order\n",
}
POINTS = {
    "limit": "the check that limits the items of an order",
    "audit": "the code that writes an audit entry",
}
AUDIT = {"audit": POINTS["audit"]}
BAR = 0.8


def shop(root: Path, files: Mapping[str, str] = SHOP) -> CodeIndex:
    for name, source in files.items():
        (root / name).write_text(source)
    return CodeIndex(root, files)


def answering() -> ScriptedJevClient:
    """P(yes) 0.9 for the limit point on the code holding the limit, else 0.1."""

    def answer(question_id: str, _question: Mapping, state: Mapping) -> float:
        code = state[ITEMS][int(question_id.rsplit("#", 1)[1])]["code"]
        limit = question_id.split("@", 1)[0] == match_check("limit").name
        return 0.9 if limit and "<= 4" in code else 0.1

    return ScriptedJevClient(nouls=answer)


class FailsAfter:
    """A provider that answers its first ``answered`` requests and fails every later one."""

    def __init__(self, answered: int) -> None:
        self.scripted = answering()
        self.model = self.scripted.model
        self.answered = answered
        self.requests = 0

    def ask(self, state, questions):
        self.requests += 1
        if self.requests > self.answered:
            raise ConnectionError("provider unreachable")
        return self.scripted.ask(state, questions)


def one_at_a_time(client, **settings) -> Judge:
    """A Judge sending one unit per request, one request at a time, in file-and-lines order."""
    return Judge(client, items_per_request=1, max_concurrency=1, **settings)


def test_a_point_at_the_bar_is_found_and_one_below_it_is_none_among_the_units_judged(tmp_path: Path) -> None:
    # Arrange: the bar is exactly the limit's best answer
    index = shop(tmp_path)
    bar = 0.9

    # Act
    result = find_all(index, Judge(answering()), POINTS, files=index.files)
    limit, audit = point_results([Round(result)], bar)

    # Assert
    assert (limit.outcome, audit.outcome) == (Outcome.FOUND, Outcome.NONE_AMONG_JUDGED)
    assert limit.render(bar) == "limit: found, best P=0.900 of 4 unit(s) judged; 0 not reached"
    assert audit.render(bar) == (
        "audit: none at the bar 0.90 among 4 unit(s) judged, best P=0.100; 0 not reached (not negative proof)"
    )


def test_units_a_spent_budget_leaves_unjudged_are_counted_by_the_source_that_listed_them(
    tmp_path: Path,
) -> None:
    # Arrange: one wave lists all four units; its one call judges the first in file order, api.py's submit
    index = shop(tmp_path)
    judge = one_at_a_time(answering(), max_calls=1)

    # Act
    result = find_all(
        index, judge, AUDIT, anchors=[LineAnchor("rules.py", 1)], files=["api.py"], names=["reject"]
    )
    [audit] = point_results([Round(result)], BAR)

    # Assert
    entered = {result_unit.symbol: result.entered_by[result_unit.id] for result_unit in result.units}
    assert entered == {
        "accept": Source.ANCHOR,
        "submit": Source.FILE,
        "cancel": Source.FILE,
        "reject": Source.NAME,
    }
    assert (audit.coverage.considered, audit.coverage.judged) == (4, 1)
    assert audit.render(BAR) == (
        "audit: none at the bar 0.80 among 1 unit(s) judged, best P=0.100; 3 not reached "
        "(1 from anchors, 1 from files, 1 from name hits) (not negative proof)"
    )


def test_a_unit_two_sources_list_counts_under_the_first(tmp_path: Path) -> None:
    # Arrange: the anchor names accept, and the file lists accept again and reject
    index = shop(tmp_path)

    # Act
    result = find_all(
        index,
        Judge(answering(), max_calls=0),
        AUDIT,
        anchors=[LineAnchor("rules.py", 1)],
        files=["rules.py"],
    )
    [audit] = point_results([Round(result)], BAR)

    # Assert
    assert audit.render(BAR) == "audit: unknown, no unit judged; 2 not reached (1 from anchors, 1 from files)"


def test_a_search_that_failed_after_judging_some_units_is_unknown_never_none(tmp_path: Path) -> None:
    # Arrange
    index = shop(tmp_path)

    # Act
    result = find_all(index, one_at_a_time(FailsAfter(1)), AUDIT, files=index.files, batches_per_wave=1)
    [audit] = point_results([Round(result)], BAR)

    # Assert
    assert (result.stopped_by, audit.outcome, audit.coverage.judged) == ("failed", Outcome.UNKNOWN, 1)
    assert audit.render(BAR) == (
        "audit: unknown, the search failed (ConnectionError: provider unreachable) after 1 unit(s) judged, "
        "best P=0.100; 3 not reached"
    )


def test_a_search_that_failed_before_judging_any_unit_says_so(tmp_path: Path) -> None:
    # Arrange
    index = shop(tmp_path)

    # Act
    result = find_all(index, one_at_a_time(FailsAfter(0)), AUDIT, files=["rules.py"], batches_per_wave=1)
    [audit] = point_results([Round(result)], BAR)

    # Assert
    assert (audit.outcome, audit.best) == (Outcome.UNKNOWN, None)
    assert audit.render(BAR) == (
        "audit: unknown, the search failed (ConnectionError: provider unreachable) "
        "before any unit was judged; 2 not reached"
    )


def test_a_point_found_before_the_search_failed_stays_found(tmp_path: Path) -> None:
    # Arrange: rules.py's accept, which holds the limit, is answered; the request for reject fails
    index = shop(tmp_path)

    # Act
    result = find_all(index, one_at_a_time(FailsAfter(1)), POINTS, files=["rules.py"], batches_per_wave=1)
    limit, audit = point_results([Round(result)], BAR)

    # Assert
    assert (result.stopped_by, limit.outcome, audit.outcome) == ("failed", Outcome.FOUND, Outcome.UNKNOWN)
    assert limit.render(BAR) == "limit: found, best P=0.900 of 1 unit(s) judged; 1 not reached"


def test_a_point_no_unit_was_judged_for_is_unknown(tmp_path: Path) -> None:
    # Arrange
    index = shop(tmp_path)

    # Act
    result = find_all(index, Judge(answering(), max_calls=0), AUDIT, files=index.files)
    [audit] = point_results([Round(result)], BAR)

    # Assert
    assert (audit.outcome, audit.best) == (Outcome.UNKNOWN, None)
    assert audit.render(BAR) == "audit: unknown, no unit judged; 4 not reached"


def test_a_caller_names_the_source_of_a_round_it_composed(tmp_path: Path) -> None:
    # Arrange: a first round over one file and a callee round the caller fed as anchors, neither paid
    index = shop(tmp_path)
    first = find_all(index, Judge(answering(), max_calls=0), AUDIT, files=["rules.py"])
    callees = find_all(
        index,
        Judge(answering(), max_calls=0),
        AUDIT,
        anchors=[LineAnchor("api.py", 1), LineAnchor("api.py", 5)],
    )

    # Act
    [audit] = point_results([Round(first), Round(callees, Source.CALLEE)], BAR)

    # Assert
    assert audit.coverage.cut == {(Source.FILE, NOT_REACHED): 2, (Source.CALLEE, NOT_REACHED): 2}
    assert audit.render(BAR) == "audit: unknown, no unit judged; 4 not reached (2 from files, 2 callees)"


def test_a_refused_unit_is_counted_refused_not_unreached(tmp_path: Path) -> None:
    # Arrange: the provider's budget is below the box the Judge measures, so it refuses the big unit
    big = "def big():\n    return '" + "x" * 3_000 + "'\n"
    index = shop(tmp_path, {**SHOP, "big.py": big})

    # Act
    result = find_all(index, Judge(BudgetedClient(2_500, default_noul=0.1)), AUDIT, files=index.files)
    [audit] = point_results([Round(result)], BAR)

    # Assert
    assert audit.coverage.cut == {(Source.FILE, REFUSED): 1}
    assert audit.render(BAR) == (
        "audit: none at the bar 0.80 among 4 unit(s) judged, best P=0.100; 0 not reached; 1 refused "
        "(not negative proof)"
    )


def test_a_unit_judged_in_an_earlier_round_is_not_cut_by_a_later_round_listing_it_again(
    tmp_path: Path,
) -> None:
    # Arrange: the first round judges both of rules.py's units; a second, unpaid round lists accept again
    index = shop(tmp_path)
    first = find_all(index, Judge(answering()), AUDIT, files=["rules.py"])
    again = find_all(index, Judge(answering(), max_calls=0), AUDIT, anchors=[LineAnchor("rules.py", 1)])

    # Act
    [audit] = point_results([Round(first), Round(again, Source.CALLEE)], BAR)

    # Assert
    assert audit.coverage == PointCoverage(considered=2, judged=2, cut={})
    assert audit.render(BAR) == (
        "audit: none at the bar 0.80 among 2 unit(s) judged, best P=0.100; 0 not reached (not negative proof)"
    )


def test_a_unit_two_rounds_leave_unjudged_counts_once_under_its_first_source_and_its_weightiest_reason(
    tmp_path: Path,
) -> None:
    # Arrange: neither round is paid; the second lists accept again with its lines already delivered
    index = shop(tmp_path)
    first = find_all(index, Judge(answering(), max_calls=0), AUDIT, files=["rules.py"])
    again = find_all(
        index,
        Judge(answering(), max_calls=0),
        AUDIT,
        anchors=[LineAnchor("rules.py", 1)],
        delivered=[RangeAnchor("rules.py", 1, 2)],
    )

    # Act
    [audit] = point_results([Round(first), Round(again, Source.CALLEE)], BAR)

    # Assert
    assert list(again.not_judged.values()) == [DELIVERED]
    assert audit.coverage == PointCoverage(considered=2, judged=0, cut={(Source.FILE, NOT_REACHED): 2})


def test_a_unit_too_large_for_one_request_is_counted_apart_from_the_units_not_reached(tmp_path: Path) -> None:
    # Arrange: one line longer than any request the Judge can send
    big = "def big():\n    return '" + "x" * 400_000 + "'\n"
    index = shop(tmp_path, {**SHOP, "big.py": big})

    # Act
    result = find_all(index, Judge(answering()), AUDIT, files=index.files)
    [audit] = point_results([Round(result)], BAR)

    # Assert
    assert audit.coverage.cut == {(Source.FILE, TOO_LARGE): 1}
    assert audit.render(BAR) == (
        "audit: none at the bar 0.80 among 4 unit(s) judged, best P=0.100; 0 not reached; "
        "1 too large to judge (not negative proof)"
    )


def test_a_later_round_that_failed_makes_the_point_unknown(tmp_path: Path) -> None:
    # Arrange: the first round judges rules.py's two units; the callee round over api.py fails at once
    index = shop(tmp_path)
    first = find_all(index, Judge(answering()), AUDIT, files=["rules.py"])
    callees = find_all(index, one_at_a_time(FailsAfter(0)), AUDIT, files=["api.py"], batches_per_wave=1)

    # Act
    [audit] = point_results([Round(first), Round(callees, Source.CALLEE)], BAR)

    # Assert
    assert audit.outcome is Outcome.UNKNOWN
    assert audit.render(BAR) == (
        "audit: unknown, the search failed (ConnectionError: provider unreachable) after 2 unit(s) judged, "
        "best P=0.100; 2 not reached"
    )
