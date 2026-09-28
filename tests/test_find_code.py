from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import replace

import pytest

from jev_navigator.directives.find_code import OPEN_FIRST, Outcome, SearchBudget, find_code
from jev_navigator.directives.places import (
    MOVES,
    Place,
    function_place,
    place_for_line,
)
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import CodeSlice, Span
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import ScriptedJevClient

TARGET = "the check that limits how many items an order may have"
_SLOT = re.compile(r"candidates\[(\d+)\]")


def scripted(
    found: Callable[[str], float], could_contain: Callable[[str], float]
) -> Callable[[str, Mapping, Mapping], float]:
    """Answers the found question from the opened code and each neighbour question from its signature."""

    def answer(question_id: str, question: Mapping, state: Mapping) -> float:
        slot = _SLOT.search(question["instructions"])
        if slot is None:
            return found(state["slice"]["code"])
        return could_contain(state["candidates"][int(slot.group(1))]["signature"])

    return answer


def start_at_place(index: CodeIndex) -> list[Place]:
    return [place_for_line(index, "app/orders.py", 6, "start")]


def test_search_follows_likely_neighbours_until_the_target_is_found(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.95 if "len(order.items) <= limit" in code else 0.05,
            could_contain=lambda signature: (
                0.9 if "check_limits" in signature or "validate_order" in signature else 0.1
            ),
        )
    )

    # Act
    result = find_code(
        sample_index, Judge(client), TARGET, start_at_place(sample_index), budget=SearchBudget(beam_width=1)
    )

    # Assert
    assert result.outcome == Outcome.FOUND
    assert result.found[0].code.span.name == "check_limits"
    assert [key.split(":")[0] for key in result.found[0].path] == [
        "app/orders.py",
        "app/validation.py",
        "app/validation.py",
    ]
    assert result.not_inspected
    assert result.steps == 3


def test_a_low_neighbour_score_keeps_the_neighbour_as_not_inspected(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))

    # Act
    result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index))

    # Assert
    assert result.outcome == Outcome.NOTHING_LEFT
    assert [visit.code.span.name for visit in result.searched] == ["place"]
    assert result.not_inspected and {entry.reason for entry in result.not_inspected} == {"deprioritized"}
    assert result.found == () and result.unsure == ()


def test_search_reports_unsure_only_when_only_unsure_places_remain(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.5, could_contain=lambda signature: 0.1))

    # Act
    result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index))

    # Assert
    assert result.outcome == Outcome.UNSURE_ONLY
    assert [visit.code.span.name for visit in result.unsure] == ["place"]


def test_search_stops_at_the_step_budget_and_lists_unopened_candidates(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.5))

    # Act
    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_steps=2, beam_width=1),
    )

    # Assert
    assert result.outcome == Outcome.BUDGET
    assert result.steps == 2
    assert result.not_inspected
    assert "budget" in {entry.reason for entry in result.not_inspected}


def test_a_beam_opens_several_places_per_round_as_separate_requests(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.9))

    # Act
    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_steps=4, beam_width=3),
    )

    # Assert
    assert result.steps == 4
    assert len(client.requests) == 4
    opened_keys = [f"{state['slice']['file']}:{state['slice']['lines']}" for state, _ in client.requests]
    assert len(set(opened_keys)) == 4


def test_the_same_place_reached_twice_is_opened_once(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))
    place = sample_index.enclosing_symbol("app/orders.py", 6)
    starts = [function_place(sample_index, place), place_for_line(sample_index, "app/orders.py", 7, "start")]

    # Act
    result = find_code(sample_index, Judge(client), TARGET, starts)

    # Assert
    assert result.steps == 1


def test_identical_code_under_two_places_is_judged_once(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))
    same_code = CodeSlice(Span("app/orders.py", 5, 7, "place"), "def place(self, order): ...")
    starts = [
        Place("first", "function", "first", lambda: same_code),
        Place("second", "function", "second", lambda: same_code),
    ]

    # Act
    result = find_code(sample_index, Judge(client), TARGET, starts)

    # Assert
    assert result.steps == 1
    assert len(client.requests) == 1


def test_jev_sees_only_the_target_the_opened_code_and_neighbour_signatures(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))

    # Act
    find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index))

    # Assert
    state, questions = client.requests[0]
    assert set(state) == {"target", "slice", "candidates"}
    assert all("goal" not in question["instructions"] for question in questions.values())


def test_beam_width_and_budgets_can_come_from_the_environment() -> None:
    # Act
    budget = SearchBudget.from_env({"JEV_NAVIGATOR_BEAM_WIDTH": "1", "JEV_NAVIGATOR_MAX_STEPS": "5"})

    # Assert
    assert budget == SearchBudget(beam_width=1, max_steps=5)


def test_empty_neighbours_are_never_offered(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))

    # Act
    find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index))

    # Assert
    offered = [candidate["signature"] for candidate in client.requests[0][0]["candidates"]]
    assert offered and not any(signature.startswith("app/__init__.py") for signature in offered)


def test_found_code_carries_its_source_and_the_path_that_reached_it(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.95 if "len(order.items) <= limit" in code else 0.05,
            could_contain=lambda signature: (
                0.9 if "validate_order" in signature or "check_limits" in signature else 0.1
            ),
        )
    )

    # Act
    result = find_code(
        sample_index, Judge(client), TARGET, start_at_place(sample_index), budget=SearchBudget(beam_width=1)
    )

    # Assert
    source = result.found[0].code.source()
    assert source["file"] == "app/validation.py" and source["commit"] == sample_index.commit
    assert source["reached_by"] == "called by validate_order"
    assert len(result.found[0].path) == 3


def test_a_stopped_search_resumes_from_its_frontier(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(
            found=lambda code: 0.95 if "len(order.items) <= limit" in code else 0.05,
            could_contain=lambda signature: (
                0.9 if "check_limits" in signature or "validate_order" in signature else 0.1
            ),
        )
    )
    judge = Judge(client)
    first = find_code(
        sample_index,
        judge,
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_steps=1, beam_width=1),
    )

    # Act
    resumed = find_code(
        sample_index, judge, TARGET, [], budget=SearchBudget(max_steps=5, beam_width=1), resume=first
    )

    # Assert
    assert first.outcome == Outcome.BUDGET
    assert resumed.outcome == Outcome.FOUND
    assert resumed.found[0].code.span.name == "check_limits"
    assert [visit.code.span.name for visit in resumed.searched].count("place") == 1


def test_capped_neighbours_are_reported_as_not_inspected(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.9))

    # Act
    result = find_code(
        sample_index,
        Judge(client),
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(neighbours_per_kind=0),
    )

    # Assert
    assert any(entry.reason == "capped" for entry in result.not_inspected)


def test_a_requested_revision_that_the_index_does_not_hold_is_an_error(sample_index: CodeIndex) -> None:
    # Arrange
    import pytest

    from jev_navigator.index.code_index import RevisionMismatchError

    # Act and Assert
    with pytest.raises(RevisionMismatchError):
        find_code(
            sample_index, Judge(ScriptedJevClient()), TARGET, start_at_place(sample_index), commit="0" * 40
        )
    find_code(
        sample_index,
        Judge(ScriptedJevClient()),
        TARGET,
        start_at_place(sample_index),
        commit=sample_index.commit,
        budget=SearchBudget(max_steps=1),
    )


def test_the_search_wording_can_be_replaced(sample_index: CodeIndex) -> None:
    # Arrange
    from jev_navigator.directives.find_code import FOUND, SearchQuestions
    from jev_navigator.judgments.questions import Check, Criterion

    own = Check(
        "holds_rule",
        "Does `slice.code` enforce what `target.description` states?",
        Criterion("It enforces it."),
        Criterion("It does not."),
    )
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))

    # Act
    find_code(
        sample_index,
        Judge(client),
        TARGET,
        start_at_place(sample_index),
        questions=SearchQuestions(found=own, open_first=None),
    )

    # Assert
    asked = client.requests[0][1]
    assert own.question_id in asked and FOUND.question_id not in asked
    assert not any(question_id.startswith("open_first") for question_id in asked)


def test_a_search_counts_only_its_own_calls_when_another_search_shares_the_judge(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    nested_results = []
    answer = scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.9)

    def answer_and_start_a_second_search(question_id: str, question: Mapping, state: Mapping) -> float:
        if not nested_results:
            nested_results.append(None)
            nested_results[0] = find_code(
                sample_index, judge, TARGET, start_at_place(sample_index), budget=SearchBudget(max_calls=3)
            )
        return answer(question_id, question, state)

    judge = Judge(ScriptedJevClient(nouls=answer_and_start_a_second_search))

    # Act
    outer = find_code(
        sample_index,
        judge,
        TARGET,
        start_at_place(sample_index),
        budget=SearchBudget(max_calls=2, beam_width=1),
    )

    # Assert
    assert nested_results[0].calls == 3
    assert outer.calls == 2
    assert outer.steps == 2
    assert judge.calls == 5


def test_a_global_call_cap_on_the_judge_ends_a_search_as_budget(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.9))
    judge = Judge(client, max_calls=2)

    # Act
    result = find_code(
        sample_index, judge, TARGET, start_at_place(sample_index), budget=SearchBudget(beam_width=1)
    )

    # Assert
    assert result.outcome == Outcome.BUDGET
    assert result.calls == 2
    assert len(client.requests) == 2


def test_find_code_with_no_moves_opens_only_its_start(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.9))

    # Act
    result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index), moves={})

    # Assert
    assert result.outcome == Outcome.NOTHING_LEFT
    assert result.steps == 1 and len(client.requests) == 1


def test_the_result_and_the_stop_step_name_the_moves_the_search_used(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))
    chosen = {"callers": MOVES["callers"], "same_file": MOVES["same_file"]}

    # Act
    chosen_result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index), moves=chosen)
    default_result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index))

    # Assert
    assert chosen_result.moves == ("callers", "same_file")
    assert chosen_result.history.steps[-1].arguments["moves"] == ["callers", "same_file"]
    assert default_result.moves == tuple(MOVES)


def open_steps(result) -> list[dict]:
    return [step.to_json() for step in result.history.steps if step.operation == "open"]


def test_open_first_offers_a_real_none_option_whose_wording_is_part_of_the_question_id(
    sample_index: CodeIndex,
) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.1))
    tie_wording = replace(OPEN_FIRST, extra_options=(("none", "No entry is more likely than the others."),))

    # Act
    budget = SearchBudget(max_steps=1)
    find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index), budget=budget)

    # Assert
    questions = client.requests[0][1]
    pick = next(question for question_id, question in questions.items() if "open_first" in question_id)
    assert pick["criteria"]["none"] == "None of the entries is likely to contain it."
    assert tie_wording.question_id != OPEN_FIRST.question_id


@pytest.mark.parametrize(("pick_score", "boosted"), [(0.15, False), (0.35, True)])
def test_a_confident_pick_moves_ahead_only_when_its_own_score_is_above_the_no_bar(
    sample_index: CodeIndex, pick_score: float, boosted: bool
) -> None:
    # Arrange
    client = ScriptedJevClient(
        nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: pick_score),
        choices={"open_first": {"0": 1.0}},
    )

    # Act
    result = find_code(
        sample_index, Judge(client), TARGET, start_at_place(sample_index), budget=SearchBudget(max_steps=2)
    )

    # Assert
    first_open = open_steps(result)[0]
    assert first_open["judgments"]["open_first"]["used"] is boosted
    reasons = [
        chosen["reason"]
        for step in result.history.steps
        if step.operation == "choose_next"
        for chosen in step.arguments["chosen"]
    ]
    assert ("open_first" in reasons) is boosted
