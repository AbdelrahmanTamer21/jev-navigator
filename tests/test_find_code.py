from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.find_code import Outcome, SearchBudget, find_code
from jev_navigator.directives.places import (
    MOVES,
    Place,
    function_place,
    neighbours,
    place_for_line,
    window_place,
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


def committed_index(root: Path, files: Mapping[str, str]) -> CodeIndex:
    commit_files(root, files)
    return CodeIndex(root, list(files))


def neighbour_signatures(index: CodeIndex, name: str) -> dict[str, str]:
    opened = index.read_slice(index.find_definition(name)[0])
    return {place.key: place.signature for place in neighbours(index, opened)}


def test_code_reached_only_through_a_reference_is_offered_both_ways(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "jobs.py": "def send_invoice(order):\n    return order\n",
            "registry.py": "from jobs import send_invoice\n\n\ndef register_jobs(scheduler):\n"
            '    scheduler.add({"invoice": send_invoice})\n',
        },
    )
    send_invoice, register_jobs = (
        index.find_definition("send_invoice")[0],
        index.find_definition("register_jobs")[0],
    )

    # Act
    from_registry = neighbour_signatures(index, "register_jobs")
    from_job = neighbour_signatures(index, "send_invoice")

    # Assert
    assert "passed on by register_jobs as collection" in from_registry[send_invoice.key]
    assert "refers to send_invoice as collection" in from_job[register_jobs.key]


def test_the_other_functions_of_the_opened_file_are_offered_whole(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {"orders.py": "def place(order):\n    return order\n\n\ndef refund(order):\n    return None\n"},
    )
    refund = index.find_definition("refund")[0]

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    assert "in the same file as place" in offered[refund.key]


def test_the_lines_before_an_opened_slice_are_offered(tmp_path: Path) -> None:
    # Arrange
    header = "".join(f"SETTING_{number} = {number}\n" for number in range(30))
    index = committed_index(tmp_path, {"orders.py": header + "\n\ndef place(order):\n    return order\n"})

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    before = [signature for signature in offered.values() if "the lines before" in signature]
    assert len(before) == 1


def test_windows_chosen_by_position_are_labelled_by_their_range_and_first_code_line(
    tmp_path: Path,
) -> None:
    # Arrange
    header = "import os\n\n\n" + "".join(f"SETTING_{number} = {number}\n" for number in range(30))
    after = "\n\nLIMIT = 5\n" + "".join(f"OTHER_{number} = {number}\n" for number in range(50))
    index = committed_index(tmp_path, {"orders.py": header + "def place(order):\n    return order\n" + after})
    opened = index.find_definition("place")[0]

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    before = next(signature for signature in offered.values() if "the lines before" in signature)
    after_place = next(signature for signature in offered.values() if "the lines after" in signature)
    assert before == f"orders.py:1-{opened.start - 1} `import os` (the lines before {opened.key})"
    last = opened.end + 40
    assert after_place == f"orders.py:{opened.end + 1}-{last} `LIMIT = 5` (the lines after {opened.key})"


def test_the_lines_after_a_place_end_at_the_last_line_even_with_a_form_feed(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(tmp_path, {"orders.py": 'def place(order):\n    return "a\fb"\n\nLIMIT = 5\n'})

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    after_place = next(signature for signature in offered.values() if "the lines after" in signature)
    assert after_place.startswith("orders.py:3-4 `LIMIT = 5`")


def test_a_window_around_a_line_outside_any_function_is_labelled_by_that_line(tmp_path: Path) -> None:
    # Arrange
    imports = "".join(f"import module_{number}\n" for number in range(14))
    index = committed_index(tmp_path, {"routes.py": imports + 'HEADER = "x-order-limit"\n' + "\n" * 20})

    # Act
    place = place_for_line(index, "routes.py", 15, 'mentions "x-order-limit"')

    # Assert
    assert place.signature == 'routes.py:5-25 line 15 `HEADER = "x-order-limit"` (mentions "x-order-limit")'


def test_quoted_keys_are_searched_across_the_scope_including_docs_and_config(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "orders.py": 'def place(order):\n    return order.get("max_items_per_order")\n',
            "config.yaml": "limits:\n  max_items_per_order: 20\n",
            "README.md": "# Orders\n\nSet `max_items_per_order` to cap orders.\n",
        },
    )

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    mentions = {
        key.split(":")[0]
        for key, signature in offered.items()
        if "mentions `max_items_per_order`" in signature
    }
    assert mentions == {"config.yaml", "README.md"}


def test_a_quoted_key_matches_whole_names_only(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "orders.py": 'def place(order):\n    return order.get("invoice")\n',
            "billing.py": "def send_invoice():\n    pass\n",
            "config.yaml": "invoice: monthly\n",
        },
    )

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    mentions = {key.split(":")[0] for key, signature in offered.items() if "mentions `invoice`" in signature}
    assert mentions == {"config.yaml"}


def test_the_caller_chooses_which_moves_list_neighbours(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "orders.py": (
                "def place(order):\n    return refund(order)\n\n\ndef refund(order):\n    return None\n"
            )
        },
    )
    opened = index.read_slice(index.find_definition("place")[0])

    def always_the_config(index: CodeIndex, code: CodeSlice) -> list[Place]:
        return [window_place(index, "orders.py", 1, "a custom move")]

    # Act
    same_file_only = neighbours(index, opened, moves={"same_file": MOVES["same_file"]})
    custom = neighbours(index, opened, moves={"custom": always_the_config})

    # Assert
    assert [place.signature.split(" (")[-1] for place in same_file_only] == ["in the same file as place)"]
    assert [place.signature.endswith("(a custom move)") for place in custom] == [True]


def test_find_code_with_no_moves_opens_only_its_start(sample_index: CodeIndex) -> None:
    # Arrange
    client = ScriptedJevClient(nouls=scripted(found=lambda code: 0.05, could_contain=lambda signature: 0.9))

    # Act
    result = find_code(sample_index, Judge(client), TARGET, start_at_place(sample_index), moves={})

    # Assert
    assert result.outcome == Outcome.NOTHING_LEFT
    assert result.steps == 1 and len(client.requests) == 1


def test_the_default_moves_cannot_be_changed_by_a_caller() -> None:
    # Act and assert
    with pytest.raises(TypeError):
        MOVES["always_open_this"] = lambda index, opened: []  # type: ignore[index]


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
