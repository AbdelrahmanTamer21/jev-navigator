"""The places the search can open and the neighbours each move lists for opened code."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.places import (
    MAX_DEFINITION_LINES,
    MOVES,
    Move,
    Place,
    neighbours,
    neighbours_and_omissions,
    place_for_line,
    range_place,
    window_place,
)
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import CodeSlice, Span


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


def test_a_window_around_a_line_outside_any_definition_is_labelled_by_that_line(tmp_path: Path) -> None:
    # Arrange
    imports = "".join(f"import module_{number}\n" for number in range(14))
    index = committed_index(tmp_path, {"routes.py": imports + 'register("x-order-limit")\n' + "\n" * 20})

    # Act
    place = place_for_line(index, "routes.py", 15, 'mentions "x-order-limit"')

    # Assert
    assert place.signature == 'routes.py:5-25 line 15 `register("x-order-limit")` (mentions "x-order-limit")'


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


def test_the_default_moves_cannot_be_changed_by_a_caller() -> None:
    # Act and assert
    with pytest.raises(TypeError):
        MOVES["always_open_this"] = lambda index, opened: []  # type: ignore[index]


def offered_from(index: CodeIndex, file: str, line: int, per_kind: int = 8) -> list[Place]:
    opened = place_for_line(index, file, line, "start").open()
    return neighbours(index, opened, per_kind)


def test_a_line_on_a_class_opens_the_class_so_code_naming_it_is_offered(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "store.py": "from typing import Protocol\n\n\nclass Store(Protocol):\n"
            "    def save(self, record) -> None: ...\n",
            "phases.py": "from store import Store\n\n\ndef export_phase(store: Store, records):\n"
            "    for record in records:\n        store.save(record)\n",
        },
    )

    # Act
    place = place_for_line(index, "store.py", 4, "start")
    offered = {candidate.key: candidate.signature for candidate in offered_from(index, "store.py", 4)}

    # Assert
    assert place.open().span.name == "Store"
    assert "refers to Store as type" in offered["phases.py:4-6"]


def test_a_line_on_a_module_constant_opens_the_constant_so_code_naming_it_is_offered(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "limits.py": 'BLOCKED_TOOLS = frozenset(\n    {"shell", "write"}\n)\n',
            "limits_test.py": "from limits import BLOCKED_TOOLS\n\n\n"
            "def test_blocked_tools_match_the_runtime():\n"
            '    runtime = {"shell", "write"}\n    assert BLOCKED_TOOLS == runtime\n',
        },
    )

    # Act
    place = place_for_line(index, "limits.py", 2, "start")
    offered = {candidate.key: candidate.signature for candidate in offered_from(index, "limits.py", 2)}

    # Assert
    assert place.open().span == index.find_definition("BLOCKED_TOOLS")[0]
    assert "refers to BLOCKED_TOOLS as condition" in offered["limits_test.py:4-6"]


def test_a_line_on_a_class_longer_than_the_limit_opens_a_window(tmp_path: Path) -> None:
    # Arrange
    attributes = "".join(f"    FIELD_{number} = {number}\n" for number in range(MAX_DEFINITION_LINES))
    index = committed_index(tmp_path, {"settings.py": "class Settings:\n" + attributes})

    # Act
    place = place_for_line(index, "settings.py", 60, "start")

    # Assert
    assert place.open().span.name == ""
    assert place.open().span.size() == 21


def test_a_constant_used_as_a_method_receiver_is_passed_on(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "redaction.py": 'import re\n\nSECRET_PATTERN = re.compile(r"key=\\w+")\n\n\n'
            'def redact(text):\n    return SECRET_PATTERN.sub("key=[hidden]", text)\n'
        },
    )

    # Act
    offered = neighbour_signatures(index, "redact")

    # Assert
    assert "passed on by redact as receiver" in offered["redaction.py:3-3"]


def numbered_functions(count: int) -> str:
    return "".join(f"def step_{number}(value):\n    return value + {number}\n\n\n" for number in range(1, count + 1))


def test_the_other_functions_of_the_file_are_offered_nearest_first(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(tmp_path, {"steps.py": numbered_functions(12)})

    # Act
    offered = [place.signature for place in neighbours(index, index.read_slice(index.find_definition("step_10")[0]))]

    # Assert
    same_file = [signature.split("`")[1] for signature in offered if "in the same file" in signature]
    assert same_file == [f"def step_{number}(value):" for number in (9, 11, 8, 12, 7, 6, 5, 4)]


def test_a_nested_function_is_offered_only_as_part_of_its_function(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "steps.py": "def outer(value):\n    def inner():\n        return value\n\n    return inner()\n\n\n"
            "def other(value):\n    return value\n"
        },
    )

    # Act
    offered = neighbour_signatures(index, "other")

    # Assert
    same_file = [signature for signature in offered.values() if "in the same file" in signature]
    assert [signature.split("`")[1] for signature in same_file] == ["def outer(value):"]


@pytest.mark.parametrize(
    "test_path", ["tests/helpers.py", "app/test_orders.py", "app/orders_test.py", "app/conftest.py", "spec/orders.py"]
)
def test_callers_in_test_files_come_after_the_other_callers(tmp_path: Path, test_path: str) -> None:
    # Arrange
    index = committed_index(
        tmp_path,
        {
            "orders.py": "def place_order(order):\n    return order\n",
            test_path: "from orders import place_order\n\n\ndef check_order():\n    place_order({})\n",
            "zz/checkout.py": "from orders import place_order\n\n\ndef checkout(order):\n    place_order(order)\n",
        },
    )

    # Act
    offered = neighbour_signatures(index, "place_order")

    # Assert
    callers = [key.split(":")[0] for key, signature in offered.items() if "calls place_order" in signature]
    assert callers == ["zz/checkout.py", test_path]


def test_the_lines_before_and_after_stay_off_the_opened_code_in_a_short_file(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(tmp_path, {"orders.py": "import os\n\ndef place(order):\n    return order\n\nLIMIT = 5\n"})

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    positional = {key for key, signature in offered.items() if "the lines" in signature}
    assert positional == {"orders.py:1-2", "orders.py:5-6"}


def a_move_offering(*places: Place) -> Move:
    return lambda index, opened: list(places)


def test_places_that_open_the_same_lines_are_offered_once(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(tmp_path, {"routes.py": "".join(f"ROUTE_{number} = {number}\n" for number in range(30))})
    opened = index.read_slice(Span("routes.py", 25, 30))
    window = window_place(index, "routes.py", 11, "mentions a key")
    same_lines = range_place(index, "routes.py", 1, 21, "the start of a co-changed file")

    # Act
    offered = neighbours(index, opened, moves={"keys": a_move_offering(window), "files": a_move_offering(same_lines)})

    # Assert
    assert [place.key for place in offered] == [window.key]


def test_a_place_wholly_inside_the_opened_code_is_not_offered(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(tmp_path, {"routes.py": "".join(f"ROUTE_{number} = {number}\n" for number in range(30))})
    opened = index.read_slice(Span("routes.py", 1, 20))
    inside = range_place(index, "routes.py", 3, 5, "inside")
    overlapping = range_place(index, "routes.py", 15, 25, "overlapping")

    # Act
    offered = neighbours(index, opened, moves={"lines": a_move_offering(inside, overlapping)})

    # Assert
    assert [place.key for place in offered] == [overlapping.key]


def test_identical_code_in_two_files_stays_two_places(tmp_path: Path) -> None:
    # Arrange
    refund = "def refund(order):\n    return None\n"
    index = committed_index(
        tmp_path,
        {
            "orders.py": "def place(order):\n    return refund(order)\n",
            "billing.py": refund,
            "legacy_billing.py": refund,
        },
    )

    # Act
    offered = neighbour_signatures(index, "place")

    # Assert
    assert {"billing.py:1-2", "legacy_billing.py:1-2"} <= set(offered)


def test_a_place_kept_by_an_earlier_move_does_not_use_a_later_moves_cap(tmp_path: Path) -> None:
    # Arrange
    index = committed_index(tmp_path, {"routes.py": "".join(f"ROUTE_{number} = {number}\n" for number in range(30))})
    opened = index.read_slice(Span("routes.py", 1, 1))
    shared = range_place(index, "routes.py", 5, 6, "shared")
    only_later = range_place(index, "routes.py", 8, 9, "only later")

    # Act
    kept, omitted = neighbours_and_omissions(
        index, opened, 1, {"first": a_move_offering(shared), "later": a_move_offering(shared, only_later)}
    )

    # Assert
    assert [place.key for place in kept] == [shared.key, only_later.key]
    assert omitted == []
