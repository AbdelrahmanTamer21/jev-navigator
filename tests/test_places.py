"""The places the search can open and the neighbours each move lists for opened code."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.places import MOVES, Place, neighbours, place_for_line, window_place
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.spans import CodeSlice


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


def test_the_default_moves_cannot_be_changed_by_a_caller() -> None:
    # Act and assert
    with pytest.raises(TypeError):
        MOVES["always_open_this"] = lambda index, opened: []  # type: ignore[index]
