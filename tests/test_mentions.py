from __future__ import annotations

import pytest

from jev_navigator.mentions import code_names_in, is_file_path, member_names, paths_in


def test_a_path_token_loses_its_leading_relative_part_and_its_trailing_punctuation() -> None:
    # Arrange
    text = (
        "See ../config/app.toml, /etc/hosts.conf and `pyproject.toml`; "
        "./loader.py:12 reads ../../shared/limits.json. Then jobs/sweep.py."
    )

    # Act
    paths = paths_in(text)

    # Assert
    assert paths == [
        "config/app.toml",
        "etc/hosts.conf",
        "pyproject.toml",
        "loader.py",
        "shared/limits.json",
        "jobs/sweep.py",
    ]


def test_a_path_is_given_once_in_order_of_first_mention_and_a_word_without_a_suffix_is_none() -> None:
    # Act
    paths = paths_in("orders.py calls `place` in orders.py, then README and Makefile")

    # Assert
    assert paths == ["orders.py"]


@pytest.mark.parametrize(
    ("span", "names_a_file"),
    [
        ("pyproject.toml", True),
        ("jobs/run", True),
        ("res.json", True),
        ("Store.flush", False),
        ("config.get", False),
        ("res.json()", False),
    ],
)
def test_a_span_names_a_file_when_it_holds_a_folder_or_ends_in_a_known_file_suffix(
    span: str, names_a_file: bool
) -> None:
    # Act
    answer = is_file_path(span)

    # Assert
    assert answer is names_a_file


def test_a_text_names_code_by_backticks_calls_and_identifier_shapes_in_order_of_mention() -> None:
    # Arrange
    text = (
        "`Store.flush()` runs before `pyproject.toml` loads; `audit` and `retry policy` follow; "
        "retry_limit and MAX_RETRIES feed placeOrder(order) and OrderBook, then send(order)."
    )

    # Act
    names = code_names_in(text)

    # Assert
    assert names == ["flush", "audit", "retry_limit", "MAX_RETRIES", "placeOrder", "OrderBook", "send"]


def test_a_bare_snake_case_word_is_its_own_rule() -> None:
    # Act
    names = code_names_in("limits are read by check_limits, not by Checks")

    # Assert
    assert names == ["check_limits"]


def test_a_symbol_names_its_members_and_a_description_names_none() -> None:
    # Act
    names = member_names("Store.flush() / cache#evict, retry via Policy.apply | Queue::push")

    # Assert
    assert names == ["flush", "evict", "push"]
