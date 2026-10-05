from __future__ import annotations

from jev_navigator.mentions import paths_in


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
