"""A place's label in a run file is its location and the name of its enclosing symbol, never its code."""

from __future__ import annotations

from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.run_files import is_place_label, place_label


def test_every_label_place_label_writes_has_the_label_form(tmp_path: Path) -> None:
    # Arrange: a named Python function, an anonymous JavaScript callback and module-level lines, all
    # parsed.
    files = {
        "app/entry.py": "from app.target import check\n\n\ndef handle():\n    return check()\n",
        "app/routes.js": "router.get('/items', (request) => {\n  return list(request);\n});\n",
    }
    commit_files(tmp_path / "repository", files)
    index = CodeIndex(tmp_path / "repository", list(files), fact_cache_dir=tmp_path / "facts")
    for file in files:
        index.functions_in(file)
    keys = ["app/entry.py:4-5", "app/routes.js:2~10", "app/entry.py:1-2"]

    # Act
    labels = [place_label(index, key) for key in keys]

    # Assert
    assert labels == ["app/entry.py:4 handle", "app/routes.js:2 <anonymous>", "app/entry.py:1"]
    assert all(is_place_label(key, label) for key, label in zip(keys, labels, strict=True))


@pytest.mark.parametrize(
    ("place_key", "stored"),
    [
        ("app/entry.py:4-5", "app/entry.py:4 `def handle():` (calls check)"),
        ("app/entry.py:1-2", "app/entry.py:1-2 `from .policy import admit` (the lines before handle)"),
        ("app/entry.py:5~10", "app/entry.py:1-9 line 5 `return check()` (calls check)"),
        ("app/entry.py:4-5", "app/entry.py:4 handle (calls check)"),
        ("app/entry.py:14-15", "app/entry.py:41 handle"),
    ],
)
def test_a_code_signature_or_another_place_s_label_is_no_label_of_the_place(
    place_key: str, stored: str
) -> None:
    # Act
    accepted = is_place_label(place_key, stored)

    # Assert
    assert not accepted
