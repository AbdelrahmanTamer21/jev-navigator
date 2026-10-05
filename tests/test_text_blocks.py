"""Text blocks: how a file JVN does not parse splits into the units a text search judges, by format."""

from __future__ import annotations

import pytest

from jev_navigator.index.languages import split_lines
from jev_navigator.index.text_blocks import TextBlock, text_blocks

README = """\
[![build](https://ci.example/badge.svg)](https://ci.example)

A shop's order service.

# Install

```bash
# not a heading: a shell comment in a fence
pip install shop
```

## macOS

brew install shop

# Configure
## Limits
Orders hold at most four items.
"""


def _blocks(path: str, text: str) -> list[tuple[str | None, int, int]]:
    return [(block.name, block.start, block.end) for block in text_blocks(path, split_lines(text))]


def test_markdown_splits_by_heading_section_named_by_its_heading_path() -> None:
    assert _blocks("README.md", README) == [
        (None, 1, 4),
        ("Install", 5, 11),
        ("Install > macOS", 12, 15),
        ("Configure", 16, 16),
        ("Configure > Limits", 17, 18),
    ]


def test_markdown_front_matter_is_not_read_for_headings() -> None:
    page = "---\n# a YAML comment\ntitle: Limits\n---\n# Limits\nAt most four.\n"

    assert _blocks("docs/limits.mdx", page) == [(None, 1, 4), ("Limits", 5, 6)]


WORKFLOW = """\
name: tests

# Runs on every dispatch.
on:
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
"""


def test_yaml_splits_by_top_level_key_and_a_comment_above_a_key_belongs_to_it() -> None:
    assert _blocks(".github/workflows/tests.yml", WORKFLOW) == [
        ("name", 1, 2),
        ("on", 3, 6),
        ("jobs", 7, 9),
    ]


def test_yaml_of_only_a_list_is_one_block() -> None:
    assert _blocks("rules.yaml", "- id: limit\n  max: 4\n- id: audit\n") == [(None, 1, 3)]


SETTINGS_JSON = """\
{
  "name": "shop",
  "limits": {
    "items": 4,
    "note": "a { in a string"
  },
  "tags": ["a", "b"]
}
"""


def test_json_splits_an_object_by_top_level_key() -> None:
    assert _blocks("settings.json", SETTINGS_JSON) == [("name", 1, 2), ("limits", 3, 6), ("tags", 7, 8)]


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "shop", "limits": {"items": 4}}\n',
        '[{"name": "shop"}]\n',
        '{"name": "shop",}\n',
    ],
    ids=["minified", "top-level array", "invalid"],
)
def test_json_that_is_not_an_object_with_keys_on_their_own_lines_is_one_block(text: str) -> None:
    assert _blocks("data.json", text) == [(None, 1, 1)]


PYPROJECT = '''\
name = "shop"
version = "1.0"
notes = """
x = 1 inside a string
"""

# Build settings.
[build-system]
requires = ["hatchling"]

[tool.ruff]
line-length = 110

[[tool.mypy.overrides]]
module = "shop.*"
'''


def test_toml_splits_by_root_key_and_table_named_by_its_key_path() -> None:
    assert _blocks("pyproject.toml", PYPROJECT) == [
        ("name", 1, 1),
        ("version", 2, 2),
        ("notes", 3, 6),
        ("build-system", 7, 10),
        ("tool.ruff", 11, 13),
        ("tool.mypy.overrides", 14, 15),
    ]


def test_a_format_without_structure_is_one_block() -> None:
    assert text_blocks("squid.conf", split_lines("http_port 3128\nacl lan src 10.0.0.0/8\n")) == (
        TextBlock(None, 1, 2),
    )
