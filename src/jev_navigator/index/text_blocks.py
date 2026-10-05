"""How a file JVN does not parse splits into blocks, by its format, without a parser.

Blocks cover every line of the file, in order, and never overlap. Markdown splits at each ``#``
heading outside a code fence and its front matter, a block named by its heading path (``Install > macOS``);
the lines before the first heading are a block of their own. YAML splits at each top-level key,
JSON at each top-level key of an object whose keys start on lines of their own, and TOML at each
root key and table header, each block named by its key path (``tool.ruff``). There the lines before
the first key belong to the first block, and comment lines right above a key belong to its block.
Any other file, and a file whose format gives no block, is one block. A block with no name has
``name`` None.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t#]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FRONT_MATTER = "---"
_YAML_KEY = re.compile(r"""^("[^"]*"|'[^']*'|[^\s#'"\-?:{\[!&*%@`|>][^#]*?)[ \t]*:(?:[ \t]|$)""")
_JSON_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|[{}\[\]:]')
_TOML_KEY_PART = r"""(?:[A-Za-z0-9_-]+|"[^"]*"|'[^']*')"""
_TOML_KEY = rf"{_TOML_KEY_PART}(?:[ \t]*\.[ \t]*{_TOML_KEY_PART})*"
_TOML_TABLE = re.compile(rf"^\[\[?[ \t]*({_TOML_KEY})[ \t]*\]\]?[ \t]*(?:#.*)?$")
_TOML_ROOT_KEY = re.compile(rf"^({_TOML_KEY})[ \t]*=")
_TOML_MULTILINE_QUOTES = ('"""', "'''")
_COMMENT = re.compile(r"^[ \t]*#")
_HEADING_PATH = " > "

Start = tuple[int, str]


@dataclass(frozen=True)
class TextBlock:
    """Lines ``start`` to ``end`` of a text file, named by its heading or key path."""

    name: str | None
    start: int
    end: int


def text_blocks(path: str, lines: Sequence[str]) -> tuple[TextBlock, ...]:
    """The blocks of ``path``'s ``lines``, by the format its suffix names."""
    if not lines:
        return ()
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _MARKDOWN_SUFFIXES:
        return _blocks(_markdown_starts(lines), len(lines), leading_block=True)
    starts = _KEYED_STARTS.get(suffix, _no_starts)(lines)
    return _blocks(starts, len(lines), leading_block=False)


def _blocks(starts: Sequence[Start], line_count: int, *, leading_block: bool) -> tuple[TextBlock, ...]:
    """Blocks from their first lines: each ends where the next starts. The lines before the first
    start are a block of their own, or with ``leading_block`` False part of the first block."""
    named: list[tuple[int, str | None]] = list(starts)
    if not named:
        named = [(1, None)]
    elif named[0][0] > 1:
        named = [(1, None), *named] if leading_block else [(1, named[0][1]), *named[1:]]
    ends = [start - 1 for start, _ in named[1:]] + [line_count]
    return tuple(TextBlock(name, start, end) for (start, name), end in zip(named, ends, strict=True))


def _markdown_starts(lines: Sequence[str]) -> list[Start]:
    starts: list[Start] = []
    path: list[tuple[int, str]] = []
    fence: str | None = None
    front_matter_end = _front_matter_end(lines)
    for number, line in enumerate(lines[front_matter_end:], front_matter_end + 1):
        fence = _fence_after(line, fence)
        heading = None if fence or _FENCE.match(line) else _ATX_HEADING.match(line)
        if heading:
            level, title = len(heading[1]), (heading[2] or "").strip() or heading[1]
            path = [*(entry for entry in path if entry[0] < level), (level, title)]
            starts.append((number, _HEADING_PATH.join(title for _, title in path)))
    return starts


def _front_matter_end(lines: Sequence[str]) -> int:
    """The line closing a front matter block that opens the file, or 0 when it has none."""
    if lines[0].strip() != _FRONT_MATTER:
        return 0
    return next((number for number, line in enumerate(lines[1:], 2) if line.strip() == _FRONT_MATTER), 0)


def _fence_after(line: str, fence: str | None) -> str | None:
    """The code fence open after ``line``: a fence opens with three or more backticks or tildes and
    closes with a run of the same character at least as long."""
    opening = _FENCE.match(line)
    if not opening:
        return fence
    run = opening[1]
    if fence is None:
        return run
    return None if run[0] == fence[0] and len(run) >= len(fence) else fence


def _yaml_starts(lines: Sequence[str]) -> list[Start]:
    keys = ((number, _YAML_KEY.match(line)) for number, line in enumerate(lines, 1))
    return [(_with_comments_above(lines, number), _unquoted(key[1])) for number, key in keys if key]


def _json_starts(lines: Sequence[str]) -> list[Start]:
    """Each top-level key of an object, at the line it starts on; none when the file is not a JSON
    object or two of its keys share a line."""
    try:
        value = json.loads("\n".join(lines))
    except ValueError:
        return []
    if not isinstance(value, dict):
        return []
    starts = _object_keys(lines)
    distinct = len({number for number, _ in starts}) == len(starts)
    return starts if distinct else []


def _object_keys(lines: Sequence[str]) -> list[Start]:
    """A JSON string never spans lines, so each line's tokens are read on their own."""
    tokens = [
        (number, token[0]) for number, line in enumerate(lines, 1) for token in _JSON_TOKEN.finditer(line)
    ]
    keys: list[Start] = []
    depth = 0
    for (number, token), (_, following) in zip(tokens, [*tokens[1:], (0, "")], strict=True):
        if token in "{[":
            depth += 1
        elif token in "}]":
            depth -= 1
        elif depth == 1 and token.startswith('"') and following == ":":
            keys.append((number, json.loads(token)))
    return keys


def _toml_starts(lines: Sequence[str]) -> list[Start]:
    starts: list[Start] = []
    in_table = False
    open_quotes: str | None = None
    for number, line in enumerate(lines, 1):
        if open_quotes is None:
            table = _TOML_TABLE.match(line)
            root_key = None if in_table else _TOML_ROOT_KEY.match(line)
            if found := table or root_key:
                starts.append((_with_comments_above(lines, number), _toml_key_path(found[1])))
            in_table = in_table or table is not None
        open_quotes = _quotes_open_after(line, open_quotes)
    return starts


def _quotes_open_after(line: str, open_quotes: str | None) -> str | None:
    """The multi-line string still open after ``line``, by counting its triple quotes."""
    for quotes in _TOML_MULTILINE_QUOTES if open_quotes is None else (open_quotes,):
        if line.count(quotes) % 2:
            return quotes if open_quotes is None else None
    return open_quotes


def _toml_key_path(key: str) -> str:
    return ".".join(_unquoted(part.strip()) for part in re.findall(_TOML_KEY_PART, key))


def _with_comments_above(lines: Sequence[str], number: int) -> int:
    """The first of the comment lines right above line ``number``, or ``number`` itself."""
    while number > 1 and _COMMENT.match(lines[number - 2]):
        number -= 1
    return number


def _unquoted(key: str) -> str:
    return key[1:-1] if len(key) > 1 and key[0] == key[-1] and key[0] in "\"'" else key


def _no_starts(lines: Sequence[str]) -> list[Start]:
    return []


_MARKDOWN_SUFFIXES = frozenset({".md", ".mdx", ".markdown"})
_KEYED_STARTS: dict[str, Callable[[Sequence[str]], list[Start]]] = {
    ".yml": _yaml_starts,
    ".yaml": _yaml_starts,
    ".json": _json_starts,
    ".toml": _toml_starts,
}
