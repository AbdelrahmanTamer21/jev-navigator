"""Places the search can open, and the neighbours code lists for each from the parser and git.

A place is concrete: a function, a window of lines, the start of a file. Its ``signature`` is the
short text Jev reads when deciding whether the target could be inside it. Jev never invents a place.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from ..index.code_index import CodeIndex
from ..index.spans import CodeSlice, Span

DEFAULT_NEIGHBOURS_PER_KIND = 8
REST_OF_FILE_LINES = 40
CO_CHANGE_HEAD_LINES = 40
_ENVIRONMENT_READ = re.compile(
    r"""(?:environ(?:\.get)?\(?\[?|getenv\(|process\.env\.)\s*["']?([A-Z][A-Z0-9_]{2,})"""
)
_QUOTED_KEY = re.compile(r"""["'`]([A-Za-z_][\w.:/\-]{3,79})["'`]""")


@dataclass(frozen=True)
class Place:
    key: str
    kind: str
    signature: str
    open: Callable[[], CodeSlice]


def function_place(index: CodeIndex, span: Span, relation: str = "") -> Place:
    first_line = index.read_slice(Span(span.file, span.start, span.start)).text.strip()
    note = f" ({relation})" if relation else ""
    signature = f"{span.file}:{span.start} `{first_line}`{note}"
    return Place(
        span.key, "function", signature, lambda: index.read_slice(span, origin=relation or "function")
    )


def window_place(index: CodeIndex, file: str, line: int, relation: str, radius: int = 10) -> Place:
    """The lines around ``line``, the line that made this place a neighbour (a call, a reference, a
    mentioned key); the signature quotes that line."""

    def open_window() -> CodeSlice:
        return index.read_window(file, line, radius, origin=relation)

    span = open_window().span
    text_line = index.read_slice(Span(file, line, line)).text.strip()
    signature = f"{file}:{span.start}-{span.end} line {line} `{text_line}` ({relation})"
    return Place(f"{file}:{line}~{radius}", "window", signature, open_window)


def range_place(index: CodeIndex, file: str, start: int, end: int, relation: str) -> Place:
    """Lines chosen by their position (before or after a place, the start of a file); no single line
    made them a neighbour, so the signature quotes their first line of code."""
    span = Span(file, start, end)
    first_code_line = next(
        (line.strip() for line in index.read_slice(span).text.split("\n") if line.strip()), ""
    )
    signature = f"{span.key} `{first_code_line}` ({relation})"
    return Place(span.key, "window", signature, lambda: index.read_slice(span, origin=relation))


def place_for_line(index: CodeIndex, file: str, line: int, relation: str) -> Place:
    enclosing = index.enclosing_symbol(file, line)
    if enclosing is not None:
        return function_place(index, enclosing, relation)
    return window_place(index, file, line, relation)


Move = Callable[[CodeIndex, CodeSlice], list[Place]]


def neighbours(
    index: CodeIndex,
    opened: CodeSlice,
    per_kind: int = DEFAULT_NEIGHBOURS_PER_KIND,
    moves: Mapping[str, Move] | None = None,
) -> list[Place]:
    return neighbours_and_omissions(index, opened, per_kind, moves)[0]


def neighbours_and_omissions(
    index: CodeIndex,
    opened: CodeSlice,
    per_kind: int = DEFAULT_NEIGHBOURS_PER_KIND,
    moves: Mapping[str, Move] | None = None,
) -> tuple[list[Place], list[Place]]:
    """The places each move lists for ``opened``, at most ``per_kind`` per move, in the order of
    ``moves`` (default ``MOVES``: callers, callees, code that refers to it without calling it, code
    it passes on without calling, the other functions of its file, lines anywhere in scope that
    mention its quoted keys or environment variables, files usually committed with it, and the lines
    before and after it). A move is any function of the index and the opened code that returns
    places, so callers can drop moves or add their own. The places cut by the cap come back
    separately, so a caller can report them as not inspected."""
    kept: list[Place] = []
    omitted: list[Place] = []
    for build in (MOVES if moves is None else moves).values():
        places = [place for place in _unique(build(index, opened)) if place.key != opened.key]
        kept += places[:per_kind]
        omitted += places[per_kind:]
    return _unique(kept), [place for place in _unique(omitted) if place.key not in {k.key for k in kept}]


def _unique(places: list[Place]) -> list[Place]:
    """One place per key, keeping the first relation found: calls and references come before file
    position, so the strongest reason a place is a neighbour is the one shown."""
    first: dict[str, Place] = {}
    for place in places:
        first.setdefault(place.key, place)
    return list(first.values())


def _callers(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    if not _is_named(opened.span):
        return []
    sites = index.find_callers(opened.span.name)
    return [
        place_for_line(index, site.file, site.line, _with_binding(f"calls {opened.span.name}", site.binding))
        for site in sites
    ]


def _callees(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    if not _is_named(opened.span):
        return []
    places = []
    for edge in index.callee_edges(opened.span):
        targets = [edge.binding.target] if edge.binding.target else index.find_definition(edge.name)
        relation = _with_binding(f"called by {opened.span.name}", edge.binding)
        places += [function_place(index, span, relation) for span in targets]
    return places


def _referenced_by(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    if not _is_named(opened.span):
        return []
    name = opened.span.name
    return [
        place_for_line(
            index, ref.file, ref.line, _with_binding(f"refers to {name} as {ref.role}", ref.binding)
        )
        for ref in index.find_references(name)
    ]


def _passed_on(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    if not _is_named(opened.span):
        return []
    places = []
    for ref in index.references_in(opened.span):
        targets = (
            [ref.binding.target] if ref.binding and ref.binding.target else index.find_definition(ref.name)
        )
        relation = _with_binding(f"passed on by {opened.span.name} as {ref.role}", ref.binding)
        places += [function_place(index, span, relation) for span in targets]
    return places


def _with_binding(relation: str, binding) -> str:
    """A name-match link is marked, so neither Jev nor the result treats it as a proven call."""
    if binding is None or binding.proven:
        return relation
    return f"{relation}, {binding.status}: {binding.reason}"


def _same_file(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    relation = f"in the same file as {opened.span.name or opened.span.key}"
    return [
        function_place(index, span, relation)
        for span in index.functions_in(opened.span.file)
        if not _overlaps(span, opened.span)
    ]


def _overlaps(first: Span, second: Span) -> bool:
    return first.start <= second.end and second.start <= first.end


def _keys_mentioned(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    """Environment variables it reads and string keys it quotes, longest (most specific) first."""
    found = _ENVIRONMENT_READ.findall(opened.text) + _QUOTED_KEY.findall(opened.text)
    keys = sorted(dict.fromkeys(found), key=len, reverse=True)
    places = []
    for key in keys:
        whole_key = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)")
        for hit in index.search_text(key):
            outside = not opened.span.contains(hit.line) or hit.file != opened.span.file
            if outside and whole_key.search(hit.text):
                places.append(place_for_line(index, hit.file, hit.line, f"mentions `{key}`"))
    return places


def _co_changed(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    places = []
    for other, commits in index.co_changed_files(opened.span.file, limit=2):
        relation = f"start of a file committed with {opened.span.file} {commits} times"
        end = min(len(index.lines(other)), CO_CHANGE_HEAD_LINES)
        places.append(range_place(index, other, 1, end, relation))
    return places


def _lines_before(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    span = opened.span
    if span.start <= 1:
        return []
    start = max(1, span.start - REST_OF_FILE_LINES)
    return [range_place(index, span.file, start, span.start - 1, f"the lines before {span.key}")]


def _rest_of_file(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    span = opened.span
    line_count = len(index.lines(span.file))
    if span.end >= line_count:
        return []
    end = min(line_count, span.end + REST_OF_FILE_LINES)
    return [range_place(index, span.file, span.end + 1, end, f"the lines after {span.key}")]


def _is_named(span: Span) -> bool:
    return bool(span.name) and not span.name.startswith("<")


def starting_places(index: CodeIndex, locations: Sequence[tuple[str, int]]) -> list[Place]:
    return [place_for_line(index, file, line, "start") for file, line in locations]


MOVES: Mapping[str, Move] = MappingProxyType(
    {
        "callers": _callers,
        "callees": _callees,
        "referenced_by": _referenced_by,
        "passed_on": _passed_on,
        "same_file": _same_file,
        "keys_mentioned": _keys_mentioned,
        "co_changed": _co_changed,
        "lines_before": _lines_before,
        "rest_of_file": _rest_of_file,
    }
)
