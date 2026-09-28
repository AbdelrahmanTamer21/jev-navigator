"""Places the search can open, and the neighbours code lists for each from the parser and git.

A place is concrete: a function, a window of lines, the start of a file. Its ``signature`` is the
short text Jev reads when deciding whether the target could be inside it. Jev never invents a place.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from ..index.code_index import CodeIndex
from ..index.spans import CodeSlice, Span, TextHit

DEFAULT_NEIGHBOURS_PER_KIND = 8
MAX_DEFINITION_LINES = 120
REST_OF_FILE_LINES = 40
CO_CHANGE_HEAD_LINES = 40
_ENVIRONMENT_READ = re.compile(
    r"""(?:environ(?:\.get)?\(?\[?|getenv\(|process\.env\.)\s*["']?([A-Z][A-Z0-9_]{2,})"""
)
_QUOTED_KEY = re.compile(r"""["'`]([A-Za-z_][\w.:/\-]{5,79})["'`]""")
_KEY_SHAPE = re.compile(r"[._:/-]|^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
MAX_KEY_HITS = 30
_TEST_DIRECTORIES = frozenset({"test", "tests", "__tests__", "spec"})
_TEST_FILE_NAME = re.compile(r"^test_|_test\.|\.test\.|\.spec\.|^conftest\.py$")


@dataclass(frozen=True)
class Place:
    key: str
    kind: str
    signature: str
    open: Callable[[], CodeSlice]


def function_place(index: CodeIndex, span: Span, relation: str = "") -> Place:
    """A function, class or declaration, opened whole."""
    first_line = index.read_slice(Span(span.file, span.start, span.start)).text.strip()
    note = f" ({relation})" if relation else ""
    signature = f"{span.file}:{span.start} `{first_line}`{note}"
    return Place(
        span.key, "function", signature, lambda: index.read_slice(span, origin=relation or "function")
    )


def window_place(
    index: CodeIndex, file: str, line: int, relation: str, radius: int = 10, name: str = ""
) -> Place:
    """The lines around ``line``, the line that made this place a neighbour (a call, a reference, a
    mentioned key); the signature quotes that line. ``name`` names the class or declaration the
    window lies in, so the moves can follow that name from the window."""

    def open_window() -> CodeSlice:
        window = index.read_window(file, line, radius).span
        return index.read_slice(replace(window, name=name), origin=relation)

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
    """The function holding ``line``. Outside every function, the class or module-level declaration
    holding it, whole when it has at most ``MAX_DEFINITION_LINES`` lines and otherwise as the window
    around the line under its name, so the moves can follow that name; outside every definition,
    the window around the line."""
    function = index.enclosing_symbol(file, line)
    if function is not None:
        return function_place(index, function, relation)
    definition = _enclosing_definition(index, file, line)
    if definition is None:
        return window_place(index, file, line, relation)
    if definition.size() <= MAX_DEFINITION_LINES:
        return function_place(index, definition, relation)
    return window_place(index, file, line, relation, name=definition.name)


def _enclosing_definition(index: CodeIndex, file: str, line: int) -> Span | None:
    definitions = (*index.symbols_in(file), *index.declarations_in(file))
    return min((span for span in definitions if span.contains(line)), key=Span.size, default=None)


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
    separately, so a caller can report them as not inspected.

    Places are one when they open the same lines of the same file, whatever their keys, and a place
    wholly inside ``opened`` is left out. The first relation found is kept: calls and references come
    before file position, so the strongest reason a place is a neighbour is the one shown. A move's
    cap counts only places no earlier move kept."""
    shown: set[str] = set()
    kept: list[Place] = []
    beyond_cap: list[Place] = []
    for build in (MOVES if moves is None else moves).values():
        new = _new_places(build(index, opened), opened.span, shown)
        kept += new[:per_kind]
        shown |= {place.open().span.key for place in new[:per_kind]}
        beyond_cap += new[per_kind:]
    return kept, _new_places(beyond_cap, opened.span, shown)


def _new_places(places: list[Place], opened: Span, shown: set[str]) -> list[Place]:
    """One place per stretch of lines, leaving out stretches in ``shown`` and inside ``opened``."""
    seen = set(shown)
    new = []
    for place in places:
        span = place.open().span
        if span.key not in seen and not _within(span, opened):
            seen.add(span.key)
            new.append(place)
    return new


def _within(inner: Span, outer: Span) -> bool:
    return inner.file == outer.file and outer.start <= inner.start and inner.end <= outer.end


def _callers(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    if not _is_named(opened.span):
        return []
    sites = sorted(index.find_callers(opened.span.name), key=lambda site: _is_test_file(site.file))
    return [
        place_for_line(index, site.file, site.line, _with_binding(f"calls {opened.span.name}", site.binding))
        for site in sites
    ]


def _callees(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    """What the opened code calls, the names called from fewest places first: a function called
    only here says more about this code than a helper called everywhere."""
    if not _is_named(opened.span):
        return []
    places = []
    edges = sorted(index.callee_edges(opened.span), key=lambda edge: index.call_site_count(edge.name))
    for edge in edges:
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


def _is_test_file(path: str) -> bool:
    directories, _, name = path.rpartition("/")
    in_test_directory = not _TEST_DIRECTORIES.isdisjoint(directories.split("/"))
    return in_test_directory or bool(_TEST_FILE_NAME.search(name))


def _same_file(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    """The other functions of the file, nearest to the opened code first; a function nested in
    another is part of that function."""
    relation = f"in the same file as {opened.span.name or opened.span.key}"
    functions = index.functions_in(opened.span.file)
    outermost = [span for span in functions if not any(_encloses(other, span) for other in functions)]
    others = [span for span in outermost if not _overlaps(span, opened.span)]
    nearest_first = sorted(others, key=lambda span: (_distance(span, opened.span), span.start))
    return [function_place(index, span, relation) for span in nearest_first]


def _overlaps(first: Span, second: Span) -> bool:
    return first.start <= second.end and second.start <= first.end


def _encloses(outer: Span, inner: Span) -> bool:
    same_lines = (outer.start, outer.end) == (inner.start, inner.end)
    return not same_lines and outer.start <= inner.start and inner.end <= outer.end


def _distance(span: Span, opened: Span) -> int:
    return opened.start - span.end if span.end < opened.start else span.start - opened.end


def _keys_mentioned(index: CodeIndex, opened: CodeSlice) -> list[Place]:
    """Lines elsewhere that mention an environment variable it reads or a key it quotes, the rarest
    key first. A quoted key has at least six characters and a key's shape: a dot, underscore, colon,
    slash or dash, or CONSTANT_CASE. A key found on more than ``MAX_KEY_HITS`` lines is too common to
    point anywhere and is skipped."""
    hits_by_key = {key: _lines_mentioning(index, opened, key) for key in _keys_in(opened.text)}
    usable = [(key, hits) for key, hits in hits_by_key.items() if 0 < len(hits) <= MAX_KEY_HITS]
    rarest_first = sorted(usable, key=lambda item: len(item[1]))
    return [
        place_for_line(index, hit.file, hit.line, f"mentions `{key}`")
        for key, hits in rarest_first
        for hit in hits
    ]


def _keys_in(code: str) -> list[str]:
    quoted = [key for key in _QUOTED_KEY.findall(code) if _KEY_SHAPE.search(key)]
    return list(dict.fromkeys(_ENVIRONMENT_READ.findall(code) + quoted))


def _lines_mentioning(index: CodeIndex, opened: CodeSlice, key: str) -> list[TextHit]:
    whole_key = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)")
    return [
        hit
        for hit in index.search_text(key, MAX_KEY_HITS + 1)
        if whole_key.search(hit.text)
        and not (hit.file == opened.span.file and opened.span.contains(hit.line))
    ]


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
