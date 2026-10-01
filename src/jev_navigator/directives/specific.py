"""Places one step more specific than a find, which the search opens before it accepts the find.

A yes can land on a layer that only names the target: an interface whose comment restates the
description while a function elsewhere does the work. For a find with no function in it (an
interface, a type, a table of values), code lists the functions defined elsewhere under the names
its best-matching lines declare. Only those that share a word with the description are kept,
best-matching first, test files last; a find with none is accepted as it is.

A find inside a function is accepted as it is. The code that calls a shared function often shares
the description's words too, so a yes there would list a caller before the very function the
description names.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from ..index.code_index import CodeIndex
from ..index.keywords import Corpus, matching_lines, target_terms
from ..index.languages import language_of
from ..index.spans import Span
from .places import Place, function_place, is_test_file

NAME_LINES = 2
# A name a line declares is followed by its parameters, its type or its value: `queueOf(`,
# `limit:`, `limit?:`, `limit =`, `Box<`.
_DECLARED = re.compile(r"([A-Za-z_$][\w$]*)\s*[?!]?\s*[(:=<]")


def specific_places(index: CodeIndex, description: str, found: Span, limit: int) -> list[Place]:
    """At most ``limit`` places more specific than the lines ``found``, best-matching first."""
    terms = target_terms(description)
    if limit <= 0 or not terms:
        return []
    if any(span.start <= found.end and found.start <= span.end for span in index.functions_in(found.file)):
        return []
    rarity = _rarity(index, terms)
    return _best_matching(_implementations(index, found, rarity), terms, rarity, limit)


def _rarity(index: CodeIndex, terms: Sequence[str]) -> dict[str, float]:
    files = tuple(file for file in index.available_files if language_of(file))
    return Corpus.of(terms, {file: "\n".join((file, *index.lines(file))) for file in files}).idf()


def _implementations(index: CodeIndex, found: Span, rarity: Mapping[str, float]) -> list[Place]:
    """The functions defined under the names declared on the find's best-matching lines, or on the
    line after one, since a comment describes the line below it."""
    lines = index.lines(found.file)[found.start - 1 : found.end]
    names: dict[str, None] = {}
    for position in matching_lines(lines, rarity)[:NAME_LINES]:
        for line in lines[position : position + 2]:
            names.update(dict.fromkeys(_DECLARED.findall(line)))
    places = []
    for name in names:
        for span in index.find_definition(name):
            outside = span.file != found.file or span.end < found.start or found.end < span.start
            if outside and span in index.functions_in(span.file):
                places.append(function_place(index, span, f"defines {name}, declared at {found.key}"))
    return places


def _best_matching(
    places: list[Place], terms: Sequence[str], rarity: Mapping[str, float], limit: int
) -> list[Place]:
    texts = {}
    for place in places:
        opened = place.open()
        texts.setdefault(place.key, "\n".join((opened.span.file, opened.text)))
    scores = dict(Corpus.of(terms, texts).ranked(rarity))
    kept = {place.key: place for place in places if scores.get(place.key)}
    ordered = sorted(
        kept.values(), key=lambda place: (is_test_file(place.open().span.file), -scores[place.key], place.key)
    )
    return ordered[:limit]
