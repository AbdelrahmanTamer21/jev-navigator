"""Rank code by the words it shares with a target description, without a model.

Code owns this ranking the way it owns the directory tree: it only orders actual files and spans
that a Choice may then offer. The rules hold for any language and script: a word is a run of
letters, split where an identifier changes case (`rasterizePages`, `ГрузПодъём`), normalized and
case-folded (`Größe` and `GRÖSSE` agree), and cut to its first ``PREFIX_LETTERS`` letters, so
`rasterized` and `rasterize`, or `Finanzierung` and `Finanzierungen`, count as one word. Text
written without spaces (Chinese, Japanese, Korean) is read as overlapping letter pairs. No word is
dropped for being common; BM25's inverse document frequency gives such words almost no weight.

Words are matched as written: a description in one language finds code that uses that language,
in its identifiers, comments or strings, and nothing else.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_LETTERS = re.compile(r"[^\W\d_]+")
PREFIX_LETTERS = 6
MIN_LETTERS = 3
K1 = 1.2
B = 0.75


def words(text: str) -> list[str]:
    """The normalized words of prose or code, identifiers split into their parts."""
    found: list[str] = []
    for run in _LETTERS.findall(text):
        for part in _case_parts(run):
            word = unicodedata.normalize("NFKC", part).casefold()
            if all(_written_without_spaces(letter) for letter in word):
                found += [word[position : position + 2] for position in range(max(1, len(word) - 1))]
            elif len(word) >= MIN_LETTERS:
                found.append(word[:PREFIX_LETTERS])
    return found


def target_terms(description: str) -> tuple[str, ...]:
    """The description's distinct words, in order."""
    return tuple(dict.fromkeys(words(description)))


def matching_lines(lines: Sequence[str], rarity: Mapping[str, float]) -> list[int]:
    """The positions of the lines that share a weighed term, those sharing the rarest terms first."""
    weighed = []
    for position, line in enumerate(lines):
        weight = sum(rarity[term] for term in set(words(line)) & rarity.keys())
        if weight > 0:
            weighed.append((weight, position))
    return [position for _weight, position in sorted(weighed, key=lambda pair: (-pair[0], pair[1]))]


def _case_parts(run: str) -> list[str]:
    """`parseHTTPHeader` becomes `parse`, `HTTP`, `Header`; a run without case stays whole."""
    parts, start = [], 0
    for position in range(1, len(run)):
        before, letter = run[position - 1], run[position]
        after = run[position + 1] if position + 1 < len(run) else ""
        if (before.islower() and letter.isupper()) or (
            before.isupper() and letter.isupper() and after.islower()
        ):
            parts.append(run[start:position])
            start = position
    parts.append(run[start:])
    return parts


def _written_without_spaces(letter: str) -> bool:
    return unicodedata.east_asian_width(letter) in ("W", "F")


@dataclass(frozen=True)
class Corpus:
    """How often each target term occurs in each named text, and each text's length in words."""

    terms: tuple[str, ...]
    counts: Mapping[str, Counter]
    lengths: Mapping[str, int]

    @classmethod
    def of(cls, terms: Sequence[str], texts: Mapping[str, str]) -> Corpus:
        wanted = set(terms)
        counts: dict[str, Counter] = {}
        lengths: dict[str, int] = {}
        for name, text in texts.items():
            found = words(text)
            counts[name] = Counter(word for word in found if word in wanted)
            lengths[name] = len(found)
        return cls(tuple(terms), counts, lengths)

    def idf(self) -> dict[str, float]:
        """Each term's rarity over these texts; a term no text holds weighs nothing."""
        total = len(self.counts)
        rarity = {}
        for term in self.terms:
            holding = sum(1 for counted in self.counts.values() if counted[term])
            rarity[term] = math.log(1 + (total - holding + 0.5) / (holding + 0.5)) if holding else 0.0
        return rarity

    def ranked(self, idf: Mapping[str, float] | None = None) -> tuple[tuple[str, float], ...]:
        """Every text sharing a term, best first; equal scores keep name order. ``idf`` defaults to
        this corpus's own, so a few spans can be weighed by how rare each term is across all files."""
        rarity = self.idf() if idf is None else idf
        average = sum(self.lengths.values()) / len(self.lengths) if self.lengths else 0.0
        scored = []
        for name in sorted(self.counts):
            norm = 1 - B + B * (self.lengths[name] / average if average else 1.0)
            score = sum(
                rarity.get(term, 0.0) * count * (K1 + 1) / (count + K1 * norm)
                for term, count in self.counts[name].items()
            )
            if score > 0:
                scored.append((name, score))
        return tuple(sorted(scored, key=lambda pair: -pair[1]))
