"""Small, pluggable text facts: a rule is a name, a regular expression and an optional filter.

``find_facts(text, rules)`` returns every match as a ``Fact`` with its offsets and matched text. The
shipped rules are examples, not policy; callers pass their own list, or wrap a rule with a filter
such as ``outside_names`` (skip matches inside a path, file name or identifier).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

MatchFilter = Callable[[str, re.Match], bool]
_NAME_JOINERS = "/._-"
_FILE_EXTENSION = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,5}$")


@dataclass(frozen=True)
class Fact:
    """``start`` and ``end`` are offsets into the text that was scanned; ``line`` counts from 1."""

    name: str
    start: int
    end: int
    line: int
    text: str


@dataclass(frozen=True)
class FactRule:
    name: str
    pattern: re.Pattern
    keep: MatchFilter | None = None

    def with_filter(self, keep: MatchFilter) -> FactRule:
        return replace(self, keep=keep)


def find_facts(text: str, rules: Sequence[FactRule]) -> tuple[Fact, ...]:
    facts = [
        Fact(rule.name, match.start(), match.end(), text.count("\n", 0, match.start()) + 1, match.group(0))
        for rule in rules
        for match in rule.pattern.finditer(text)
        if rule.keep is None or rule.keep(text, match)
    ]
    return tuple(sorted(facts, key=lambda fact: (fact.start, fact.name)))


def outside_names(text: str, match: re.Match) -> bool:
    """False when the match sits inside a path, file name or identifier: it touches ``/ . - _`` or a
    letter or digit, within a whitespace-delimited token that holds a path separator or ends in a file
    extension, or it is glued to letters or digits on either side."""
    before = text[match.start() - 1] if match.start() > 0 else " "
    after = text[match.end()] if match.end() < len(text) else " "
    if before.isalnum() or after.isalnum():
        return False
    token = _token_around(text, match.start(), match.end())
    joined = before in _NAME_JOINERS or after in _NAME_JOINERS
    return not (joined and ("/" in token or _FILE_EXTENSION.search(token.rstrip(".,;:)")) or "_" in token))


def _token_around(text: str, start: int, end: int) -> str:
    left = text.rfind(" ", 0, start)
    left = max(left, text.rfind("\n", 0, start)) + 1
    right_candidates = [index for index in (text.find(" ", end), text.find("\n", end)) if index != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left:right].strip("`'\"()[]")


TODO_WITHOUT_OWNER = FactRule(
    "todo_without_owner", re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b(?!\s*[(:]\s*@?\w+[)])(?!\s*@\w+)")
)
DATE = FactRule("date", re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"))
TICKET_REFERENCE = FactRule(
    "ticket_reference", re.compile(r"(?<![\w&])#\d{2,}\b|\b[A-Z][A-Z0-9]+-\d+\b|\bPR\s*#?\d+\b")
)
COMMENTED_OUT_CODE = FactRule(
    "commented_out_code",
    re.compile(
        r"(?m)^\s*(?:#+|//+)\s?(?:(?:if|for|while|return|def|class|import|from|const|let|var|function)\b.*"
        r"|.*[;{}]\s*|[\w.]+\(.*\)\s*|\w+\s*=\s*\S.*)$"
    ),
)
DEFAULT_COMMENT_RULES: tuple[FactRule, ...] = (TODO_WITHOUT_OWNER, COMMENTED_OUT_CODE, DATE, TICKET_REFERENCE)
