"""Structural excerpts over parser facts and literal source lines.

Keep units of at most 25 physical lines whole. Larger units retain signatures,
decorators, query matches with two context lines, controlling branch headers and
exits. Every removed run is named with its original line numbers.
"""

from __future__ import annotations

import io
import keyword
import re
import tokenize
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from ..index.code_index import CodeIndex
from ..index.imports import without_comments
from ..index.languages import CLASS_KINDS, FUNCTION_KINDS, grammar_of, sgconfig_of
from ..index.tools import ast_grep_rules
from ..index.units import Unit
from ..mentions import code_names_in, names_from_text, spelling_variants

STOP_WORDS = frozenset(
    [
        "a",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "behavior",
        "being",
        "by",
        "can",
        "code",
        "could",
        "described",
        "did",
        "do",
        "does",
        "each",
        "every",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "json",
        "may",
        "md",
        "mjs",
        "must",
        "no",
        "not",
        "of",
        "on",
        "only",
        "or",
        "our",
        "py",
        "relevant",
        "should",
        "shown",
        "test",
        "tests",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "unit",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "whose",
        "with",
        "without",
        "would",
        "yaml",
        "your",
    ]
)


_SyntaxSpan = tuple[str, tuple[int, int], tuple[int, int]]


@lru_cache(maxsize=32_768)
def _words(text: str) -> frozenset[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return frozenset(
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9]*", text)
        if len(word) > 2 and word.lower() not in STOP_WORDS
    )


def batch_query(index: CodeIndex, files: list[str], concerns: list[str]) -> str:
    """Actual concerns in order, followed by sorted identifiers from the batch's source."""
    names: set[str] = set()
    for path in files:
        if path not in index.files:
            continue
        text = "\n".join(index.lines(path))
        if path.endswith(".py"):
            names.update(
                token.string
                for token in tokenize.generate_tokens(io.StringIO(text).readline)
                if token.type == tokenize.NAME and not keyword.iskeyword(token.string)
            )
        else:
            names.update(code_names_in(without_comments(text, path)))
        facts = index.facts_in_files([path]).get(path)
        if facts:
            names.update(call.name for call in facts.calls)
            names.update(ref.name for ref in facts.references)
            names.update(span.name for span in index.symbols_in(path) if span.name)
            names.update(span.name for span in index.declarations_in(path) if span.name)
    return " ".join([*concerns, *sorted(names)])


def _syntax_rule(language: str, role: str, kind: str, field: str | None = None) -> str:
    rule = f"id: {role}_{kind}\nlanguage: {grammar_of(language)}\nrule:\n  kind: {kind}"
    if field:
        rule += f"\n  has:\n    field: {field}\n    pattern: $PART"
    return rule


def _syntax_rules(language: str) -> str:
    grammar = grammar_of(language)
    rules = [
        _syntax_rule(language, "signature", kind, "body")
        for kind in (*FUNCTION_KINDS[grammar], *CLASS_KINDS[grammar])
    ]
    conditions = {
        "if_statement": "consequence",
        "while_statement": "body",
        "for_statement": "body",
    }
    if language == "python":
        rules.append(_syntax_rule(language, "decorator", "decorator"))
        conditions.update(
            {
                "elif_clause": "consequence",
                "else_clause": "body",
                "except_clause": None,
                "with_statement": "body",
                "try_statement": "body",
                "finally_clause": None,
            }
        )
        exits = ("return_statement", "raise_statement")
    else:
        conditions.update(
            {
                "for_in_statement": "body",
                "switch_statement": "body",
                "catch_clause": "body",
                "else_clause": None,
                "try_statement": "body",
                "finally_clause": None,
                "do_statement": "body",
            }
        )
        exits = ("return_statement", "throw_statement")
    rules.extend(_syntax_rule(language, "condition", kind, field) for kind, field in conditions.items())
    rules.extend(
        _syntax_rule(language, "exit", kind) for kind in (*exits, "break_statement", "continue_statement")
    )
    return "\n---\n".join(rules)


def _node_range(match: dict[str, Any]) -> tuple[int, int]:
    location = match["range"]
    start = location["start"]["line"] + 1
    end = location["end"]["line"] + (location["end"]["column"] > 0)
    return start, max(start, end)


def _add(selected: set[int], interval: tuple[int, int], visible: set[int]) -> None:
    start, end = interval
    selected.update(number for number in range(start, end + 1) if number in visible)


def _syntax_span(match: dict[str, Any], language: str) -> _SyntaxSpan:
    role = match["ruleId"].split("_", 1)[0]
    interval = _node_range(match)
    part = match.get("metaVariables", {}).get("single", {}).get("PART")
    body_start = part["range"]["start"]["line"] + 1 if part else interval[0] + 1
    end = max(interval[0], body_start - (language == "python")) if part else interval[0]
    return role, interval, (interval[0], end)


def _condition_closure(
    selected: set[int],
    conditions: list[tuple[tuple[int, int], tuple[int, int]]],
    visible: set[int],
    language: str | None,
) -> None:
    for interval, header in sorted(conditions, key=lambda item: item[0][1] - item[0][0]):
        if any(interval[0] <= number <= interval[1] for number in selected):
            _add(selected, header, visible)
            if language != "python":
                _add(selected, (interval[1], interval[1]), visible)


def _matching_context(lines: list[str], visible: set[int], query: str) -> set[int]:
    terms = _words(query)
    variants = {variant for name in names_from_text(query).code for variant in spelling_variants(name)}
    pattern = (
        re.compile(r"(?<![\w$])(?:" + "|".join(map(re.escape, sorted(variants))) + r")(?![\w$])")
        if variants
        else None
    )
    selected: set[int] = set()
    for number in visible:
        text = lines[number - 1]
        if _words(text) & terms or pattern is not None and pattern.search(text):
            _add(selected, (number - 2, number + 2), visible)
    return selected


class StructuralExcerpts:
    """Reuse compact syntax per file while rendering each query's source excerpts."""

    def __init__(self, index: CodeIndex, whole_unit_lines: int = 25) -> None:
        self.index = index
        self.whole_unit_lines = whole_unit_lines
        self.refused: dict[str, str] = {}
        self._structures: dict[str, tuple[str | None, list[_SyntaxSpan]]] = {}

    def _structure(self, path: str) -> tuple[str | None, list[_SyntaxSpan]]:
        if path not in self._structures:
            self._structures[path] = self._load_structure(path)
        return self._structures[path]

    def _load_structure(self, path: str) -> tuple[str | None, list[_SyntaxSpan]]:
        facts = self.index.facts_in_files([path]).get(path)
        if not facts or not facts.language or facts.refusal:
            if facts and facts.refusal:
                self.refused[path] = facts.refusal
            return None, []
        refused: dict[str, str] = {}
        matches = ast_grep_rules(
            _syntax_rules(facts.language),
            [path],
            self.index.root,
            config=sgconfig_of(facts.language),
            refused=refused,
        )
        spans = [_syntax_span(match, facts.language) for match in matches]
        if refused:
            self.refused.update(refused)
            return None, []
        return facts.language, spans

    def render(
        self, unit: Unit, query: str, *, mask_lines: Callable[[str, list[str]], list[str]] | None = None
    ) -> str:
        """Render literal masked source, preserving physical numbers and explicit gaps."""
        source = list(self.index.lines(unit.path))
        lines = mask_lines(unit.path, source) if mask_lines else source
        visible = {number for start, end in unit.ranges for number in range(start, end + 1)}
        selected = (
            self._selected(unit.path, visible, source, query)
            if len(visible) > self.whole_unit_lines
            else visible
        )
        body: list[str] = []
        for start, end in unit.ranges:
            cursor = start
            while cursor <= end:
                kept = cursor in selected
                last = cursor
                while last < end and (last + 1 in selected) == kept:
                    last += 1
                if kept:
                    body.extend(f"{number}: {lines[number - 1]}" for number in range(cursor, last + 1))
                else:
                    body.append(f"... ELIDED lines {cursor}-{last} ({last - cursor + 1} lines) ...")
                cursor = last + 1
        return f"===== {unit.path} =====\n" + "\n".join(body)

    def _selected(self, path: str, visible: set[int], lines: list[str], query: str) -> set[int]:
        language, matches = self._structure(path)
        selected: set[int] = set()
        conditions: list[tuple[tuple[int, int], tuple[int, int]]] = []
        exits: list[tuple[int, int]] = []
        for role, interval, header in matches:
            if role == "signature" and interval[0] in visible:
                _add(selected, header, visible)
                if language != "python":
                    _add(selected, (interval[1], interval[1]), visible)
            elif role == "decorator":
                _add(selected, interval, visible)
            elif role == "condition":
                conditions.append((interval, header))
            elif role == "exit":
                exits.append(interval)
        if not selected:
            selected.add(min(visible))
        selected.update(_matching_context(lines, visible, query))
        _condition_closure(selected, conditions, visible, language)
        for interval in exits:
            _add(selected, interval, visible)
        _condition_closure(selected, conditions, visible, language)
        return selected
