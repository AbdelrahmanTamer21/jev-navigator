"""Scope-wide tables built with one ast-grep scan each: symbols and declarations per file, every call
site, and every non-call reference. Lookups then filter a table instead of running ast-grep again.

Each scan runs in batches of ``BATCH_FILES`` files, so one slow batch cannot fail the index: a batch
that times out is logged and its files are reported as unparsed, and every other file still counts.
The structure scan also matches the grammar's ERROR nodes: a file the parser could only recover
partially (Flow types in a JavaScript file, say) is reported as unparsed too. Its matched symbols and
calls still count — recovery keeps what it could — but whatever the ERROR nodes swallowed is unknown,
not absent, exactly as for a timed-out batch.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import tools
from .languages import (
    CLASS_KINDS,
    DECLARATION_RULES,
    FLOW_LANGUAGE,
    FLOW_SGCONFIG,
    FUNCTION_KINDS,
    declared_name,
    function_name,
    grammar_of,
    language_for,
    language_of,
    reference_rules,
)
from .spans import Span

LinesOf = Callable[[str], Sequence[str]]
BATCH_FILES = 100

logger = logging.getLogger(__name__)


@dataclass
class Unparsed:
    """Files a scan could not parse — a timed-out batch, or grammar ERROR nodes the parser only
    recovered partially; lookups treat them as holding nothing they have not matched."""

    files_by_scan: dict[str, set[str]] = field(default_factory=dict)

    def add(self, scan: str, files: Sequence[str]) -> None:
        self.files_by_scan.setdefault(scan, set()).update(files)

    @property
    def files(self) -> frozenset[str]:
        return frozenset(file for files in self.files_by_scan.values() for file in files)


@dataclass(frozen=True)
class FileStructure:
    functions: tuple[Span, ...]
    symbols: tuple[Span, ...]
    declarations: tuple[Span, ...]


@dataclass(frozen=True)
class CallMatch:
    file: str
    line: int
    name: str
    receiver: str | None


@dataclass(frozen=True, order=True)
class ReferenceMatch:
    file: str
    line: int
    role: str
    name: str


def scan_structure(
    files: Sequence[str], root: Path, lines_of: LinesOf, unparsed: Unparsed
) -> dict[str, FileStructure]:
    matches = _batched("structure", files, lines_of, _structure_rules, root, unparsed)
    functions: dict[str, set[Span]] = {file: set() for file in files}
    classes: dict[str, set[Span]] = {file: set() for file in files}
    declarations: dict[str, set[Span]] = {file: set() for file in files}
    for match in matches:
        file, start, end = match["file"], _line_of(match), match["range"]["end"]["line"] + 1
        if match["ruleId"] == _ERROR_RULE:
            # The grammar reports ERROR nodes here: whatever recovery swallowed is unknown, while the
            # symbols it did keep are still matched below.
            unparsed.add("structure", [file])
            continue
        first_line = lines_of(file)[start - 1]
        if match["ruleId"] == "declaration":
            declarations[file].add(Span(file, start, end, declared_name(first_line)))
        else:
            target = functions if match["ruleId"] == "function" else classes
            target[file].add(Span(file, start, end, function_name(first_line)))
    return {
        file: FileStructure(
            _ordered(functions[file]),
            _ordered(functions[file] | classes[file]),
            tuple(sorted(declarations[file])),
        )
        for file in files
    }


def scan_calls(
    files: Sequence[str], root: Path, unparsed: Unparsed, lines_of: LinesOf
) -> tuple[CallMatch, ...]:
    """Every call whose callee ends in a plain name, with the receiver before the last dot."""
    found = []
    for match in _batched("calls", files, lines_of, _call_rules, root, unparsed):
        expression = match["metaVariables"]["single"]["CALLEE"]["text"]
        name = last_identifier(expression)
        if name:
            found.append(CallMatch(match["file"], _line_of(match), name, receiver_of(expression)))
    return tuple(sorted(found, key=lambda call: (call.file, call.line)))


def scan_references(
    files: Sequence[str], root: Path, unparsed: Unparsed, lines_of: LinesOf
) -> tuple[ReferenceMatch, ...]:
    matches = _batched("references", files, lines_of, reference_rules, root, unparsed)
    return tuple(
        sorted(
            {
                ReferenceMatch(match["file"], _line_of(match), match["ruleId"], match["text"])
                for match in matches
            }
        )
    )


def _batched(
    scan: str,
    files: Sequence[str],
    lines_of: LinesOf,
    rules_of: Callable[[Sequence[str]], str],
    root: Path,
    unparsed: Unparsed,
) -> list[dict]:
    matches: list[dict] = []
    for config, group, languages in _scan_groups(files, lines_of):
        rules = rules_of(languages)
        for start in range(0, len(group), BATCH_FILES):
            batch = group[start : start + BATCH_FILES]
            try:
                matches += tools.ast_grep_rules(rules, batch, root, config=config)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "ast-grep timed out on %d files in the %s scan; they count as unparsed",
                    len(batch),
                    scan,
                )
                unparsed.add(scan, batch)
    return matches


def receiver_of(expression: str) -> str | None:
    head, dot, _ = expression.replace("?.", ".").rpartition(".")
    return head if dot else None


def last_identifier(expression: str) -> str:
    tail = expression.replace("?.", ".").split(".")[-1]
    return tail if tail.isidentifier() else ""


_ERROR_RULE = "parse_error"


def _structure_rules(languages: Sequence[str]) -> str:
    documents = []
    for language in languages:
        documents.append(_kind_rule("function", language, FUNCTION_KINDS[language]))
        documents.append(_kind_rule("class", language, CLASS_KINDS[language]))
        documents.append(
            f"id: declaration\nlanguage: {grammar_of(language)}\nrule:\n{DECLARATION_RULES[language]}"
        )
        documents.append(f"id: {_ERROR_RULE}\nlanguage: {grammar_of(language)}\nrule:\n  kind: ERROR")
    return "\n---\n".join(documents)


def _call_rules(languages: Sequence[str]) -> str:
    return "\n---\n".join(
        f"id: call\nlanguage: {grammar_of(language)}\nrule:\n  pattern: $CALLEE($$$)"
        for language in languages
    )


def _kind_rule(rule_id: str, language: str, kinds: Sequence[str]) -> str:
    listed = "".join(f"\n    - kind: {kind}" for kind in kinds)
    return f"id: {rule_id}\nlanguage: {grammar_of(language)}\nrule:\n  any:{listed}"


def _scan_groups(files: Sequence[str], lines_of: LinesOf) -> list[tuple[str | None, list[str], list[str]]]:
    """(sgconfig, files, languages) per invocation: every non-flow file scanned together exactly as
    before, and the ``@flow`` files in their own invocation, where the config's ``languageGlobs``
    parses the JavaScript suffixes with the tsx grammar. The globs are global per invocation, so
    mixing the two would re-parse plain JavaScript files too."""
    plain: list[str] = []
    flow: list[str] = []
    for file in files:
        if language_for(file, lines_of(file)) == FLOW_LANGUAGE:
            flow.append(file)
        else:
            plain.append(file)
    groups: list[tuple[str | None, list[str], list[str]]] = []
    if plain:
        groups.append((None, plain, _languages(plain)))
    if flow:
        groups.append((FLOW_SGCONFIG, flow, [FLOW_LANGUAGE]))
    return groups


def _languages(files: Sequence[str]) -> list[str]:
    return sorted({language for file in files if (language := language_of(file))})


def _ordered(spans: set[Span]) -> tuple[Span, ...]:
    return tuple(sorted(spans, key=lambda span: (span.start, -span.end)))


def _line_of(match: dict) -> int:
    return match["range"]["start"]["line"] + 1
