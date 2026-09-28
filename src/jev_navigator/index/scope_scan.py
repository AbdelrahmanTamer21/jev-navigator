"""Syntax facts extracted together in one ast-grep pass over each requested file set.

Each requested file set is handed to one ast-grep scan, which schedules parsing across its own worker
pool without reparsing arbitrary fixed-size batches. The structure rules also match the grammar's
ERROR nodes: a file the parser could only recover
partially (Flow types in a JavaScript file, say) is reported as unparsed too. Its matched symbols and
calls still count — recovery keeps what it could — but whatever the ERROR nodes swallowed is unknown,
not absent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import tools
from .languages import (
    CLASS_KINDS,
    DECLARATION_RULES,
    FUNCTION_KINDS,
    declared_name,
    function_name,
    language_of,
    reference_rules,
)
from .spans import Span

LinesOf = Callable[[str], Sequence[str]]
logger = logging.getLogger(__name__)


@dataclass
class Unparsed:
    """Files with grammar ERROR nodes the parser only recovered partially.

    Lookups keep recovered matches while treating anything the parser omitted as unknown.
    """

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


@dataclass(frozen=True)
class FileFacts:
    structure: FileStructure
    calls: tuple[CallMatch, ...]
    references: tuple[ReferenceMatch, ...]
    incomplete: bool = False


def scan_facts(
    files: Sequence[str], root: Path, lines_of: LinesOf, unparsed: Unparsed
) -> dict[str, FileFacts]:
    """Extract structure, calls, and references in one parser pass over each supplied file."""
    matches = _batched(
        "facts",
        "\n---\n".join((_structure_rules(files), _call_rules(files), reference_rules())),
        files,
        root,
        unparsed,
    )
    structure = _structure_from_matches(
        files,
        lines_of,
        unparsed,
        (match for match in matches if match["ruleId"] in {"function", "class", "declaration", _ERROR_RULE}),
    )
    calls = _calls_from_matches(match for match in matches if match["ruleId"] == "call")
    references = _references_from_matches(
        match
        for match in matches
        if match["ruleId"] not in {"function", "class", "declaration", _ERROR_RULE, "call"}
    )
    return {
        file: FileFacts(
            structure[file],
            tuple(call for call in calls if call.file == file),
            tuple(reference for reference in references if reference.file == file),
            file in unparsed.files,
        )
        for file in files
    }


def scan_structure(
    files: Sequence[str], root: Path, lines_of: LinesOf, unparsed: Unparsed
) -> dict[str, FileStructure]:
    matches = _batched("structure", _structure_rules(files), files, root, unparsed)
    return _structure_from_matches(files, lines_of, unparsed, matches)


def _structure_from_matches(files, lines_of, unparsed, matches):
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


def scan_calls(files: Sequence[str], root: Path, unparsed: Unparsed) -> tuple[CallMatch, ...]:
    """Every call whose callee ends in a plain name, with the receiver before the last dot."""
    return _calls_from_matches(_batched("calls", _call_rules(files), files, root, unparsed))


def _call_rules(files: Sequence[str]) -> str:
    return "\n---\n".join(
        f"id: call\nlanguage: {language}\nrule:\n  pattern: $CALLEE($$$)" for language in _languages(files)
    )


def _calls_from_matches(matches) -> tuple[CallMatch, ...]:
    found = []
    for match in matches:
        expression = match["metaVariables"]["single"]["CALLEE"]["text"]
        name = last_identifier(expression)
        if name:
            found.append(CallMatch(match["file"], _line_of(match), name, receiver_of(expression)))
    return tuple(sorted(found, key=lambda call: (call.file, call.line)))


def scan_references(files: Sequence[str], root: Path, unparsed: Unparsed) -> tuple[ReferenceMatch, ...]:
    matches = _batched("references", reference_rules(), files, root, unparsed)
    return _references_from_matches(matches)


def _references_from_matches(matches) -> tuple[ReferenceMatch, ...]:
    return tuple(
        sorted(
            {
                ReferenceMatch(match["file"], _line_of(match), match["ruleId"], match["text"])
                for match in matches
            }
        )
    )


def _batched(scan: str, rules: str, files: Sequence[str], root: Path, unparsed: Unparsed) -> list[dict]:
    del scan, unparsed
    return tools.ast_grep_rules(rules, files, root)


def receiver_of(expression: str) -> str | None:
    head, dot, _ = expression.replace("?.", ".").rpartition(".")
    return head if dot else None


def last_identifier(expression: str) -> str:
    tail = expression.replace("?.", ".").split(".")[-1]
    return tail if tail.isidentifier() else ""


_ERROR_RULE = "parse_error"


def _structure_rules(files: Sequence[str]) -> str:
    documents = []
    for language in _languages(files):
        documents.append(_kind_rule("function", language, FUNCTION_KINDS[language]))
        documents.append(_kind_rule("class", language, CLASS_KINDS[language]))
        documents.append(f"id: declaration\nlanguage: {language}\nrule:\n{DECLARATION_RULES[language]}")
        documents.append(f"id: {_ERROR_RULE}\nlanguage: {language}\nrule:\n  kind: ERROR")
    return "\n---\n".join(documents)


def _kind_rule(rule_id: str, language: str, kinds: Sequence[str]) -> str:
    listed = "".join(f"\n    - kind: {kind}" for kind in kinds)
    return f"id: {rule_id}\nlanguage: {language}\nrule:\n  any:{listed}"


def _languages(files: Sequence[str]) -> list[str]:
    return sorted({language for file in files if (language := language_of(file))})


def _ordered(spans: set[Span]) -> tuple[Span, ...]:
    return tuple(sorted(spans, key=lambda span: (span.start, -span.end)))


def _line_of(match: dict) -> int:
    return match["range"]["start"]["line"] + 1
