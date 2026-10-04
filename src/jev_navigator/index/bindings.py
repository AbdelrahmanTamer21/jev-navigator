"""How sure the index is that a call reaches a given definition.

Calls are found by name in the syntax tree; a name match is not a resolved binding. Each call gets a
status: ``resolved`` when a definition in the same file or an import naming it proves the target,
``candidate`` when a name or a repository import mapping suggests a target without proving it, and
``unresolved`` when no definition exists in the index scope. A host with a real
resolver (a code-intelligence service, a TypeScript alias resolver, an LSP) injects it as a
``BindingResolver``; its answer wins whenever it returns one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .imports import ImportFact
from .spans import Span


class BindingStatus(StrEnum):
    RESOLVED = "resolved"
    CANDIDATE = "candidate"
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Binding:
    status: str
    reason: str
    target: Span | None = None

    @property
    def proven(self) -> bool:
        return self.status == BindingStatus.RESOLVED


class BindingResolver(Protocol):
    def resolve_call(self, file: str, line: int, name: str, receiver: str | None) -> Binding | None: ...


@dataclass(frozen=True)
class CallFacts:
    """What the index knows about one call when no injected resolver answers. ``top_level`` holds the
    definitions no class or function contains: only those can be named from their file's module
    scope, or by an import."""

    file: str
    name: str
    receiver: str | None
    definitions: Sequence[Span]
    top_level: Sequence[Span]
    imported_from: Sequence[ImportFact]
    # Files that could hold a definition of ``name`` the index never saw.
    unparsed: frozenset[str] = frozenset()


def binding_from_facts(facts: CallFacts) -> Binding:
    """``unknown`` when the definition may sit where the index could not parse: no definition was
    found, or the import or a definition names a file that could hold one unseen. Missing evidence is
    never absence."""
    unparsed_import = [fact.path for fact in facts.imported_from if fact.path in facts.unparsed]
    unparsed_definitions = [span.file for span in facts.definitions if span.file in facts.unparsed]
    if facts.unparsed and (not facts.definitions or unparsed_import or unparsed_definitions):
        files = ", ".join(sorted(unparsed_import or unparsed_definitions or facts.unparsed)[:5])
        return Binding(BindingStatus.UNKNOWN, f"{facts.name} may be defined in files not parsed: {files}")
    if not facts.definitions:
        return Binding(BindingStatus.UNRESOLVED, f"no definition of {facts.name} in the index scope")
    if facts.receiver is not None:
        count = len(facts.definitions)
        return Binding(
            BindingStatus.CANDIDATE,
            f"method call on {facts.receiver}; receiver type not resolved ({count} definitions)",
        )
    same_file = _one_per_definition([span for span in facts.top_level if span.file == facts.file])
    if len(same_file) == 1:
        return Binding(BindingStatus.RESOLVED, "defined in the same file", same_file[0])
    if same_file:
        return Binding(
            BindingStatus.CANDIDATE, f"{len(same_file)} definitions of {facts.name} in the same file"
        )
    imported = [
        (span, fact)
        for span in _one_per_definition(facts.top_level)
        for fact in facts.imported_from
        if span.file == fact.path
    ]
    if len(imported) == 1:
        span, fact = imported[0]
        if fact.proven:
            return Binding(BindingStatus.RESOLVED, f"imported from {span.file}", span)
        return Binding(BindingStatus.CANDIDATE, f"import suggests {span.file}: {fact.reason}")
    if imported:
        paths = ", ".join(dict.fromkeys(span.file for span, _ in imported))
        return Binding(BindingStatus.CANDIDATE, f"import suggests multiple definitions: {paths}")
    count = len(facts.definitions)
    return Binding(
        BindingStatus.CANDIDATE, f"name match only; {count} definitions in scope and no import names it"
    )


def _one_per_definition(spans: Sequence[Span]) -> list[Span]:
    """The first span of each definition: a declaration and the function or class it holds overlap,
    `const load =\n  () => 2` is one definition of `load` over lines 1 to 2 and 2 to 2."""
    kept: list[Span] = []
    for span in spans:
        if not any(
            other.file == span.file and other.start <= span.end and span.start <= other.end for other in kept
        ):
            kept.append(span)
    return kept
