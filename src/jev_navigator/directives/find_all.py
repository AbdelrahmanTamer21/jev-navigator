"""Seed-first semantic enumeration: expand relationships, judge, then search outside them.

The host obtains seeds through find_code or symbol lookup. This composition examines function
bodies. Static reachability prioritizes candidates; it never proves their semantic relevance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from .. import operations
from ..index.code_index import CodeIndex
from ..index.languages import language_of
from ..index.spans import Span
from ..judgments.judge import CheckResult, Judge
from ..judgments.questions import Check
from ..judgments.thresholds import NoulVerdict
from .find_code import FOUND

CONTAINS_IMPLEMENTATION = replace(
    FOUND,
    instructions=FOUND.instructions.replace("slice.code", "{item}.code"),
    yes=replace(FOUND.yes, what=FOUND.yes.what.replace("slice.code", "{item}.code")),
    no=replace(FOUND.no, what=FOUND.no.what.replace("slice.code", "{item}.code")),
)


@dataclass(frozen=True)
class FindAllResult:
    target: str
    graph: operations.TraceGraph
    judged: tuple[CheckResult, ...]
    remaining_files: tuple[str, ...]
    unparsed_files: frozenset[str]
    unsupported_files: tuple[str, ...]
    unavailable_files: Mapping[str, str]
    stopped_by: str
    calls: int

    @property
    def matched(self) -> tuple[CheckResult, ...]:
        return tuple(answer for answer in self.judged if answer.verdict == NoulVerdict.YES)

    @property
    def uncertain(self) -> tuple[CheckResult, ...]:
        return tuple(answer for answer in self.judged if answer.verdict == NoulVerdict.UNSURE)

    @property
    def negative(self) -> tuple[CheckResult, ...]:
        return tuple(answer for answer in self.judged if answer.verdict == NoulVerdict.NO)

    @property
    def coverage(self) -> str:
        """Examination coverage, never proof that the semantic answers are correct."""
        if self.stopped_by != "scope_examined" or self.remaining_files:
            return "partial"
        if self.unparsed_files or self.unsupported_files or self.unavailable_files:
            return "scope_incomplete"
        return "functions_examined"


def find_all(
    index: CodeIndex,
    judge: Judge,
    target: str,
    seeds: Sequence[Span],
    *,
    include_disconnected: bool = True,
    check: Check = CONTAINS_IMPLEMENTATION,
    cancelled: Callable[[], bool] | None = None,
) -> FindAllResult:
    """Expand from concrete seeds, batch a property check, then examine remaining functions.

    Seeds are candidate locations, not assumed matches. Candidate and unresolved graph bindings
    remain unchanged. The fallback includes disconnected and differently named functions. An
    empty seed list runs just that fallback; include_disconnected=False reports partial coverage.

    Judge owns packing, caching, secrets and caller-selected live-call budgets. This composition
    has no file/function/call cap and does not truncate bodies. Provider errors propagate with
    their cause; the Judge journal retains the requests and responses actually made.
    """
    judge = judge.scope()
    graph = operations.trace_graph(index, seeds, cancelled=cancelled)
    judged: list[CheckResult] = []
    seen: set[str] = set()
    remaining_files = list(index.files)
    stop = "connected_component"

    def stopped() -> bool:
        return cancelled is not None and cancelled()

    def assess(spans: Sequence[Span]) -> None:
        fresh = {span.key: span for span in spans if span.key not in seen}
        items = []
        for span in fresh.values():
            code = index.read_slice(span, origin="findall")
            items.append({**code.source(), "span_key": span.key, "name": span.name, "code": code.text})
        if items:
            judged.extend(judge.check_each(check, items, {"target": {"description": target}}))
        seen.update(fresh)

    if graph.stop == "cancelled" or stopped():
        stop = "cancelled"
    else:
        # Graphs may reach classes/declarations too. They remain in the graph; the answer unit is
        # a function body, so only function candidates are judged here.
        functions = set(index.functions_in_files(tuple(dict.fromkeys(span.file for span in graph.functions))))
        assess([span for span in graph.functions if span in functions])
        if include_disconnected and not stopped():
            pending = index.functions_in_files(index.available_files)
            if not stopped():
                assess(pending)
                remaining_files.clear()
                stop = "scope_examined"
        if stopped():
            stop = "cancelled"

    return FindAllResult(
        target,
        graph,
        tuple(judged),
        tuple(remaining_files),
        index.observed_unparsed_files,
        tuple(file for file in index.files if not language_of(file)),
        index.unavailable_files,
        stop,
        judge.calls,
    )
