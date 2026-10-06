"""What a search examined for each point, and the outcome that holds over it.

A point is ``found`` when a unit's answer reaches the bar. Otherwise it is ``none_among_judged`` over
the units judged, with every unit the search left unjudged counted by the source that listed it and
the reason, so "none" never claims more than was examined. It is ``unknown`` when no unit was judged
or the search failed, never "none". Counts are in units, each counted once over a search's rounds: a
unit judged in any round is judged, and a cut unit counts as judged when any of its pieces was.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ..index.units import Unit
from .find_all import DELIVERED, NOT_REACHED, REFUSED, TOO_LARGE, FindAllResult
from .frontier import Source

UNJUDGED_PRECEDENCE = (NOT_REACHED, REFUSED, TOO_LARGE, DELIVERED)
"""The reason a unit left unjudged at several places or in several rounds counts under: the first
here, since a place not reached could still change the answer and one refused might on a retry."""
_SOURCE_LABELS = {
    Source.ANCHOR: "from anchors",
    Source.FILE: "from files",
    Source.NAME: "from name hits",
    Source.CALLEE: "callees",
}
_CUT_LABELS = ((REFUSED, "refused"), (TOO_LARGE, "too large to judge"))


class Outcome(StrEnum):
    FOUND = "found"
    NONE_AMONG_JUDGED = "none_among_judged"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Round:
    """One find_all search of a composed search. With ``source``, every unit the round lists first
    counts under the source the caller fed it as; without it, under the source find_all recorded."""

    result: FindAllResult
    source: Source | None = None


@dataclass(frozen=True)
class PointCoverage:
    """The units a search ``considered``, the ones it ``judged`` for the point, and the rest ``cut``,
    counted by the source that listed them and the reason they stayed unjudged."""

    considered: int
    judged: int
    cut: Mapping[tuple[Source, str], int]

    def unjudged(self, reason: str) -> Counter[Source]:
        return Counter({source: count for (source, why), count in self.cut.items() if why == reason})


@dataclass(frozen=True)
class PointResult:
    """One point's outcome, its best probability (None when no unit was judged) and its coverage.
    ``failure`` is the failed round's error when the search failed."""

    point: str
    outcome: Outcome
    best: float | None
    coverage: PointCoverage
    failure: str = ""

    def render(self, bar: float) -> str:
        cuts = _cuts_text(self.coverage)
        judged = self.coverage.judged
        if self.outcome is Outcome.FOUND:
            return f"{self.point}: found, best P={self.best:.3f} of {judged} unit(s) judged; {cuts}"
        if self.outcome is Outcome.NONE_AMONG_JUDGED:
            return (
                f"{self.point}: none at the bar {bar:.2f} among {judged} unit(s) judged, "
                f"best P={self.best:.3f}; {cuts} (not negative proof)"
            )
        if not self.failure:
            return f"{self.point}: unknown, no unit judged; {cuts}"
        examined = self._examined_before_failing()
        return f"{self.point}: unknown, the search failed ({self.failure}) {examined}; {cuts}"

    def _examined_before_failing(self) -> str:
        if not self.coverage.judged:
            return "before any unit was judged"
        return f"after {self.coverage.judged} unit(s) judged, best P={self.best:.3f}"


def point_results(rounds: Sequence[Round], bar: float) -> tuple[PointResult, ...]:
    """Each point's result over ``rounds``, the rounds of one search, each asking the same points."""
    search = _Search.of(rounds)
    points = dict.fromkeys(point for round_ in rounds for point in round_.result.targets)
    return tuple(search.point_result(point, bar) for point in points)


@dataclass(frozen=True)
class _Search:
    """``sources`` holds each unit the rounds listed, under its first source; ``reasons`` each unit
    left unjudged at some place, under the reason that takes precedence."""

    rounds: Sequence[Round]
    sources: Mapping[str, Source]
    reasons: Mapping[str, str]
    failure: str

    @classmethod
    def of(cls, rounds: Sequence[Round]) -> _Search:
        return cls(rounds, _first_sources(rounds), _unjudged_reasons(rounds), _failure(rounds))

    def point_result(self, point: str, bar: float) -> PointResult:
        best_of = self._best_per_unit(point)
        best = max(best_of.values(), default=None)
        coverage = PointCoverage(len(self.sources), len(best_of), self._cut(best_of))
        outcome = _outcome(best, coverage.judged, self.failure, bar)
        return PointResult(point, outcome, best, coverage, self.failure)

    def _cut(self, judged: Mapping[str, float]) -> dict[tuple[Source, str], int]:
        unjudged = (unit_id for unit_id in self.reasons if unit_id not in judged)
        return dict(Counter((self.sources[unit_id], self.reasons[unit_id]) for unit_id in unjudged))

    def _best_per_unit(self, point: str) -> dict[str, float]:
        best: dict[str, float] = {}
        for round_ in self.rounds:
            for score in round_.result.scores(point):
                best[score.unit.id] = max(score.probability, best.get(score.unit.id, score.probability))
        return best


def _outcome(best: float | None, judged: int, failure: str, bar: float) -> Outcome:
    if best is not None and best >= bar:
        return Outcome.FOUND
    if not judged or failure:
        return Outcome.UNKNOWN
    return Outcome.NONE_AMONG_JUDGED


def _first_sources(rounds: Iterable[Round]) -> dict[str, Source]:
    sources: dict[str, Source] = {}
    for round_ in rounds:
        for unit in round_.result.units:
            sources.setdefault(unit.id, round_.source or round_.result.entered_by[unit.id])
    return sources


def _unjudged_reasons(rounds: Iterable[Round]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for round_ in rounds:
        unit_of = _unit_of_place(round_.result.units)
        for place_id, reason in round_.result.not_judged.items():
            unit_id = unit_of[place_id]
            reasons[unit_id] = min(reasons.get(unit_id, reason), reason, key=UNJUDGED_PRECEDENCE.index)
    return reasons


def _unit_of_place(units: Iterable[Unit]) -> dict[str, str]:
    """Each place a unit is judged or left at, the unit itself or one of its pieces, with the unit."""
    return {place: unit.id for unit in units for place in (unit.id, *map(unit.piece_id, unit.pieces))}


def _failure(rounds: Iterable[Round]) -> str:
    failed = next((round_.result.failure for round_ in rounds if round_.result.stopped_by == "failed"), None)
    return "" if failed is None else f"{type(failed).__name__}: {failed}"


def _cuts_text(coverage: PointCoverage) -> str:
    unreached = coverage.unjudged(NOT_REACHED)
    text = f"{sum(unreached.values())} not reached{_split(unreached)}"
    for reason, label in _CUT_LABELS:
        if count := sum(coverage.unjudged(reason).values()):
            text += f"; {count} {label}"
    return text


def _split(unreached: Counter[Source]) -> str:
    present = [source for source in Source if unreached[source]]
    if len(present) < 2:
        return ""
    return " (" + ", ".join(f"{unreached[source]} {_SOURCE_LABELS[source]}" for source in present) + ")"
