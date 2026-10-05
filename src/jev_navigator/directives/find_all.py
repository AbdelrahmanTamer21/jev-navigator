"""Judge every unit of a population against described targets: one Noul per unit per target.

The population is units (``index/units.py``), each one a listing lists: first the units the caller's
line and range anchors name, then the units of the named files, then the units holding each hit of
the named texts, names with fewer hits first, so a common word never decides which hits of a rare
name are seen. Code finds the units; Jev judges each one whole, or a unit larger than its room in a
request by its pieces. The Judge's call cap is the only budget: no code step is capped. The result
keeps every raw answer with its place; ranking and any bar belong to the caller.

``find_all`` judges code units only. ``find_all_text`` judges the text units of the files JVN does
not parse, the same way and with the same question, and never a code unit; ``find_text`` is the text
search that stops once a unit is found. A caller gives a text search its own Judge, so it never
spends code search's budget.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

from ..index.code_index import CodeIndex
from ..index.languages import language_read
from ..index.scope import is_lockfile
from ..index.spans import TextHit
from ..index.units import (
    Anchor,
    AnchorResolution,
    Item,
    LineAnchor,
    Piece,
    RangeAnchor,
    Reading,
    Unit,
    UnresolvedAnchor,
    best_piece,
    items_to_judge,
    list_units,
    read_ranges,
    resolve_anchors,
)
from ..judgments.judge import CallCapReachedError, CheckResult, Judge, Refusal
from ..judgments.questions import Check, item_path, serialized_chars
from ..judgments.thresholds import NoulVerdict
from .find_code import search_failure

ITEMS = "items"
TARGETS = "targets"
DELIVERED = "already delivered by the caller"
TOO_LARGE = "too large to judge"
NOT_REACHED = "not reached: the search stopped first"
REFUSED = "refused: the provider or the final secret scan refused its request, so its answer is unknown"
FOUND = "found"
FIND_TEXT_TARGET = "target"
BATCHES_PER_WAVE = 16
"""How many requests' worth of places one wave hands the Judge, in the population's order. The Judge
sends a wave's batches together and orders them by place, so the population's order holds between
waves, and exactly only at one batch per wave. Like ``items_per_request`` it shapes the batches, and
so the answer store's keys; it does not follow the Judge's concurrency, so that setting never moves
them."""

T = TypeVar("T")


class Source(StrEnum):
    """How a unit entered a population: find_all lists the units of the caller's anchors (``ANCHOR``),
    then of its files (``FILE``), then of each name's hits (``NAME``). A caller composing searches names
    the rest, such as a round of callees (``CALLEE``) it fed as anchors."""

    ANCHOR = "anchor"
    FILE = "file"
    NAME = "name"
    CALLEE = "callee"


def match_check(target: str) -> Check:
    """The one question asked of every unit for ``target``: no criteria, the description in the
    shared state, as jgrep asks it."""
    return Check(
        f"match_{target}",
        f"Look only at `{{item}}`. Does that code match the description in `{TARGETS}.{target}`?",
    )


@dataclass(frozen=True)
class NameHits:
    """A name's hits: how many the text search found in the files the search reads, how many the search
    reached before it stopped, and how many of those named no listed unit (a line of a file the
    reading leaves out, or of top-level code a listing leaves out, such as imports)."""

    found: int
    reached: int
    without_unit: int


@dataclass(frozen=True)
class UnitScore:
    """A unit's answer for one target: its own, or for a cut unit its best judged piece's."""

    unit: Unit
    answer: CheckResult
    piece: Piece | None

    @property
    def probability(self) -> float:
        return self.answer.probability


@dataclass(frozen=True)
class FindAllResult:
    """``units`` are every unit of the population, in place order; ``judged`` each target's answers,
    each naming its place; ``not_judged`` each unit or piece id left unjudged with the reason.
    ``room`` is what one unit's code had in a request and ``batches_per_wave`` the wave size, both
    None when the search never started. ``unlisted`` files gave no units and ``unresolved`` caller
    anchors named none. ``refusals`` keeps each refused place's error; the place is ``not_judged``
    as ``REFUSED`` and the search goes on past it. ``entered_by`` gives the source each unit entered
    the population by, the first when several listed it."""

    targets: Mapping[str, str]
    room: int | None
    batches_per_wave: int | None
    units: tuple[Unit, ...]
    judged: Mapping[str, tuple[CheckResult, ...]]
    not_judged: Mapping[str, str]
    unlisted: Mapping[str, str]
    unresolved: tuple[UnresolvedAnchor, ...]
    names: Mapping[str, NameHits]
    unparsed_files: frozenset[str]
    stopped_by: str
    calls: int
    failure: Exception | None = None
    refusals: tuple[Refusal, ...] = ()
    entered_by: Mapping[str, Source] = field(default_factory=dict)

    @classmethod
    def not_started(cls, targets: Mapping[str, str], stopped_by: str) -> FindAllResult:
        """The result of a search that stopped before it listed anything."""
        return cls(
            dict(targets),
            None,
            None,
            (),
            {target: () for target in targets},
            {},
            {},
            (),
            {},
            frozenset(),
            stopped_by,
            0,
        )

    def scores(self, target: str) -> tuple[UnitScore, ...]:
        """Every judged unit's answer for ``target``, in place order."""
        answers = {answer.place.id: answer for answer in self.judged[target] if answer.place is not None}
        probabilities = {place_id: answer.probability for place_id, answer in answers.items()}
        scored = (_unit_score(unit, answers, probabilities) for unit in self.units)
        return tuple(score for score in scored if score is not None)

    @property
    def coverage(self) -> str:
        """Examination coverage, never proof that the semantic answers are correct."""
        reasons = set(self.not_judged.values())
        if self.stopped_by != "scope_examined" or NOT_REACHED in reasons:
            return "partial"
        if self.unlisted or self.unresolved or self.unparsed_files or reasons & {TOO_LARGE, REFUSED}:
            return "scope_incomplete"
        return "units_examined"


def find_all(
    index: CodeIndex,
    judge: Judge,
    targets: Mapping[str, str],
    *,
    files: Sequence[str] = (),
    anchors: Sequence[Anchor] = (),
    names: Sequence[str] = (),
    delivered: Sequence[RangeAnchor] = (),
    completed: Mapping[str, Sequence[CheckResult]] | None = None,
    cancelled: Callable[[], bool] | None = None,
    batches_per_wave: int = BATCHES_PER_WAVE,
) -> FindAllResult:
    """Judge the units ``anchors`` name, then the units of ``files``, then the units holding each hit
    of ``names``, rarest name first, in waves of ``batches_per_wave`` requests' worth, until the
    judge's call cap stops it.

    ``targets`` maps a name (an identifier) to a description; each unit is asked one question per
    target, all in the same request. ``delivered`` names the lines the caller already shows: a unit
    or piece whose every line lies in them is not judged, one with a line outside them is. ``completed``
    holds each target's answers from an earlier run over the same code; a place answered for every
    target is not asked again. A failed request ends the search ``failed`` with ``failure`` holding
    the error (see ``search_failure``), Ctrl-C ends it ``cancelled``, and either way ``judged`` keeps
    every answer that arrived.
    """
    search = _begin(index, judge, targets, delivered, completed, cancelled, batches_per_wave, Reading.CODE)
    return _finished(search, anchors, files, names)


def find_all_text(
    index: CodeIndex,
    judge: Judge,
    targets: Mapping[str, str],
    *,
    files: Sequence[str] = (),
    anchors: Sequence[Anchor] = (),
    names: Sequence[str] = (),
    delivered: Sequence[RangeAnchor] = (),
    completed: Mapping[str, Sequence[CheckResult]] | None = None,
    cancelled: Callable[[], bool] | None = None,
    batches_per_wave: int = BATCHES_PER_WAVE,
) -> FindAllResult:
    """``find_all`` over text units (``Reading.TEXT``): the files JVN does not parse, each in the blocks
    its format gives, a code file named ``unlisted``. A name's hits in code and in lockfiles are left
    out, so a common word never floods the search with a lockfile's pieces; a lockfile ``files`` or
    ``anchors`` name is judged."""
    search = _begin(index, judge, targets, delivered, completed, cancelled, batches_per_wave, Reading.TEXT)
    return _finished(search, anchors, files, names)


def find_text(
    index: CodeIndex,
    judge: Judge,
    description: str,
    *,
    files: Sequence[str] = (),
    anchors: Sequence[Anchor] = (),
    names: Sequence[str] = (),
    delivered: Sequence[RangeAnchor] = (),
    cancelled: Callable[[], bool] | None = None,
    batches_per_wave: int = BATCHES_PER_WAVE,
) -> FindAllResult:
    """``find_all_text`` for the one target ``description``, named ``FIND_TEXT_TARGET``, ending
    ``found`` after the first wave in which a unit's answer is yes by the judge's thresholds."""
    targets = {FIND_TEXT_TARGET: description}
    search = _begin(index, judge, targets, delivered, None, cancelled, batches_per_wave, Reading.TEXT)
    search.stops_when_found = True
    return _finished(search, anchors, files, names)


def _finished(
    search: _Search, anchors: Sequence[Anchor], files: Sequence[str], names: Sequence[str]
) -> FindAllResult:
    try:
        search.run(anchors, files, names)
    except (KeyboardInterrupt, Exception) as error:  # noqa: BLE001 - _stop_by owns how an error ends a search
        return search.ended(error)
    return search.ended(None)


async def find_all_async(
    index: CodeIndex,
    judge: Judge,
    targets: Mapping[str, str],
    *,
    files: Sequence[str] = (),
    anchors: Sequence[Anchor] = (),
    names: Sequence[str] = (),
    delivered: Sequence[RangeAnchor] = (),
    completed: Mapping[str, Sequence[CheckResult]] | None = None,
    cancelled: Callable[[], bool] | None = None,
    batches_per_wave: int = BATCHES_PER_WAVE,
) -> FindAllResult:
    """``find_all`` with each wave's requests sent concurrently through the Judge's async form, for an
    async client. Listing, resolving and reading code run in a worker thread, so the event loop stays
    free. ``cancelled`` is read before each parse and between waves, since the Judge's async form
    reads none; a cancelled task's ``CancelledError`` is never caught."""
    search = _begin(index, judge, targets, delivered, completed, cancelled, batches_per_wave, Reading.CODE)
    return await _finished_async(search, anchors, files, names)


async def find_all_text_async(
    index: CodeIndex,
    judge: Judge,
    targets: Mapping[str, str],
    *,
    files: Sequence[str] = (),
    anchors: Sequence[Anchor] = (),
    names: Sequence[str] = (),
    delivered: Sequence[RangeAnchor] = (),
    completed: Mapping[str, Sequence[CheckResult]] | None = None,
    cancelled: Callable[[], bool] | None = None,
    batches_per_wave: int = BATCHES_PER_WAVE,
) -> FindAllResult:
    """``find_all_text`` the way ``find_all_async`` runs ``find_all``."""
    search = _begin(index, judge, targets, delivered, completed, cancelled, batches_per_wave, Reading.TEXT)
    return await _finished_async(search, anchors, files, names)


async def _finished_async(
    search: _Search, anchors: Sequence[Anchor], files: Sequence[str], names: Sequence[str]
) -> FindAllResult:
    try:
        await search.run_async(anchors, files, names)
    except (KeyboardInterrupt, Exception) as error:  # noqa: BLE001 - _stop_by owns how an error ends a search
        return search.ended(error)
    return search.ended(None)


def _begin(
    index: CodeIndex,
    judge: Judge,
    targets: Mapping[str, str],
    delivered: Sequence[RangeAnchor],
    completed: Mapping[str, Sequence[CheckResult]] | None,
    cancelled: Callable[[], bool] | None,
    batches_per_wave: int,
    reading: Reading,
) -> _Search:
    _require_identifiers(targets)
    if batches_per_wave < 1:
        raise ValueError("batches_per_wave must be at least 1")
    search = _Search(index, judge.scope(), targets, delivered, cancelled, batches_per_wave, reading)
    search.resume(completed or {})
    return search


def _stop_by(error: KeyboardInterrupt | Exception | None) -> tuple[str, Exception | None]:
    """How an error ends a search: a call cap ``budget``, Ctrl-C ``cancelled``, any other error
    ``failed`` holding it (``search_failure`` decides), and no error ``scope_examined``, or ``found``
    for a search that stops once found (``_Search.ended``)."""
    if error is None:
        return "scope_examined", None
    if isinstance(error, CallCapReachedError):
        return "budget", None
    if isinstance(error, KeyboardInterrupt):
        return "cancelled", None
    return "failed", search_failure(error)


def _require_identifiers(targets: Mapping[str, str]) -> None:
    if not targets:
        raise ValueError("find_all needs at least one target")
    if bad := [target for target in targets if not target.isidentifier()]:
        raise ValueError(f"target names must be identifiers, so a question can point at them: {bad}")


class _Search:
    def __init__(
        self,
        index: CodeIndex,
        judge: Judge,
        targets: Mapping[str, str],
        delivered: Sequence[RangeAnchor],
        cancelled: Callable[[], bool] | None,
        batches_per_wave: int,
        reading: Reading,
    ) -> None:
        self.index = index
        self.reading = reading
        self.stops_when_found = False
        self.found_one = False
        self.batches_per_wave = batches_per_wave
        self.judge = judge
        self.targets = dict(targets)
        self.checks = [match_check(target) for target in targets]
        self.target_of = {check.name: target for check, target in zip(self.checks, targets, strict=True)}
        self.shared = {TARGETS: self.targets}
        self.room = _room(judge, index, self.checks, self.shared)
        self.delivered = _lines_by_file(delivered)
        self.cancelled = cancelled
        self.answered: set[tuple[str, str]] = set()
        self.judged: dict[str, list[CheckResult]] = {target: [] for target in targets}
        self.units: dict[str, Unit] = {}
        self.entered_by: dict[str, Source] = {}
        self.not_judged: dict[str, str] = {}
        self.unlisted: dict[str, str] = {}
        self.unresolved: list[UnresolvedAnchor] = []
        self.refusals: list[Refusal] = []
        self.found: dict[str, int] = {}
        self.reached: Counter[str] = Counter()
        self.without_unit: Counter[str] = Counter()

    def stopped(self) -> bool:
        return self.cancelled is not None and self.cancelled()

    def resume(self, completed: Mapping[str, Sequence[CheckResult]]) -> None:
        for target in self.targets:
            for answer in completed.get(target, ()):
                self._record(target, answer)

    def run(self, anchors: Sequence[Anchor], files: Sequence[str], names: Sequence[str]) -> None:
        for places in self._waves(anchors, files, names):
            self._judge(places)
            if self.found_one:
                return

    async def run_async(self, anchors: Sequence[Anchor], files: Sequence[str], names: Sequence[str]) -> None:
        waves = self._waves(anchors, files, names)
        while (wave := await asyncio.to_thread(self._next_wave, waves)) is not None:
            places, entries = wave
            if self.stopped():
                return
            async for name, answer in self.judge.iter_check_every_async(
                self.checks, entries, self.shared, list_name=ITEMS, places=places, refusals=self.refusals
            ):
                self._record(self.target_of[name], answer)
            if self.found_one:
                return

    def ended(self, error: KeyboardInterrupt | Exception | None) -> FindAllResult:
        stop, failure = _stop_by(error)
        if self.stopped() and failure is None:
            stop = "cancelled"
        elif error is None and self.found_one:
            stop = FOUND
        return self.result(stop, failure)

    def _waves(
        self, anchors: Sequence[Anchor], files: Sequence[str], names: Sequence[str]
    ) -> Iterator[list[Item]]:
        return _chunks(
            self._population(anchors, files, names), self.judge.items_per_request * self.batches_per_wave
        )

    def _next_wave(self, waves: Iterator[list[Item]]) -> tuple[list[Item], list[dict]] | None:
        """The next wave's places and the entries Jev reads for them; None when the population is spent."""
        places = next(waves, None)
        return None if places is None else (places, self._entries(places))

    def _entries(self, places: Sequence[Item]) -> list[dict]:
        return [
            {"file": place.file, "code": read_ranges(self.index, place.file, place.ranges)}
            for place in places
        ]

    def _population(
        self, anchors: Sequence[Anchor], files: Sequence[str], names: Sequence[str]
    ) -> Iterator[Item]:
        """Every place still to judge, in order: the anchored units', the files' units', then the
        places of the units holding each name's hits, rarest name first. A request's worth of hits is
        resolved together, and only when the next wave needs it."""
        if self.stopped():
            return
        yield from self._admitted(self._anchored(anchors), Source.ANCHOR)
        if self.stopped():
            return
        yield from self._admitted(self._listed(files), Source.FILE)
        if not names or self.stopped():
            return
        hits = {name: self._hits(name) for name in dict.fromkeys(names)}
        self.found = {name: len(found) for name, found in hits.items()}
        for chunk in _chunks(_rarest_first(hits), self.judge.items_per_request):
            if self.stopped():
                return
            yield from self._admitted(self._units_of(chunk), Source.NAME)

    def _hits(self, name: str) -> tuple[TextHit, ...]:
        hits = self.index.search_text(name)
        if self.reading is Reading.CODE:
            return hits
        return tuple(hit for hit in hits if _read_by_name_in_text(hit.file))

    def _anchored(self, anchors: Sequence[Anchor]) -> tuple[Unit, ...]:
        resolution = self._resolved(anchors)
        self.unresolved.extend(resolution.unresolved)
        return resolution.units

    def _listed(self, files: Sequence[str]) -> tuple[Unit, ...]:
        listing = list_units(self.index, files, box_chars=self.room, reading=self.reading)
        self.unlisted.update(listing.unlisted)
        return listing.units

    def _resolved(self, anchors: Sequence[Anchor]) -> AnchorResolution:
        return resolve_anchors(
            self.index, anchors, box_chars=self.room, listed_only=True, reading=self.reading
        )

    def _units_of(self, chunk: Sequence[tuple[str, TextHit]]) -> tuple[Unit, ...]:
        anchors = [LineAnchor(hit.file, hit.line) for _, hit in chunk]
        resolution = self._resolved(anchors)
        missed = {problem.anchor for problem in resolution.unresolved}
        for (name, _), anchor in zip(chunk, anchors, strict=True):
            self.reached[name] += 1
            self.without_unit[name] += anchor in missed
        return resolution.units

    def _admitted(self, units: Iterable[Unit], source: Source) -> list[Item]:
        """The places still to judge of the units not seen before, each unit registered on first sight
        with the ``source`` that listed it."""
        places = []
        for unit in units:
            if unit.id not in self.units:
                self.units[unit.id] = unit
                self.entered_by[unit.id] = source
                places += self._pending_places(unit)
        return places

    def _pending_places(self, unit: Unit) -> list[Item]:
        for piece in unit.too_large_pieces:
            self.not_judged[unit.piece_id(piece)] = TOO_LARGE
        pending = []
        for place in items_to_judge(unit):
            if self._already_delivered(place):
                self.not_judged[place.id] = DELIVERED
            elif self._answered(place):
                continue
            elif not self._fits_alone(place):
                self.not_judged[place.id] = TOO_LARGE
            else:
                self.not_judged[place.id] = NOT_REACHED
                pending.append(place)
        return pending

    def _fits_alone(self, place: Item) -> bool:
        """Whether the request asking every target about ``place`` alone fits the Judge's limits as
        masked: masking can lengthen code past the room measured before it."""
        [entry] = self._entries([place])
        return self.judge.fits_alone(self.checks, entry, self.shared, ITEMS)

    def _already_delivered(self, place: Item) -> bool:
        lines = self.delivered.get(place.file, frozenset())
        return all(line in lines for first, last in place.ranges for line in range(first, last + 1))

    def _judge(self, places: Sequence[Item]) -> None:
        if not places or self.stopped():
            return
        for name, answer in self.judge.iter_check_every(
            self.checks,
            self._entries(places),
            self.shared,
            list_name=ITEMS,
            cancelled=self.cancelled,
            places=places,
            refusals=self.refusals,
        ):
            self._record(self.target_of[name], answer)

    def _record(self, target: str, answer: CheckResult) -> None:
        if answer.place is None:
            raise ValueError("a Find All answer names its place")
        if (target, answer.place.id) in self.answered:
            return
        self.answered.add((target, answer.place.id))
        self.judged[target].append(answer)
        self.found_one |= self.stops_when_found and answer.verdict is NoulVerdict.YES
        if self._answered(answer.place):
            self.not_judged.pop(answer.place.id, None)

    def _answered(self, place: Item) -> bool:
        return all((target, place.id) in self.answered for target in self.targets)

    def result(self, stop: str, failure: Exception | None) -> FindAllResult:
        refused = {
            refusal.place.id
            for refusal in self.refusals
            if refusal.place is not None and refusal.place.id in self.not_judged
        }
        return FindAllResult(
            self.targets,
            self.room,
            self.batches_per_wave,
            tuple(sorted(self.units.values(), key=_unit_order)),
            {target: tuple(sorted(answers, key=_answer_order)) for target, answers in self.judged.items()},
            self.not_judged | dict.fromkeys(refused, REFUSED),
            dict(self.unlisted),
            tuple(self.unresolved),
            {
                name: NameHits(found, self.reached[name], self.without_unit[name])
                for name, found in self.found.items()
            },
            self.index.observed_unparsed_files,
            stop,
            self.judge.calls,
            failure,
            tuple(self.refusals),
            dict(self.entered_by),
        )


def _room(judge: Judge, index: CodeIndex, checks: Sequence[Check], shared: Mapping) -> int:
    """The room one unit's code has in a request: the client's box less what a one-item request
    carries beside the code (the shared targets, the longest path in scope, and the longest of the
    item's questions), measured as the box measures it."""
    longest_path = max(index.files, key=serialized_chars, default="")
    beside = {**shared, ITEMS: [{"file": longest_path, "code": ""}]}
    longest_question = max(serialized_chars(check.to_question(item_path(ITEMS, 0))) for check in checks)
    return judge.input_limits.box_chars - serialized_chars(beside) - longest_question + serialized_chars("")


def _read_by_name_in_text(file: str) -> bool:
    """A text search reaches by a name's hits only text files, never a lockfile."""
    return not language_read(file) and not is_lockfile(file)


def _lines_by_file(regions: Sequence[RangeAnchor]) -> dict[str, frozenset[int]]:
    lines: dict[str, set[int]] = {}
    for region in regions:
        lines.setdefault(region.file, set()).update(range(region.start, region.end + 1))
    return {file: frozenset(numbers) for file, numbers in lines.items()}


def _rarest_first(hits: Mapping[str, Sequence[TextHit]]) -> Iterator[tuple[str, TextHit]]:
    for name in sorted(hits, key=lambda name: (len(hits[name]), name)):
        for hit in hits[name]:
            yield name, hit


def _chunks(values: Iterable[T], size: int) -> Iterator[list[T]]:
    chunk: list[T] = []
    for value in values:
        chunk.append(value)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _unit_score(
    unit: Unit, answers: Mapping[str, CheckResult], probabilities: Mapping[str, float]
) -> UnitScore | None:
    if not unit.pieces:
        answer = answers.get(unit.id)
        return None if answer is None else UnitScore(unit, answer, None)
    piece = best_piece(unit, probabilities)
    return None if piece is None else UnitScore(unit, answers[unit.piece_id(piece)], piece)


def _unit_order(unit: Unit) -> tuple:
    return unit.path, unit.start, unit.end, unit.id


def _answer_order(answer: CheckResult) -> tuple:
    """Answers in file and line order, whatever order they arrived in."""
    place = answer.place
    return place.file, place.ranges[0][0], place.ranges[-1][1], place.id
