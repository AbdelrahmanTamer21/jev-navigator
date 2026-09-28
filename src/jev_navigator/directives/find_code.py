"""find_code: open places best first until the code a description names is found, bounded by code.

Each opened place gets one request with two kinds of yes/no question: does this code contain the
target, and, per neighbour code lists, could the target be inside that neighbour. An optional pick
names the neighbour to open first. Jev sees only the target description, the opened code and the
neighbour signatures; it never sees the search history, and code alone decides what happens next.

Each round opens the top ``beam_width`` places of the queue at once. Neighbours merge into one
best-first queue; a visited set removes overlapping paths, and an in-run cache skips code already
judged. With ``beam_width=1`` this is a plain sequential best-first search.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum

from ..history import (
    DEFAULT_QUESTION_RESERVE,
    DEFAULT_STOP_SECTIONS,
    JEV_STATE_TOKEN_LIMIT,
    FetchedSpan,
    History,
    HistoryJudgment,
    HistoryOutcome,
    HistoryStep,
    judge_history,
    judge_history_async,
)
from ..index.code_index import CodeIndex
from ..index.spans import CodeSlice
from ..judgments.judge import CallCapReachedError, Judge
from ..judgments.questions import Check, Criterion, Pick, content_hash
from ..judgments.thresholds import NoulVerdict, Thresholds
from .places import Place, neighbours_and_omissions

FOUND = Check(
    name="contains_target",
    instructions="Does `slice.code` contain the code described in `target.description`?",
    yes=Criterion(
        "A line or block in `slice.code` is itself the code the description names.",
        examples=(
            "Description 'the check that limits items per order' and the slice holds "
            "`if len(order.items) > limit: raise`.",
        ),
    ),
    no=Criterion(
        "`slice.code` only calls, mentions or sits near that code, or does something else entirely.",
        not_for="Code that merely has a similar name.",
        examples=(
            "The slice calls `check_limits(order)` but the comparison is inside `check_limits`.",
            "The slice formats an invoice.",
        ),
    ),
)
COULD_CONTAIN = Check(
    name="could_contain_target",
    instructions=(
        "Does `{item}.preview`, under the signature `{item}.signature`, suggest that this function itself"
        " implements the code described in `target.description`?"
    ),
    yes=Criterion(
        "The signature and preview suggest this function itself implements the description.",
        examples=(
            "Description 'the check that limits items per order' and the candidate is "
            "`def check_limits(order):` reading a limit setting.",
        ),
    ),
    no=Criterion(
        "The function is unrelated, or it only uses or wraps code that implements the description.",
        examples=(
            "A route handler that calls `place_order`, when the description is the item limit check.",
            "A logging helper.",
        ),
    ),
)
OPEN_FIRST = Pick(
    name="open_first",
    instructions=(
        "Which entry of `candidates` most likely contains the code described in `target.description`?"
    ),
)


@dataclass(frozen=True)
class SearchQuestions:
    """The wording find_code asks with. Replace any part; a new wording is a new question id."""

    found: Check = FOUND
    could_contain: Check = COULD_CONTAIN
    open_first: Pick | None = OPEN_FIRST


DEFAULT_SEARCH_QUESTIONS = SearchQuestions()
NO_CLEAR_FIRST = "none"
NO_CLEAR_FIRST_TEXT = "No entry is more likely than the others."


class Outcome(StrEnum):
    FOUND = "found"
    STOP_RULE = "stop_rule"
    UNSURE_ONLY = "unsure_only"
    NOTHING_LEFT = "nothing_left"
    BUDGET = "budget"


@dataclass(frozen=True)
class SearchBudget:
    max_depth: int = 3
    max_steps: int = 24
    max_calls: int = 24
    beam_width: int = 3
    neighbours_per_kind: int = 8
    preview_lines: int = 8

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> SearchBudget:
        """Library defaults, overridden by ``JEV_NAVIGATOR_<FIELD>`` variables; read once at the edge."""
        environment = os.environ if environment is None else environment
        found = {
            name: int(environment[f"JEV_NAVIGATOR_{name.upper()}"])
            for name in (
                "max_depth",
                "max_steps",
                "max_calls",
                "beam_width",
                "neighbours_per_kind",
                "preview_lines",
            )
            if f"JEV_NAVIGATOR_{name.upper()}" in environment
        }
        return replace(cls(), **found)


@dataclass(frozen=True)
class Visit:
    """An opened place: its code, the path from a start place, and the found verdict."""

    place_key: str
    code: CodeSlice
    path: tuple[str, ...]
    probability: float
    verdict: NoulVerdict


@dataclass(frozen=True)
class NotInspected:
    """A place the search did not open. ``reason`` is ``budget`` (still worth opening when the budget
    ran out), ``deprioritized`` (its signature scored low; that only lowered its priority, it was never
    judged), ``capped`` (cut by the per-kind neighbour cap) or ``depth`` (beyond the depth limit)."""

    place_key: str
    signature: str
    reason: str
    priority: float
    depth: int
    path: tuple[str, ...]
    place: Place = field(compare=False, repr=False)


@dataclass(frozen=True)
class FindResult:
    """Three explicit sets: ``found``; ``searched`` (bodies judged and not the target) plus ``unsure``;
    and ``not_inspected``, the frontier a later call can resume from with ``resume=``."""

    outcome: Outcome
    found: tuple[Visit, ...]
    searched: tuple[Visit, ...]
    unsure: tuple[Visit, ...]
    not_inspected: tuple[NotInspected, ...]
    steps: int
    calls: int
    visited: frozenset[str] = frozenset()
    judged_code: frozenset[str] = frozenset()
    history: History | None = None
    stop_judgment: HistoryJudgment | None = None


@dataclass(order=True)
class _Queued:
    priority: float
    order: int
    place: Place = field(compare=False)
    depth: int = field(compare=False)
    path: tuple[str, ...] = field(compare=False)


@dataclass
class _Search:
    target: Mapping
    thresholds: Thresholds
    budget: SearchBudget
    questions: SearchQuestions = DEFAULT_SEARCH_QUESTIONS
    stop_rule: StopRule | None = None
    history: History = field(default_factory=History)
    stop_judgment: HistoryJudgment | None = None
    queue: list[_Queued] = field(default_factory=list)
    visited: set[str] = field(default_factory=set)
    judged_code: set[str] = field(default_factory=set)
    found: list[Visit] = field(default_factory=list)
    searched: list[Visit] = field(default_factory=list)
    unsure: list[Visit] = field(default_factory=list)
    set_aside: list[NotInspected] = field(default_factory=list)
    steps: int = 0
    cap_reached: bool = False
    counter: itertools.count = field(default_factory=itertools.count)

    def push(self, place: Place, probability: float, depth: int, path: tuple[str, ...]) -> None:
        if place.key in self.visited:
            return
        if depth > self.budget.max_depth:
            self.set_aside.append(
                NotInspected(place.key, place.signature, "depth", probability, depth, path, place)
            )
            return
        heapq.heappush(self.queue, _Queued(-probability, next(self.counter), place, depth, path))

    def next_beam(self, calls_left: int) -> list[_Queued]:
        beam = []
        while self.queue and len(beam) < min(self.budget.beam_width, calls_left):
            item = heapq.heappop(self.queue)
            if item.place.key not in self.visited:
                self.visited.add(item.place.key)
                beam.append(item)
        return beam

    def worth_opening(self) -> bool:
        return any(-item.priority > self.thresholds.noul_no_at for item in self.queue)


def find_code(
    index: CodeIndex,
    judge: Judge,
    target_description: str,
    start: Sequence[Place],
    *,
    budget: SearchBudget | None = None,
    thresholds: Mapping[str, float] | None = None,
    questions: SearchQuestions = DEFAULT_SEARCH_QUESTIONS,
    resume: FindResult | None = None,
    commit: str | None = None,
    stop_rule: StopRule | None = None,
) -> FindResult:
    """``commit``: the revision the caller means; the index must hold exactly it. ``resume``: continue
    a stopped search from its frontier with a fresh budget. ``stop_rule`` (off by default): after each
    round the caller's check is asked over the history; a yes ends the search with outcome
    ``stop_rule``. Each round's places are asked concurrently in threads; ``find_code_async`` is the
    same search for an async client."""
    search, judge = _begin(
        index, judge, target_description, start, budget, thresholds, questions, resume, commit, stop_rule
    )
    while (stop := _stop_reason(search, judge)) is None:
        opened = _open_round(index, search, judge)
        if not opened:
            continue
        with ThreadPoolExecutor(max_workers=len(opened)) as pool:
            responses = list(pool.map(lambda opening: _ask_within_cap(judge, search, opening), opened))
        _merge_round(search, opened, responses)
        _apply_stop_rule(judge, search)
    return _result(search, stop, judge)


async def find_code_async(
    index: CodeIndex,
    judge: Judge,
    target_description: str,
    start: Sequence[Place],
    *,
    budget: SearchBudget | None = None,
    thresholds: Mapping[str, float] | None = None,
    questions: SearchQuestions = DEFAULT_SEARCH_QUESTIONS,
    resume: FindResult | None = None,
    commit: str | None = None,
    stop_rule: StopRule | None = None,
) -> FindResult:
    """``find_code`` with each round's places sent concurrently with ``asyncio.gather``; budgets,
    masking, the store, the journal and the history work exactly as in ``find_code``."""
    search, judge = _begin(
        index, judge, target_description, start, budget, thresholds, questions, resume, commit, stop_rule
    )
    while (stop := _stop_reason(search, judge)) is None:
        opened = _open_round(index, search, judge)
        if not opened:
            continue
        responses = await asyncio.gather(
            *(_ask_within_cap_async(judge, search, opening) for opening in opened)
        )
        _merge_round(search, opened, responses)
        await _apply_stop_rule_async(judge, search)
    return _result(search, stop, judge)


def _begin(
    index: CodeIndex,
    judge: Judge,
    target_description: str,
    start: Sequence[Place],
    budget: SearchBudget | None,
    thresholds: Mapping[str, float] | None,
    questions: SearchQuestions,
    resume: FindResult | None,
    commit: str | None,
    stop_rule: StopRule | None,
) -> tuple[_Search, Judge]:
    if commit is not None:
        index.require_commit(commit)
    target = {"description": target_description}
    search = _Search(
        target,
        judge.effective(thresholds),
        budget or SearchBudget(),
        questions,
        stop_rule,
        stop_rule.new_history(target) if stop_rule else History(sections={SUBJECT: target}),
    )
    if resume is not None:
        _restore(search, resume)
    for place in start:
        search.push(place, 1.0, 0, (place.key,))
    return search, judge.scope()


def _open_round(index: CodeIndex, search: _Search, judge: Judge) -> list[_Opening]:
    beam = search.next_beam(_calls_left(search, judge))
    _record_choice(search, beam)
    return [opening for item in beam if (opening := _open(index, search, item)) is not None]


def _merge_round(search: _Search, opened: list[_Opening], responses: list) -> None:
    for opening, response in zip(opened, responses, strict=True):
        if response is None:
            _set_aside_unasked(search, opening)
        else:
            _merge(search, opening, response)


SUBJECT = "subject"


@dataclass(frozen=True)
class StopRule:
    """A caller-defined yes/no check over the history, asked after each round. It reads only the
    ``sections`` it selects: by default ``history``, every step with its code and judgments; select
    ``("fetched",)`` for the code and sources alone, without the search's own judgments. The history
    declares ``subject`` (the target description) and any ``context`` sections the caller adds, such
    as the code shown with a comment; ``shared`` is extra state outside the history."""

    check: Check
    shared: Mapping = field(default_factory=dict)
    budget_tokens: int = JEV_STATE_TOKEN_LIMIT - DEFAULT_QUESTION_RESERVE
    sections: tuple[str, ...] = DEFAULT_STOP_SECTIONS
    context: Mapping[str, object] = field(default_factory=dict)

    def new_history(self, subject: Mapping) -> History:
        return History(budget_tokens=self.budget_tokens, sections={SUBJECT: subject, **self.context})


def _apply_stop_rule(judge: Judge, search: _Search) -> None:
    rule = search.stop_rule
    if rule is None:
        return
    try:
        search.stop_judgment = judge_history(
            judge, search.history, rule.check, rule.shared, sections=rule.sections
        )
    except CallCapReachedError:
        search.cap_reached = True


async def _apply_stop_rule_async(judge: Judge, search: _Search) -> None:
    rule = search.stop_rule
    if rule is None:
        return
    try:
        search.stop_judgment = await judge_history_async(
            judge, search.history, rule.check, rule.shared, sections=rule.sections
        )
    except CallCapReachedError:
        search.cap_reached = True


def _restore(search: _Search, previous: FindResult) -> None:
    search.visited |= previous.visited
    search.judged_code |= previous.judged_code
    search.searched += previous.searched
    search.unsure += previous.unsure
    for entry in previous.not_inspected:
        heapq.heappush(
            search.queue, _Queued(-entry.priority, next(search.counter), entry.place, entry.depth, entry.path)
        )


def _stop_reason(search: _Search, judge: Judge) -> Outcome | None:
    if search.found:
        return Outcome.FOUND
    if search.stop_judgment is not None and search.stop_judgment.outcome == HistoryOutcome.FOUND:
        return Outcome.STOP_RULE
    if search.cap_reached or search.steps >= search.budget.max_steps or _calls_left(search, judge) == 0:
        return Outcome.BUDGET
    if not search.worth_opening():
        return Outcome.UNSURE_ONLY if search.unsure else Outcome.NOTHING_LEFT
    return None


def _calls_left(search: _Search, judge: Judge) -> int:
    own = search.budget.max_calls - judge.calls
    shared = judge.calls_left()
    return max(0, own if shared is None else min(own, shared))


@dataclass(frozen=True)
class _Opening:
    item: _Queued
    code: CodeSlice
    candidates: list[Place]
    capped: tuple[NotInspected, ...] = ()


def _open(index: CodeIndex, search: _Search, item: _Queued) -> _Opening | None:
    """Opens a place in code; the same code reached by another path is not judged twice."""
    code = item.place.open()
    fingerprint = content_hash(code.text)
    if fingerprint in search.judged_code:
        return None
    search.judged_code.add(fingerprint)
    search.visited.add(code.key)
    search.steps += 1
    if item.depth >= search.budget.max_depth:
        return _Opening(item, code, [])
    candidates, omitted = neighbours_and_omissions(index, code, search.budget.neighbours_per_kind)
    capped = tuple(
        NotInspected(
            place.key, place.signature, "capped", 0.5, item.depth + 1, (*item.path, place.key), place
        )
        for place in omitted
        if place.key not in search.visited
    )
    search.set_aside.extend(capped)
    unseen = [place for place in candidates if place.key not in search.visited]
    return _Opening(item, code, [place for place in unseen if place.open().text.strip()], capped)


@dataclass(frozen=True)
class _OpeningRequest:
    state: Mapping
    questions: Mapping
    sources: Mapping


def _opening_request(search: _Search, opening: _Opening) -> _OpeningRequest:
    code, candidates = opening.code, opening.candidates
    state = {
        "target": search.target,
        "slice": {"file": code.span.file, "lines": f"{code.span.start}-{code.span.end}", "code": code.text},
        "candidates": [_candidate_state(place, search.budget.preview_lines) for place in candidates],
    }
    asked = search.questions
    questions = {asked.found.question_id: asked.found.to_question()}
    for slot in range(len(candidates)):
        questions[f"{asked.could_contain.question_id}#{slot}"] = asked.could_contain.to_question(
            f"candidates[{slot}]"
        )
    if asked.open_first is not None and len(candidates) > 1:
        options = {str(slot): place.signature for slot, place in enumerate(candidates)}
        questions[asked.open_first.question_id] = asked.open_first.to_question(
            {**options, NO_CLEAR_FIRST: NO_CLEAR_FIRST_TEXT}
        )
    sources = {asked.found.question_id: code.source()}
    sources.update(
        {
            f"{asked.could_contain.question_id}#{slot}": {"place": place.key}
            for slot, place in enumerate(candidates)
        }
    )
    return _OpeningRequest(state, questions, sources)


def _ask_within_cap(judge: Judge, search: _Search, opening: _Opening):
    """None when a judge's global cap, shared with other callers, ran out before this request."""
    request = _opening_request(search, opening)
    try:
        return judge.ask(
            request.state, request.questions, thresholds=search.thresholds, sources=request.sources
        )
    except CallCapReachedError:
        search.cap_reached = True
        return None


async def _ask_within_cap_async(judge: Judge, search: _Search, opening: _Opening):
    request = _opening_request(search, opening)
    try:
        return await judge.ask_async(
            request.state, request.questions, thresholds=search.thresholds, sources=request.sources
        )
    except CallCapReachedError:
        search.cap_reached = True
        return None


def _set_aside_unasked(search: _Search, opening: _Opening) -> None:
    item = opening.item
    search.visited -= {item.place.key, opening.code.key}
    search.judged_code.discard(content_hash(opening.code.text))
    search.steps -= 1
    search.set_aside.append(
        NotInspected(
            item.place.key, item.place.signature, "budget", -item.priority, item.depth, item.path, item.place
        )
    )


def _merge(search: _Search, opening: _Opening, response) -> None:
    item, code, candidates = opening.item, opening.code, opening.candidates
    found_probability = response.noul(search.questions.found.question_id).probability
    visit = Visit(
        item.place.key, code, item.path, found_probability, search.thresholds.noul_verdict(found_probability)
    )
    if visit.verdict == NoulVerdict.YES:
        search.found.append(visit)
    elif visit.verdict == NoulVerdict.UNSURE:
        search.unsure.append(visit)
    else:
        search.searched.append(visit)
    first = _confident_first(search, response)
    set_aside_before = len(search.set_aside)
    offered = []
    for slot, place in enumerate(candidates):
        probability = response.noul(f"{search.questions.could_contain.question_id}#{slot}").probability
        verdict = search.thresholds.noul_verdict(probability)
        preferred = slot == first and verdict != NoulVerdict.NO
        priority = 1.0 + probability if preferred else probability
        search.push(place, priority, item.depth + 1, (*item.path, place.key))
        offered.append(
            {"place": place.key, "signature": place.signature, "probability": probability, "verdict": verdict}
        )
    not_opened = [*opening.capped, *search.set_aside[set_aside_before:]]
    search.history.append(_open_step(search, opening, visit, offered, response, not_opened))


_DECISIONS = {NoulVerdict.YES: "found", NoulVerdict.UNSURE: "unsure", NoulVerdict.NO: "searched"}


def _open_step(
    search: _Search,
    opening: _Opening,
    visit: Visit,
    offered: list[dict],
    response,
    not_opened: list[NotInspected],
) -> HistoryStep:
    """What was opened, what Jev answered about it and its neighbours, and what code set aside."""
    judgments: dict[str, object] = {
        "contains_target": {"probability": visit.probability, "verdict": visit.verdict},
        "could_contain": offered,
    }
    pick = search.questions.open_first
    if pick is not None and pick.question_id in response.answers:
        answer = response.choice(pick.question_id)
        judgments["open_first"] = {
            "choice": _picked_place(answer.choice, opening.candidates),
            "confidence": answer.confidence,
            "used": _confident_first(search, response) is not None,
        }
    if not_opened:
        judgments["not_opened"] = [_frontier_entry(entry) for entry in not_opened]
    return HistoryStep(
        "open",
        {"place": visit.place_key, "depth": opening.item.depth, "path": list(visit.path)},
        (FetchedSpan(visit.code.source(), visit.code.text),),
        judgments,
        _DECISIONS[visit.verdict],
    )


def _picked_place(choice: str, candidates: list[Place]) -> str:
    return choice if choice == NO_CLEAR_FIRST else candidates[int(choice)].key


def _record_choice(search: _Search, beam: list[_Queued]) -> None:
    if not beam:
        return
    chosen = [
        {
            "place": item.place.key,
            "priority": -item.priority,
            "depth": item.depth,
            "reason": _choice_reason(item),
        }
        for item in beam
    ]
    search.history.append(
        HistoryStep(
            "choose_next",
            {"chosen": chosen, "still_queued": len(search.queue)},
            decision=f"open {len(chosen)} of the best-scored places",
        )
    )


def _choice_reason(item: _Queued) -> str:
    """``start`` for a caller's start place, ``open_first`` when Jev's confident pick raised it above
    every score, else ``queue_score`` (its could_contain probability)."""
    if item.depth == 0:
        return "start"
    return "open_first" if -item.priority > 1.0 else "queue_score"


def _frontier_entry(entry: NotInspected) -> dict:
    return {"place": entry.place_key, "reason": entry.reason, "priority": entry.priority}


def _confident_first(search: _Search, response) -> int | None:
    pick = search.questions.open_first
    if pick is None or pick.question_id not in response.answers:
        return None
    answer = response.choice(pick.question_id)
    if answer.choice == NO_CLEAR_FIRST or not search.thresholds.choice_is_confident(answer.confidence):
        return None
    return int(answer.choice)


def _candidate_state(place: Place, preview_lines: int) -> dict:
    """The signature line plus the first lines of the candidate's code, so the judgment rests on
    more than a name."""
    preview = "\n".join(place.open().text.splitlines()[:preview_lines])
    return {"signature": place.signature, "preview": preview}


def _result(search: _Search, outcome: Outcome, judge: Judge) -> FindResult:
    left = [item for item in search.queue if item.place.key not in search.visited]
    frontier = [
        NotInspected(
            item.place.key,
            item.place.signature,
            _reason(search, item),
            -item.priority,
            item.depth,
            item.path,
            item.place,
        )
        for item in sorted(left)
    ]
    frontier += [entry for entry in search.set_aside if entry.place_key not in search.visited]
    not_inspected = tuple({entry.place_key: entry for entry in frontier}.values())
    search.history.append(_stop_step(search, outcome, not_inspected))
    return FindResult(
        outcome,
        tuple(search.found),
        tuple(search.searched),
        tuple(search.unsure),
        not_inspected,
        search.steps,
        judge.calls,
        frozenset(search.visited),
        frozenset(search.judged_code),
        search.history,
        search.stop_judgment,
    )


def _stop_step(search: _Search, outcome: Outcome, not_inspected: tuple[NotInspected, ...]) -> HistoryStep:
    judgments: dict[str, object] = {"not_inspected": [_frontier_entry(entry) for entry in not_inspected]}
    if search.stop_judgment is not None:
        judgments["last_stop_check"] = {
            "probability": search.stop_judgment.probability,
            "outcome": search.stop_judgment.outcome,
        }
    return HistoryStep("stop", {"outcome": outcome}, (), judgments, f"stopped: {outcome}")


def _reason(search: _Search, item: _Queued) -> str:
    return "budget" if -item.priority > search.thresholds.noul_no_at else "deprioritized"
