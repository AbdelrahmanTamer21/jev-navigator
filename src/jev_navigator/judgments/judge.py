"""Jev judgments as building blocks: yes/no checks over supplied items, picks over code-built
options, and one function-calling decision per step.

Every request is masked, scanned, hashed and looked up in the answer store before any call, and
every fresh answer is stored with the thresholds that were in force.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from .answers import JevResponse, NoulAnswer, response_to_raw
from .client import AsyncJevClient, JevClient
from .journal import Journal, JournalRequest, RawResponse
from .questions import Check, Pick, Rate, content_hash, item_path, request_sha256
from .secrets import (
    Masker,
    Scanner,
    SecretMasker,
    SecretScanner,
    mask_value,
    refuse_if_secret,
    safe_options,
)
from .store import AnswerRecord, AnswerStore
from .thresholds import NoulVerdict, Thresholds

logger = logging.getLogger(__name__)

MAX_STATE_CHARS = 60_000
CODE_FIELD = "code"
ROUTE_QUESTION = "route"
_DEFAULT_MASKER = SecretMasker()
_DEFAULT_SCANNER = SecretScanner()


class CallCapReachedError(RuntimeError):
    """A call would exceed the ``max_calls`` cap of this judge or of a judge it was scoped from."""


@dataclass(frozen=True)
class CheckResult:
    """``probability`` is Jev's raw P(yes); ``verdict`` applies the current yes/no band. Callers may
    apply any band of their own to ``probability``. ``request_sha256`` identifies the masked request
    that answered it, also when the answer came from the store."""

    item: Mapping
    probability: float
    verdict: NoulVerdict
    from_store: bool
    request_sha256: str


@dataclass(frozen=True)
class PickResult:
    """The whole distribution is kept: ``probabilities`` per option and the API's ``confidence``.
    ``confident`` applies the current threshold; callers may apply any rule of their own instead."""

    choice: str
    confidence: float
    probabilities: Mapping[str, float]
    confident: bool
    request_sha256: str


@dataclass(frozen=True)
class ScoreResult:
    score: float
    probabilities: Mapping[str, float]
    confidence: float
    request_sha256: str


@dataclass(frozen=True)
class AllAnswers:
    """Typed results by question name, each with its raw probabilities."""

    checks: Mapping[str, CheckResult]
    picks: Mapping[str, PickResult]
    scores: Mapping[str, ScoreResult]
    request_sha256: str


@dataclass(frozen=True)
class CallOffer:
    """One operation Jev may call, with the closed set of inputs code found for it this step."""

    name: str
    description: str
    argument: Pick
    options: Mapping[str, str]


@dataclass(frozen=True)
class CallDecision:
    operation: str
    argument: str
    route: PickResult
    argument_pick: PickResult

    @property
    def confident(self) -> bool:
        return self.route.confident and self.argument_pick.confident

    @property
    def request_sha256(self) -> str:
        return self.route.request_sha256


class Judge:
    """``calls`` counts requests sent (store hits are free). ``scope()`` gives one caller, such as a
    single search, its own counter on the same client, store and journal; every scope adds its calls
    to its parent, and ``max_calls`` caps a judge together with all of its scopes."""

    def __init__(
        self,
        client: JevClient | AsyncJevClient,
        *,
        masker: Masker | None = _DEFAULT_MASKER,
        scanner: Scanner | None = _DEFAULT_SCANNER,
        store: AnswerStore | None = None,
        thresholds: Thresholds | None = None,
        served_model: str | None = None,
        journal: Journal | None = None,
        max_calls: int | None = None,
    ) -> None:
        self.client = client
        self.masker = masker
        self.scanner = scanner
        self.store = store
        self.thresholds = thresholds or Thresholds()
        self.served_model = served_model
        self.journal = journal
        self.max_calls = max_calls
        self.calls = 0
        self.input_tokens = 0
        self._parent: Judge | None = None
        self._bookkeeping = threading.Lock()

    def scope(self) -> Judge:
        child = copy.copy(self)
        child.max_calls = None
        child.calls = 0
        child.input_tokens = 0
        child._parent = self
        return child

    def calls_left(self) -> int | None:
        """The calls this judge may still send under its own and its parents' caps; None when uncapped."""
        caps = [judge.max_calls - judge.calls for judge in self._chain() if judge.max_calls is not None]
        return max(0, min(caps)) if caps else None

    def effective(self, *overrides: Mapping[str, float] | None) -> Thresholds:
        thresholds = self.thresholds
        for layer in overrides:
            thresholds = thresholds.updated(layer)
        return thresholds

    def check_each(
        self,
        check: Check,
        items: Sequence[Mapping],
        shared: Mapping | None = None,
        *,
        list_name: str = "items",
        thresholds: Thresholds | None = None,
    ) -> list[CheckResult]:
        """One yes/no answer per item, batched into as few requests as the state size allows.
        Items already judged by the same question and model come from the store."""
        plan = self._check_plan(check, items, shared, list_name, thresholds)
        for batch in plan.batches:
            plan.answer(
                batch, self.ask(batch.state, batch.questions, thresholds=plan.thresholds, **batch.extras)
            )
        return plan.results()

    async def check_each_async(
        self,
        check: Check,
        items: Sequence[Mapping],
        shared: Mapping | None = None,
        *,
        list_name: str = "items",
        thresholds: Thresholds | None = None,
    ) -> list[CheckResult]:
        """``check_each`` with its batches sent concurrently."""
        plan = self._check_plan(check, items, shared, list_name, thresholds)
        responses = await asyncio.gather(
            *(
                self.ask_async(batch.state, batch.questions, thresholds=plan.thresholds, **batch.extras)
                for batch in plan.batches
            )
        )
        for batch, response in zip(plan.batches, responses, strict=True):
            plan.answer(batch, response)
        return plan.results()

    def ask_all(
        self,
        state: Mapping,
        *,
        checks: Sequence[Check] = (),
        picks: Sequence[tuple[Pick, Mapping[str, str]]] = (),
        scores: Sequence[Rate] = (),
        thresholds: Thresholds | None = None,
    ) -> AllAnswers:
        """Many independent questions over one state, in one request."""
        request = self._all_request(checks, picks, scores, thresholds)
        return request.answers(state, self.ask(state, request.questions, thresholds=request.thresholds))

    async def ask_all_async(
        self,
        state: Mapping,
        *,
        checks: Sequence[Check] = (),
        picks: Sequence[tuple[Pick, Mapping[str, str]]] = (),
        scores: Sequence[Rate] = (),
        thresholds: Thresholds | None = None,
    ) -> AllAnswers:
        request = self._all_request(checks, picks, scores, thresholds)
        response = await self.ask_async(state, request.questions, thresholds=request.thresholds)
        return request.answers(state, response)

    def pick(
        self, pick: Pick, options: Mapping[str, str], state: Mapping, *, thresholds: Thresholds | None = None
    ) -> PickResult | None:
        """The option Jev picks, or None when masking left no option to offer."""
        questions = self._pick_questions(pick, options)
        if questions is None:
            return None
        thresholds = thresholds or self.thresholds
        return _pick_result(self.ask(state, questions, thresholds=thresholds), pick.question_id, thresholds)

    async def pick_async(
        self, pick: Pick, options: Mapping[str, str], state: Mapping, *, thresholds: Thresholds | None = None
    ) -> PickResult | None:
        questions = self._pick_questions(pick, options)
        if questions is None:
            return None
        thresholds = thresholds or self.thresholds
        response = await self.ask_async(state, questions, thresholds=thresholds)
        return _pick_result(response, pick.question_id, thresholds)

    def choose_call(
        self,
        route: Pick,
        offers: Sequence[CallOffer],
        state: Mapping,
        *,
        thresholds: Thresholds | None = None,
    ) -> CallDecision | None:
        """Function calling in one request: which operation to run, and each operation's input,
        asked for every operation at once; only the chosen operation's answer is read."""
        request = self._call_request(route, offers, thresholds)
        if request is None:
            return None
        return request.decision(self.ask(state, request.questions, thresholds=request.thresholds))

    async def choose_call_async(
        self,
        route: Pick,
        offers: Sequence[CallOffer],
        state: Mapping,
        *,
        thresholds: Thresholds | None = None,
    ) -> CallDecision | None:
        request = self._call_request(route, offers, thresholds)
        if request is None:
            return None
        return request.decision(await self.ask_async(state, request.questions, thresholds=request.thresholds))

    def ask(
        self,
        state: Mapping,
        questions: Mapping,
        *,
        thresholds: Thresholds,
        item_keys: Mapping[str, str] | None = None,
        sources: Mapping[str, Mapping] | None = None,
        skeleton: Mapping | None = None,
    ) -> JevResponse:
        """Masks, scans, hashes and looks up the store; only a miss sends, and every fresh answer is
        recorded. The async variant shares every step except the send."""
        prepared = self._prepare(state, questions)
        if prepared.stored is not None:
            return prepared.stored
        self._reserve_call()
        response = self._dispatch(prepared)
        return self._finish(prepared, response, thresholds, item_keys, sources, skeleton)

    async def ask_async(
        self,
        state: Mapping,
        questions: Mapping,
        *,
        thresholds: Thresholds,
        item_keys: Mapping[str, str] | None = None,
        sources: Mapping[str, Mapping] | None = None,
        skeleton: Mapping | None = None,
    ) -> JevResponse:
        prepared = self._prepare(state, questions)
        if prepared.stored is not None:
            return prepared.stored
        self._reserve_call()
        response = await self._dispatch_async(prepared)
        return self._finish(prepared, response, thresholds, item_keys, sources, skeleton)

    def _prepare(self, state: Mapping, questions: Mapping) -> _Prepared:
        state = mask_value(state, self.masker) if self.masker else state
        questions = mask_value(questions, self.masker) if self.masker else questions
        refuse_if_secret(state, questions, self.scanner)
        request_hash = request_sha256(state, questions)
        stored = self.store.by_request(request_hash) if self.store else None
        accepted = stored.response() if stored is not None and self._accepts(stored.model) else None
        return _Prepared(state, questions, request_hash, accepted)

    def _finish(
        self,
        prepared: _Prepared,
        response: JevResponse,
        thresholds: Thresholds,
        item_keys: Mapping[str, str] | None,
        sources: Mapping[str, Mapping] | None,
        skeleton: Mapping | None,
    ) -> JevResponse:
        with self._bookkeeping:
            for judge in self._chain():
                judge.served_model = response.model
                judge.input_tokens += response.input_tokens
            self._record(prepared, response, thresholds, item_keys or {}, sources or {}, skeleton or {})
        return JevResponse(response.answers, response.model, response.input_tokens, prepared.request_hash)

    def _reserve_call(self) -> None:
        with self._bookkeeping:
            if self.calls_left() == 0:
                raise CallCapReachedError("the call cap of this judge is used up")
            for judge in self._chain():
                judge.calls += 1

    def _chain(self) -> Iterator[Judge]:
        judge: Judge | None = self
        while judge is not None:
            yield judge
            judge = judge._parent

    def _dispatch(self, prepared: _Prepared) -> JevResponse:
        if _is_async(self.client):
            raise TypeError("this Jev client is async; use the judge's *_async methods")
        request_id = self._journal_request(prepared)
        raw: RawResponse | None = None
        try:
            if hasattr(self.client, "send"):
                raw = self.client.send(prepared.state, prepared.questions)
                self._journal_response(request_id, raw)
                return self.client.parse(raw)
            response = self.client.ask(prepared.state, prepared.questions)
            self._journal_response(request_id, RawResponse.from_decoded(response_to_raw(response)))
            return response
        except Exception as error:
            self._journal_failure(request_id, error, raw)
            raise

    async def _dispatch_async(self, prepared: _Prepared) -> JevResponse:
        """Awaits an async client; a sync client runs in a worker thread."""
        request_id = self._journal_request(prepared)
        raw: RawResponse | None = None
        try:
            if hasattr(self.client, "send"):
                raw = await _awaited(self.client.send, prepared.state, prepared.questions)
                self._journal_response(request_id, raw)
                return self.client.parse(raw)
            response = await _awaited(self.client.ask, prepared.state, prepared.questions)
            self._journal_response(request_id, RawResponse.from_decoded(response_to_raw(response)))
            return response
        except Exception as error:
            self._journal_failure(request_id, error, raw)
            raise

    def _journal_request(self, prepared: _Prepared) -> str | None:
        if self.journal is None:
            return None
        request = JournalRequest(prepared.request_hash, self.client.model, prepared.state, prepared.questions)
        return self.journal.record_request(request)

    def _journal_response(self, request_id: str | None, raw: RawResponse) -> None:
        if self.journal is not None and request_id is not None:
            self.journal.record_response(request_id, raw)

    def _journal_failure(self, request_id: str | None, error: Exception, raw: RawResponse | None) -> None:
        if self.journal is not None and request_id is not None:
            self.journal.record_failure(request_id, f"{type(error).__name__}: {error}", raw)

    def _check_plan(
        self,
        check: Check,
        items: Sequence[Mapping],
        shared: Mapping | None,
        list_name: str,
        thresholds: Thresholds | None,
    ) -> _CheckPlan:
        masked_items = [mask_value(item, self.masker) if self.masker else item for item in items]
        masked_shared = mask_value(shared or {}, self.masker) if self.masker else (shared or {})
        plan = _CheckPlan(items, thresholds or self.thresholds)
        for position, item in enumerate(masked_items):
            stored = self._stored_item(check, item, masked_shared)
            if stored is not None:
                plan.answered[position] = stored
        pending = [position for position in range(len(items)) if position not in plan.answered]
        plan.batches = [
            self._batch(check, positions, masked_items, shared or {}, masked_shared, list_name)
            for positions in _batches(pending, masked_items, shared or {})
        ]
        return plan

    def _batch(
        self,
        check: Check,
        positions: list[int],
        items: list[Mapping],
        shared: Mapping,
        masked_shared: Mapping,
        list_name: str,
    ) -> _Batch:
        batch_items = [items[position] for position in positions]
        question_ids = {position: f"{check.question_id}#{slot}" for slot, position in enumerate(positions)}
        questions = {
            question_ids[position]: check.to_question(item_path(list_name, slot))
            for slot, position in enumerate(positions)
        }
        extras = {
            "item_keys": {
                self._item_key(check, items[position], masked_shared): question_ids[position]
                for position in positions
            },
            "sources": {
                question_ids[position]: _source_of(items[position])
                for position in positions
                if _source_of(items[position])
            },
            "skeleton": _skeleton(list_name, questions, batch_items, masked_shared),
        }
        return _Batch({**shared, list_name: batch_items}, questions, question_ids, extras)

    def _all_request(
        self,
        checks: Sequence[Check],
        picks: Sequence[tuple[Pick, Mapping[str, str]]],
        scores: Sequence[Rate],
        thresholds: Thresholds | None,
    ) -> _AllRequest:
        offered = {pick.name: safe_options(options, self.masker) for pick, options in picks}
        questions = {check.question_id: check.to_question() for check in checks}
        questions |= {pick.question_id: pick.to_question(offered[pick.name]) for pick, _ in picks}
        questions |= {rate.question_id: rate.to_question() for rate in scores}
        pick_questions = tuple(pick for pick, _ in picks)
        return _AllRequest(
            tuple(checks), pick_questions, tuple(scores), questions, thresholds or self.thresholds
        )

    def _pick_questions(self, pick: Pick, options: Mapping[str, str]) -> dict | None:
        offered = safe_options(options, self.masker)
        return {pick.question_id: pick.to_question(offered)} if offered else None

    def _call_request(
        self, route: Pick, offers: Sequence[CallOffer], thresholds: Thresholds | None
    ) -> _CallRequest | None:
        usable = {offer.name: offer for offer in offers if safe_options(offer.options, self.masker)}
        if not usable:
            return None
        questions = {
            ROUTE_QUESTION: route.to_question({name: offer.description for name, offer in usable.items()})
        }
        for name, offer in usable.items():
            questions[_argument_id(name, offer)] = offer.argument.to_question(
                safe_options(offer.options, self.masker)
            )
        return _CallRequest(usable, questions, thresholds or self.thresholds)

    def _stored_item(self, check: Check, item: Mapping, shared: Mapping) -> _ItemAnswer | None:
        if self.store is None or not self._knows_model():
            return None
        stored = self.store.by_item(self._item_key(check, item, shared), self._model_filter())
        if stored is None or not isinstance(stored.answer, NoulAnswer):
            return None
        return _ItemAnswer(stored.answer.probability, True, stored.request_sha256)

    def _item_key(self, check: Check, item: Mapping, shared: Mapping) -> str:
        """Item content, the shared state the question refers to, and the question with its wording."""
        return f"{content_hash(item)}|{content_hash(shared)}|{check.question_id}"

    def _accepts(self, stored_model: str) -> bool:
        return self._knows_model() and self._model_filter() in (None, stored_model)

    def _knows_model(self) -> bool:
        return self._replays_any_model() or self.served_model is not None

    def _model_filter(self) -> str | None:
        return None if self._replays_any_model() else self.served_model

    def _replays_any_model(self) -> bool:
        return getattr(self.client, "replays_any_model", False)

    def _record(
        self,
        prepared: _Prepared,
        response: JevResponse,
        thresholds: Thresholds,
        item_keys: Mapping[str, str],
        sources: Mapping[str, Mapping],
        skeleton: Mapping,
    ) -> None:
        if self.store is None:
            return
        self.store.put(
            AnswerRecord(
                request_sha256=prepared.request_hash,
                question_ids=tuple(prepared.questions),
                answers={question_id: answer.to_json() for question_id, answer in response.answers.items()},
                model=response.model,
                input_tokens=response.input_tokens,
                thresholds=thresholds.as_dict(),
                item_keys=dict(item_keys),
                sources=dict(sources),
                skeleton=dict(skeleton),
                request={"state": prepared.state, "questions": prepared.questions},
            )
        )


@dataclass(frozen=True)
class _ItemAnswer:
    probability: float
    from_store: bool
    request_sha256: str


@dataclass(frozen=True)
class _Prepared:
    """A masked, scanned and hashed request, with the stored answer when the store has one."""

    state: Mapping
    questions: Mapping
    request_hash: str
    stored: JevResponse | None


@dataclass(frozen=True)
class _Batch:
    state: Mapping
    questions: Mapping
    question_ids: Mapping[int, str]
    extras: Mapping[str, Mapping]


@dataclass
class _CheckPlan:
    items: Sequence[Mapping]
    thresholds: Thresholds
    answered: dict[int, _ItemAnswer] = field(default_factory=dict)
    batches: list[_Batch] = field(default_factory=list)

    def answer(self, batch: _Batch, response: JevResponse) -> None:
        for position, question_id in batch.question_ids.items():
            probability = response.noul(question_id).probability
            self.answered[position] = _ItemAnswer(probability, response.from_store, response.request_sha256)

    def results(self) -> list[CheckResult]:
        return [
            CheckResult(
                self.items[position],
                answer.probability,
                self.thresholds.noul_verdict(answer.probability),
                answer.from_store,
                answer.request_sha256,
            )
            for position, answer in sorted(self.answered.items())
        ]


@dataclass(frozen=True)
class _AllRequest:
    checks: tuple[Check, ...]
    picks: tuple[Pick, ...]
    scores: tuple[Rate, ...]
    questions: Mapping
    thresholds: Thresholds

    def answers(self, state: Mapping, response: JevResponse) -> AllAnswers:
        return AllAnswers(
            {check.name: _check_result(response, check, state, self.thresholds) for check in self.checks},
            {pick.name: _pick_result(response, pick.question_id, self.thresholds) for pick in self.picks},
            {rate.name: _score_result(response, rate.question_id) for rate in self.scores},
            response.request_sha256,
        )


@dataclass(frozen=True)
class _CallRequest:
    usable: Mapping[str, CallOffer]
    questions: Mapping
    thresholds: Thresholds

    def decision(self, response: JevResponse) -> CallDecision:
        route_result = _pick_result(response, ROUTE_QUESTION, self.thresholds)
        chosen = self.usable[route_result.choice]
        argument_result = _pick_result(response, _argument_id(chosen.name, chosen), self.thresholds)
        return CallDecision(chosen.name, argument_result.choice, route_result, argument_result)


def _is_async(client: object) -> bool:
    method = getattr(client, "send", None) or client.ask
    return inspect.iscoroutinefunction(method)


async def _awaited(method, *arguments):
    if inspect.iscoroutinefunction(method):
        return await method(*arguments)
    return await asyncio.to_thread(method, *arguments)


def _check_result(response: JevResponse, check: Check, state: Mapping, thresholds: Thresholds) -> CheckResult:
    probability = response.noul(check.question_id).probability
    return CheckResult(
        state, probability, thresholds.noul_verdict(probability), response.from_store, response.request_sha256
    )


def _score_result(response: JevResponse, question_id: str) -> ScoreResult:
    answer = response.score(question_id)
    return ScoreResult(answer.score, answer.probabilities, answer.confidence, response.request_sha256)


def _skeleton(list_name: str, questions: Mapping, items: list[Mapping], shared: Mapping) -> dict:
    """Everything needed to rebuild a batched request except the code itself and the shared state:
    the code is re-read from each item's file and lines, and only hashes of it are kept."""
    return {
        "list_name": list_name,
        "questions": dict(questions),
        "items": [{key: value for key, value in item.items() if key != CODE_FIELD} for item in items],
        "item_code_sha256": [content_hash(item.get(CODE_FIELD, "")) for item in items],
        "shared_sha256": content_hash(shared),
    }


def _source_of(item: Mapping) -> dict:
    """The location fields an item carries (``file``, ``lines``, ``commit``), if any."""
    return {key: item[key] for key in ("file", "lines", "commit") if key in item}


def _pick_result(response: JevResponse, question_id: str, thresholds: Thresholds) -> PickResult:
    answer = response.choice(question_id)
    return PickResult(
        answer.choice,
        answer.confidence,
        answer.probabilities,
        thresholds.choice_is_confident(answer.confidence),
        response.request_sha256,
    )


def _argument_id(operation: str, offer: CallOffer) -> str:
    return f"{operation}.{offer.argument.question_id}"


def _batches(pending: list[int], items: list[Mapping], shared: Mapping) -> list[list[int]]:
    budget = MAX_STATE_CHARS - len(json.dumps(shared))
    batches: list[list[int]] = []
    current: list[int] = []
    used = 0
    for position in pending:
        size = len(json.dumps(items[position]))
        if current and used + size > budget:
            batches.append(current)
            current, used = [], 0
        current.append(position)
        used += size
    return [*batches, current] if current else batches
