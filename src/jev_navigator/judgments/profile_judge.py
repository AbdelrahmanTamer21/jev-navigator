"""find_all's Judge, asking a question set in place of find_all's one match question.

This collection adapter shares the library's question profiles. find_all hands the Judge one wave at a time
(``iter_check_every``); here every place of the wave is asked the set's questions for every point, in
the batches the Judge forms, and find_all gets back one rank per place and point under its own match
name, which code composes from the set's answers. The waves, the call cap and the places stay find_all's,
so a set changes only what is asked.

Masking: a point's text is sent as written unless it holds a secret. With a ``request_masker``, the items
are masked as JVN masks them, each value hidden in one item hidden in all of them; the point text and
the set's own state fields get the masker's own rules and lose only copies of values of
``BY_CONTENT_MIN_CHARS`` or more characters, the length from which JVN matches a copy anywhere. A short
value hidden in an item (``"a"``, ``"local"``) is a word collision in prose, and copying it into the
point text changed the question in 15 to 24 percent of requests. The threat kept covered is a real
secret that a finder quoted into its text; the Judge's final scan still reads every text sent. Without
a ``request_masker`` nothing is masked: the simulator sends nothing anywhere.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from jev_navigator.directives.find_all import ITEMS, TARGETS, match_check
from jev_navigator.judgments.judge import CheckResult, Judge
from jev_navigator.judgments.questions import Check, content_hash
from jev_navigator.judgments.secret_shapes import BY_CONTENT_MIN_CHARS
from jev_navigator.judgments.secrets import Masker, mask_everywhere, masked_values
from jev_navigator.judgments.thresholds import Thresholds

from .profiles import QuestionProfile


@dataclass(frozen=True)
class AskedAnswer:
    """One raw answer as asked: the place and point, the question, the code as the index holds it
    (``entry_sha256`` of the unmasked item), the set's own state fields the question reads
    (``shared_sha256``), the request that answered it and the model that served it."""

    place_id: str
    file: str
    ranges: tuple[tuple[int, int], ...]
    entry_sha256: str
    target: str
    point_sha256: str
    shared_sha256: str
    question: str
    question_id: str
    probability: float
    request_sha256: str
    served_model: str | None


def shared_sha256(state: Mapping) -> str:
    """The state fields a question reads beside its item and its point, such as the role definitions."""
    return content_hash({name: value for name, value in state.items() if name not in (ITEMS, TARGETS)})


@dataclass(frozen=True)
class _Unmasked:
    """The hashes of what one wave asks about, taken before masking: each place's item, each point's
    text and the set's own state fields."""

    entries: Mapping[str, str]
    points: Mapping[str, str]
    shared: str


class ProfileJudge(Judge):
    def __init__(
        self,
        client,
        question_set: QuestionProfile,
        *,
        request_masker: Masker | None = None,
        on_answer: Callable[[AskedAnswer], None] | None = None,
        **options,
    ) -> None:
        super().__init__(client, masker=None, **options)
        self.question_set = question_set
        self.request_masker = request_masker
        self.on_answer = on_answer

    def iter_check_every(
        self,
        checks: Sequence[Check],
        items: Sequence[Mapping],
        shared: Mapping | None = None,
        *,
        list_name: str = "items",
        thresholds: Thresholds | None = None,
        cancelled: Callable[[], bool] | None = None,
        places=None,
        refusals=None,
        keep_order: bool = False,
    ) -> Iterator[tuple[str, CheckResult]]:
        """find_all's ``checks`` are its match questions, one per point; the points themselves are read
        from ``shared``, and each comes back once every question of the set has its answer."""
        asked, sent_items, sent_shared, unmasked, per_target = self._prepare_set(items, shared, places)
        answers = {}
        for name, result in super().iter_check_every(
            [check for check, _, _ in asked.values()],
            sent_items,
            sent_shared,
            list_name=list_name,
            thresholds=thresholds,
            cancelled=cancelled,
            places=places,
            refusals=refusals,
            keep_order=keep_order,
        ):
            composed = self._consume(name, result, asked, unmasked, answers, per_target, thresholds)
            if composed is not None:
                yield composed

    async def iter_check_every_async(
        self,
        checks,
        items,
        shared=None,
        *,
        list_name="items",
        thresholds=None,
        places=None,
        refusals=None,
        keep_order=False,
    ):
        """The async pack path asks and composes the same question set as the sync search."""
        asked, sent_items, sent_shared, unmasked, per_target = self._prepare_set(items, shared, places)
        answers = {}
        async for name, result in super().iter_check_every_async(
            [check for check, _, _ in asked.values()],
            sent_items,
            sent_shared,
            list_name=list_name,
            thresholds=thresholds,
            places=places,
            refusals=refusals,
            keep_order=keep_order,
        ):
            composed = self._consume(name, result, asked, unmasked, answers, per_target, thresholds)
            if composed is not None:
                yield composed

    def _prepare_set(self, items, shared, places):
        asked = self.question_set.asked(shared[TARGETS])
        sent_items, sent_shared = self._as_sent(items, shared, [check for check, _, _ in asked.values()])
        unmasked = _Unmasked(
            {place.id: content_hash(item) for place, item in zip(places, items, strict=True)},
            {target: content_hash(point) for target, point in shared[TARGETS].items()},
            shared_sha256({**shared, **self.question_set.shared}),
        )
        per_target = Counter(target for _, _, target in asked.values())
        return asked, sent_items, sent_shared, unmasked, per_target

    def _consume(self, name, result, asked, unmasked, answers, per_target, thresholds):
        check, question, target = asked[name]
        self._report(result, check, question, target, unmasked)
        place_answers = answers.setdefault((result.place.id, target), {})
        place_answers[question] = result
        if len(place_answers) == per_target[target]:
            return match_check(target).name, self._ranked(place_answers, thresholds or self.thresholds)
        return None

    def fits_alone(
        self, checks: Sequence[Check], item: Mapping, shared: Mapping, list_name: str = "items"
    ) -> bool:
        """Whether the set's questions about ``item`` alone fit, sent as this Judge sends them."""
        set_checks = [check for check, _, _ in self.question_set.asked(shared[TARGETS]).values()]
        [sent_item], sent_shared = self._as_sent([item], shared, set_checks)
        return super().fits_alone(set_checks, sent_item, sent_shared, list_name)

    def _as_sent(
        self, items: Sequence[Mapping], shared: Mapping, checks: Sequence[Check]
    ) -> tuple[list[Mapping], Mapping]:
        """The items and the shared state with the set's fields, masked as the module says."""
        shared = {**shared, **self.question_set.shared}
        if self.request_masker is None:
            return list(items), shared
        hidden = masked_values(
            [*items, shared, *(check.to_question() for check in checks)], self.request_masker
        )
        long_copies = frozenset(value for value in hidden if len(value) >= BY_CONTENT_MIN_CHARS)
        sent_shared = mask_everywhere(shared, self.request_masker, long_copies)
        # JVN treats targets as wording and deliberately does not hide copied values there.
        # This adapter's existing contract hides proved long secrets copied into a point.
        sent_shared[TARGETS] = {
            target: mask_everywhere({"value": point}, self.request_masker, long_copies)["value"]
            for target, point in shared[TARGETS].items()
        }
        return (
            [mask_everywhere(item, self.request_masker, hidden) for item in items],
            sent_shared,
        )

    def _report(
        self,
        result: CheckResult,
        check: Check,
        question: str,
        target: str,
        unmasked: _Unmasked,
    ) -> None:
        if self.on_answer is None:
            return
        place = result.place
        self.on_answer(
            AskedAnswer(
                place.id,
                place.file,
                tuple(tuple(run) for run in place.ranges),
                unmasked.entries[place.id],
                target,
                unmasked.points[target],
                unmasked.shared,
                question,
                check.question_id,
                result.probability,
                result.request_sha256,
                self.served_model,
            )
        )

    def _ranked(self, answers: Mapping[str, CheckResult], thresholds: Thresholds) -> CheckResult:
        """One place's rank for one point: the set's own answer when it asks one question, else the
        composed rank, carrying the request of the answers it came from."""
        if len(answers) == 1:
            return next(iter(answers.values()))
        first = next(iter(answers.values()))
        if isinstance(self.question_set, QuestionProfile):
            return self.question_set.compose(answers, thresholds)
        rank = self.question_set.rank({question: answer.probability for question, answer in answers.items()})
        return CheckResult(
            first.item,
            rank,
            thresholds.noul_verdict(rank),
            all(answer.from_store for answer in answers.values()),
            first.request_sha256,
            None,
            first.place,
            dict(answers),
        )
