"""Label only caller-selected source pieces. The existing Judge owns batching and storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..index.units import Item
from .judge import CheckResult, Judge, Refusal
from .profiles import ROLE_QUESTIONS, QuestionProfile

_ROLE_LABELS = QuestionProfile("role-labels", ROLE_QUESTIONS)
_LABEL_BATCH_SIZE = 16


@dataclass(frozen=True)
class LabelPiece:
    """One selected piece and its exact visible text; no surrounding source is fetched."""

    place: Item
    code: str


@dataclass(frozen=True)
class PieceRoles:
    """Independent raw answers per point and role for one supplied piece."""

    piece: LabelPiece
    answers: Mapping[str, Mapping[str, CheckResult]]

    @property
    def probabilities(self) -> dict[str, dict[str, float]]:
        return {
            target: {role: answer.probability for role, answer in roles.items()}
            for target, roles in self.answers.items()
        }


@dataclass(frozen=True)
class RoleLabellingResult:
    """Labels in supplied piece order. Refusals remain unknown, never zero probabilities."""

    pieces: tuple[PieceRoles, ...]
    refusals: tuple[Refusal, ...]
    calls: int


class _Labelling:
    def __init__(self, judge: Judge, pieces: Sequence[LabelPiece], targets: Mapping[str, str]):
        self.pieces = tuple(pieces)
        ids = [piece.place.id for piece in pieces]
        if len(set(ids)) != len(ids):
            raise ValueError("Selected pieces must have distinct place ids")
        if pieces and not targets:
            raise ValueError("Role labelling needs at least one target")
        self.judge = judge.scope()
        self.judge.items_per_request = _LABEL_BATCH_SIZE
        self.asked = _ROLE_LABELS.asked(targets)
        self.checks = [check for check, _, _ in self.asked.values()]
        self.items = [{"file": piece.place.file, "code": piece.code} for piece in pieces]
        self.shared = {"targets": dict(targets)}
        self.places = [piece.place for piece in pieces]
        self.answers = {key: {target: {} for target in targets} for key in ids}
        self.refusals: list[Refusal] = []

    def consume(self, name: str, answer: CheckResult) -> None:
        _, role, target = self.asked[name]
        self.answers[answer.place.id][target][role] = answer

    def result(self) -> RoleLabellingResult:
        return RoleLabellingResult(
            tuple(PieceRoles(piece, self.answers[piece.place.id]) for piece in self.pieces),
            tuple(self.refusals),
            self.judge.calls,
        )


def label_roles(
    judge: Judge, pieces: Sequence[LabelPiece], targets: Mapping[str, str]
) -> RoleLabellingResult:
    """Ask all six roles per piece and target, sixteen selected pieces per request.

    The Judge splits oversized requests by item and retains every raw answer in its configured
    store. Give this step its own caller-owned call allowance. Labels never change selection.
    """
    work = _Labelling(judge, pieces, targets)
    for name, answer in work.judge.iter_check_every(
        work.checks,
        work.items,
        work.shared,
        places=work.places,
        refusals=work.refusals,
        keep_order=True,
    ):
        work.consume(name, answer)
    return work.result()


async def label_roles_async(
    judge: Judge, pieces: Sequence[LabelPiece], targets: Mapping[str, str]
) -> RoleLabellingResult:
    """The same role-labelled pieces through the existing asynchronous Judge."""
    work = _Labelling(judge, pieces, targets)
    async for name, answer in work.judge.iter_check_every_async(
        work.checks,
        work.items,
        work.shared,
        places=work.places,
        refusals=work.refusals,
        keep_order=True,
    ):
        work.consume(name, answer)
    return work.result()
