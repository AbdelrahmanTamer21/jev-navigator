"""Versioned search questions and their code composition, independent of caller domains.

The role wording is proposal-v2, candidate SHA-256
 d1b1d84d125b65e361a56d69386836eccbb2e490ff0906103ac83555d9132a2f.
Required roles affect retention only. Search stopping remains the search policy's.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files

from .judge import CheckResult
from .questions import Check, Criterion, content_hash
from .thresholds import Thresholds

LOCAL_ROLES = ("decide", "guard", "value", "effect")
ROLE_QUESTIONS = json.loads(files(__package__).joinpath("role_questions.json").read_text())


def bind_question(value, item: str, point: str):
    """Bind measured state references without changing wording or criteria."""
    if isinstance(value, str):
        return value.replace("`unit", f"`{item}").replace("`point`", f"`{point}`")
    if isinstance(value, dict):
        return {key: bind_question(part, item, point) for key, part in value.items()}
    if isinstance(value, list):
        return [bind_question(part, item, point) for part in value]
    return value


@dataclass(frozen=True)
class RoleAnswers:
    """Raw role observations reusable by search and offline allocation."""

    unit_id: str
    probabilities: Mapping[str, float]

    @property
    def relevance(self) -> float:
        return max(self.probabilities[role] for role in (*LOCAL_ROLES, "satisfied"))

    @property
    def forwarding_only(self) -> bool:
        return (
            self.probabilities["delegates"] >= 0.80
            and max(self.probabilities[role] for role in LOCAL_ROLES) <= 0.20
        )


def compose_roles(answers: Sequence[RoleAnswers], required: Sequence[str]) -> dict:
    """Measured role winners, uncovered roles and forwarding observations, without a stop rule."""
    best = {
        role: max(answers, key=lambda answer: answer.probabilities[role], default=None) for role in required
    }
    covered = {
        role for role, answer in best.items() if answer is not None and answer.probabilities[role] >= 0.80
    }
    return {
        "best_by_role": best,
        "uncovered_roles": set(required) - covered,
        "ranked": sorted(answers, key=lambda answer: -answer.relevance),
        "follow_units": [answer.unit_id for answer in answers if answer.forwarding_only],
    }


@dataclass(frozen=True)
class QuestionProfile:
    """A selectable question contract. Callers explicitly choose retention roles."""

    name: str
    templates: Mapping

    shared: Mapping = field(default_factory=dict)

    def questions(self, target: str, point: str = "") -> tuple[Check, ...]:
        checks = []
        for role, template in self.templates.items():
            question = bind_question(template, "{item}", f"targets.{target}")
            criteria = question.get("criteria", {})
            checks.append(
                Check(
                    f"{role}_{target}",
                    question["instructions"],
                    yes=Criterion(**criteria["true"]) if criteria else None,
                    no=Criterion(**criteria["false"]) if criteria else None,
                )
            )
        return tuple(checks)

    def asked(self, targets: Mapping[str, str]) -> dict:
        """Checks by name, with their role and target, for collection adapters."""
        return {
            check.name: (check, check.name.removesuffix(f"_{target}"), target)
            for target in targets
            for check in self.questions(target)
        }

    def identity(self) -> str:
        """Versioned meaning shared by search and offline collection."""
        return f"{self.name}@{content_hash(self.templates)}"

    def rank(self, probabilities: Mapping[str, float]) -> float:
        """The measured scalar composition, with delegation retained separately."""
        if "match" in self.templates:
            return probabilities["match"]
        return RoleAnswers("", probabilities).relevance

    def compose(self, answers: Mapping[str, CheckResult], thresholds: Thresholds) -> CheckResult:
        """Keep raw judgments, composing relevance only after every answer arrives."""
        first = next(iter(answers.values()))
        if self.name == "match":
            return first
        relevance = self.rank({role: answer.probability for role, answer in answers.items()})
        return CheckResult(
            first.item,
            relevance,
            thresholds.noul_verdict(relevance),
            all(answer.from_store for answer in answers.values()),
            first.request_sha256,
            None,
            first.place,
            dict(answers),
        )


def forwarding_only(answers: Mapping[str, CheckResult]) -> bool:
    """Delegation is forwarding-only only when all local roles are low."""
    return RoleAnswers("", {role: answer.probability for role, answer in answers.items()}).forwarding_only


def retain_roles(scores: Sequence, required: Sequence[str]) -> tuple:
    """Best per required role first, then stable relevance order, with uncertain best retained."""
    if unknown := set(required) - set(LOCAL_ROLES):
        raise ValueError(f"unknown local roles: {sorted(unknown)}")
    observed = [
        RoleAnswers(
            score.unit.id, {role: answer.probability for role, answer in score.answer.components.items()}
        )
        for score in scores
    ]
    composed = compose_roles(observed, required)
    by_id = {score.unit.id: score for score in scores}
    ordered = [
        by_id[answer.unit_id]
        for answer in (*filter(None, composed["best_by_role"].values()), *composed["ranked"])
    ]
    retained = {}
    for score in ordered:
        retained.setdefault(score.unit.id, score)
    return tuple(retained.values())


MATCH = QuestionProfile(
    "match",
    {
        "match": {
            "type": "noul",
            "instructions": "Look only at `unit`. Does that code match the description in `point`?",
        }
    },
)
ROLES_V2 = QuestionProfile("roles-v2", ROLE_QUESTIONS)
