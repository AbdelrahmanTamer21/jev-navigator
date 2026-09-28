"""Question templates a directive author writes once: concrete, with backticked state paths.

A question's id carries a hash of its wording, so a reworded question is a new question and an
executed question is never changed underneath its stored answer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MAX_CHOICE_OPTIONS = 255
ITEM_PLACEHOLDER = "{item}"


@dataclass(frozen=True)
class Criterion:
    what: str
    not_for: str = ""
    examples: tuple[str, ...] = ()

    def to_json(self) -> dict:
        described: dict = {"what": self.what}
        if self.not_for:
            described["not_for"] = self.not_for
        if self.examples:
            described["examples"] = list(self.examples)
        return described


@dataclass(frozen=True)
class Check:
    """A yes/no judgment about concrete supplied state. ``{item}`` in the instructions stands for
    one entry of a batched list, for example "Is `{item}.code` the implementation that
    `doc.sentence` describes?"."""

    name: str
    instructions: str
    yes: Criterion
    no: Criterion

    @property
    def question_id(self) -> str:
        return f"{self.name}@{wording_hash(self.to_question())}"

    def to_question(self, item_path: str = "") -> dict:
        question = {
            "type": "noul",
            "instructions": self.instructions,
            "criteria": {"true": self.yes.to_json(), "false": self.no.to_json()},
        }
        if not item_path:
            return question
        return json.loads(json.dumps(question).replace(ITEM_PLACEHOLDER, item_path))


@dataclass(frozen=True)
class Pick:
    """A Choice among options code supplies at call time."""

    name: str
    instructions: str

    @property
    def question_id(self) -> str:
        return f"{self.name}@{wording_hash({'instructions': self.instructions})}"

    def to_question(self, options: Mapping[str, str]) -> dict:
        if len(options) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"{self.name} has {len(options)} options; the API accepts {MAX_CHOICE_OPTIONS}")
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(options)}


@dataclass(frozen=True)
class Rate:
    """A Score question: ``levels`` are ordered descriptions, level 0 first, each understandable alone."""

    name: str
    instructions: str
    levels: tuple[str, ...]

    @property
    def question_id(self) -> str:
        return f"{self.name}@{wording_hash(self.to_question())}"

    def to_question(self) -> dict:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.levels)}


def wording_hash(question: Mapping) -> str:
    encoded = json.dumps(question, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:8]


def request_sha256(state: Mapping, questions: Mapping) -> str:
    encoded = json.dumps({"state": state, "questions": questions}, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def item_path(list_name: str, position: int) -> str:
    return f"{list_name}[{position}]"


def numbered(values: Sequence[str]) -> dict[str, str]:
    return {str(position): value for position, value in enumerate(values)}
