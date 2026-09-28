"""Typed Jev answers, independent of any SDK, in the form the answer store keeps them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]
    confidence: float

    @classmethod
    def from_probabilities(cls, probabilities: Mapping[str, float]) -> ChoiceAnswer:
        """An answer whose confidence is derived as the API does: (n * max - 1) / (n - 1)."""
        winner = max(probabilities, key=probabilities.__getitem__)
        return cls(winner, dict(probabilities), distribution_confidence(probabilities))

    def to_json(self) -> dict:
        return {
            "type": "choice",
            "choice": self.choice,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class NoulAnswer:
    probability: float

    def to_json(self) -> dict:
        return {"type": "noul", "noul": self.probability}


@dataclass(frozen=True)
class ScoreAnswer:
    """``score`` is the probability-weighted level; ``probabilities`` are keyed by level number as text."""

    score: float
    probabilities: Mapping[str, float]
    confidence: float

    @classmethod
    def from_probabilities(cls, probabilities: Mapping[str, float]) -> ScoreAnswer:
        expected = sum(int(level) * probability for level, probability in probabilities.items())
        return cls(expected, dict(probabilities), distribution_confidence(probabilities))

    def to_json(self) -> dict:
        return {
            "type": "score",
            "score": self.score,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
        }


Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer


@dataclass(frozen=True)
class JevResponse:
    answers: Mapping[str, Answer]
    model: str
    input_tokens: int = 0
    request_sha256: str = ""
    from_store: bool = False
    extra: Mapping[str, object] = field(default_factory=dict)

    def choice(self, question_id: str) -> ChoiceAnswer:
        answer = self.answers[question_id]
        if not isinstance(answer, ChoiceAnswer):
            raise TypeError(f"{question_id} is not a Choice answer")
        return answer

    def score(self, question_id: str) -> ScoreAnswer:
        answer = self.answers[question_id]
        if not isinstance(answer, ScoreAnswer):
            raise TypeError(f"{question_id} is not a Score answer")
        return answer

    def noul(self, question_id: str) -> NoulAnswer:
        answer = self.answers[question_id]
        if not isinstance(answer, NoulAnswer):
            raise TypeError(f"{question_id} is not a Noul answer")
        return answer


def answer_from_json(raw: Mapping) -> Answer:
    if raw["type"] == "noul":
        return NoulAnswer(float(raw["noul"]))
    probabilities = {str(key): float(value) for key, value in raw["probabilities"].items()}
    if raw["type"] == "score":
        return ScoreAnswer(float(raw["score"]), probabilities, float(raw["confidence"]))
    confidence = raw.get("confidence")
    if confidence is None:
        return ChoiceAnswer.from_probabilities(probabilities)
    return ChoiceAnswer(str(raw["choice"]), probabilities, float(confidence))


def distribution_confidence(probabilities: Mapping[str, float]) -> float:
    option_count = len(probabilities)
    if option_count < 2:
        return 1.0
    return (option_count * max(probabilities.values()) - 1) / (option_count - 1)


def response_from_raw(raw: Mapping) -> JevResponse:
    """Parses a raw System One response: ``{"model", "usage": {"input_tokens"}, "answers": {...}}``."""
    answers = {question_id: answer_from_json(answer) for question_id, answer in raw["answers"].items()}
    usage = raw.get("usage") or {}
    return JevResponse(answers, str(raw["model"]), int(usage.get("input_tokens") or 0))


def response_to_raw(response: JevResponse) -> dict:
    return {
        "model": response.model,
        "usage": {"input_tokens": response.input_tokens},
        "answers": {question_id: answer.to_json() for question_id, answer in response.answers.items()},
    }
