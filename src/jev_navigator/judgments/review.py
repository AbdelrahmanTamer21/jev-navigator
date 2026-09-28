"""Capture a function's exact first request, to review its questions before any paid call.

Run the function once against ``CapturingJevClient`` (it never calls Jev), then write the captured,
already-masked request with ``export_for_review``: one JSON file with the request and, per question,
what code does with the answer, ready for a person or a question-review tool.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .answers import ChoiceAnswer, JevResponse, NoulAnswer, ScoreAnswer
from .client import LATEST_JEV


@dataclass
class CapturingJevClient:
    """Answers every question with a neutral value in its own shape (Noul, Choice or Score) and keeps
    the requests it was given."""

    model: str = LATEST_JEV
    requests: list[tuple[Mapping, Mapping]] = field(default_factory=list)

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        self.requests.append((state, questions))
        return JevResponse(
            {question_id: _neutral(question) for question_id, question in questions.items()}, self.model
        )


def export_for_review(
    state: Mapping,
    questions: Mapping,
    intended_uses: Mapping[str, str],
    path: Path,
    *,
    case_id: str,
    group_id: str,
    revision_id: str = "v1",
) -> Path:
    """``intended_uses`` says, per question id, what code does with the answer."""
    missing = set(questions) - set(intended_uses)
    if missing:
        raise ValueError(f"every question needs an intended use; missing: {sorted(missing)}")
    candidate = {
        "case_id": case_id,
        "group_id": group_id,
        "revision_id": revision_id,
        "request": {"model": LATEST_JEV, "state": dict(state), "questions": dict(questions)},
        "intended_uses": dict(intended_uses),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    return path


def _neutral(question: Mapping):
    if question["type"] == "noul":
        return NoulAnswer(0.5)
    if question["type"] == "score":
        levels = [str(level) for level in range(len(question["criteria"]))]
        return ScoreAnswer.from_probabilities(_uniform(levels))
    return ChoiceAnswer.from_probabilities(_uniform(list(question["criteria"])))


def _uniform(options: list[str]) -> dict[str, float]:
    return {option: 1 / len(options) for option in options}
