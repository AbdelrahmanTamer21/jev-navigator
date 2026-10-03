"""LlmStep: an opt-in building block a user adds to their own directive. Nothing else calls an LLM.

The user defines all four parts:

- ``when``: a predicate over the Jev results that decides whether the LLM is called at all;
- ``context``: builds, from those results, exactly what the LLM gets;
- ``answer``: the answer contract, which renders the prompt and parses the reply;
- ``connector``: which LLM (see ``jev_navigator.connectors``).

The library only guards what is sent: the context is masked, the rendered prompt is scanned and
refused on a hit, each call is stored (prompt hash, reply, parsed answer), and an optional budget
stops further calls. A reply that does not parse is retried once with the parse error.

``run`` never returns ``None``: the result carries a ``status`` (``not_requested``,
``budget_exhausted``, ``answered``, ``parse_failed``) so callers can inspect the answer and the
attempt count. Every provider attempt is persisted through the guard before the connector is
called (its identity, connector, model and exact prompt hash), the reply is persisted before it
is parsed or a retry starts, and the parse outcome and provider failures are persisted separately.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar
from uuid import uuid4

from .judgments.questions import content_hash
from .judgments.secrets import (
    Masker,
    Scanner,
    SecretMasker,
    SecretScanner,
    mask_by_content,
    refuse_if_secret,
)

Parsed = TypeVar("Parsed")
Results = TypeVar("Results")


class ReplyParseError(ValueError):
    pass


class Connector(Protocol):
    name: str
    model: str

    def complete(self, prompt: str) -> str: ...


class AnswerContract(Protocol[Parsed]):
    def render(self, context: Mapping, parse_error: str = "") -> str: ...

    def parse(self, reply: str, context: Mapping) -> Parsed: ...


class LlmStatus(StrEnum):
    """The four outcomes of ``LlmStep.run``; the string value is the status spelled out."""

    NOT_REQUESTED = "not_requested"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ANSWERED = "answered"
    PARSE_FAILED = "parse_failed"


@dataclass(frozen=True)
class LlmCall(Generic[Parsed]):
    """One ``LlmStep.run`` outcome. ``answer`` is set only when ``status`` is ``answered``;
    ``parse_error`` says why the last reply never parsed, and ``attempts`` counts provider calls."""

    status: LlmStatus
    answer: Parsed | None = None
    reply: str | None = None
    parse_error: str = ""
    attempts: int = 0
    connector: str = ""
    model: str = ""
    prompt_sha256: str = ""


@dataclass
class LlmGuard:
    """Masking and the pre-send scan are on by default; ``store_path`` appends one JSON line per call."""

    masker: Masker | None = field(default_factory=SecretMasker)
    scanner: Scanner | None = field(default_factory=SecretScanner)
    store_path: Path | None = None
    max_calls: int | None = None
    calls: int = 0

    def budget_left(self) -> bool:
        return self.max_calls is None or self.calls < self.max_calls

    def safe_context(self, context: Mapping) -> Mapping:
        return mask_by_content(context, self.masker) if self.masker else context

    def check_prompt(self, prompt: str) -> None:
        refuse_if_secret({"prompt": prompt}, {}, self.scanner)

    def record(self, step_name: str, call: LlmCall) -> None:
        """One summary line per finished call, as before (plus the status)."""
        self._append(
            {
                "event": "call",
                "status": call.status.value,
                "attempts": call.attempts,
                "answer": _jsonable(call.answer),
                "reply": call.reply,
                "prompt_sha256": call.prompt_sha256,
                "connector": call.connector,
                "model": call.model,
                "parse_error": call.parse_error,
            },
            step_name,
        )

    def record_attempt(
        self, step_name: str, attempt: str, connector: str, model: str, prompt_sha256: str
    ) -> None:
        """Persist the attempt identity before the connector is called."""
        self._append(
            {
                "event": "attempt",
                "attempt": attempt,
                "connector": connector,
                "model": model,
                "prompt_sha256": prompt_sha256,
            },
            step_name,
        )

    def record_reply(self, step_name: str, attempt: str, reply: str) -> None:
        """Persist the exact reply before it is parsed or a retry starts."""
        self._append({"event": "reply", "attempt": attempt, "reply": reply}, step_name)

    def record_parse(self, step_name: str, attempt: str, parse_error: str = "") -> None:
        """Persist the parse outcome; an empty ``parse_error`` means the reply parsed."""
        self._append({"event": "parse", "attempt": attempt, "parse_error": parse_error}, step_name)

    def record_failure(self, step_name: str, attempt: str, error: str) -> None:
        """Persist a provider failure against its already-created attempt."""
        self._append({"event": "failure", "attempt": attempt, "error": error}, step_name)

    def _append(self, line: dict, step_name: str) -> None:
        if self.store_path is None:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "kind": "llm_step",
            "step": step_name,
            "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **line,
        }
        with self.store_path.open("a") as lines:
            lines.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass
class LlmStep(Generic[Results, Parsed]):
    name: str
    when: Callable[[Results], bool]
    context: Callable[[Results], Mapping]
    answer: AnswerContract[Parsed]
    connector: Connector
    guard: LlmGuard = field(default_factory=LlmGuard)

    def run(self, results: Results) -> LlmCall[Parsed]:
        """``not_requested`` when ``when`` says no, ``budget_exhausted`` when the budget is used up;
        otherwise the call and its answer, or ``parse_failed`` when no reply ever parsed."""
        if not self.when(results):
            return LlmCall(LlmStatus.NOT_REQUESTED)
        if not self.guard.budget_left():
            return LlmCall(LlmStatus.BUDGET_EXHAUSTED)
        context = self.guard.safe_context(self.context(results))
        call = self._ask(context)
        self.guard.record(self.name, call)
        return call

    def _ask(self, context: Mapping, parse_error: str = "", attempt: int = 1) -> LlmCall[Parsed]:
        prompt = self.answer.render(context, parse_error)
        self.guard.check_prompt(prompt)
        self.guard.calls += 1
        attempt_id = uuid4().hex
        prompt_sha256 = content_hash(prompt)
        self.guard.record_attempt(
            self.name, attempt_id, self.connector.name, self.connector.model, prompt_sha256
        )
        try:
            reply = self.connector.complete(prompt)
        except Exception as error:
            self.guard.record_failure(self.name, attempt_id, f"{type(error).__name__}: {error}")
            raise
        self.guard.record_reply(self.name, attempt_id, reply)
        try:
            parsed = self.answer.parse(reply, context)
        except ReplyParseError as error:
            self.guard.record_parse(self.name, attempt_id, str(error))
            if attempt == 1 and self.guard.budget_left():
                return self._ask(context, str(error), attempt=2)
            return LlmCall(
                status=LlmStatus.PARSE_FAILED,
                reply=reply,
                parse_error=str(error),
                attempts=attempt,
                connector=self.connector.name,
                model=self.connector.model,
                prompt_sha256=prompt_sha256,
            )
        self.guard.record_parse(self.name, attempt_id)
        return LlmCall(
            status=LlmStatus.ANSWERED,
            answer=parsed,
            reply=reply,
            attempts=attempt,
            connector=self.connector.name,
            model=self.connector.model,
            prompt_sha256=prompt_sha256,
        )


@dataclass(frozen=True)
class PickFromOptions:
    """Helper contract: the context must hold ``options`` (key to description); the reply is
    ``{"<answer_field>": "<key>"}`` and must name one of those keys."""

    instructions: str
    answer_field: str = "choice"

    def render(self, context: Mapping, parse_error: str = "") -> str:
        options = "\n".join(f"- `{key}`: {text}" for key, text in context["options"].items())
        rest = {key: value for key, value in context.items() if key != "options"}
        shape = f'{{"{self.answer_field}": "<option key>"}}'
        return _prompt(self.instructions, rest, f"## Options\n{options}", shape, parse_error)

    def parse(self, reply: str, context: Mapping) -> str:
        value = _single_json_object(reply).get(self.answer_field)
        if value not in context["options"]:
            raise ReplyParseError(f"{self.answer_field} must be one of the listed option keys, got {value!r}")
        return value


@dataclass(frozen=True)
class JsonContract:
    """Helper contract: the reply is one JSON object with at least ``required`` fields of the given types."""

    instructions: str
    required: Mapping[str, type]
    example: str = ""

    def render(self, context: Mapping, parse_error: str = "") -> str:
        example = self.example or json.dumps(
            {name: f"<{kind.__name__}>" for name, kind in self.required.items()}
        )
        return _prompt(self.instructions, context, "", example, parse_error)

    def parse(self, reply: str, context: Mapping) -> dict:
        parsed = _single_json_object(reply)
        for name, kind in self.required.items():
            if not isinstance(parsed.get(name), kind):
                raise ReplyParseError(f"field {name!r} must be a {kind.__name__}")
        return parsed


def _prompt(instructions: str, context: Mapping, extra: str, answer_shape: str, parse_error: str) -> str:
    retry = f"\nYour previous reply could not be used: {parse_error}\n" if parse_error else ""
    return (
        f"## Task\n{instructions}\n\n"
        f"## Context\n```json\n{json.dumps(context, indent=1, sort_keys=True)}\n```\n\n"
        f"{extra}\n{retry}\nAnswer with one line of JSON and nothing else: {answer_shape}\n"
    )


def _single_json_object(reply: str) -> dict:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        raise ReplyParseError("the reply holds no JSON object")
    try:
        parsed = json.loads(reply[start : end + 1])
    except json.JSONDecodeError as error:
        raise ReplyParseError(f"the JSON did not parse: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise ReplyParseError("the reply must be one JSON object")
    return parsed


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
