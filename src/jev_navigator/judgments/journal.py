"""A durable record of every attempt, kept apart from the parsed-answer store.

The judge records the exact (masked) request before dispatch and the raw response before parsing:
the body bytes as received, the HTTP status and the content type, with the decoded form optional. A
transport error or a response that fails to parse is recorded as a failure. Hosts inject their
own journal (their runtime's, an evaluation journal); ``JsonlJournal`` is a simple local one.

A request holds code, and the library cannot know whose code it is. So by default ``JsonlJournal``
keeps only the request hash, the question ids and a hash of the state; the full request text is kept
only with ``keep_request_text=True``, which is meant for your own or open-source code.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .questions import content_hash


@dataclass(frozen=True)
class JournalRequest:
    """``body`` is the request as handed to the client, keys in the order they were built."""

    request_sha256: str
    requested_model: str
    state: Mapping
    questions: Mapping
    body: bytes = b""


@dataclass(frozen=True)
class RawResponse:
    """A response as received. ``body`` is the exact bytes when the client captured them
    (``exact``); a client that only returns a decoded response is recorded with its JSON re-encoded,
    ``exact=False`` and no status. ``sent_body`` is the exact request body the client sent, when its
    transport captured it."""

    body: bytes
    status: int | None
    content_type: str
    decoded: Mapping | None = None
    exact: bool = True
    sent_body: bytes | None = None

    @classmethod
    def from_decoded(cls, decoded: Mapping) -> RawResponse:
        body = json.dumps(decoded, sort_keys=True).encode()
        return cls(body, None, "application/json", decoded, exact=False)

    def json(self) -> Mapping:
        return self.decoded if self.decoded is not None else json.loads(self.body)


class Journal(Protocol):
    def record_request(self, request: JournalRequest) -> str:
        """Returns the id later records refer to."""
        ...

    def record_response(self, request_id: str, response: RawResponse) -> None: ...

    def record_failure(self, request_id: str, error: str, response: RawResponse | None = None) -> None: ...


class JsonlJournal:
    def __init__(self, path: Path, *, keep_request_text: bool = False) -> None:
        self.path = Path(path)
        self.keep_request_text = keep_request_text

    def record_request(self, request: JournalRequest) -> str:
        request_id = uuid.uuid4().hex
        fields = _request_with_text(request) if self.keep_request_text else _request_without_text(request)
        self._append({"kind": "request", "request_id": request_id, **fields})
        return request_id

    def record_response(self, request_id: str, response: RawResponse) -> None:
        self._append({"kind": "response", "request_id": request_id, **self._response_fields(response)})

    def record_step(self, step: Mapping) -> None:
        """Lets a ``History`` record every appended step in the same file."""
        self._append({"kind": "history_step", "step": dict(step)})

    def record_failure(self, request_id: str, error: str, response: RawResponse | None = None) -> None:
        fields = self._response_fields(response) if response is not None else {}
        self._append({"kind": "failure", "request_id": request_id, "error": error, **fields})

    def _response_fields(self, response: RawResponse) -> dict:
        """The sent body holds code, so it is kept only with ``keep_request_text``."""
        fields = _response_fields(response)
        if self.keep_request_text and response.sent_body is not None:
            fields["sent_body_base64"] = _base64(response.sent_body)
        return fields

    def _append(self, line: dict) -> None:
        line["recorded_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as lines:
            lines.write(json.dumps(line, sort_keys=True, default=str) + "\n")


def _response_fields(response: RawResponse) -> dict:
    return {
        "body_base64": _base64(response.body),
        "status": response.status,
        "content_type": response.content_type,
        "exact": response.exact,
    }


def _request_with_text(request: JournalRequest) -> dict:
    """The state and questions for reading (written with sorted keys) and the body in its exact order."""
    fields = {key: value for key, value in asdict(request).items() if key != "body"}
    return {**fields, "body_base64": _base64(request.body)}


def _base64(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


def _request_without_text(request: JournalRequest) -> dict:
    return {
        "request_sha256": request.request_sha256,
        "requested_model": request.requested_model,
        "question_ids": list(request.questions),
        "state_sha256": content_hash(request.state),
    }
