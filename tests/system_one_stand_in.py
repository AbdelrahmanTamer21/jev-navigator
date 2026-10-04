"""A local System One stand-in server for tests: it answers every question yes, fails with a status,
or refuses for input size the way Jev does, and never reaches a hosted provider."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Event, Thread

SIZE_REFUSAL = {"detail": {"error_type": "max_tokens_exceeded"}}
"""Jev's body for a request over its input limit, sent with HTTP 400."""


@dataclass
class StandIn:
    """A local System One server's address, the request bodies it received, and two gates: ``arrived``
    is set when a request comes in, and a held server answers only once ``release`` is set."""

    url: str
    received: list[dict] = field(default_factory=list)
    arrived: Event = field(default_factory=Event)
    release: Event = field(default_factory=Event)
    throttled: int = 0
    in_flight: int = 0
    flight: Condition = field(default_factory=Condition)


@contextmanager
def stand_in(
    model: str,
    *,
    status: int = 200,
    refuses_size: bool = False,
    held: bool = False,
    max_in_flight: int | None = None,
) -> Iterator[StandIn]:
    """A System One server that answers every question yes, fails every request with ``status``, or,
    when it ``refuses_size``, refuses every request as too large. A ``held`` server keeps each answer
    back until the test sets ``release``. With ``max_in_flight``, a request beyond that many in flight
    gets HTTP 429 and is counted in ``throttled``, and each admitted request stays open briefly, so
    concurrent requests really overlap."""
    serving = StandIn("")
    if not held:
        serving.release.set()
    if refuses_size:
        status = 400
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(model, status, serving, max_in_flight))
    serving.url = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever)
    try:
        thread.start()
        yield serving
    finally:
        serving.release.set()
        server.shutdown()
        server.server_close()
        if thread.is_alive():
            thread.join()


def _handler(
    model: str, status: int, serving: StandIn, max_in_flight: int | None
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            serving.received.append(payload)
            serving.arrived.set()
            serving.release.wait()
            admitted = _enter(serving, max_in_flight)
            try:
                answered_status = status if admitted else 429
                answered = _answers(model, payload) if answered_status == 200 else _failure(answered_status)
                self._reply(answered_status, json.dumps(answered).encode())
            finally:
                _leave(serving)

        def _reply(self, answered_status: int, body: bytes) -> None:
            self.send_response(answered_status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    return Handler


def _answers(model: str, payload: Mapping) -> dict:
    answers = {}
    for name, question in payload["questions"].items():
        if question["type"] == "noul":
            answers[name] = {"type": "noul", "noul": 0.95}
        else:
            labels = list(question["criteria"])
            answers[name] = {
                "type": "choice",
                "choice": labels[0],
                "confidence": 1.0,
                "probabilities": {label: float(label == labels[0]) for label in labels},
            }
    return {"model": model, "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 10}}


def _failure(status: int) -> dict:
    return SIZE_REFUSAL if status == 400 else {"detail": "unavailable"}


OVERLAP_SECONDS = 0.3
"""How long a limited server keeps each admitted request open, so a client sending more than the
limit at once is caught with a request over it."""


def _enter(serving: StandIn, max_in_flight: int | None) -> bool:
    """Whether the server admits one more request in flight. An admitted request stays open until a
    request over the limit arrives or ``OVERLAP_SECONDS`` pass, so overlapping sends really overlap."""
    with serving.flight:
        serving.in_flight += 1
        serving.flight.notify_all()
        if max_in_flight is None:
            return True
        if serving.in_flight > max_in_flight:
            serving.throttled += 1
            return False
        serving.flight.wait_for(lambda: serving.in_flight > max_in_flight, timeout=OVERLAP_SECONDS)
        return True


def _leave(serving: StandIn) -> None:
    with serving.flight:
        serving.in_flight -= 1
