"""A local System One stand-in server for tests: it answers every question yes, fails with a status,
or refuses for input size the way Jev does, and never reaches a hosted provider."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

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


@contextmanager
def stand_in(
    model: str, *, status: int = 200, refuses_size: bool = False, held: bool = False
) -> Iterator[StandIn]:
    """A System One server that answers every question yes, fails every request with ``status``, or,
    when it ``refuses_size``, refuses every request as too large. A ``held`` server keeps each answer
    back until the test sets ``release``."""
    serving = StandIn("")
    if not held:
        serving.release.set()
    if refuses_size:
        status = 400
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(model, status, serving))
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


def _handler(model: str, status: int, serving: StandIn) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            serving.received.append(payload)
            serving.arrived.set()
            serving.release.wait()
            answered = _answers(model, payload) if status == 200 else _failure(status)
            body = json.dumps(answered).encode()
            self.send_response(status)
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
