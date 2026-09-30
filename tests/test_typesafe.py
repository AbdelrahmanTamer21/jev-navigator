from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from jev_navigator.directives.find_code import Outcome, SearchBudget, find_code
from jev_navigator.directives.places import range_place
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.journal import JsonlJournal
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.judgments.thresholds import Thresholds

STATE = {"slice": {"file": "counter.py", "code": "def add_one(x):\n    return x + 1"}}
QUESTIONS = {"adds_one": {"type": "noul", "instructions": "Does `slice.code` add one?"}}


def _jev_server(exchanges: list[tuple[bytes, bytes]]) -> ThreadingHTTPServer:
    """A local Jev endpoint recording the exact bytes of every request it answers, and its answer.

    Those pairs are ground truth for what crossed the wire, so a journal or store entry is checked
    against them rather than against another copy of what the library thinks it sent.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            sent = self.rfile.read(int(self.headers["content-length"]))
            answers = {id_: {"type": "noul", "noul": 0.9} for id_ in json.loads(sent)["questions"]}
            served = json.dumps(
                {
                    "model": "jev-1.13.0",
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                    "answers": answers,
                }
            ).encode()
            exchanges.append((sent, served))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(served)))
            self.end_headers()
            self.wfile.write(served)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, name="jev-local-server", daemon=True).start()
    return server


def _stop(server: ThreadingHTTPServer) -> None:
    server.shutdown()
    server.server_close()


def test_cancel_aborts_an_active_official_sdk_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production adapter owns cancellation through the SDK's async HTTP boundary."""
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    entered = threading.Event()
    release = threading.Event()
    requests = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            nonlocal requests
            self.rfile.read(int(self.headers["content-length"]))
            requests += 1
            entered.set()
            release.wait()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    finished = threading.Event()
    failure: list[BaseException] = []

    def ask() -> None:
        try:
            client.ask(
                {"code": "return wanted"},
                {"match": {"type": "noul", "instructions": "Does code return wanted?"}},
            )
        except BaseException as error:
            failure.append(error)
        finally:
            finished.set()

    request_thread = threading.Thread(target=ask)
    request_thread.start()
    try:
        assert entered.wait(2), "the SDK request did not reach the local HTTP server"

        client.cancel()

        assert finished.wait(2), "cancellation left the SDK request blocked"
        assert isinstance(failure[0], concurrent.futures.CancelledError)
        with pytest.raises(concurrent.futures.CancelledError):
            client.ask(
                {"code": "a late beam request"},
                {"match": {"type": "noul", "instructions": "Does code match?"}},
            )
        assert requests == 1
    finally:
        release.set()
        request_thread.join()
        client.close()
        server.shutdown()
        server.server_close()
        server_thread.join()


def test_sigint_returns_the_active_http_place_as_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installed sync search and official SDK share one prompt cancellation boundary."""
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    all_entered = threading.Event()
    release = threading.Event()
    entered = 0
    entered_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            nonlocal entered
            self.rfile.read(int(self.headers["content-length"]))
            with entered_lock:
                entered += 1
                if entered == 2:
                    all_entered.set()
            release.wait()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    (tmp_path / "policy.py").write_text(
        "def first(item):\n    return item\n\ndef second(item):\n    return item\n"
    )
    index = CodeIndex(tmp_path, ["policy.py"])
    places = [
        range_place(index, "policy.py", 1, 2, "candidate"),
        range_place(index, "policy.py", 4, 5, "candidate"),
    ]

    def interrupt_when_sent() -> None:
        all_entered.wait()
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = threading.Thread(target=interrupt_when_sent)
    interrupter.start()
    try:
        result = find_code(
            index,
            Judge(client),
            "the policy",
            [],
            budget=SearchBudget(beam_width=2),
            moves={},
            initial_candidates=[(place, 1.0) for place in places],
        )

        assert result.outcome == Outcome.CANCELLED
        assert {entry.place_key for entry in result.not_inspected} == {place.key for place in places}
        assert {entry.reason for entry in result.not_inspected} == {"cancelled"}
    finally:
        interrupter.join()
        release.set()
        client.close()
        server.shutdown()
        server.server_close()
        server_thread.join()


def test_the_production_transport_journals_the_exact_bytes_it_sent_and_received(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`exact` in the journal means "these are the wire bytes", so prove it over a real socket.

    The journal's `exact` flag and the answer store's `sent_exact` flag describe different things: the
    first is the response as received, the second is a kept copy of the request. This pins the first.
    """
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    exchanges: list[tuple[bytes, bytes]] = []
    server = _jev_server(exchanges)
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    client = TypeSafeJevClient()
    journal_path = tmp_path / "journal.jsonl"
    judge = Judge(client, journal=JsonlJournal(journal_path, keep_request_text=True))

    try:
        answer = judge.ask(STATE, QUESTIONS, thresholds=Thresholds())
    finally:
        client.close()
        _stop(server)

    # Assert
    assert answer.noul("adds_one").probability == 0.9
    sent, served = exchanges[0]
    journaled = json.loads(journal_path.read_text().splitlines()[1])
    assert (journaled["status"], journaled["content_type"], journaled["exact"]) == (
        200,
        "application/json",
        True,
    )
    assert base64.b64decode(journaled["body_base64"]) == served
    captured = base64.b64decode(journaled["sent_body_base64"])
    assert captured == sent
    assert b'"model"' in captured  # the wire body names the model; the library's handover does not


def test_the_answer_store_keeps_no_request_bytes_by_default_even_after_a_real_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `sent_exact: false` answer record is the privacy default, not evidence of a lost capture."""
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    exchanges: list[tuple[bytes, bytes]] = []
    server = _jev_server(exchanges)
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    silent_journal = JsonlJournal(tmp_path / "silent-journal.jsonl", keep_request_text=True)
    keeping_journal = JsonlJournal(tmp_path / "keeping-journal.jsonl", keep_request_text=True)
    silent = JsonlAnswerStore(tmp_path / "silent-answers.jsonl")
    keeping = JsonlAnswerStore(tmp_path / "keeping-answers.jsonl", keep_requests=True)
    client = TypeSafeJevClient()

    try:
        Judge(client, store=silent, journal=silent_journal).ask(STATE, QUESTIONS, thresholds=Thresholds())
        Judge(client, store=keeping, journal=keeping_journal).ask(STATE, QUESTIONS, thresholds=Thresholds())
    finally:
        client.close()
        _stop(server)

    # Assert: the journal kept the wire bytes of the request and the answer of the response either
    # way, so the answer store's own flag below cannot be read as "the transport captured nothing".
    for journal in (silent_journal, keeping_journal):
        line = json.loads(journal.path.read_text().splitlines()[1])
        assert (line["exact"], line["status"]) == (True, 200)
        assert base64.b64decode(line["sent_body_base64"]) == exchanges[0][0]

    dropped = silent.records()[0]
    assert (dropped.sent_exact, dropped.sent_body_base64, dropped.request) == (False, None, None)
    with pytest.raises(ValueError, match="keep_requests"):
        dropped.sent_request()

    kept = keeping.records()[0]
    assert kept.sent_exact is True
    assert base64.b64decode(kept.sent_body_base64) == exchanges[1][0]
    assert kept.sent_request() == (STATE, QUESTIONS)


def test_a_jev_route_asks_the_sdk_with_the_routes_own_timeout_and_retries() -> None:
    pytest.importorskip("typesafe_sdk")
    import httpx2
    from typesafe_sdk import TypeSafeRateLimitError

    from jev_navigator.adapters.routes import routes_from_env

    timeouts: list[dict] = []

    def busy(request: httpx2.Request) -> httpx2.Response:
        timeouts.append(request.extensions["timeout"])
        return httpx2.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "busy"}})

    (route,) = routes_from_env(
        {
            "SYSTEM_ONE_ROUTES": "jev",
            "TYPESAFE_API_KEY": "local-test-key",
            "SYSTEM_ONE_JEV_TIMEOUT": "2.5",
            "SYSTEM_ONE_JEV_RETRIES": "0",
        },
        transport=httpx2.MockTransport(busy),
    )

    try:
        with pytest.raises(TypeSafeRateLimitError):
            route.client.ask(STATE, QUESTIONS)
    finally:
        route.client.close()
    assert [timeout["read"] for timeout in timeouts] == [2.5]
