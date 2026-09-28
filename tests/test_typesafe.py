from __future__ import annotations

import concurrent.futures
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from jev_navigator.directives.find_code import Outcome, SearchBudget, find_code
from jev_navigator.directives.places import range_place
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge


def test_cancel_aborts_an_active_official_sdk_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production adapter owns cancellation through the SDK's async HTTP boundary."""
    pytest.importorskip("typesafe_sdk")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    entered = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self.rfile.read(int(self.headers["content-length"]))
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

    entered = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self.rfile.read(int(self.headers["content-length"]))
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
    (tmp_path / "policy.py").write_text("def policy(item):\n    return item\n")
    index = CodeIndex(tmp_path, ["policy.py"])
    place = range_place(index, "policy.py", 1, 2, "candidate")

    def interrupt_when_sent() -> None:
        entered.wait()
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = threading.Thread(target=interrupt_when_sent)
    interrupter.start()
    try:
        result = find_code(
            index,
            Judge(client),
            "the policy",
            [],
            budget=SearchBudget(beam_width=1),
            moves={},
            initial_candidates=[(place, 1.0)],
        )

        assert result.outcome == Outcome.CANCELLED
        assert [(entry.place_key, entry.reason) for entry in result.not_inspected] == [
            (place.key, "cancelled")
        ]
    finally:
        interrupter.join()
        release.set()
        client.close()
        server.shutdown()
        server.server_close()
        server_thread.join()
