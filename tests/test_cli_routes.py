"""`SYSTEM_ONE_ROUTES` through the real `jvn find`: each route is a local stand-in server, and the
run never reads the machine's own key or settings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

pytest.importorskip("typesafe_sdk")

DEAD_ENDPOINT = "http://127.0.0.1:9"
"""Where a request goes when no stand-in is named: a closed local port, never the hosted service."""


@dataclass
class StandIn:
    """A local System One server's address and the request bodies it received."""

    url: str
    received: list[dict] = field(default_factory=list)


@contextmanager
def stand_in(model: str, *, status: int = 200) -> Iterator[StandIn]:
    """A System One server that answers every question yes, or fails every request with ``status``."""
    handler = _handler(model, status)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    serving = StandIn(f"http://127.0.0.1:{server.server_port}", handler.received)
    thread = Thread(target=server.serve_forever)
    try:
        thread.start()
        yield serving
    finally:
        server.shutdown()
        server.server_close()
        if thread.is_alive():
            thread.join()


def _handler(model: str, status: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        received: list[dict] = []

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            self.received.append(payload)
            answered = _answers(model, payload) if status == 200 else {"detail": "unavailable"}
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


def run_find(tmp_path: Path, settings: Mapping[str, str]) -> subprocess.CompletedProcess:
    """`jvn find` on a two-function repository, with only ``settings`` as its System One setup and a
    fake key, so no request can reach a hosted provider."""
    repository = tmp_path / "repo"
    repository.mkdir(exist_ok=True)
    (repository / "admit.py").write_text(
        "def entry(items):\n    return admit(items)\n\ndef admit(items):\n    return len(items) <= 3\n"
    )
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith(("SYSTEM_ONE_", "TYPESAFE_"))
    }
    environment |= {"HOME": str(home), "TYPESAFE_API_KEY": "local-test-key"}
    environment |= {"TYPESAFE_BASE_URL": DEAD_ENDPOINT, **settings}
    command = [sys.executable, "-c", "from jev_navigator.cli import main; raise SystemExit(main())"]
    arguments = ["find", "the item-count check", "--repo", str(repository), "--start", "admit.py:1"]
    arguments += ["--max-calls", "4", "--out", str(tmp_path / "pack")]
    return subprocess.run(
        [*command, *arguments], env=environment, text=True, capture_output=True, timeout=120
    )


def journal_attempts(tmp_path: Path) -> list[dict]:
    lines = (tmp_path / "pack" / "journal.jsonl").read_text().splitlines()
    return [record for record in map(json.loads, lines) if record["kind"] == "http_attempt"]


def test_the_first_named_route_answers_the_first_request(tmp_path: Path) -> None:
    # Arrange
    with stand_in("drex-test") as drex, stand_in("jev-test") as jev:
        settings = {
            "SYSTEM_ONE_ROUTES": "drex,jev",
            "SYSTEM_ONE_DREX_ENDPOINT": drex.url,
            "SYSTEM_ONE_DREX_MODEL": "drex-test",
            "SYSTEM_ONE_JEV_ENDPOINT": jev.url,
            "SYSTEM_ONE_JEV_MODEL": "jev-test",
            "TYPESAFE_BASE_URL": jev.url,
        }

        # Act
        result = run_find(tmp_path, settings)

    # Assert
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert drex.received and not jev.received
    manifest = json.loads((tmp_path / "pack" / "manifest.json").read_text())
    assert manifest["provider"]["served_model"] == "drex-test"


def test_a_failing_route_falls_back_to_the_next_and_both_attempts_are_journaled(tmp_path: Path) -> None:
    # Arrange
    with stand_in("drex-test", status=503) as drex, stand_in("jev-test") as jev:
        settings = {
            "SYSTEM_ONE_ROUTES": "drex,jev",
            "SYSTEM_ONE_DREX_ENDPOINT": drex.url,
            "SYSTEM_ONE_DREX_MODEL": "drex-test",
            "SYSTEM_ONE_JEV_ENDPOINT": jev.url,
            "SYSTEM_ONE_JEV_MODEL": "jev-test",
        }

        # Act
        result = run_find(tmp_path, settings)

    # Assert
    assert result.returncode == 0, result.stderr
    assert drex.received and jev.received
    first_request = [(attempt["route"], attempt["status"]) for attempt in journal_attempts(tmp_path)[:4]]
    assert first_request == [("drex", 503)] * 3 + [("jev", 200)]


def test_without_routes_the_default_jev_client_answers(tmp_path: Path) -> None:
    # Arrange
    with stand_in("jev-test") as jev:
        # Act
        result = run_find(tmp_path, {"TYPESAFE_BASE_URL": jev.url})

    # Assert
    assert result.returncode == 0, result.stderr
    assert jev.received
    assert all("route" not in attempt for attempt in journal_attempts(tmp_path))


def test_an_incomplete_route_is_refused_by_name_before_any_request(tmp_path: Path) -> None:
    # Arrange
    with stand_in("jev-test") as jev:
        # Act
        result = run_find(tmp_path, {"SYSTEM_ONE_ROUTES": "decider,jev", "TYPESAFE_BASE_URL": jev.url})

    # Assert
    assert result.returncode == 1
    assert "route 'decider' is incomplete" in result.stderr
    assert not jev.received
