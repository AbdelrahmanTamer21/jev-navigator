"""Durable provider journal with concise live terminal progress."""

from __future__ import annotations

import json
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from .judgments.journal import JournalRequest, JsonlJournal, RawResponse

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class TerminalProgress:
    def __init__(self, journal_path: Path | None, *, verbose: bool = False) -> None:
        self.journal_path = journal_path
        self.verbose = verbose
        self.started = monotonic()
        self.phase_name = "starting"
        self.requests = 0
        self.responses = 0
        self.failures = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._tty = sys.stderr.isatty()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._event(f"journal {self.journal_path}" if self.journal_path else "local analysis")
        if self._tty:
            self._thread = threading.Thread(target=self._spin, name="jvn-progress", daemon=True)
            self._thread.start()

    def phase(self, name: str) -> None:
        with self._lock:
            self.phase_name = name
        self._event(f"phase {name}")

    def scan(self, name: str, event: str, file_count: int) -> None:
        phase = f"scanning {name} ({file_count:,} code files)"
        with self._lock:
            self.phase_name = phase if event == "started" else f"{name} scan {event}"
        self._event(f"{phase} {event}; {self.elapsed():.1f}s elapsed")

    def request(self, request_id: str, request: JournalRequest) -> None:
        with self._lock:
            self.requests += 1
            number = self.requests
        state = request.state
        source = state.get("slice", {}) if isinstance(state, dict) else {}
        file = source.get("file", "") if isinstance(source, dict) else ""
        lines = source.get("lines", "") if isinstance(source, dict) else ""
        if isinstance(lines, (list, tuple)) and len(lines) == 2:
            lines = f"{lines[0]}-{lines[1]}"
        location = f" {file}:{lines}" if file else ""
        question_ids = ", ".join(request.questions)
        self._event(
            f"request {number} started{location}; purpose {question_ids}; {self.elapsed():.1f}s elapsed"
        )
        if self.verbose:
            self._event(
                "masked request "
                + json.dumps({"state": request.state, "questions": request.questions}, default=str)
            )

    def response(self, request_id: str, response: RawResponse) -> None:
        usage = _usage(response)
        with self._lock:
            self.responses += 1
            self.input_tokens += usage[0]
            self.output_tokens += usage[1]
            number = self.responses
        model = _model(response)
        self._event(
            f"response {number} received"
            f"{f' from {model}' if model else ''}; usage +{usage[0]} in/+{usage[1]} out; "
            f"total {self.input_tokens} in/{self.output_tokens} out; {self.elapsed():.1f}s elapsed"
        )

    def failure(self, request_id: str, error: str) -> None:
        with self._lock:
            self.failures += 1
        self._event(f"request failed: {error}")

    def close(self, outcome: str) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._clear_spinner()
        self._event(
            f"{outcome}; {self.requests} requests, {self.responses} responses, "
            f"{self.input_tokens} input tokens, {self.output_tokens} output tokens, "
            f"{self.elapsed():.1f}s elapsed"
        )

    def elapsed(self) -> float:
        return monotonic() - self.started

    def _spin(self) -> None:
        position = 0
        while not self._stop.wait(0.1):
            with self._lock:
                line = (
                    f"{_SPINNER[position % len(_SPINNER)]} {self.phase_name} · {self.elapsed():.1f}s · "
                    f"{self.requests} requests/{self.responses} responses · "
                    f"{self.input_tokens} in/{self.output_tokens} out"
                )
            sys.stderr.write(f"\r{line[:160]:<160}")
            sys.stderr.flush()
            position += 1

    def _event(self, message: str) -> None:
        with self._lock:
            self._clear_spinner()
            stamp = datetime.now(UTC).isoformat(timespec="seconds")
            print(f"jvn [{stamp}] {message}", file=sys.stderr, flush=True)

    def _clear_spinner(self) -> None:
        if self._tty:
            sys.stderr.write("\r" + " " * 160 + "\r")


class ProgressJournal(JsonlJournal):
    def __init__(self, path: Path, progress: TerminalProgress) -> None:
        super().__init__(path, keep_request_text=True)
        self.progress = progress

    def record_request(self, request: JournalRequest) -> str:
        request_id = super().record_request(request)
        self.progress.request(request_id, request)
        return request_id

    def record_response(self, request_id: str, response: RawResponse) -> None:
        super().record_response(request_id, response)
        self.progress.response(request_id, response)

    def record_failure(self, request_id: str, error: str, response: RawResponse | None = None) -> None:
        super().record_failure(request_id, error, response)
        self.progress.failure(request_id, error)

    def record_terminal(self, outcome: str) -> None:
        self._append({"kind": "terminal", "outcome": outcome})


def _usage(response: RawResponse) -> tuple[int, int]:
    try:
        raw = response.json()
    except Exception:  # noqa: BLE001 - progress never replaces the durable parser failure
        return 0, 0
    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    if not isinstance(usage, dict):
        return 0, 0
    return _integer(usage.get("input_tokens")), _integer(usage.get("output_tokens"))


def _model(response: RawResponse) -> str:
    try:
        raw = response.json()
    except Exception:  # noqa: BLE001 - progress never replaces the durable parser failure
        return ""
    model = raw.get("model", "") if isinstance(raw, dict) else ""
    return str(model) if model else ""


def _integer(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0
