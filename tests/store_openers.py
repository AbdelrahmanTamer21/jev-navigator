"""A worker for tests that open one shared answer store from many processes at once."""

from __future__ import annotations

from pathlib import Path

from jev_navigator.judgments.store import SqliteAnswerStore


def open_and_write_in_each_trial(paths: list[str], barrier, failures) -> None:
    """Open the store at each path together with the other workers, then write to it; every error is
    reported by name and message."""
    for path in paths:
        barrier.wait(timeout=60)
        try:
            SqliteAnswerStore(Path(path)).put_refusal("request", "route", 1)
        except Exception as error:  # noqa: BLE001 - the test reports every failure an opener meets
            failures.append(f"{type(error).__name__}: {error}")
