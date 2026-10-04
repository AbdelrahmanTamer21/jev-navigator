"""A provider failure ends a CLI search with the same resumable state a cancel writes."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import pytest
from test_cli_run_logs import TARGET, limit_client, marked_repository

from jev_navigator import cli
from jev_navigator.cli import create_evidence_pack
from jev_navigator.directives.find_all import CONTAINS_IMPLEMENTATION
from jev_navigator.directives.find_code import SearchBudget
from jev_navigator.judgments.questions import request_sha256
from jev_navigator.testing import ScriptedJevClient

START = "app/entry.py:5"


class ProviderError(RuntimeError):
    """A failure the provider reports, such as a 503."""


class FailsOnRequest:
    """Answers like ``script``, except that request number ``failing`` raises ``error`` once. It
    records every request it receives, answered or not."""

    def __init__(self, script: ScriptedJevClient, failing: int, error: Exception) -> None:
        self.script = script
        self.failing = failing
        self.error = error
        self.received: list[tuple[Mapping, Mapping]] = []
        self.model = script.model

    def ask(self, state: Mapping, questions: Mapping):
        self.received.append((state, questions))
        if len(self.received) == self.failing:
            raise self.error
        return self.script.ask(state, questions)

    def close(self) -> None:
        pass


def provider_error() -> ProviderError:
    error = ProviderError("Jev answered 503")
    error.__cause__ = ConnectionResetError("connection reset by peer")
    return error


def hashes(requests: list[tuple[Mapping, Mapping]]) -> list[str]:
    return [request_sha256(state, questions) for state, questions in requests]


def closable(client: ScriptedJevClient) -> ScriptedJevClient:
    client.close = lambda: None
    return client


def use_clients(monkeypatch: pytest.MonkeyPatch, clients: Iterator) -> None:
    monkeypatch.setattr(cli, "_load_typesafe_environment", lambda environment: None)
    monkeypatch.setattr(cli, "TypeSafeJevClient", lambda: next(clients))


def find_command(repository: Path, output: Path, store: Path, *options: str) -> list[str]:
    command = ["find", TARGET, "--repo", str(repository), "--prefix", "app/", "--start", START]
    return [*command, "--max-calls", "5", "--out", str(output), "--answer-store", str(store), *options]


def uninterrupted_requests(repository: Path, tmp_path: Path) -> tuple[dict, list[str]]:
    client = limit_client()
    manifest = create_evidence_pack(
        repository,
        ("app/",),
        TARGET,
        (START,),
        tmp_path / "whole",
        SearchBudget(max_calls=5),
        client,
        answer_store=tmp_path / "whole-answers.sqlite",
    )
    return manifest, hashes(client.requests)


def manifest_of(folder: Path) -> dict:
    return json.loads((folder / "manifest.json").read_text())


def journal_failures(folder: Path) -> list[dict]:
    records = [json.loads(line) for line in (folder / "journal.jsonl").read_text().splitlines()]
    return [record for record in records if record["kind"] == "failure"]


def test_a_failed_find_exits_1_with_the_error_and_its_resume_reaches_the_uninterrupted_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    whole, expected = uninterrupted_requests(repository, tmp_path)
    store = tmp_path / "answers.sqlite"
    failing = FailsOnRequest(limit_client(), failing=2, error=provider_error())
    resuming = closable(limit_client())
    use_clients(monkeypatch, iter([failing, resuming]))
    first, second = tmp_path / "first", tmp_path / "second"

    # Act
    failed_status = cli.main(find_command(repository, first, store))
    failed_stderr = capsys.readouterr().err
    resumed_status = cli.main(find_command(repository, second, store, "--resume", str(first)))

    # Assert
    failed = manifest_of(first)
    assert failed_status == 1
    assert "Jev answered 503" in failed_stderr
    assert f"--resume {first.resolve()}" in failed_stderr
    assert failed["search"]["outcome"] == "failed"
    assert failed["search"]["failure"]["type"] == "ProviderError"
    assert failed["search"]["failure"]["message"] == "Jev answered 503"
    assert failed["search"]["failure"]["causes"] == [
        {"type": "ConnectionResetError", "message": "connection reset by peer"}
    ]
    assert failed["search"]["failure"]["request_id"] in [row["request_id"] for row in journal_failures(first)]
    assert failed["search"]["failure"]["route"] is None
    assert (first / "resume.json").is_file()
    resumed = manifest_of(second)
    assert resumed_status == 0
    assert resumed["search"]["outcome"] == whole["search"]["outcome"] == "found"
    assert resumed["search"]["found"] == whole["search"]["found"]
    sent = hashes(failing.received)
    assert sent[:1] + hashes(resuming.requests) == expected
    assert sent[1] == expected[1]


def test_a_new_find_after_a_failed_one_runs_as_usual(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    store = tmp_path / "answers.sqlite"
    use_clients(
        monkeypatch,
        iter([FailsOnRequest(limit_client(), failing=2, error=provider_error()), closable(limit_client())]),
    )
    cli.main(find_command(repository, tmp_path / "failed", store))

    # Act
    status = cli.main(find_command(repository, tmp_path / "fresh", store))

    # Assert
    assert status == 0
    assert manifest_of(tmp_path / "fresh")["search"]["outcome"] == "found"


def asks(question_id: str) -> Callable[[tuple[Mapping, Mapping]], bool]:
    return lambda request: any(asked.startswith(question_id) for asked in request[1])


def test_a_find_all_whose_seed_search_fails_never_starts_its_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    repository = marked_repository(tmp_path / "repository")
    failing = FailsOnRequest(limit_client(), failing=2, error=provider_error())
    use_clients(monkeypatch, iter([failing]))
    command = find_command(repository, tmp_path / "findall", tmp_path / "answers.sqlite")
    command[0] = "findall"

    # Act
    status = cli.main(command)

    # Assert
    assert status == 1
    assert manifest_of(tmp_path / "findall")["search"]["outcome"] == "failed"
    assert not any(map(asks(CONTAINS_IMPLEMENTATION.question_id), failing.received))
    assert (tmp_path / "findall" / "resume.json").is_file()
