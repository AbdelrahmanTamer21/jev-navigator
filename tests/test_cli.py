from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from git_repos import commit_files

from jev_navigator.cli import (
    SCHEMA_VERSION,
    _load_typesafe_environment,
    _scope_warning,
    create_evidence_pack,
    main,
)
from jev_navigator.directives.find_code import SearchBudget
from jev_navigator.testing import ScriptedJevClient


def test_evidence_pack_runs_the_real_index_and_search_boundary(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/entry.py": "from .policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "app/policy.py": "def admit(item):\n    return len(item) <= 3\n",
        },
    )
    output = tmp_path / "evidence"
    client = ScriptedJevClient(
        nouls=lambda question_id, question, state: (
            0.96 if "len(item) <= 3" in state["slice"]["code"] else 0.04
        ),
        choices={"open_first": {"0": 1.0}},
    )

    manifest = create_evidence_pack(
        repository,
        ("app/",),
        "the check that limits the number of items",
        ("app/entry.py:4",),
        output,
        SearchBudget(max_depth=2, max_steps=3, max_calls=3, beam_width=1),
        client,
        fact_cache_dir=tmp_path / "fact-cache",
    )

    written = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == written["schema_version"] == SCHEMA_VERSION
    assert written["source"]["revision"] == repository_commit(repository)
    assert written["navigator"]["package_version"] == "0.1.0"
    assert len(written["navigator"]["source_tree_sha256"]) == 64
    assert written["provider"] == {
        "input_tokens": 200,
        "requested_model": "jev-scripted",
        "served_model": "jev-scripted",
    }
    assert written["search"]["outcome"] == "found"
    assert written["search"]["found"][0]["source"]["file"] == "app/policy.py"
    assert written["search"]["found"][0]["probability"] == 0.96
    assert written["search"]["history"][-1]["operation"] == "stop"
    assert "Not inspected" in (output / "report.md").read_text()
    assert (output / "journal.jsonl").read_text()


def test_evidence_pack_chooses_a_real_entry_when_no_start_is_supplied(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "app/entry.py": "from .policy import admit\n\ndef handle(item):\n    return admit(item)\n",
            "app/policy.py": "def admit(item):\n    return len(item) <= 3\n",
        },
    )
    client = ScriptedJevClient(
        nouls=lambda question_id, question, state: (
            0.96 if "len(item) <= 3" in state["slice"]["code"] else 0.04
        ),
        choices={"automatic_entry_path": {"1": 1.0}},
    )

    manifest = create_evidence_pack(
        repository,
        (),
        "the check that limits the number of items",
        (),
        tmp_path / "evidence-auto",
        SearchBudget(max_depth=2, max_steps=3, max_calls=3, beam_width=1),
        client,
        fact_cache_dir=tmp_path / "fact-cache",
    )

    assert manifest["search"]["outcome"] == "found"
    assert manifest["search"]["calls"] == 2
    assert manifest["search"]["entry_calls"] == 1
    assert manifest["entry_selection"]["selected_file"] == "app/policy.py"
    assert manifest["search"]["found"][0]["source"]["file"] == "app/policy.py"


def test_zero_choice_probability_remains_zero_in_the_uninspected_frontier(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    commit_files(
        repository,
        {
            "policy.py": (
                "def first():\n    return 'wrong'\n\n"
                "def target():\n    return 'wanted'\n\n"
                "def third():\n    return 'also wrong'\n"
            )
        },
    )
    client = ScriptedJevClient(
        nouls=lambda question_id, question, state: 0.04,
        choices={"automatic_entry_span": {"0": 0.8, "1": 0.0, "2": 0.2}},
    )

    manifest = create_evidence_pack(
        repository,
        (),
        "wanted",
        (),
        tmp_path / "zero-probability",
        SearchBudget(max_steps=1, max_calls=2, beam_width=1),
        client,
        fact_cache_dir=tmp_path / "fact-cache",
    )

    target = next(item for item in manifest["search"]["not_inspected"] if "target" in item["signature"])
    assert target["priority"] == 0.0


def test_cancelled_navigation_writes_a_resumable_evidence_pack(tmp_path: Path) -> None:
    # Arrange
    repository = tmp_path / "repository"
    commit_files(repository, {"policy.py": "def policy(item):\n    return item\n"})

    class InterruptingClient:
        model = "interrupting"

        def ask(self, state, questions):
            del state, questions
            raise KeyboardInterrupt

    output = tmp_path / "cancelled-evidence"

    # Act
    manifest = create_evidence_pack(
        repository,
        (),
        "the policy",
        ("policy.py:1",),
        output,
        SearchBudget(beam_width=1),
        InterruptingClient(),
        fact_cache_dir=tmp_path / "fact-cache",
    )

    # Assert
    assert manifest["search"]["outcome"] == "cancelled"
    assert manifest["search"]["steps"] == 0
    assert manifest["search"]["not_inspected"][0]["reason"] == "cancelled"
    assert "Outcome: **cancelled**" in (output / "report.md").read_text()
    records = [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]
    assert any(record["kind"] == "history_step" for record in records)
    assert records[-1]["kind"] == "terminal"
    assert records[-1]["outcome"] == "cancelled"


def test_main_maps_a_cancelled_pack_to_the_shell_interrupt_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    from jev_navigator import cli

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(
        cli,
        "create_evidence_pack",
        lambda *args, **kwargs: {"search": {"outcome": "cancelled", "calls": 1}},
    )

    # Act
    status = main(["find", "the policy", "--repo", str(tmp_path), "--out", str(tmp_path / "out")])

    # Assert
    assert status == 130


def repository_commit(repository: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_each_existing_typesafe_environment_value_wins_independently(tmp_path: Path) -> None:
    pytest.importorskip("dotenv")
    path = tmp_path / "env"
    path.write_text("TYPESAFE_API_KEY=file-key\nTYPESAFE_BASE_URL=http://file.example/gateway\n")
    environment = {"TYPESAFE_API_KEY": "process-key"}

    _load_typesafe_environment(environment, path)

    assert environment == {
        "TYPESAFE_API_KEY": "process-key",
        "TYPESAFE_BASE_URL": "http://file.example/gateway",
    }


def test_user_dotenv_key_is_loaded_without_shell_evaluation(tmp_path: Path) -> None:
    pytest.importorskip("dotenv")
    path = tmp_path / "env"
    path.write_text(
        "TYPESAFE_API_KEY='file-value'\n"
        "TYPESAFE_BASE_URL='http://127.0.0.1:4777/jvn'\n"
        "UNRELATED=$(touch should-not-run)\n"
    )
    environment: dict[str, str] = {}

    _load_typesafe_environment(environment, path)

    assert environment == {
        "TYPESAFE_API_KEY": "file-value",
        "TYPESAFE_BASE_URL": "http://127.0.0.1:4777/jvn",
    }
    assert not (tmp_path / "should-not-run").exists()


def test_dotenv_base_url_reaches_the_real_sdk_system_one_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("typesafe_sdk")
    pytest.importorskip("dotenv")
    from jev_navigator.adapters.typesafe import TypeSafeJevClient

    received: list[tuple[str, bytes]] = []
    response = (
        b'{"model":"jev-1.13.0","usage":{"input_tokens":1,"output_tokens":1},'
        b'"answers":{"match":{"type":"noul","noul":0.9}}}'
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            body = self.rfile.read(int(self.headers["content-length"]))
            received.append((self.path, body))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
        path = tmp_path / "env"
        path.write_text(
            "TYPESAFE_API_KEY=local-viewer-key\n"
            f"TYPESAFE_BASE_URL=http://127.0.0.1:{server.server_port}/jvn\n"
        )
        _load_typesafe_environment(os.environ, path)

        answer = TypeSafeJevClient().ask(
            {"code": "return wanted"},
            {"match": {"type": "noul", "instructions": "Does code return wanted?"}},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert received and received[0][0] == "/jvn/v1/systemone"
    assert b'"code":"return wanted"' in received[0][1]
    assert answer.noul("match").probability == 0.9


def test_large_scope_warning_starts_above_twenty_thousand_files() -> None:
    assert _scope_warning(20_000) is None
    assert "20,001" in _scope_warning(20_001)
