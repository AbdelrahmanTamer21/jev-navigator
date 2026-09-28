from __future__ import annotations

import json
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.cli import (
    SCHEMA_VERSION,
    _load_typesafe_api_key,
    _scope_warning,
    create_evidence_pack,
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
    )

    target = next(
        item for item in manifest["search"]["not_inspected"] if "target" in item["signature"]
    )
    assert target["priority"] == 0.0


def repository_commit(repository: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_existing_environment_key_wins_over_the_user_dotenv_file(tmp_path: Path) -> None:
    path = tmp_path / "env"
    path.write_text("TYPESAFE_API_KEY=file-value\n")
    environment = {"TYPESAFE_API_KEY": "process-value"}

    _load_typesafe_api_key(environment, path)

    assert environment["TYPESAFE_API_KEY"] == "process-value"


def test_user_dotenv_key_is_loaded_without_shell_evaluation(tmp_path: Path) -> None:
    pytest.importorskip("dotenv")
    path = tmp_path / "env"
    path.write_text("TYPESAFE_API_KEY='file-value'\nUNRELATED=$(touch should-not-run)\n")
    environment: dict[str, str] = {}

    _load_typesafe_api_key(environment, path)

    assert environment == {"TYPESAFE_API_KEY": "file-value"}
    assert not (tmp_path / "should-not-run").exists()


def test_large_scope_warning_starts_above_twenty_thousand_files() -> None:
    assert _scope_warning(20_000) is None
    assert "20,001" in _scope_warning(20_001)
