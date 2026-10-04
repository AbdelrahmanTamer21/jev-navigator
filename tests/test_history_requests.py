"""Every Jev judgment a history step records names the answer it came from: the request's hash and the
question id, so a run pack joins each decision to its journal request and response without guessing
from journal order."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from test_cli_run_logs import TARGET, limit_client, marked_repository

from jev_navigator.cli import create_evidence_pack
from jev_navigator.directives.find_code import SearchBudget


def run_pack(tmp_path: Path) -> Path:
    output = tmp_path / "pack"
    create_evidence_pack(
        marked_repository(tmp_path / "repository"),
        ("app/",),
        TARGET,
        ("app/entry.py:5",),
        output,
        SearchBudget(max_calls=5, beam_width=1),
        limit_client(),
        fact_cache_dir=tmp_path / "fact-cache",
    )
    return output


def journal(output: Path) -> list[dict]:
    return [json.loads(line) for line in (output / "journal.jsonl").read_text().splitlines()]


def answered_probability(records: list[dict], source: dict) -> float:
    """The probability the journal's response holds for ``source``'s question, reached through the
    request row with ``source``'s hash, then that request's response row."""
    requests = [
        record
        for record in records
        if record["kind"] == "request" and record["request_sha256"] == source["request_sha256"]
    ]
    assert [source["question_id"] in request["question_ids"] for request in requests] == [True]
    response = next(
        record
        for record in records
        if record["kind"] == "response" and record["request_id"] == requests[0]["request_id"]
    )
    answer = json.loads(base64.b64decode(response["body_base64"]))["answers"][source["question_id"]]
    return answer["noul"]


def test_each_judgment_of_an_open_step_joins_to_the_answer_that_decided_it(tmp_path: Path) -> None:
    # Arrange
    output = run_pack(tmp_path)

    # Act
    records = journal(output)

    # Assert
    steps = [record["step"] for record in records if record["kind"] == "history_step"]
    opens = [step for step in steps if step["operation"] == "open"]
    assert opens
    for step in opens:
        contains = step["judgments"]["contains_target"]
        assert answered_probability(records, contains["answered_by"]) == contains["probability"]
        for offered in step["judgments"]["could_contain"]:
            assert answered_probability(records, offered["answered_by"]) == offered["probability"]


def test_a_place_a_choose_next_step_opens_names_the_answer_that_scored_it(tmp_path: Path) -> None:
    # Arrange
    output = run_pack(tmp_path)

    # Act
    records = journal(output)

    # Assert
    steps = [record["step"] for record in records if record["kind"] == "history_step"]
    scored = [
        chosen
        for step in steps
        if step["operation"] == "choose_next"
        for chosen in step["arguments"]["chosen"]
        if chosen["reason"] in ("queue_score", "open_first")
    ]
    assert scored
    for chosen in scored:
        assert answered_probability(records, chosen["scored_by"]) == chosen["priority"]
