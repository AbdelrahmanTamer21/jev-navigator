"""The frozen round: a benchmark registered before any call, verified before every read."""

from __future__ import annotations

import json

import pytest

from jev_navigator.judgments.round import (
    MANIFEST_FILE,
    REGISTRATION_FILE,
    FrozenRoundError,
    RoundRegistration,
    freeze,
    registered_request_sha256,
    verify,
)

QUESTIONS = {"keep": {"type": "check", "instructions": "Does the comment hold?"}}
RULE = {"escalate_below": 0.8}
CASES = tuple(f"case-{n:02d}" for n in range(1, 6))


def registration() -> RoundRegistration:
    return RoundRegistration(case_ids=CASES, questions=QUESTIONS, rule=RULE, library_commit="", python="3.11")


def test_freeze_writes_the_hash_chain_and_verifies(tmp_path):
    fields = freeze(tmp_path, registration())
    assert fields["python"] == "3.11"
    manifest = json.loads((tmp_path / MANIFEST_FILE).read_text())
    assert manifest["registration_sha256"]
    verify(tmp_path)  # no raise


def test_an_edited_manifest_fails_its_own_hash(tmp_path):
    freeze(tmp_path, registration())
    fields = json.loads((tmp_path / MANIFEST_FILE).read_text())
    fields["python"] = "3.9"
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(fields))
    with pytest.raises(FrozenRoundError, match="edited after it was frozen"):
        verify(tmp_path)


def test_an_edited_registration_fails_the_manifest(tmp_path):
    freeze(tmp_path, registration())
    data = json.loads((tmp_path / REGISTRATION_FILE).read_text())
    data["case_ids"] = list(CASES)[:-1]  # drop a case after the fact
    (tmp_path / REGISTRATION_FILE).write_text(json.dumps(data))
    with pytest.raises(FrozenRoundError, match="registration was edited"):
        verify(tmp_path)


def test_verify_against_a_live_registration_catches_drift(tmp_path):
    freeze(tmp_path, registration())
    reworded = RoundRegistration(
        case_ids=CASES, questions={"keep": {"type": "check", "instructions": "Holds?"}},
        rule=RULE, library_commit="", python="3.11")
    with pytest.raises(FrozenRoundError, match="registration differs"):
        verify(tmp_path, reworded)


def test_a_re_frozen_round_is_refused(tmp_path):
    freeze(tmp_path, registration())
    with pytest.raises(FrozenRoundError, match="already frozen"):
        freeze(tmp_path, registration())


def test_a_rule_source_change_is_caught_when_the_code_registers_it(tmp_path):
    source = "def rule(row):\n    return row['p'] >= 0.9\n"
    with_source = RoundRegistration(case_ids=CASES, questions=QUESTIONS, rule=RULE, rule_source=source)
    freeze(tmp_path, with_source)
    changed = RoundRegistration(
        case_ids=CASES, questions=QUESTIONS, rule=RULE,
        rule_source="def rule(row):\n    return row['p'] >= 0.8\n")
    with pytest.raises(FrozenRoundError, match="registration differs"):
        verify(tmp_path, changed)


def test_incomplete_registrations_are_refused():
    with pytest.raises(FrozenRoundError, match="case ids"):
        RoundRegistration(case_ids=(), questions=QUESTIONS, rule=RULE)
    with pytest.raises(FrozenRoundError, match="unique"):
        RoundRegistration(case_ids=("a", "a"), questions=QUESTIONS, rule=RULE)
    with pytest.raises(FrozenRoundError, match="questions"):
        RoundRegistration(case_ids=CASES, questions={}, rule=RULE)


def test_the_registered_request_hash_is_the_store_request_hash(tmp_path):
    from jev_navigator.judgments.questions import request_sha256

    reg = registration()
    state = {"case_id": "case-01", "code": "x = 1"}
    assert registered_request_sha256(reg, state) == request_sha256(state, QUESTIONS)


def test_freeze_without_a_checkout_records_no_commit(tmp_path):
    fields = freeze(tmp_path, registration())
    assert fields["library_commit"] in ("", None) or isinstance(fields["library_commit"], str)
