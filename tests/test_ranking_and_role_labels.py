"""Ranking and selected-only labels over a git repository of real JVN source."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from git_repos import commit_all, write_files

from jev_navigator.directives.find_all import find_all, find_all_async
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import items_to_judge, list_units, read_ranges
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.questions import content_hash, request_body
from jev_navigator.judgments.store import JsonlAnswerStore
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient

FIXTURE = Path(__file__).parent / "fixtures/j1_admitted_contract.json"
# Independently frozen from the original admitted six-role contract.
ROLE_WORDING_HASH = "9acec9bd7c4285787bb10075477541fc26cc737a789f9df4d5431c9322d80fd5"
PROBABILITIES = {
    "decide": 0.13,
    "guard": 0.57,
    "value": 0.32,
    "effect": 0.19,
    "delegates": 0.91,
    "satisfied": 0.44,
}


@pytest.fixture
def source_repository(tmp_path):
    source = Path(__file__).parents[1] / "src/jev_navigator"
    files = ("judgments/questions.py", "judgments/thresholds.py", "judgments/item_keys.py", "cache_root.py")
    write_files(tmp_path, {name: (source / name).read_text() for name in files})
    commit_all(tmp_path)
    return CodeIndex.from_git(tmp_path)


def selected_pieces(index, count=20):
    from jev_navigator.judgments.role_labels import LabelPiece

    units = list_units(index, index.files, box_chars=76800).units
    pieces = [
        LabelPiece(place, read_ranges(index, place.file, place.ranges))
        for unit in units
        for place in items_to_judge(unit)
    ]
    # Caller selection deliberately differs from the index's file-and-line order.
    selected = sorted(pieces, key=lambda piece: len(piece.code))[:count][::-1]
    assert len(selected) == count and len(pieces) > count
    return selected


def role_probability(question_id, question, state):
    role = question_id.split("_", 1)[0]
    return PROBABILITIES[role]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_ranking_wire_matches_the_admitted_j1_contract(source_repository, asynchronous):
    admitted = json.loads(FIXTURE.read_text())
    provider = ScriptedJevClient(default_noul=0.37)
    client = AsyncScriptedJevClient(provider) if asynchronous else provider
    search = find_all_async if asynchronous else find_all
    result = search(
        source_repository,
        Judge(client, max_calls=1, masker=None, scanner=None),
        admitted["state_targets"],
        files=source_repository.files,
        batches_per_wave=1,
    )
    if asynchronous:
        result = asyncio.run(result)
    assert result.failure is None and result.calls == 1
    [sent] = provider.requests
    state, questions = sent
    assert list(state) == ["targets", "items"]
    assert state["targets"] == admitted["state_targets"]
    assert len(state["items"]) == 16
    assert all(list(item) == ["file", "code"] for item in state["items"])
    assert request_body(state, questions) == request_body(state, admitted["questions"])
    assert len(result.judged["p0"]) == 16
    assert all(answer.probability == 0.37 and answer.source() is not None for answer in result.judged["p0"])


def test_resumed_ranking_keeps_earlier_units_and_the_better_observation(source_repository):
    targets = {"p0": "The locally implemented condition."}
    files = list(source_repository.files)
    earlier = find_all(
        source_repository, Judge(ScriptedJevClient(default_noul=0.8)), targets, files=files[:1], hops=()
    )
    later = find_all(
        source_repository, Judge(ScriptedJevClient(default_noul=0.2)), targets, files=files[:2], hops=()
    )
    combined = later.ranked_with("p0", earlier)
    earlier_ids = {score.unit.id for score in earlier.scores("p0")}
    later_ids = {score.unit.id for score in later.scores("p0")}
    assert earlier_ids < later_ids
    assert {score.unit.id for score in combined} == earlier_ids | later_ids
    assert len(combined) == len(later_ids)
    assert [score.probability for score in combined] == sorted(
        (0.8 if score.unit.id in earlier_ids else 0.2 for score in combined), reverse=True
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_labels_exact_selected_pieces_and_points_store_raw_answers_and_replay(
    source_repository, tmp_path, asynchronous
):
    from jev_navigator.judgments.profiles import ROLE_QUESTIONS
    from jev_navigator.judgments.role_labels import label_roles, label_roles_async

    pieces = selected_pieces(source_repository)
    targets = {"p0": "The local condition.", "p1": "The value being passed."}
    store_path = tmp_path / "roles.jsonl"
    provider = ScriptedJevClient(nouls=role_probability)
    client = AsyncScriptedJevClient(provider) if asynchronous else provider
    store = JsonlAnswerStore(store_path, keep_requests=True)
    # Labelling enforces 16 even when the caller's ranking Judge uses another batch size.
    judge = Judge(client, store=store, masker=None, scanner=None, items_per_request=32)
    label = label_roles_async if asynchronous else label_roles

    def run(judge):
        result = label(judge, pieces, targets)
        return asyncio.run(result) if asynchronous else result

    result = run(judge)
    assert result.calls == judge.calls == 2 and not result.refusals
    assert judge.items_per_request == 32
    assert [len(state["items"]) for state, _ in provider.requests] == [16, 4]
    assert [item for state, _ in provider.requests for item in state["items"]] == [
        {"file": piece.place.file, "code": piece.code} for piece in pieces
    ]
    assert content_hash(ROLE_QUESTIONS) == ROLE_WORDING_HASH
    assert [piece.piece for piece in result.pieces] == pieces
    records = {record.request_sha256: record for record in store.records()}
    assert len(records) == 2
    for piece in result.pieces:
        assert piece.probabilities == dict.fromkeys(targets, PROBABILITIES)
        for target, answers in piece.answers.items():
            assert set(answers) == set(PROBABILITIES)
            for role, answer in answers.items():
                assert answer.place == piece.piece.place and answer.source() is not None
                record = records[answer.request_sha256]
                assert record.answers[answer.question_id]["noul"] == PROBABILITIES[role]
                assert answer.question_id.startswith(f"{role}_{target}@")
    for state, questions in provider.requests:
        assert state["targets"] == targets and set(state) == {"targets", "items"}
        assert len(questions) == 6 * len(targets) * len(state["items"])

    # Reopen the real persisted store; identical batch company reuses every raw answer.
    reopened = JsonlAnswerStore(store_path, keep_requests=True)
    cached = run(Judge(client, store=reopened, masker=None, scanner=None, served_model=provider.model))
    assert cached.calls == 0 and len(provider.requests) == 2
    assert [piece.probabilities for piece in cached.pieces] == [
        piece.probabilities for piece in result.pieces
    ]
    assert all(
        answer.from_store
        for piece in cached.pieces
        for roles in piece.answers.values()
        for answer in roles.values()
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("refusal", [False, True])
def test_label_oversize_splits_by_item_without_changing_selected_text(
    source_repository, asynchronous, refusal
):
    from jev_navigator.judgments.client import InputBudgetExceededError, InputLimits
    from jev_navigator.judgments.role_labels import label_roles, label_roles_async

    pieces = selected_pieces(source_repository, 16)
    attempts = []

    class SizeBoundProvider(ScriptedJevClient):
        def send(self, state, questions):
            attempts.append(state)
            if refusal and len(state["items"]) > 4:
                raise InputBudgetExceededError("fixture provider refuses more than four items")
            return super().send(state, questions)

    provider = SizeBoundProvider(nouls=role_probability)
    if not refusal:
        provider.input_limits = InputLimits(box_chars=1600)
    client = AsyncScriptedJevClient(provider) if asynchronous else provider
    if not refusal:
        client.input_limits = provider.input_limits
    judge = Judge(client, masker=None, scanner=None, max_concurrency=1)
    label = label_roles_async if asynchronous else label_roles
    result = label(judge, pieces, {"p0": "The described behavior."})
    if asynchronous:
        result = asyncio.run(result)
    assert not result.refusals
    assert result.calls == len(attempts) > 1
    assert [label.piece for label in result.pieces] == pieces
    assert all(label.probabilities == {"p0": PROBABILITIES} for label in result.pieces)
    assert [item for state, _ in provider.requests for item in state["items"]] == [
        {"file": piece.place.file, "code": piece.code} for piece in pieces
    ]
    if refusal:
        assert len(attempts[0]["items"]) == 16
        assert all(len(state["items"]) <= 4 for state, _ in provider.requests)
    else:
        assert all(len(state["items"]) < 16 for state, _ in provider.requests)
        assert all(
            not provider.input_limits.exceeded_by(state, questions) for state, questions in provider.requests
        )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unlabelled_oversize_piece_stays_unknown_and_empty_selection_sends_nothing(
    source_repository, asynchronous
):
    from jev_navigator.judgments.client import InputLimits
    from jev_navigator.judgments.role_labels import LabelPiece, label_roles, label_roles_async

    small = selected_pieces(source_repository, 1)[0]
    units = list_units(source_repository, source_repository.files, box_chars=76800).units
    candidates = [
        LabelPiece(place, read_ranges(source_repository, place.file, place.ranges))
        for unit in units
        for place in items_to_judge(unit)
    ]
    large = max(candidates, key=lambda piece: len(piece.code))
    pieces = [small, large]
    provider = ScriptedJevClient(nouls=role_probability)
    provider.input_limits = InputLimits(box_chars=850)
    client = AsyncScriptedJevClient(provider) if asynchronous else provider
    client.input_limits = provider.input_limits
    label_roles_step = label_roles_async if asynchronous else label_roles
    result = label_roles_step(Judge(client, masker=None, scanner=None), pieces, {"p0": "The behavior."})
    if asynchronous:
        result = asyncio.run(result)
    assert result.calls == 1
    assert result.refusals
    assert [item for state, _ in provider.requests for item in state["items"]] == [
        {"file": small.place.file, "code": small.code}
    ]
    for refusal in result.refusals:
        label = next(label for label in result.pieces if label.piece.place == refusal.place)
        assert label.probabilities == {"p0": {}}
    assert [label.piece for label in result.pieces] == pieces
    previous_calls = len(provider.requests)
    empty = label_roles(Judge(provider), (), {})
    assert empty.pieces == empty.refusals == () and empty.calls == 0
    assert len(provider.requests) == previous_calls
