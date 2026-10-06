"""Profile contracts at real repository search interfaces, with only inference scripted."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from git_repos import commit_all

from jev_navigator.directives.find_all import find_all, find_all_async, find_all_text, find_all_text_async
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.judgments.judge import Judge
from jev_navigator.judgments.profiles import LOCAL_ROLES, ROLES_V2, forwarding_only
from jev_navigator.judgments.questions import content_hash
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient

# Canonical questions hash, independently frozen from the approved candidate.
APPROVED_QUESTIONS = "9acec9bd7c4285787bb10075477541fc26cc737a789f9df4d5431c9322d80fd5"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("text", [False, True])
def test_profile_batches_real_units_and_retains_roles(tmp_path: Path, asynchronous: bool, text: bool):
    paths = []
    for number in range(33):
        path = f"unit{number:02d}.md" if text else f"unit{number:02d}.py"
        body = f"unit {number}" if text else f"def unit{number}():\n    return {number}\n"
        (tmp_path / path).write_text(body)
        paths.append(path)
    commit_all(tmp_path)
    index = CodeIndex.from_git(tmp_path)
    probabilities = {
        "unit00": {"decide": 0.60},
        "unit01": {"guard": 0.70, "delegates": 0.95},
        "unit02": {"value": 0.65},
        "unit03": {"effect": 0.75},
        "unit04": {"satisfied": 0.99},
        "unit05": {"delegates": 0.95},
    }

    def answer(question_id, question, state):
        role = question_id.split("_p0@", 1)[0]
        slot = int(re.search(r"items\[(\d+)\]", question["instructions"])[1])
        unit = Path(state["items"][slot]["file"]).stem
        return probabilities.get(unit, {}).get(role, 0.10)

    provider = ScriptedJevClient(nouls=answer)
    client = AsyncScriptedJevClient(provider) if asynchronous else provider
    search = (
        (find_all_text_async if text else find_all_async)
        if asynchronous
        else (find_all_text if text else find_all)
    )
    result = search(
        index,
        Judge(client, masker=None, scanner=None),
        {"p0": "the requested behavior"},
        files=paths,
        question_profile=ROLES_V2,
        required_roles=LOCAL_ROLES,
    )
    if asynchronous:
        result = asyncio.run(result)

    assert result.failure is None
    assert sorted(len(state["items"]) for state, _ in provider.requests) == [1, 16, 16]
    for state, questions in provider.requests:
        assert len(questions) == 6 * len(state["items"])
        for slot in range(len(state["items"])):
            assert {name.split("_p0@", 1)[0] for name in questions if name.endswith(f"#{slot}")} == {
                *LOCAL_ROLES,
                "delegates",
                "satisfied",
            }
    assert content_hash(ROLES_V2.templates) == APPROVED_QUESTIONS
    scores = {Path(score.unit.path).stem: score for score in result.scores("p0")}
    assert scores["unit04"].probability == 0.99
    assert scores["unit01"].probability == 0.70
    assert len(scores["unit01"].answer.components) == 6
    assert not forwarding_only(scores["unit01"].answer.components)
    assert forwarding_only(scores["unit05"].answer.components)
    assert [Path(score.unit.path).stem for score in result.ranked("p0")[:5]] == [
        "unit00",
        "unit01",
        "unit02",
        "unit03",
        "unit04",
    ]
    assert result.stopped_by == "scope_examined"  # Missing high-band roles do not extend search.


def test_resumed_profile_retains_prior_units_with_new_role_winners(tmp_path: Path):
    (tmp_path / "first.py").write_text("def decide():\n    return True\n")
    (tmp_path / "next.py").write_text("def guard():\n    return False\n")
    commit_all(tmp_path)
    index = CodeIndex.from_git(tmp_path)
    provider = ScriptedJevClient(nouls={"decide_p0": 0.6, "guard_p0": 0.7}, default_noul=0.1)
    judge = Judge(provider, masker=None, scanner=None)
    options = {"question_profile": ROLES_V2, "required_roles": ("decide", "guard")}
    first = find_all(index, judge, {"p0": "the controls"}, files=["first.py"], **options)
    # Distinct observations for the new unit, independent of the resumed result.
    provider.nouls = {"decide_p0": 0.1, "guard_p0": 0.9}
    second = find_all(index, judge, first.targets, files=["next.py"], completed=first.judged, **options)
    assert first.failure is second.failure is None
    assert [score.unit.path for score in second.ranked_with("p0", first)] == ["first.py", "next.py"]
    assert len(provider.requests) == 2


def test_role_profile_keeps_todays_settling_when_local_roles_are_uncovered(tmp_path: Path):
    from jev_navigator.directives.frontier import VALUE

    for number in range(3):
        (tmp_path / f"unit{number}.py").write_text(f"def unit{number}():\n    return {number}\n")
    commit_all(tmp_path)
    index = CodeIndex.from_git(tmp_path)
    provider = ScriptedJevClient(nouls={"satisfied_p0": 0.99}, default_noul=0.1)
    result = find_all(
        index,
        Judge(provider, items_per_request=1),
        {"p0": "the requested behavior"},
        files=index.files,
        question_profile=ROLES_V2,
        required_roles=LOCAL_ROLES,
        policy=VALUE,
        batches_per_wave=1,
    )
    assert result.failure is None
    assert result.stopped_by == "settled"
    assert len(provider.requests) == 1
