"""Starting evidence remains starting evidence when a search resumes."""

import asyncio

import pytest

from jev_navigator.directives.find_code import SearchBudget, find_code, find_code_async
from jev_navigator.directives.places import place_for_line
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import AsyncScriptedJevClient, ScriptedJevClient


@pytest.mark.parametrize("asynchronous", [False, True])
def test_unopened_starts_keep_their_role_after_resume(sample_index, asynchronous):
    # Arrange: a budget interruption happens before the caller's evidence is opened.
    start = place_for_line(sample_index, "app/validation.py", 11, "start")
    client = ScriptedJevClient(default_noul=0.95)
    judge = Judge(AsyncScriptedJevClient(client) if asynchronous else client)

    def search(starts, **options):
        arguments = (sample_index, judge, "the item limit check", starts)
        return (
            asyncio.run(find_code_async(*arguments, moves={}, **options))
            if asynchronous
            else find_code(*arguments, moves={}, **options)
        )

    interrupted = search([start], budget=SearchBudget(max_steps=0))

    # Act: resume the same search with enough budget to inspect its waiting start.
    resumed = search([], resume=interrupted)

    # Assert: rediscovering the caller's own evidence is never a new finding.
    assert not interrupted.starts and len(interrupted.not_inspected) == 1
    assert not resumed.found
    assert [visit.place_key for visit in resumed.starts] == [start.key]
