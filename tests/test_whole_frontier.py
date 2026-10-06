from __future__ import annotations

import asyncio
from pathlib import Path

from shop_search import shop_index

from jev_navigator.composition import FrontierConfiguration
from jev_navigator.judgments.judge import Judge
from jev_navigator.sources import (
    ANCHORS,
    CALLEES,
    FILE_WORDS,
    FILES,
    IMPORTS,
    LITERALS,
    NAMED_FILES,
    TEXT_NAMED_FILES,
)
from jev_navigator.testing import ScriptedJevClient


def test_budget_keeps_the_entire_reached_population_and_preserves_file_first_batches(tmp_path: Path) -> None:
    matched = "\n".join(f"def check_{i}():\n    return {i}\n" for i in range(35))
    index = shop_index(
        tmp_path,
        {
            "z/limits.py": matched,
            "a/unrelated.py": "def other():\n    return 99\n",
            "docs/limits.md": "# Limits\nEvery answer matters.\n",
        },
    )
    client = ScriptedJevClient()
    config = FrontierConfiguration(max_calls=1, sources=(FILE_WORDS, FILES), hops=())
    result = asyncio.run(config.search(index, Judge(client), {"p": "check limits"}, files=["a/unrelated.py"]))
    assert result.stopped_by == "budget"
    assert len(result.units) == 37
    assert len(result.judged["p"]) == 16
    assert len(result.not_judged) == 21
    [state, _] = client.requests[0]
    assert state["items"][0]["file"] == "docs/limits.md"
    assert all(item["file"] == "z/limits.py" for item in state["items"][1:])
    client = ScriptedJevClient()
    result = asyncio.run(
        FrontierConfiguration(max_calls=3, sources=config.sources, hops=()).search(
            index,
            Judge(client),
            {"p": "check limits"},
            files=["a/unrelated.py"],
        )
    )
    assert result.stopped_by == "scope_examined"
    assert len(result.judged["p"]) == 37
    assert [len(state["items"]) for state, _ in client.requests] == [16, 16, 5]
    assert client.requests[-1][0]["items"][-1]["file"] == "a/unrelated.py"


def test_follow_imported_owner_callee_and_literal_even_after_a_no_answer(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {
            "entry.py": "from worker import execute\n\ndef entry():\n    return execute()\n",
            "worker.py": "from helper import decide\n\ndef execute():\n    return decide()\n",
            "helper.py": "def decide():\n    return 'FEATURE_LIMIT'\n",
            "other.py": "def decide():\n    return 'WRONG_OWNER'\n",
            "config.toml": "FEATURE_LIMIT = 7\n",
        },
    )
    result = asyncio.run(
        FrontierConfiguration(
            sources=(FILES,),
            hops=(CALLEES, IMPORTS, LITERALS),
        ).search(index, Judge(ScriptedJevClient(default_noul=0.0)), {"p": "entry"}, files=["entry.py"])
    )
    assert result.stopped_by == "scope_examined", result.failure
    assert {score.unit.path for score in result.scores("p")} == {
        "entry.py",
        "worker.py",
        "helper.py",
        "config.toml",
    }
    assert not result.not_judged


def test_follow_named_file_and_its_text_to_a_second_named_file(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {
            "app.py": "def run():\n    return 'docs/policy.md'\n",
            "docs/policy.md": "# Policy\nRead config/limits.toml for the limit.\n",
            "config/limits.toml": "maximum = 7\n",
        },
    )
    result = asyncio.run(
        FrontierConfiguration(sources=(FILES,), hops=(NAMED_FILES, TEXT_NAMED_FILES)).search(
            index,
            Judge(ScriptedJevClient()),
            {"p": "policy"},
            files=["app.py"],
        )
    )
    assert result.stopped_by == "scope_examined", result.failure
    assert {unit.path for unit in result.units} == {"app.py", "docs/policy.md", "config/limits.toml"}
    assert len(result.scores("p")) == 3


def test_zero_budget_discovers_files_without_calling_the_client(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path, {"app.py": "def run():\n    return 1\n", "policy.md": "# Policy\nKeep it.\n"}
    )
    client = ScriptedJevClient()
    result = asyncio.run(
        FrontierConfiguration(max_calls=0, sources=(FILE_WORDS, FILES, ANCHORS)).search(
            index,
            Judge(client),
            {"p": "policy"},
            files=["app.py"],
        )
    )
    assert len(result.units) == 2
    assert len(result.not_judged) == 2
    assert result.calls == 0
    assert not client.requests
