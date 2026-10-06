from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from shop_search import shop_index

from jev_navigator.composition import SearchConfiguration, reserve_calls
from jev_navigator.directives.find_all import NameHits, find_all, find_all_text
from jev_navigator.judgments.judge import Judge
from jev_navigator.sources import TEXT_FILE_NAMES
from jev_navigator.testing import ScriptedJevClient


def test_code_cannot_spend_the_calls_reserved_for_named_text(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {
            "app.py": "def run():\n    return 1\n\ndef later():\n    return 2\n",
            "docs/limits.md": "# Limits\nThe limit is configured by MAX_ITEMS.\n",
            "config.toml": "[limits]\nMAX_ITEMS = 4\n",
        },
    )
    judge = Judge(ScriptedJevClient(), max_calls=3, items_per_request=1)
    code, text = asyncio.run(
        SearchConfiguration("named-evidence", 1, 2).search(
            index,
            judge,
            {"p": "MAX_ITEMS in docs/limits.md and config.toml"},
            files=["app.py"],
        )
    )
    assert code.stopped_by == "budget"
    assert {score.unit.path for score in text.scores("p")} == {"docs/limits.md", "config.toml"}
    assert judge.calls == 3


def test_continuation_gets_its_reserved_calls_after_discovery_exhausts_its_share(tmp_path: Path) -> None:
    index = shop_index(
        tmp_path,
        {
            "first.py": "def a():\n    return 1\n\ndef b():\n    return 2\n",
            "effect.py": "def perform():\n    return 3\n",
        },
    )
    judge = Judge(ScriptedJevClient(), max_calls=2, items_per_request=1)
    stages = reserve_calls(judge, {"discover": 1, "continue": 1})
    first = find_all(index, stages["discover"], {"p": "perform"}, files=["first.py"])
    second = find_all(index, stages["continue"], {"p": "perform"}, files=["effect.py"])
    assert first.stopped_by == "budget"
    assert [score.unit.symbol for score in second.scores("p")] == ["perform"]
    assert judge.calls == 2
    with pytest.raises(ValueError, match="remaining"):
        reserve_calls(judge, {"another": 1})


def test_text_named_by_stem_reaches_the_file_without_a_content_match(tmp_path: Path) -> None:
    index = shop_index(tmp_path, {"docs/Policy.md": "# Rules\nKeep every answer.\n"})
    _, text = asyncio.run(
        SearchConfiguration("named-evidence", 0, 1).search(
            index,
            Judge(ScriptedJevClient(), max_calls=1),
            {"p": "read `Policy`"},
        )
    )
    assert [score.unit.path for score in text.scores("p")] == ["docs/Policy.md"]
    assert text.names["Policy"] == NameHits(found=1, reached=1, without_unit=0)


@pytest.mark.parametrize("scoped", [False, True])
def test_reservations_cannot_exceed_any_ancestor_cap(scoped: bool) -> None:
    parent = Judge(ScriptedJevClient(), max_calls=1)
    judge = parent.scope().scope() if scoped else parent
    with pytest.raises(ValueError, match="remaining"):
        reserve_calls(judge, {"stage": 2})


def test_outstanding_reservations_keep_their_calls_from_other_scopes(tmp_path: Path) -> None:
    index = shop_index(tmp_path, {"app.py": "def run():\n    return 1\n"})
    parent = Judge(ScriptedJevClient(), max_calls=2)
    reserved = reserve_calls(parent.scope(), {"stage": 2})["stage"]
    with pytest.raises(ValueError, match="remaining"):
        reserve_calls(parent, {"competing": 2})
    blocked = find_all(index, parent.scope(), {"p": "run"}, files=["app.py"])
    assert blocked.stopped_by == "budget"
    nested = reserve_calls(reserved.scope(), {"first": 1, "second": 1})
    for stage in nested.values():
        result = find_all(index, stage.scope(), {"p": "run"}, files=["app.py"])
        assert len(result.scores("p")) == 1
    assert parent.calls == reserved.calls == 2


@pytest.mark.parametrize("content", ["# Rules\nKeep every answer.\n", ""])
def test_named_file_coverage_counts_resolution_even_without_units(tmp_path: Path, content: str) -> None:
    index = shop_index(tmp_path, {"docs/Policy.md": content})
    result = find_all_text(
        index,
        Judge(ScriptedJevClient()),
        {"p": "read Policy"},
        names=["Policy"],
        sources=(TEXT_FILE_NAMES,),
    )
    assert [score.unit.path for score in result.scores("p")] == (["docs/Policy.md"] if content else [])
    assert result.names["Policy"] == NameHits(found=1, reached=1, without_unit=0 if content else 1)
