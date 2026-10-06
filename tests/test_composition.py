from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from shop_search import shop_index

from jev_navigator.composition import SearchConfiguration, reserve_calls
from jev_navigator.directives.find_all import find_all
from jev_navigator.judgments.judge import Judge
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
