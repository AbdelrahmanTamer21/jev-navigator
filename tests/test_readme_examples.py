"""Runs the README's library examples as they are written, so a change that breaks one breaks a test."""

from __future__ import annotations

import re
from pathlib import Path

from conftest import labelled
from shop_search import SHOP, shop_index

from jev_navigator.judgments.judge import Judge

README = Path(__file__).parent.parent / "README.md"
PADDING = {
    f"pad/module_{number}.py": f"def helper_{number}(value):\n    return value + {number}\n"
    for number in range(40)
}


def readme_example(marker: str) -> str:
    """The python block that follows ``<!-- example: marker -->`` in the README."""
    found = re.search(
        rf"<!-- example: {re.escape(marker)} -->\n```python\n(.*?)```", README.read_text(), re.DOTALL
    )
    assert found, f"the README has no example marked {marker!r}"
    return found.group(1)


def test_the_frontier_composition_example_settles_both_targets_on_their_best_units(tmp_path: Path) -> None:
    # Arrange: the shop plus forty unrelated functions, so one wave of sixteen one-unit requests never
    # reaches every unit; check_limit raises for the limit and refund negates an invoice
    index = shop_index(tmp_path, {**SHOP, **PADDING})
    client = labelled({("limit", "raise ValueError"): 0.93, ("refund", "-invoice(order)"): 0.9})
    judge = Judge(client, items_per_request=1, max_concurrency=1)
    printed: list[str] = []
    namespace = {
        "index": index,
        "judge": judge,
        "print": lambda *values: printed.append(" ".join(map(str, values))),
    }

    # Act
    exec(readme_example("frontier composition"), namespace)

    # Assert
    result = namespace["result"]
    assert printed == [
        "limit orders/limits.py check_limit 0.93",
        "refund billing/invoice.py refund 0.9",
        "settled ('limit', 'refund')",
    ]
    assert [source.name for source in result.sources] == [
        "anchor",
        "file",
        "name",
        "definition",
        "caller",
        "callee",
        "model",
    ]
    assert len(client.requests) < len(result.units)
