"""A small shop repository in Python, TypeScript and Go, and one find_all request over it: an anchor,
seven files and two names, asked two units per request and two requests per wave. The frontier tests run
it, and ``fixtures/stage_order_requests.json`` holds the requests find_all sent for it at 8a15a936,
before the frontier existed, written by ``frozen_requests``."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

from git_repos import commit_all, write_files

from jev_navigator.directives.find_all import find_all
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import LineAnchor
from jev_navigator.judgments.judge import Judge
from jev_navigator.testing import ScriptedJevClient

SHOP = {
    "orders/service.py": (
        "from orders.limits import check_limit\n\n\n"
        "def place_order(order):\n    check_limit(order)\n    return order\n\n\n"
        "def cancel_order(order):\n    return order\n"
    ),
    "orders/limits.py": (
        "MAX_ITEMS = 4\n\n\n"
        "def check_limit(order):\n    if len(order.items) > MAX_ITEMS:\n"
        '        raise ValueError("too many items")\n'
    ),
    "orders/api.ts": (
        "export function submitOrder(order) {\n"
        '  return fetch("/orders", { method: "POST", body: order });\n}\n\n'
        'export function cancelOrder(id) {\n  return fetch(`/orders/${id}`, { method: "DELETE" });\n}\n'
    ),
    "billing/invoice.py": (
        "def invoice(order):\n    return sum(item.price for item in order.items)\n\n\n"
        "def refund(order):\n    return -invoice(order)\n"
    ),
    "tests/test_limits.py": (
        "from orders.limits import MAX_ITEMS, check_limit\n\n\n"
        "def test_check_limit_refuses_one_item_over(order_factory):\n"
        "    order = order_factory(MAX_ITEMS + 1)\n    try:\n        check_limit(order)\n"
        "    except ValueError:\n        return\n"
        '    raise AssertionError("no error")\n'
    ),
    "orders/copy.py": "def cancel_order(order):\n    return order\n",
    "lib/rates.go": "package lib\n\nfunc Rate() int {\n\treturn 4\n}\n",
}
TARGETS = {
    "limit": "the check that refuses an order over the item limit",
    "refund": "the code that refunds an order",
}
REQUEST = {
    "anchors": [LineAnchor("orders/service.py", 5)],
    "files": [
        "orders/api.ts",
        "billing/invoice.py",
        "tests/test_limits.py",
        "orders/limits.py",
        "orders/copy.py",
        "lib/rates.go",
        "orders/service.py",
    ],
    "names": ["MAX_ITEMS", "check_limit"],
    "batches_per_wave": 2,
}
FROZEN = Path(__file__).parent / "fixtures" / "stage_order_requests.json"


def shop_index(root: Path, files: Mapping[str, str] = SHOP) -> CodeIndex:
    write_files(root, files)
    commit_all(root)
    return CodeIndex.from_git(root)


def sent_requests(
    index: CodeIndex, max_calls: int | None = None, request: Mapping = REQUEST, **settings
) -> list[list[list]]:
    """Each request find_all sends for ``request``, as its items' files and first code lines, in order.
    One request at a time, so the order is the search's own."""
    client = ScriptedJevClient(default_noul=0.1)
    judge = Judge(client, items_per_request=2, max_concurrency=1, max_calls=max_calls)
    find_all(index, judge, TARGETS, **request, **settings)
    return [[_file_and_first_line(item) for item in state["items"]] for state, _ in client.requests]


def _file_and_first_line(item: Mapping) -> list[str]:
    return [item["file"], item["code"].splitlines()[0]]


def frozen_requests(root: Path) -> dict:
    """The requests of a whole search and of one capped at three calls."""
    index = shop_index(root)
    return {"uncapped": sent_requests(index), "three_calls": sent_requests(index, max_calls=3)}


if __name__ == "__main__":
    print(json.dumps(frozen_requests(Path(sys.argv[1])), indent=2))
