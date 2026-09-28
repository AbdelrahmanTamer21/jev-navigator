"""A small real repository (Python and TypeScript, two commits) that the index tests run against."""

from __future__ import annotations

from pathlib import Path

import pytest
from git_repos import git, write_files

from jev_navigator.index.code_index import CodeIndex

ORDER_SERVICE = '''\
from app.validation import validate_order


class OrderService:
    def place(self, order):
        validate_order(order)
        return self.store.save(order)


def cancel(order_id):
    """Cancels an order that has not shipped."""
    order = load(order_id)
    return archive(order)
'''

VALIDATION = """\
LIMITS_KEY = "orders.max_items"


def validate_order(order):
    if not order.items:
        raise ValueError("empty order")
    return check_limits(order)


def check_limits(order):
    limit = read_setting("orders.max_items")
    return len(order.items) <= limit
"""

ROUTES = """\
import { handleOrder } from "./handlers";

export function registerRoutes(app) {
  app.post("/orders", (req, res) => handleOrder(req, res));
}
"""

HANDLERS = """\
export function handleOrder(req, res) {
  const order = parseOrder(req.body);
  return res.json(order);
}

export const parseOrder = (body) => JSON.parse(body);
"""

COMMENTS = """\
import functools

# Retries the payment twice before giving up.
@functools.cache
def charge(amount):
    return amount


# Normalises a currency code.

def normalise(code):
    return code.upper()


def total(items):
    subtotal = sum(items)  # adds every item price
    if subtotal > 100:  # large orders get a discount
        subtotal = subtotal * 0.9
        subtotal = round(subtotal, 2)

    return subtotal


# A basket of items.
class Basket:
    items = []
"""

SECRET_CONFIG = """\
API_TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
"""


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    write_files(
        root, {"app/__init__.py": "", "app/orders.py": ORDER_SERVICE, "app/validation.py": VALIDATION}
    )
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "orders and validation")
    write_files(
        root,
        {
            "web/routes.ts": ROUTES,
            "web/handlers.ts": HANDLERS,
            "app/settings.py": SECRET_CONFIG,
            "app/comments.py": COMMENTS,
            "app/validation.py": VALIDATION + "\n\ndef noop():\n    return None\n",
        },
    )
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "routes, and a validation change")
    write_files(
        root,
        {
            "app/orders.py": ORDER_SERVICE + "\n",
            "app/validation.py": VALIDATION + "\n\ndef noop():\n    return 1\n",
        },
    )
    git(root, "commit", "-qam", "orders and validation change together")
    return root


@pytest.fixture
def sample_index(sample_repo: Path) -> CodeIndex:
    return CodeIndex.from_git(sample_repo, fact_cache_dir=sample_repo.parent / "fact-cache")
