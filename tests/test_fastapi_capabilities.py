"""Must-pass capabilities for jev-navigator's moves, on a small FastAPI app with a GraphQL-style schema.

Each test is one break the hard benchmark found on saleor (named in the test), rebuilt small. The tests
use only the library's public moves: open a start place, list what the moves offer with the current uncapped
defaults, and check the place the true path needs is among the kept offers.

Copied in from ``jev-navigator-evals/builder/test_fastapi_capabilities.py`` so the benchmark's must-pass
capabilities run in the library's own test command. The lookalike cases in ``test_places.py`` and
``test_scope_scan.py`` stay: each of those pins one lookup on a short snippet, while every case here walks
the whole built index - a start place opened with ``starting_places``, then every default move without a
result cap through ``neighbours_and_omissions``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.places import Place, neighbours_and_omissions, starting_places
from jev_navigator.index.code_index import CodeIndex

MAIN = """\
from fastapi import Depends, FastAPI

from app.deps import require_user
from app.routes import orders

app = FastAPI()
app.include_router(orders.router, dependencies=[Depends(require_user)])
"""

DEPS = """\
from fastapi import Header, HTTPException


def require_user(authorization: str = Header(default="")):
    if not authorization:
        raise HTTPException(status_code=401)
    return authorization
"""

ROUTES = """\
from fastapi import APIRouter

from app.services import place_order

router = APIRouter()


@router.post("/orders")
def create_order(payload: dict):
    return place_order(payload)
"""

SCHEMA = """\
from app.mutations import CreateOrder


class Mutations:
    create_order = CreateOrder.Field()
"""

MUTATIONS = """\
class BaseMutation:
    @classmethod
    def Field(cls):
        return cls

    @classmethod
    def mutate(cls, root, info, **data):
        return cls.perform_mutation(root, info, **data)


class CreateOrder(BaseMutation):
    @classmethod
    def perform_mutation(cls, root, info, **data):
        return data
"""

SERVICES = """\
from app.events import call_event
from app.plugins import PluginsManager


def place_order(payload):
    manager = PluginsManager()
    order = dict(payload)
    call_event(manager.order_created, order)
    return order
"""

EVENTS = """\
def call_event(handler, *arguments):
    return handler(*arguments)
"""

PLUGINS = """\
class PluginsManager:
    def order_created(self, order):
        return self.notify("order_created", order)

    def notify(self, event, payload):
        return event, payload
"""

FILES = {
    "app/__init__.py": "",
    "app/main.py": MAIN,
    "app/deps.py": DEPS,
    "app/routes/__init__.py": "",
    "app/routes/orders.py": ROUTES,
    "app/schema.py": SCHEMA,
    "app/mutations.py": MUTATIONS,
    "app/services.py": SERVICES,
    "app/events.py": EVENTS,
    "app/plugins.py": PLUGINS,
}


@pytest.fixture
def shop_index(tmp_path: Path) -> CodeIndex:
    """The shop as one real repository: the files are written and committed, then indexed from git."""
    root = tmp_path / "shop"
    commit_files(root, FILES)
    return CodeIndex.from_git(root)


def _kept_offers(index: CodeIndex, file: str, line: int) -> list[Place]:
    start = starting_places(index, [(file, line)])[0]
    kept, _omitted = neighbours_and_omissions(index, start.open())
    return kept


def _offers_symbol(offers: list[Place], file: str, first_line: int) -> bool:
    return any(place.key.startswith(f"{file}:{first_line}-") for place in offers)


def test_a_class_body_line_offers_the_class_it_names(shop_index: CodeIndex) -> None:
    """saleor S1, S2 and S7: `checkout_create = CheckoutCreate.Field()` in the schema's class body opens
    as a window with no name, so no move reaches CheckoutCreate or the mutate it inherits."""
    # Act
    offers = _kept_offers(shop_index, "app/schema.py", 5)

    # Assert
    assert _offers_symbol(offers, "app/mutations.py", 11)


def test_a_registration_line_offers_the_dependency_it_names(shop_index: CodeIndex) -> None:
    """documenso D1 and parse-server P5 in Python form: a module-level registration line names the code
    that runs later (`Depends(require_user)`), and a window offers no move to it."""
    # Act
    offers = _kept_offers(shop_index, "app/main.py", 7)

    # Assert
    assert _offers_symbol(offers, "app/deps.py", 4)


def test_a_bound_method_passed_as_an_argument_is_offered(shop_index: CodeIndex) -> None:
    """saleor S3: `cls.call_event(manager.product_created, product)` hands on a bound method; the moves
    follow bare names passed on, not attributes."""
    # Act
    offers = _kept_offers(shop_index, "app/services.py", 5)

    # Assert
    assert _offers_symbol(offers, "app/plugins.py", 2)
