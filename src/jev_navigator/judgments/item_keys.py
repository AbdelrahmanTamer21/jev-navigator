"""The keys an item's stored answer is found by: one owner for both formats.

The strict key is the one production lookups use. It holds the masked item, the masked shared state,
the question with its wording, and the batch the item was asked in, because Jev's answer about an item
can move with its batch mates. The relaxed key leaves the batch out, so an answer can be found in any
company: a simulator replaying another configuration over the same code, or a later lookup once the
company shift is measured. It hashes the item before masking, because masking hides a value found in
one item in every item of the wave, so a masked item's bytes depend on its wave. It holds only the
shared state the question names, so another point's text does not change it. Both keys are hashes;
no code reaches a store through them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from .questions import Check, content_hash

RELAXED_PREFIX = "relaxed:"
_STATE_PATH = re.compile(r"`([A-Za-z_]\w*(?:\.\w+)*)`")
_ABSENT = object()


def strict_item_key(check: Check, item: Mapping, shared: Mapping, mates: str) -> str:
    """The masked item, the masked shared state, the question with its wording, and its batch mates."""
    return f"{content_hash(item)}|{content_hash(shared)}|{check.question_id}|{mates}"


def relaxed_item_key(check: Check, item: Mapping, shared: Mapping) -> str:
    """The unmasked item, the shared state the question names, and the question with its wording."""
    named = content_hash(named_state(check, shared))
    return f"{RELAXED_PREFIX}{content_hash(item)}|{named}|{check.question_id}"


def named_state(check: Check, shared: Mapping) -> dict:
    """The shared values the question's wording names by a backticked path, such as `doc.sentence`;
    a backticked word that names no shared value is left out."""
    named = {}
    for path in _STATE_PATH.findall(str(check.to_question())):
        value = _value_at(shared, path)
        if value is not _ABSENT:
            named[path] = value
    return named


def _value_at(shared: Mapping, path: str) -> object:
    value: object = shared
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _ABSENT
        value = value[part]
    return value
