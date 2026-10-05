"""Per-object caches that never hold the object they belong to."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from functools import wraps
from typing import TypeVar

_Result = TypeVar("_Result")


def memoized(method: Callable[..., _Result]) -> Callable[..., _Result]:
    """Caches ``method``'s result per object and positional arguments, in the object's own
    ``__dict__``. A ``functools.cache`` of a bound method stored on the object would hold the object
    in a reference cycle, so a dropped object would stay in memory until the cycle collector ran."""
    slot = f"_memoized_{method.__name__}"

    @wraps(method)
    def cached(owner: object, *args: Hashable) -> _Result:
        results = owner.__dict__.setdefault(slot, {})
        if args not in results:
            results[args] = method(owner, *args)
        return results[args]

    return cached
