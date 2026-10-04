"""Keep secrets out of every Jev request: mask what is found, then scan the final request and refuse on a hit.

Masking works by content: every value the masker hides anywhere in a request is hidden everywhere
in it, so a token found in an assignment is also hidden where a relation text or another item quotes
it. The built-in masker and scanner are lightweight and on by default. A host with a stronger scanner
passes its own objects; turning either off must be explicit (``masker=None`` or ``scanner=None``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Protocol

from .secret_shapes import BY_CONTENT_MIN_CHARS, MASK, hide_secrets


class Masker(Protocol):
    """``mask`` hides secrets in one text; ``masked_values`` lists the values it hides there, so they
    can be hidden everywhere else in the request too."""

    def mask(self, text: str) -> str: ...

    def masked_values(self, text: str) -> list[str]: ...


class Scanner(Protocol):
    def findings(self, text: str) -> list[str]: ...


class SecretInRequestError(RuntimeError):
    """The final pre-send scan found a secret; the request was not sent."""


@dataclass(frozen=True)
class SecretMasker:
    """Masks secret values and keeps code: private-key blocks, common token shapes, Bearer values,
    env-file values, quoted, bare and fallback values under secret-named keys, literal arguments to
    secret-named calls, and high-entropy quoted values in assignments. A value that is a reference
    (an identifier, dotted path, call, env lookup or interpolation) is code and stays.

    ``masked_values`` lists only values of at least ``BY_CONTENT_MIN_CHARS`` characters: a shorter
    value is masked where it stands, because hiding a short word everywhere would blank ordinary code."""

    def mask(self, text: str) -> str:
        return hide_secrets(text)[0]

    def masked_values(self, text: str) -> list[str]:
        return [value for value in hide_secrets(text)[1] if len(value) >= BY_CONTENT_MIN_CHARS]


@dataclass(frozen=True)
class SecretScanner:
    """Reports what the built-in masker would have masked; used as the final check before sending."""

    def findings(self, text: str) -> list[str]:
        return [value[:4] for value in hide_secrets(text)[1]]


def mask_request(state: Mapping, questions: Mapping, masker: Masker) -> tuple[Mapping, Mapping, frozenset]:
    """The masked state and questions, and every value that was hidden in either."""
    values = masked_values([state, questions], masker)
    return mask_everywhere(state, masker, values), mask_everywhere(questions, masker, values), values


def mask_by_content(value: object, masker: Masker) -> object:
    """Masks nested JSON-like data so that a value hidden in one string is hidden in all of them."""
    return mask_everywhere(value, masker, masked_values(value, masker))


def masked_values(value: object, masker: Masker) -> frozenset[str]:
    """Every value the masker hides anywhere inside nested JSON-like data, keys included."""
    return frozenset(found for text in dict.fromkeys(_strings(value)) for found in masker.masked_values(text))


def mask_everywhere(value: object, masker: Masker, values: frozenset[str]) -> object:
    """Masks every string by the masker's rules, then hides each of ``values`` wherever it still
    appears. Keys are left as they are; ``refuse_if_secret`` refuses a request with one in a key."""
    longest_first = sorted(values - {MASK}, key=len, reverse=True)

    @cache
    def hide(text: str) -> str:
        text = masker.mask(text)
        for secret in longest_first:
            text = text.replace(secret, MASK)
        return text

    return _each_string(value, hide)


def safe_options(options: Mapping[str, str], masker: Masker | None) -> dict[str, str]:
    """Options whose key would change under masking are dropped: a secret is never offered as a choice."""
    if masker is None:
        return dict(options)
    return {key: masker.mask(text) for key, text in options.items() if masker.mask(key) == key}


def refuse_if_secret(
    state: Mapping, questions: Mapping, scanner: Scanner | None, masked: frozenset[str] = frozenset()
) -> None:
    """Refuses when a value masked elsewhere is still in the request (it can only sit in a key), or
    when the scanner finds a secret."""
    texts = _strings(state) + _strings(questions)
    if any(value in text for text in texts for value in masked):
        raise SecretInRequestError("a masked value is still in the request, in a key; nothing was sent")
    if scanner is None:
        return
    for text in texts:
        if scanner.findings(text):
            raise SecretInRequestError("the final scan found a secret in the request; nothing was sent")


def _each_string(value: object, change: Callable[[str], str]) -> object:
    if isinstance(value, str):
        return change(value)
    if isinstance(value, Mapping):
        return {key: _each_string(item, change) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_each_string(item, change) for item in value]
    return value


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for key, item in value.items() for text in [str(key), *_strings(item)]]
    if isinstance(value, list | tuple):
        return [text for item in value for text in _strings(item)]
    return []
