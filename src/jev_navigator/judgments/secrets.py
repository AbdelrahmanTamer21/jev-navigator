"""Keep secrets out of every Jev request: mask what is found, then scan the final request and refuse on a hit.

Masking works by content: every value the masker hides anywhere in a request is hidden everywhere
in it, so a token found in an assignment is also hidden where a relation text or another item quotes
it. The built-in masker and scanner are lightweight and on by default. A host with a stronger scanner
passes its own objects; turning either off must be explicit (``masker=None`` or ``scanner=None``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Protocol

from .secret_shapes import (
    BY_CONTENT_MIN_CHARS,
    HIGH_ENTROPY_MIN_CHARS,
    MASK,
    TOKEN_CHARACTER_CLASS,
    hide_secrets,
    is_high_entropy,
)

_SHORT_NUMBER = re.compile(r"[\d.,:_+-]{1,7}")

__all__ = [
    "BY_CONTENT_MIN_CHARS",
    "HIGH_ENTROPY_MIN_CHARS",
    "MASK",
    "TOKEN_CHARACTER_CLASS",
    "Masker",
    "Scanner",
    "SecretInRequestError",
    "SecretMasker",
    "SecretScanner",
    "is_high_entropy",
    "mask_by_content",
    "mask_everywhere",
    "mask_request",
    "masked_values",
    "refuse_if_secret",
    "safe_options",
]


class Masker(Protocol):
    """``mask`` hides secrets in one text; ``masked_values`` lists the values it hides there, so they
    can be hidden everywhere else in the request too."""

    def mask(self, text: str, path: str | None = None) -> str: ...

    def masked_values(self, text: str, path: str | None = None) -> list[str]: ...


class Scanner(Protocol):
    def findings(self, text: str, path: str | None = None) -> list[str]: ...


class SecretInRequestError(RuntimeError):
    """The final pre-send scan found a secret; the request was not sent."""


@dataclass(frozen=True)
class SecretMasker:
    """Masks secret values and keeps code: private-key blocks, common token shapes, Bearer values,
    env-file values, quoted, bare and fallback values under secret-named keys, literal arguments to
    secret-named calls, and high-entropy quoted values in assignments. A value that is a reference
    (an identifier, dotted path, call, env lookup or interpolation) is code and stays.

    ``masked_values`` lists every masked value but a short number (``"1.5"``, ``"0"``), so request masking
    hides each copy elsewhere too: anywhere for a value of ``BY_CONTENT_MIN_CHARS`` or more characters,
    as a whole word for a shorter one (see ``_copy_pattern``).

    ``path`` is the file the text comes from. In a config file, or in text from no file, an unquoted value
    under a secret key is a value (``POSTGRES_PASSWORD: example``); in code it stays (``token: str``)."""

    def mask(self, text: str, path: str | None = None) -> str:
        return hide_secrets(text, path)[0]

    def masked_values(self, text: str, path: str | None = None) -> list[str]:
        return [value for value in hide_secrets(text, path)[1] if not _SHORT_NUMBER.fullmatch(value)]


@dataclass(frozen=True)
class SecretScanner:
    """Reports what the built-in masker would have masked; used as the final check before sending."""

    def findings(self, text: str, path: str | None = None) -> list[str]:
        return [value[:4] for value in hide_secrets(text, path)[1]]


def mask_request(state: Mapping, questions: Mapping, masker: Masker) -> tuple[Mapping, Mapping, frozenset]:
    """The masked state and questions, and every value that was hidden in either."""
    values = masked_values([state, questions], masker)
    return mask_everywhere(state, masker, values), mask_everywhere(questions, masker, values), values


def mask_by_content(value: object, masker: Masker) -> object:
    """Masks nested JSON-like data so that a value hidden in one string is hidden in all of them."""
    return mask_everywhere(value, masker, masked_values(value, masker))


def masked_values(value: object, masker: Masker) -> frozenset[str]:
    """Every value the masker hides anywhere inside nested JSON-like data, keys included."""
    return frozenset(
        found for text, path in dict.fromkeys(_strings(value)) for found in masker.masked_values(text, path)
    )


def mask_everywhere(value: object, masker: Masker, values: frozenset[str]) -> object:
    """Masks every string by the masker's rules, then hides each of ``values`` wherever it still
    appears. Keys are left as they are; ``refuse_if_secret`` refuses a request with one in a key."""
    copies = [_copy_pattern(secret) for secret in sorted(values - {MASK}, key=len, reverse=True)]

    @cache
    def hide(text: str, path: str | None) -> str:
        text = masker.mask(text, path)
        for copy in copies:
            text = copy.sub(MASK, text)
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
    copies = [_copy_pattern(value) for value in masked - {MASK}]
    if any(copy.search(text) for text, _ in texts for copy in copies):
        raise SecretInRequestError("a masked value is still in the request, in a key; nothing was sent")
    if scanner is None:
        return
    for text, path in texts:
        if scanner.findings(text, path):
            raise SecretInRequestError("the final scan found a secret in the request; nothing was sent")


def _copy_pattern(value: str) -> re.Pattern[str]:
    """Where a masked value's copies stand: anywhere for a value of ``BY_CONTENT_MIN_CHARS`` or more
    characters, and as a whole word for a shorter one, so ``hunter2`` is hidden but ``hunter2x`` stays."""
    if len(value) >= BY_CONTENT_MIN_CHARS:
        return re.compile(re.escape(value))
    return re.compile(rf"(?<![\w$]){re.escape(value)}(?![\w$])")


def _each_string(value: object, change: Callable[[str, str | None], str], path: str | None = None) -> object:
    """Changes every string, each with the file it comes from (see ``_file_of``)."""
    if isinstance(value, str):
        return change(value, path)
    if isinstance(value, Mapping):
        inner = _file_of(value, path)
        return {key: _each_string(item, change, inner) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_each_string(item, change, path) for item in value]
    return value


def _strings(value: object, path: str | None = None) -> list[tuple[str, str | None]]:
    """Every string, keys included, with the file it comes from."""
    if isinstance(value, str):
        return [(value, path)]
    if isinstance(value, Mapping):
        inner = _file_of(value, path)
        return [pair for key, item in value.items() for pair in [(str(key), None), *_strings(item, inner)]]
    if isinstance(value, list | tuple):
        return [pair for item in value for pair in _strings(item, path)]
    return []


def _file_of(mapping: Mapping, outer: str | None) -> str | None:
    """A mapping's ``file`` names the file its strings come from (a slice's code, a candidate's lines)."""
    file = mapping.get("file")
    return file if isinstance(file, str) else outer
