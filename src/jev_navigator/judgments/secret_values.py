"""Which keys hold a secret and which values are code: the vocabulary every masking rule shares.

A key holds a secret when a secret word is one of its parts at a snake, kebab or camel boundary.
It is secret when the secret word ends it (``DB_PASSWORD``, ``authToken``, ``db_pass``,
``credentials``), suffixed when other words follow (``SECRET_KEY_BASE``, ``GH_TOKEN_RO``), and naming
when a naming word ends it (``SECRET_ENV``, ``token_url``, ``credentialsMountPath``); ``max_tokens``,
``tokenizer`` and ``bypass`` are none of these. A value is code when it refers to
something: an identifier, dotted path, call, a whole ``${...}``, a command substitution ``$(...)``,
or ``$NAME`` outside single quotes. Every pattern here matches a possessive run that starts only at
a run boundary, so matching is linear in the line length.
"""

from __future__ import annotations

import re

KEY = r"(?<![\w$.-])(?P<key>[A-Za-z_$][\w$.-]*+)"
SEPARATOR = r"(?:[\"']?:|[\"']?[ \t]*=)(?![:=>])[ \t]*"

CODE_REFERENCE = re.compile(
    r"\$?[A-Za-z_]\w*(?:\.\$?[A-Za-z_]\w*)*|\$\d+|\d+(?:-\d+)+|-?(?:0[xob][\da-fA-F_]+|\d[\d_]*(?:\.\d+)*[A-Za-z%]{0,4})"
)
CALL_OR_INDEX = re.compile(r"[A-Za-z_$][\w$.]*\s*[(\[]")
DOTTED_PATH = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
NAME_LITERAL = re.compile(r"[A-Z][A-Z0-9_]*|[a-z]+(?:[-_./][a-z]+)*")

_KEY_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_SECRET_WORDS = (
    ("secret", "access", "key"),
    ("secret", "key"),
    ("access", "key"),
    ("private", "key"),
    ("api", "key"),
    ("apikey",),
    ("secretkey",),
    ("password",),
    ("passwd",),
    ("pass",),
    ("credentials",),
    ("pwd",),
    ("secret",),
    ("token",),
    ("credential",),
)
_NAMING_SUFFIXES = frozenset(
    {
        "name", "path", "dir", "directory", "file", "header", "ref", "url", "uri", "type", "kind",
        "field", "label", "annotation", "mount", "env", "endpoint", "host",
    }
)  # fmt: skip
RANDOM_VALUE_MIN_CHARS = 16
ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
_INTERPOLATION = re.compile(r"\$\{[^{}]*\}|\$\([^()]*\)")
_VARIABLE = re.compile(r"\$[A-Za-z_]\w*")
_NAME_SHAPED = re.compile(r"[A-Za-z_][A-Za-z_.\-]*")
_PATH_SHAPED = re.compile(r"(?:~|\.{1,2})?/?[\w.@-]+(?:/[\w.@\[\]-]+)+/?|/[\w.@-]*")
_URL_SHAPED = re.compile(r"[a-z][a-z0-9+.-]*://\S+")


def key_kind(key: str) -> str | None:
    """The key's kind: "secret", "suffixed", "naming" or None, as the module docstring describes."""
    parts = [part.lower() for piece in re.split(r"[._\-$]+", key) for part in _KEY_PART.findall(piece)]
    end = _last_secret_word_end(parts)
    if end is None:
        return None
    tail = parts[end:]
    if not tail:
        return "secret"
    return "naming" if tail[-1] in _NAMING_SUFFIXES else "suffixed"


def _last_secret_word_end(parts: list[str]) -> int | None:
    """Where the last secret word among the parts ends, or None when no part is one."""
    for start in range(len(parts) - 1, -1, -1):
        for word in _SECRET_WORDS:
            if tuple(parts[start : start + len(word)]) == word:
                return start + len(word)
    return None


def is_literal(value: str, quote: str = '"') -> bool:
    """Whether a value is written out rather than referring to a variable or running a command. A value
    that interpolates into a name or a path (``GITLAB_CLIENT_SECRET_${slug}``, ``"$CI_TMP/adc.json"``)
    builds that name or path, so it is code too; outside single quotes ``$NAME`` interpolates, and a
    value it builds into a path (``"$CI_TMP/adc.json"``) is code, while ``"my$ecret"`` stays a literal."""
    if _INTERPOLATION.fullmatch(value) or "$(" in value:
        return False
    if quote == "'":
        return "${" not in value or not _builds_a_name_or_path(_INTERPOLATION.sub("", value))
    if "${" in value and _builds_a_name_or_path(_INTERPOLATION.sub("", value)):
        return False
    if "$" in value and _PATH_SHAPED.fullmatch(_VARIABLE.sub("", _INTERPOLATION.sub("", value))):
        return False
    return not _VARIABLE.fullmatch(value)


def _builds_a_name_or_path(rest: str) -> bool:
    rest = rest or "_"
    return bool(_NAME_SHAPED.fullmatch(rest) or _PATH_SHAPED.fullmatch(rest))


def names_something(value: str) -> bool:
    """A name, a path or a URL: what a naming key (``SECRET_ENV``, ``token_url``) may hold unmasked."""
    value = _INTERPOLATION.sub("", value) or "_"
    return bool(
        _NAME_SHAPED.fullmatch(value) or _PATH_SHAPED.fullmatch(value) or _URL_SHAPED.fullmatch(value)
    )


_PLAIN_WORD = re.compile(r"[^\s\"'`(){}\[\]=<>!&|?+*;,:]+")
_CODE_WORDS = frozenset(
    {
        "and", "as", "async", "await", "else", "for", "function", "if", "in", "instanceof", "is", "lambda",
        "new", "not", "of", "or", "return", "typeof", "void", "yield",
    }
)  # fmt: skip


def is_plain_words(value: str) -> bool:
    """Two or more words of a plain YAML or ini scalar (``correct horse battery``), not an expression."""
    words = value.split()
    return (
        len(words) >= 2
        and all(_PLAIN_WORD.fullmatch(word) for word in words)
        and not _CODE_WORDS.intersection(words)
    )


def looks_generated(value: str) -> bool:
    """A long undotted run of letters and digits (a hex key, ``whsec_`` plus 32 characters) is a value,
    even though it parses as an identifier."""
    return (
        len(value) >= RANDOM_VALUE_MIN_CHARS
        and "." not in value
        and any(c.isdigit() for c in value)
        and any(c.isalpha() for c in value)
    )
