"""Which keys hold a secret and which values are code: the vocabulary every masking rule shares.

A key is secret when its last part is a secret word (``DB_PASSWORD``, ``authToken``,
``password_hash``) and naming when a naming word follows that word (``SECRET_ENV``, ``token_url``,
``secretName``); ``max_tokens`` and ``tokenizer`` are neither. A value is code when it refers to
something: an identifier, dotted path, call, a whole ``${...}``, a command substitution ``$(...)``,
or ``$NAME`` outside single quotes. Every pattern here matches a possessive run that starts only at
a run boundary, so matching is linear in the line length.
"""

from __future__ import annotations

import re

KEY = r"(?<![\w$.-])(?P<key>[A-Za-z_$][\w$.-]*+)"
SEPARATOR = r"(?:[\"']?:|[\"']?[ \t]*=)(?![:=>])[ \t]*"

CODE_REFERENCE = re.compile(
    r"\$?[A-Za-z_]\w*(?:\.\$?[A-Za-z_]\w*)*|\$\d+|-?(?:0[xob][\da-fA-F_]+|\d[\d_]*(?:\.\d+)*[A-Za-z%]{0,4})"
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
    ("pwd",),
    ("secret",),
    ("token",),
    ("credential",),
)
_CREDENTIAL_SUFFIXES = frozenset({"hash", "digest", "salt"})
_NAMING_SUFFIXES = frozenset(
    {
        "name", "path", "dir", "directory", "file", "header", "ref", "url", "uri", "type", "kind",
        "field", "label", "annotation", "mount", "env",
    }
)  # fmt: skip
_INTERPOLATION = re.compile(r"\$\{[^{}]*\}|\$\([^()]*\)")
_VARIABLE = re.compile(r"\$[A-Za-z_]\w*")
_NAME_SHAPED = re.compile(r"[A-Za-z_][A-Za-z_.\-]*")
_PATH_SHAPED = re.compile(r"(?:~|\.{1,2})?/?[\w.@-]+(?:/[\w.@\[\]-]+)+/?|/[\w.@-]*")
_URL_SHAPED = re.compile(r"[a-z][a-z0-9+.-]*://\S+")


def key_kind(key: str) -> str | None:
    """ "secret", "naming" or None, as the module docstring describes."""
    parts = [part.lower() for piece in re.split(r"[._\-$]+", key) for part in _KEY_PART.findall(piece)]
    if _ends_with_secret_word(parts):
        return "secret"
    if parts and parts[-1] in _CREDENTIAL_SUFFIXES and _ends_with_secret_word(parts[:-1]):
        return "secret"
    if parts and parts[-1] in _NAMING_SUFFIXES and _ends_with_secret_word(parts[:-1]):
        return "naming"
    return None


def _ends_with_secret_word(parts: list[str]) -> bool:
    return any(tuple(parts[-len(word) :]) == word for word in _SECRET_WORDS if len(parts) >= len(word))


def is_literal(value: str, quote: str = '"') -> bool:
    """Whether a value is written out rather than referring to a variable or running a command. A value
    that interpolates into a name (``GITLAB_CLIENT_SECRET_${slug}``) builds a name, so it is code too."""
    if _INTERPOLATION.fullmatch(value) or "$(" in value:
        return False
    if "${" in value:
        return not _NAME_SHAPED.fullmatch(_INTERPOLATION.sub("", value) or "_")
    return quote == "'" or not _VARIABLE.fullmatch(value)


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
