"""What a text mentions by path. This module is the one owner of whether a token in a text is a path."""

from __future__ import annotations

import re

PATH_TOKEN = re.compile(r"[\w./-]*[\w-]\.[A-Za-z0-9]{1,8}")
"""A file name with a suffix of one to eight letters or digits, perhaps under folders: ``ci.yml``,
``jobs/sweep.py``. A word without a suffix, such as ``Makefile``, is not a path token."""
_LEADING_RELATIVE = re.compile(r"^(?:\.{1,2}/|/)+")


def paths_in(text: str) -> list[str]:
    """The path tokens of ``text`` in order of first mention, each once and without a leading
    ``./``, ``../`` or ``/``: ``../config/app.toml`` gives ``config/app.toml``."""
    tokens = (_LEADING_RELATIVE.sub("", token) for token in PATH_TOKEN.findall(text))
    return list(dict.fromkeys(token for token in tokens if token))
