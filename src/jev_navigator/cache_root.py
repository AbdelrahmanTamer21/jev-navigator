"""The one folder every JVN cache lives in."""

from __future__ import annotations

import os
from pathlib import Path


def cache_root() -> Path:
    """``$XDG_CACHE_HOME/jev-navigator``, or ``~/.cache/jev-navigator`` when the variable is unset or
    relative. The XDG specification ignores a relative path, which would otherwise put the cache inside
    whatever repository JVN runs in."""
    configured = Path(os.environ.get("XDG_CACHE_HOME", ""))
    base = configured if configured.is_absolute() else Path.home() / ".cache"
    return base / "jev-navigator"
