"""TypeScript path aliases (``compilerOptions.paths`` and ``baseUrl``) from the nearest tsconfig.json.

The config is read from disk under the index root, following relative ``extends`` chains. Comments
and trailing commas are allowed, as TypeScript allows them. Package ``extends`` (``@tsconfig/...``)
are not followed. The mapped targets come back as root-relative paths without a suffix; the import
resolver then tries the usual suffixes and ``index`` files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

CONFIG_NAME = "tsconfig.json"
_MAX_EXTENDS = 8
_COMMENT_OR_STRING = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', re.S)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


@dataclass(frozen=True)
class ScriptPaths:
    """``base`` is the root-relative directory that alias targets and bare specifiers resolve from."""

    base: str
    paths: tuple[tuple[str, tuple[str, ...]], ...]
    has_base_url: bool

    def candidates(self, specifier: str) -> list[str]:
        """Root-relative paths the specifier may name, in the order TypeScript tries them."""
        mapped = [
            _join(self.base, target.replace("*", wildcard))
            for pattern, targets in self.paths
            if (wildcard := _match(pattern, specifier)) is not None
            for target in targets
        ]
        if self.has_base_url:
            mapped.append(_join(self.base, specifier))
        return mapped


def nearest_script_paths(root: Path, directory: str) -> ScriptPaths | None:
    """The alias settings of the tsconfig.json nearest to ``directory`` (root-relative), or None."""
    current = PurePosixPath(directory)
    while True:
        config = root / current / CONFIG_NAME
        if config.is_file():
            return _script_paths(root, config)
        if str(current) in ("", "."):
            return None
        current = current.parent


def _script_paths(root: Path, config: Path) -> ScriptPaths | None:
    """A child config overrides its parents. ``baseUrl`` is relative to the config that sets it;
    without one, ``paths`` targets are relative to the config that sets ``paths``."""
    base_url_dir: Path | None = None
    paths_dir: Path | None = None
    paths: dict = {}
    for path in reversed(_extends_chain(config)):
        compiler = _read(path).get("compilerOptions", {})
        if "baseUrl" in compiler:
            base_url_dir = path.parent / compiler["baseUrl"]
        if "paths" in compiler:
            paths_dir, paths = path.parent, compiler["paths"]
    base_dir = base_url_dir or paths_dir
    if base_dir is None:
        return None
    base = normalised(base_dir.resolve().relative_to(root.resolve()).as_posix())
    patterns = tuple((pattern, tuple(targets)) for pattern, targets in paths.items())
    return ScriptPaths(base, patterns, base_url_dir is not None)


def _extends_chain(config: Path) -> list[Path]:
    chain = [config]
    while len(chain) < _MAX_EXTENDS:
        parent = _read(chain[-1]).get("extends")
        if not isinstance(parent, str) or not parent.startswith("."):
            break
        target = (chain[-1].parent / parent).resolve()
        target = target if target.suffix == ".json" else target.with_name(target.name + ".json")
        if not target.is_file():
            break
        chain.append(target)
    return chain


def _read(config: Path) -> dict:
    text = config.read_text(errors="replace")
    without_comments = _COMMENT_OR_STRING.sub(_keep_strings, text)
    try:
        loaded = json.loads(_TRAILING_COMMA.sub(r"\1", without_comments))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _keep_strings(match: re.Match) -> str:
    return match.group(0) if match.group(0).startswith('"') else ""


def _match(pattern: str, specifier: str) -> str | None:
    """The part of ``specifier`` that the pattern's ``*`` stands for, or None when it does not match."""
    if "*" not in pattern:
        return "" if pattern == specifier else None
    prefix, suffix = pattern.split("*", 1)
    long_enough = len(specifier) >= len(prefix) + len(suffix)
    if not (long_enough and specifier.startswith(prefix) and specifier.endswith(suffix)):
        return None
    return specifier[len(prefix) : len(specifier) - len(suffix)]


def _join(base: str, target: str) -> str:
    return normalised(f"{base}/{target}" if base else target)


def normalised(path: str) -> str:
    """A root-relative POSIX path with ``.`` and ``..`` segments folded away."""
    parts: list[str] = []
    for part in PurePosixPath(path).parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)
