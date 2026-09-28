"""Import statements read line by line and resolved to files inside the index's scope."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_PYTHON_FROM = re.compile(r"^\s*from\s+(\.*[\w.]*)\s+import\b")
_PYTHON_IMPORT = re.compile(r"^\s*import\s+([\w.]+)")
_SCRIPT_IMPORT = re.compile(
    r"""(?:\bfrom\s+|\brequire\(\s*|\bimport\s*\(\s*|^\s*import\s+)['"]([^'"]+)['"]"""
)
_SCRIPT_SUFFIXES = (".ts", ".tsx", ".js", ".mjs", ".cjs", ".jsx")
_PYTHON_ROOTS = ("", "src/")


def imported_modules(source: str, path: str) -> list[str]:
    """The module specifiers a file imports, in source order."""
    patterns = (_PYTHON_FROM, _PYTHON_IMPORT) if path.endswith(".py") else (_SCRIPT_IMPORT,)
    specifiers = []
    for line in source.splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                specifiers.append(match.group(1))
                break
    return specifiers


def resolve_import(specifier: str, importer: str, scope: frozenset[str]) -> str | None:
    """The scope file a specifier names, or None for packages and files outside the scope."""
    if importer.endswith(".py"):
        return _resolve_python(specifier, importer, scope)
    return _resolve_script(specifier, importer, scope)


def _resolve_python(specifier: str, importer: str, scope: frozenset[str]) -> str | None:
    dots = len(specifier) - len(specifier.lstrip("."))
    module_path = specifier.lstrip(".").replace(".", "/")
    if dots:
        base = PurePosixPath(importer).parents[dots - 1]
        roots = [f"{base}/" if str(base) != "." else ""]
    else:
        roots = list(_PYTHON_ROOTS)
    for root in roots:
        for candidate in (f"{root}{module_path}.py", f"{root}{module_path}/__init__.py"):
            if candidate in scope:
                return candidate
    return None


def _resolve_script(specifier: str, importer: str, scope: frozenset[str]) -> str | None:
    if not specifier.startswith("."):
        return None
    joined = _normalise(PurePosixPath(importer).parent / specifier)
    candidates = [joined, *(joined + suffix for suffix in _SCRIPT_SUFFIXES)]
    candidates += [f"{joined}/index{suffix}" for suffix in _SCRIPT_SUFFIXES]
    return next((candidate for candidate in candidates if candidate in scope), None)


def _normalise(path: PurePosixPath) -> str:
    parts: list[str] = []
    for part in path.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


_PYTHON_FROM_NAMES = re.compile(r"^\s*from\s+(\.*[\w.]*)\s+import\s+\(?([^)#]+)")
_SCRIPT_NAMED = re.compile(r"""^\s*import\s+(?:\w+\s*,\s*)?\{([^}]*)\}\s*from\s+['"]([^'"]+)['"]""")
_SCRIPT_DEFAULT = re.compile(r"""^\s*import\s+(\w+)\s*(?:,|\s+from)\s*.*?['"]([^'"]+)['"]""")


def imported_names(source: str, path: str) -> dict[str, str]:
    """Local name to module specifier, for names imported by name (``from m import a as b``,
    ``import { a as b } from "m"``, ``import a from "m"``)."""
    names: dict[str, str] = {}
    for line in source.splitlines():
        if path.endswith(".py"):
            match = _PYTHON_FROM_NAMES.search(line)
            if match:
                names.update(
                    {_local(part): match.group(1) for part in match.group(2).split(",") if part.strip()}
                )
            continue
        named = _SCRIPT_NAMED.search(line)
        if named:
            names.update({_local(part): named.group(2) for part in named.group(1).split(",") if part.strip()})
        default = _SCRIPT_DEFAULT.search(line)
        if default and not named:
            names[default.group(1)] = default.group(2)
    return names


def _local(part: str) -> str:
    return part.split(" as ")[-1].strip()
