"""Import statements, including ones that span several lines, resolved to files inside the index's
scope. TypeScript and JavaScript specifiers also resolve through the path aliases of the nearest
tsconfig.json when the caller passes them."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from .tsconfig import ScriptPaths, normalised

_PYTHON_FROM = re.compile(r"^[ \t]*from\s+(\.*[\w.]*)\s+import\s+(\([^)]*\)|[^\n]*)", re.M)
_PYTHON_IMPORT = re.compile(r"^[ \t]*import\s+([\w.]+)", re.M)
_SCRIPT_FROM = re.compile(
    r"""^[ \t]*(import|export)\s+(?:type\s+)?"""
    r"""((?:(?!\n[ \t]*(?:import|export)\b)[\w$*\s{},])*?)\s*from\s*['"]([^'"]+)['"]""",
    re.M,
)
_SCRIPT_COMMENT_OR_STRING = re.compile(
    r""""(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`|//[^\n]*|/\*.*?\*/""", re.S
)
_SCRIPT_BARE = re.compile(r"""(?:\brequire\(\s*|\bimport\s*\(\s*|^[ \t]*import\s+)['"]([^'"]+)['"]""", re.M)
_SCRIPT_SUFFIXES = (".ts", ".tsx", ".js", ".mjs", ".cjs", ".jsx")
_PYTHON_ROOTS = ("", "src/")


def imported_modules(source: str, path: str) -> list[str]:
    """The module specifiers a file imports, in source order, each once."""
    if path.endswith(".py"):
        found = [(match.start(), match.group(1)) for match in _PYTHON_FROM.finditer(source)]
        found += [(match.start(), match.group(1)) for match in _PYTHON_IMPORT.finditer(source)]
    else:
        code = _without_script_comments(source)
        found = [(match.start(), match.group(3)) for match in _SCRIPT_FROM.finditer(code)]
        found += [(match.start(), match.group(1)) for match in _SCRIPT_BARE.finditer(code)]
    return list(dict.fromkeys(specifier for _, specifier in sorted(found)))


def resolve_import(
    specifier: str, importer: str, scope: frozenset[str], script_paths: ScriptPaths | None = None
) -> str | None:
    """The scope file a specifier names, or None for packages and files outside the scope.
    ``script_paths`` are the importer's tsconfig aliases, used for non-relative script specifiers."""
    if importer.endswith(".py"):
        return _resolve_python(specifier, importer, scope)
    return _resolve_script(specifier, importer, scope, script_paths)


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


def _resolve_script(
    specifier: str, importer: str, scope: frozenset[str], script_paths: ScriptPaths | None
) -> str | None:
    if specifier.startswith("."):
        bases = [normalised(f"{PurePosixPath(importer).parent}/{specifier}")]
    else:
        bases = script_paths.candidates(specifier) if script_paths else []
    return next((file for base in bases for file in _script_files(base) if file in scope), None)


def _script_files(base: str) -> list[str]:
    return [
        base,
        *(base + suffix for suffix in _SCRIPT_SUFFIXES),
        *(f"{base}/index{suffix}" for suffix in _SCRIPT_SUFFIXES),
    ]


_PYTHON_COMMENT = re.compile(r"#[^\n]*")
_SCRIPT_DEFAULT_NAME = re.compile(r"^\s*([\w$]+)\s*(?:,|$)")
_SCRIPT_BRACES = re.compile(r"\{([^}]*)\}")
_SCRIPT_DIRECT_EXPORT = re.compile(
    r"^[ \t]*export\s+(?!default\b)(?:declare\s+)?(?:async\s+)?"
    r"(?:function|class|const|let|var|enum|interface|type)\s+([\w$]+)",
    re.M,
)
_SCRIPT_EXPORT_LIST = re.compile(r"^[ \t]*export\s*\{([^}]*)\}(?!\s*from\b)", re.M)


def imported_names(source: str, path: str) -> dict[str, str]:
    """Local name to module specifier, for names imported by name (``from m import a as b``, also
    parenthesised over several lines; ``import { a as b } from "m"`` and ``import a from "m"``, also
    over several lines). Type-only names are included; namespace imports are not."""
    if path.endswith(".py"):
        return {
            _local(part): match.group(1)
            for match in _PYTHON_FROM.finditer(source)
            for part in _PYTHON_COMMENT.sub("", match.group(2)).strip("()\n ").split(",")
            if part.strip() and part.strip() != "*"
        }
    names: dict[str, str] = {}
    for match in _SCRIPT_FROM.finditer(_without_script_comments(source)):
        keyword, clause, specifier = match.groups()
        if keyword != "import":
            continue
        default = _SCRIPT_DEFAULT_NAME.match(clause)
        if default:
            names[default.group(1)] = specifier
        for braces in _SCRIPT_BRACES.findall(clause):
            names.update({_local(part): specifier for part in braces.split(",") if part.strip()})
    return names


def reexported_names(source: str, path: str) -> tuple[tuple[frozenset[str] | None, str], ...]:
    """Names re-exported from each script module; ``None`` means an ``export *`` wildcard."""
    if path.endswith(".py"):
        return ()
    exports = []
    for match in _SCRIPT_FROM.finditer(_without_script_comments(source)):
        keyword, clause, specifier = match.groups()
        if keyword != "export":
            continue
        stripped = clause.strip()
        if stripped == "*":
            exports.append((None, specifier))
            continue
        names = frozenset(
            _local(part)
            for braces in _SCRIPT_BRACES.findall(clause)
            for part in braces.split(",")
            if part.strip()
        )
        if names:
            exports.append((names, specifier))
    return tuple(exports)


def directly_exported_names(source: str, path: str) -> frozenset[str]:
    """Names this script module itself exports, excluding names available only inside the module."""
    if path.endswith(".py"):
        return frozenset()
    code = _without_script_comments(source)
    names = {match.group(1) for match in _SCRIPT_DIRECT_EXPORT.finditer(code)}
    names.update(
        _local(part)
        for match in _SCRIPT_EXPORT_LIST.finditer(code)
        for part in match.group(1).split(",")
        if part.strip()
    )
    return frozenset(names)


def _without_script_comments(source: str) -> str:
    """The source with ``//`` and ``/* */`` comments removed; string literals are kept whole, so a
    ``//`` inside a string is not taken for a comment."""
    return _SCRIPT_COMMENT_OR_STRING.sub(_keep_literal, source)


def _keep_literal(match: re.Match) -> str:
    text = match.group(0)
    return "" if text.startswith("/") else text


def _local(part: str) -> str:
    return part.split(" as ")[-1].strip().removeprefix("type ").strip()
