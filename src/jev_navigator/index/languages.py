"""Which ast-grep language parses a file, which syntax nodes are functions, and how to read their names."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
}

FUNCTION_KINDS = {
    "python": ("function_definition",),
    "typescript": ("function_declaration", "method_definition", "arrow_function", "function_expression"),
    "tsx": ("function_declaration", "method_definition", "arrow_function", "function_expression"),
    "javascript": ("function_declaration", "method_definition", "arrow_function", "function_expression"),
}

CLASS_KINDS = {
    "python": ("class_definition",),
    "typescript": ("class_declaration",),
    "tsx": ("class_declaration",),
    "javascript": ("class_declaration",),
}

_SCRIPT_DECLARATIONS = """  any:
    - kind: type_alias_declaration
    - kind: interface_declaration
    - kind: enum_declaration
    - kind: lexical_declaration
      inside:
        any:
          - kind: program
          - kind: export_statement"""

DECLARATION_RULES = {
    "python": """  kind: assignment
  inside:
    kind: expression_statement
    inside:
      kind: module""",
    "typescript": _SCRIPT_DECLARATIONS,
    "tsx": _SCRIPT_DECLARATIONS,
    "javascript": """  kind: lexical_declaration
  inside:
    any:
      - kind: program
      - kind: export_statement""",
}

_DECLARED_NAME = re.compile(
    r"^\s*(?:export\s+)?(?:declare\s+)?(?:(?:type|interface|enum|const|let|var)\s+)?(\w+)"
)

_NAME_PATTERNS = (
    re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)"),
    re.compile(r"^\s*(?:async\s+)?def\s+(\w+)"),
    re.compile(r"\bfunction\s*\*?\s*(\w+)"),
    re.compile(r"\b(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:function\b|\()"),
    re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|async\s+|get\s+|set\s+)*(\w+)\s*\("),
    re.compile(r"(\w+)\s*[:=]\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>)"),
)


def language_of(path: str) -> str | None:
    return LANGUAGE_BY_SUFFIX.get(PurePosixPath(path).suffix)


def declared_name(first_line: str) -> str:
    """The name a module-level assignment, constant, type, interface or enum declares."""
    match = _DECLARED_NAME.search(first_line)
    return match.group(1) if match else "<anonymous>"


def function_name(first_line: str, line_before: str = "") -> str:
    """The declared name on a function's first line, or on the line that assigns it."""
    for candidate in (first_line, line_before):
        for pattern in _NAME_PATTERNS:
            match = pattern.search(candidate)
            if match and match.group(1) not in _NOT_NAMES:
                return match.group(1)
    return "<anonymous>"


_NOT_NAMES = frozenset({"if", "for", "while", "switch", "catch", "return", "function", "async"})


_SCRIPT_REFERENCE_PARENTS = {
    "argument": ("kind: arguments",),
    "decorator": ("kind: decorator",),
    "collection": ("kind: pair\nfield: value", "kind: array"),
    "assignment": ("kind: variable_declarator\nfield: value", "kind: assignment_expression\nfield: right"),
    "export": ("kind: export_specifier", "kind: export_statement"),
    "return": ("kind: return_statement",),
}

REFERENCE_PARENTS = {
    "python": {
        "argument": ("kind: argument_list", "kind: keyword_argument\nfield: value"),
        "decorator": ("kind: decorator",),
        "collection": ("kind: pair\nfield: value", "kind: list", "kind: tuple", "kind: set"),
        "assignment": ("kind: assignment\nfield: right",),
        "return": ("kind: return_statement",),
    },
    "typescript": _SCRIPT_REFERENCE_PARENTS,
    "tsx": _SCRIPT_REFERENCE_PARENTS,
    "javascript": _SCRIPT_REFERENCE_PARENTS,
}

_SHORTHAND_PROPERTY = "shorthand_property_identifier"


def reference_rules(name: str | None = None) -> str:
    """ast-grep rules, one per language and role, matching an identifier (``name``, or any) whose
    direct parent gives it that role. Calls and imports have no role, so they never match."""
    documents = []
    for language, roles in REFERENCE_PARENTS.items():
        for role, parents in roles.items():
            documents.append(_reference_rule(role, language, "identifier", name, parents))
        if language != "python":
            documents.append(_reference_rule("collection", language, _SHORTHAND_PROPERTY, name, ()))
    return "\n---\n".join(documents)


def _reference_rule(role: str, language: str, kind: str, name: str | None, parents: tuple[str, ...]) -> str:
    lines = [f"id: {role}", f"language: {language}", "rule:", f"  kind: {kind}"]
    if name is not None:
        lines.append(f"  regex: ^{re.escape(name)}$")
    if parents:
        lines.append("  any:")
        for parent in parents:
            first, *rest = parent.split("\n")
            lines += ["    - inside:", "        stopBy: neighbor", f"        {first}"]
            lines += [f"        {extra}" for extra in rest]
    return "\n".join(lines)
