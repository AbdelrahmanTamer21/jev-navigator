"""Which ast-grep language parses a file, which syntax nodes are functions, and how to read their names."""

from __future__ import annotations

import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ReferenceRole:
    """An identifier of ``kind`` has the role ``name`` when a node listed in ``inside`` holds it and
    no node listed in ``not_inside`` does, and its text does not match ``not_regex``. Each listed node
    is an ast-grep relational rule about the direct parent, unless it sets its own ``stopBy``."""

    name: str
    inside: tuple[str, ...] = ()
    kind: str = "identifier"
    not_inside: tuple[str, ...] = ()
    not_regex: str = ""


_PYTHON_ROLES = (
    ReferenceRole("argument", ("kind: argument_list", "kind: keyword_argument\nfield: value")),
    ReferenceRole("decorator", ("kind: decorator",)),
    ReferenceRole("collection", ("kind: pair\nfield: value", "kind: list", "kind: tuple", "kind: set")),
    ReferenceRole("assignment", ("kind: assignment\nfield: right",)),
    ReferenceRole("return", ("kind: return_statement",)),
    ReferenceRole("receiver", ("kind: attribute\nfield: object",), not_regex="^(self|cls)$"),
    ReferenceRole("type", ("kind: type\nstopBy: end",)),
    ReferenceRole(
        "condition",
        (
            "kind: comparison_operator",
            "kind: boolean_operator",
            "kind: not_operator",
            "kind: assert_statement",
            "field: condition\nany:\n  - kind: if_statement\n  - kind: elif_clause\n  - kind: while_statement",
        ),
    ),
)

_COMPARISON_OR_LOGIC = r"^(===|!==|==|!=|<|>|<=|>=|&&|\|\||\?\?|instanceof|in)$"

_SCRIPT_ROLES = (
    ReferenceRole("argument", ("kind: arguments",)),
    ReferenceRole("decorator", ("kind: decorator",)),
    ReferenceRole("collection", ("kind: pair\nfield: value", "kind: array")),
    ReferenceRole("collection", kind="shorthand_property_identifier"),
    ReferenceRole(
        "assignment", ("kind: variable_declarator\nfield: value", "kind: assignment_expression\nfield: right")
    ),
    ReferenceRole("export", ("kind: export_specifier", "kind: export_statement")),
    ReferenceRole("return", ("kind: return_statement",)),
    ReferenceRole("receiver", ("kind: member_expression\nfield: object",)),
    ReferenceRole(
        "condition",
        (
            f"kind: binary_expression\nhas:\n  field: operator\n  regex: {_COMPARISON_OR_LOGIC}",
            "kind: unary_expression\nhas:\n  field: operator\n  regex: ^!$",
            "kind: ternary_expression\nfield: condition",
            "kind: parenthesized_expression\ninside:\n  stopBy: neighbor\n  field: condition\n  any:\n"
            "    - kind: if_statement\n    - kind: while_statement\n    - kind: do_statement",
        ),
    ),
)

_TYPED_SCRIPT_ROLES = (
    *_SCRIPT_ROLES,
    ReferenceRole(
        "type",
        kind="type_identifier",
        not_inside=(
            "field: name\nany:\n  - kind: interface_declaration\n  - kind: type_alias_declaration\n"
            "  - kind: class_declaration\n  - kind: abstract_class_declaration\n  - kind: type_parameter",
        ),
    ),
)

REFERENCE_ROLES = {
    "python": _PYTHON_ROLES,
    "typescript": _TYPED_SCRIPT_ROLES,
    "tsx": _TYPED_SCRIPT_ROLES,
    "javascript": _SCRIPT_ROLES,
}


def reference_rules() -> str:
    """ast-grep rules, one per language and role, matching every identifier that has that role. Calls
    and imports have no role, so they never match."""
    return "\n---\n".join(
        _role_rule(language, role) for language, roles in REFERENCE_ROLES.items() for role in roles
    )


def _role_rule(language: str, role: ReferenceRole) -> str:
    lines = [f"id: {role.name}", f"language: {language}", "rule:", f"  kind: {role.kind}"]
    if role.inside:
        lines += ["  any:", *_inside_entries(role.inside, "    ")]
    exclusions = [f"      - regex: {role.not_regex}"] if role.not_regex else []
    exclusions += _inside_entries(role.not_inside, "      ")
    if exclusions:
        lines += ["  not:", "    any:", *exclusions]
    return "\n".join(lines)


def _inside_entries(parents: tuple[str, ...], indent: str) -> list[str]:
    lines = []
    for parent in parents:
        lines.append(f"{indent}- inside:")
        if not any(line.startswith("stopBy:") for line in parent.split("\n")):
            lines.append(f"{indent}    stopBy: neighbor")
        lines += [f"{indent}    {line}" for line in parent.split("\n")]
    return lines
