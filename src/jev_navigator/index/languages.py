"""Which ast-grep language parses a file, which syntax nodes are functions, and how to read their names.

A ``.js`` file whose leading comments carry the ``@flow`` pragma parses as ``flow``: flow uses
ast-grep's available ``tsx`` grammar, selected
per scan through a ``languageGlobs`` sgconfig. What that grammar cannot recover still surfaces as
ERROR nodes, so incomplete coverage stays visible."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
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

_SCRIPT_FUNCTIONS = (
    "function_declaration",
    "generator_function_declaration",
    "method_definition",
    "arrow_function",
    "function_expression",
    "generator_function",
)

FUNCTION_KINDS = {
    "python": ("function_definition",),
    "typescript": _SCRIPT_FUNCTIONS,
    "tsx": _SCRIPT_FUNCTIONS,
    "javascript": _SCRIPT_FUNCTIONS,
}

CLASS_KINDS = {
    "python": ("class_definition",),
    "typescript": ("class_declaration", "abstract_class_declaration", "class"),
    "tsx": ("class_declaration", "abstract_class_declaration", "class"),
    "javascript": ("class_declaration", "class"),
}

# A function or class expression is called by the name that holds it: `const save = function
# inner() {}` is called as `save`. So an expression is named by the declarator, class field, object
# key, assignment or default value (`onError = () => {}` in a parameter list or a destructuring) it
# is the value of, looking through the grammar's parentheses and type casts, and only then by its
# own name. A callback passed as an argument, `it("works", () => ...)`, is held
# by no name and stays anonymous. Each grammar lists only the node kinds it has: one unknown kind
# makes ast-grep reject the whole scan. The tables are keyed by grammar, see ``grammar_of``.
EXPRESSION_KINDS = frozenset({"arrow_function", "function_expression", "generator_function", "class"})
_SCRIPT_HOLDERS = (
    ("variable_declarator", "name"),
    ("public_field_definition", "name"),
    ("pair", "key"),
    ("assignment_expression", "left"),
    ("required_parameter", "pattern"),
    ("assignment_pattern", "left"),
    ("object_assignment_pattern", "left"),
)
NAME_HOLDERS = {
    "python": (),
    "typescript": _SCRIPT_HOLDERS,
    "tsx": _SCRIPT_HOLDERS,
    "javascript": (
        ("variable_declarator", "name"),
        ("field_definition", "property"),
        ("pair", "key"),
        ("assignment_expression", "left"),
        ("assignment_pattern", "left"),
        ("object_assignment_pattern", "left"),
    ),
}
_TSX_WRAPPERS = ("parenthesized_expression", "as_expression", "satisfies_expression", "non_null_expression")
NAME_WRAPPERS = {
    "python": (),
    "typescript": (*_TSX_WRAPPERS, "type_assertion"),
    "tsx": _TSX_WRAPPERS,
    "javascript": ("parenthesized_expression",),
}

# An object literal's functions and classes are its properties: `const api = { fetch() {} }` defines
# `api.fetch`, never a name `fetch` that its module or an importer can call.
OBJECT_KINDS = {"python": (), "typescript": ("object",), "tsx": ("object",), "javascript": ("object",)}

# A function or class assigned to a property, `foo.bar = function () {}`, gives its module no name.
# Assigned to `exports.x` or `module.exports.x`, or listed in `module.exports = {...}`, it is one of
# the module's CommonJS exports, which another module imports by its name.
PROPERTY_TARGET = "{kind: assignment_expression, has: {field: left, kind: member_expression}}"
_MODULE_EXPORTS = "kind: member_expression, regex: '^module[.]exports$'"
_EXPORTS = f"[{{kind: identifier, regex: '^exports$'}}, {{{_MODULE_EXPORTS}}}]"
COMMONJS_EXPORT_TARGET = (
    "{kind: assignment_expression, has: {field: left, kind: member_expression, "
    f"has: {{field: object, any: {_EXPORTS}}}}}}}"
)
COMMONJS_EXPORTS_OBJECT = (
    "{kind: object, inside: {field: right, kind: assignment_expression, "
    f"has: {{field: left, {_MODULE_EXPORTS}}}}}}}"
)
COMMONJS_EXPORT_PAIR = f"{{kind: pair, inside: {COMMONJS_EXPORTS_OBJECT}}}"

# ast-grep prints every node a rule's relations match, so a relation to a large ancestor (the
# program, a module statement, an object literal) printed that ancestor once per match, and the
# parser's output grew with matches times file size. Every such relation sits under a double
# negation, `not: {not: ...}`: it holds the same and prints only the match.
#
# Module-level declarations by what may name them, each a rule per grammar that has such
# declarations: a type alias or interface only a type, a `const`, `let` or `var`, or a TypeScript
# `declare function`, only a value, and an enum or a Python assignment (which may be a type alias)
# both. A module-level declaration sits in the program, in an export, or, in TypeScript, in a
# `declare` that does.
_VARIABLES = "any: [{kind: lexical_declaration}, {kind: variable_declaration}]"
_IN_MODULE = "inside: {any: [{kind: program}, {kind: export_statement}]}"
_IN_TYPED_MODULE = (
    "inside: {any: [{kind: program}, {kind: export_statement}, "
    f"{{kind: ambient_declaration, {_IN_MODULE}}}]}}"
)
_MODULE_VARIABLES = f"{{{_VARIABLES}, not: {{not: {{{_IN_MODULE}}}}}}}"
_TYPED_MODULE_VARIABLES = f"{{{_VARIABLES}, not: {{not: {{{_IN_TYPED_MODULE}}}}}}}"
_AMBIENT_FUNCTIONS = (
    f"{{kind: function_signature, not: {{not: {{inside: {{kind: ambient_declaration, {_IN_MODULE}}}}}}}}}"
)
_SCRIPT_TYPES = "  any: [{kind: type_alias_declaration}, {kind: interface_declaration}]"
_SCRIPT_VALUES = f"  any: [{_MODULE_VARIABLES}]"
_TYPED_SCRIPT_VALUES = f"  any: [{_TYPED_MODULE_VARIABLES}, {_AMBIENT_FUNCTIONS}]"
_SCRIPT_ENUMS = "  kind: enum_declaration"
TYPE_DECLARATIONS = {"typescript": _SCRIPT_TYPES, "tsx": _SCRIPT_TYPES}
VALUE_DECLARATIONS = {
    "typescript": _TYPED_SCRIPT_VALUES,
    "tsx": _TYPED_SCRIPT_VALUES,
    "javascript": _SCRIPT_VALUES,
}
TYPE_AND_VALUE_DECLARATIONS = {
    "python": (
        "  kind: assignment\n  not: {not: {inside: {kind: expression_statement, inside: {kind: module}}}}"
    ),
    "typescript": _SCRIPT_ENUMS,
    "tsx": _SCRIPT_ENUMS,
}

# The name nodes a declaration binds, one match per name: both names of `const a = 1, b = 2`, each
# name a destructuring pulls out, and each target of `first, second = 1, 2`. A default value, a
# computed key, an attribute or an item binds no name, and destructuring a `require(...)` imports its
# names rather than declaring them. The scan pairs each name with the innermost declaration holding
# it, so a name declared inside another declaration's value stays its own.
_PATTERN_EXCLUSIONS = """  not:
    any:
      - inside:
          stopBy: end
          field: right
          any: [{kind: assignment_pattern}, {kind: object_assignment_pattern}]
      - inside: {stopBy: end, kind: computed_property_name}
      - inside:
          stopBy: end
          kind: variable_declarator
          all:
            - has: {field: name, kind: object_pattern}
            - has: {field: value, kind: call_expression, has: {field: function, regex: '^require$'}}"""
_SCRIPT_NAME_KINDS = "{kind: identifier}, {kind: shorthand_property_identifier_pattern}"
_SCRIPT_DECLARED_NAMES = f"""  any: [{_SCRIPT_NAME_KINDS}]
  all:
    - not:
        not:
          inside:
            stopBy: end
            field: name
            kind: variable_declarator
            inside: {_MODULE_VARIABLES}
{_PATTERN_EXCLUSIONS}"""
_TYPED_SCRIPT_DECLARED_NAMES = f"""  any: [{_SCRIPT_NAME_KINDS}, {{kind: type_identifier}}]
  all:
    - not:
        not:
          inside:
            stopBy: end
            field: name
            any:
              - kind: variable_declarator
                inside: {_TYPED_MODULE_VARIABLES}
              - {_AMBIENT_FUNCTIONS}
              - kind: type_alias_declaration
              - kind: interface_declaration
              - kind: enum_declaration
{_PATTERN_EXCLUSIONS}"""
DECLARED_NAME_RULES = {
    "python": """  kind: identifier
  all:
    - not:
        not:
          inside:
            stopBy: end
            field: left
            kind: assignment
            inside: {stopBy: end, kind: expression_statement, inside: {kind: module}}
  not:
    inside: {stopBy: end, any: [{kind: attribute}, {kind: subscript}]}""",
    "typescript": _TYPED_SCRIPT_DECLARED_NAMES,
    "tsx": _TYPED_SCRIPT_DECLARED_NAMES,
    "javascript": _SCRIPT_DECLARED_NAMES,
}

# The installed ast-grep supports tsx but not Flow. Route marked files through tsx;
# unsupported Flow constructs remain visible through ERROR nodes.
FLOW_LANGUAGE = "flow"
FUNCTION_KINDS[FLOW_LANGUAGE] = FUNCTION_KINDS["tsx"]
CLASS_KINDS[FLOW_LANGUAGE] = CLASS_KINDS["tsx"]
OBJECT_KINDS[FLOW_LANGUAGE] = OBJECT_KINDS["tsx"]
TYPE_DECLARATIONS[FLOW_LANGUAGE] = _SCRIPT_TYPES
VALUE_DECLARATIONS[FLOW_LANGUAGE] = _TYPED_SCRIPT_VALUES
TYPE_AND_VALUE_DECLARATIONS[FLOW_LANGUAGE] = _SCRIPT_ENUMS
DECLARED_NAME_RULES[FLOW_LANGUAGE] = _TYPED_SCRIPT_DECLARED_NAMES

# ast-grep reads `languageGlobs` only from a config file: a scan of flow files passes this sgconfig,
# which parses every JavaScript suffix with the tsx grammar. Plain-JS files are scanned in their own
# invocation without it, so their grammar is unchanged.
FLOW_SGCONFIG = 'languageGlobs:\n  tsx:\n    - "*.js"\n    - "*.jsx"\n    - "*.mjs"\n    - "*.cjs"\n'


def grammar_of(language: str) -> str:
    """The ast-grep language whose grammar parses ``language`` (flow rides on the tsx grammar)."""
    return "tsx" if language == FLOW_LANGUAGE else language


def language_of(path: str) -> str | None:
    return LANGUAGE_BY_SUFFIX.get(PurePosixPath(path).suffix)


def language_for(path: str, lines: Sequence[str] | None = None) -> str | None:
    """The language ``path`` parses as: like ``language_of``, but a JavaScript file whose leading
    comments carry the ``@flow`` pragma parses as ``flow`` (with the tsx grammar)."""
    language = language_of(path)
    if language == "javascript" and lines is not None and has_flow_pragma(lines):
        return FLOW_LANGUAGE
    return language


_FLOW_PRAGMA = re.compile(r"@flow\b")


def has_flow_pragma(lines: Sequence[str]) -> bool:
    """True when a leading comment of the source carries the ``@flow`` pragma. Only comments before
    the first line of code count — never ``@flow`` in a string or the body — and the scan stops at
    that first code line, however far down it sits: leading comments may be arbitrarily long. A
    byte-order mark or shebang may precede the comments."""
    in_block = False
    for index, line in enumerate(lines):
        text = line[1:] if index == 0 and line.startswith("\ufeff") else line
        position = 0
        while True:
            rest = text[position:]
            stripped = rest.lstrip()
            if not stripped:
                break
            position = len(text) - len(stripped)
            if in_block:
                end = stripped.find("*/")
                if _FLOW_PRAGMA.search(stripped if end < 0 else stripped[:end]):
                    return True
                if end < 0:
                    break
                in_block = False
                position += end + 2
            elif stripped.startswith("//"):
                if _FLOW_PRAGMA.search(stripped[2:]):
                    return True
                break
            elif stripped.startswith("/*"):
                in_block = True
                position += 2
            elif index == 0 and stripped.startswith("#!"):
                break
            else:
                return False
    return False


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


_PYTHON_SUPERCLASSES = (
    "kind: argument_list\ninside:\n  stopBy: neighbor\n  kind: class_definition\n  field: superclasses"
)

_PYTHON_ROLES = (
    ReferenceRole(
        "argument",
        ("kind: argument_list", "kind: keyword_argument\nfield: value"),
        not_inside=(_PYTHON_SUPERCLASSES,),
    ),
    ReferenceRole(
        "argument",
        ("kind: argument_list", "kind: keyword_argument\nfield: value"),
        kind="attribute",
    ),
    ReferenceRole("decorator", ("kind: decorator",)),
    ReferenceRole("collection", ("kind: pair\nfield: value", "kind: list", "kind: tuple", "kind: set")),
    ReferenceRole("assignment", ("kind: assignment\nfield: right",)),
    ReferenceRole("return", ("kind: return_statement",)),
    ReferenceRole("receiver", ("kind: attribute\nfield: object",), not_regex="^(self|cls)$"),
    ReferenceRole("type", ("kind: type\nstopBy: end",)),
    ReferenceRole("base", (_PYTHON_SUPERCLASSES,)),
    ReferenceRole(
        "condition",
        (
            "kind: comparison_operator",
            "kind: boolean_operator",
            "kind: not_operator",
            "kind: assert_statement",
            "field: condition\nany:\n  - kind: if_statement\n  - kind: elif_clause\n"
            "  - kind: while_statement",
        ),
    ),
)

_COMPARISON_OR_LOGIC = r"^(===|!==|==|!=|<|>|<=|>=|&&|\|\||\?\?|instanceof|in)$"

_SCRIPT_ROLES = (
    ReferenceRole("argument", ("kind: arguments",)),
    ReferenceRole("argument", ("kind: arguments",), kind="member_expression"),
    ReferenceRole("decorator", ("kind: decorator",)),
    ReferenceRole("collection", ("kind: pair\nfield: value", "kind: array")),
    ReferenceRole("collection", kind="shorthand_property_identifier"),
    ReferenceRole(
        "assignment", ("kind: variable_declarator\nfield: value", "kind: assignment_expression\nfield: right")
    ),
    ReferenceRole("export", ("kind: export_specifier", "kind: export_statement")),
    ReferenceRole("return", ("kind: return_statement",)),
    ReferenceRole("receiver", ("kind: member_expression\nfield: object",)),
    ReferenceRole("base", ("kind: class_heritage",)),
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
    ReferenceRole("base", ("kind: extends_clause",)),
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
REFERENCE_ROLES[FLOW_LANGUAGE] = _TYPED_SCRIPT_ROLES


_EXPORT = "{field: declaration, kind: export_statement}"
_AMBIENT_EXPORT = f"{{kind: ambient_declaration, inside: {_EXPORT}}}"
_VARIABLES = "[{kind: lexical_declaration}, {kind: variable_declaration}]"
_EXPORTED_NAME = f"""  inside:
    field: name
    any:
      - inside: {{field: declaration, kind: export_statement, not: {{has: {{regex: '^default$'}}}}}}
      - kind: variable_declarator
        inside: {{any: {_VARIABLES}, inside: {_EXPORT}}}"""
_TYPED_EXPORTED_NAME = f"""{_EXPORTED_NAME}
      - inside: {_AMBIENT_EXPORT}
      - kind: variable_declarator
        inside: {{any: {_VARIABLES}, inside: {_AMBIENT_EXPORT}}}"""
# The name node of each declaration an ``export`` statement makes, one match per name, so the
# declaration's body (a nested function, a template literal) never names the export. A default
# export has no name of its own.
EXPORTED_NAMES = {
    "typescript": f"  any: [{{kind: identifier}}, {{kind: type_identifier}}]\n{_TYPED_EXPORTED_NAME}",
    "tsx": f"  any: [{{kind: identifier}}, {{kind: type_identifier}}]\n{_TYPED_EXPORTED_NAME}",
    "javascript": f"  kind: identifier\n{_EXPORTED_NAME}",
}


def export_rules(languages: Iterable[str]) -> str:
    """ast-grep rules for the script export surface: the names exported declarations make, and the
    ``{ ... }`` clause specifiers that carry aliased names. Python has no such kinds, so it
    contributes no rules."""
    documents = []
    for language in languages:
        if language == "python":
            continue
        grammar = grammar_of(language)
        documents.append(f"id: export_surface\nlanguage: {grammar}\nrule:\n{EXPORTED_NAMES[grammar]}")
        documents.append(f"id: export_specifier\nlanguage: {grammar}\nrule:\n  kind: export_specifier")
    return "\n---\n".join(documents)


def reference_rules(languages: Iterable[str]) -> str:
    """ast-grep rules, one per listed language and role, matching every identifier that has that role.
    Calls and imports have no role, so they never match."""
    return "\n---\n".join(
        _role_rule(language, role) for language in languages for role in REFERENCE_ROLES[language]
    )


def _role_rule(language: str, role: ReferenceRole) -> str:
    lines = [f"id: {role.name}", f"language: {grammar_of(language)}", "rule:", f"  kind: {role.kind}"]
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
