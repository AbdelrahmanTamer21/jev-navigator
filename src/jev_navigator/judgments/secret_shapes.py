"""What a secret value looks like in code and config, and what stays code.

``hide_secrets`` applies every rule in order and returns the masked text with the values it
hid. A value under a secret-named key is hidden; a value that is a reference (an identifier,
dotted path, call, env lookup or interpolation) is code and stays. The structural rules (shell
words, quoted values with escapes or no closing quote, YAML block and continued values, and
nested flow values) follow the analysis engine's audit masker, whose corpus both sides test.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterator

MASK = "[MASKED]"
HIGH_ENTROPY_BITS_PER_CHAR = 4.0

Span = tuple[int, int]

_SECRET_WORD = (
    r"(?i:secret[_-]?access[_-]?key|secret[_-]?key|access[_-]?key|private[_-]?key|api[_-]?key"
    r"|password|passwd|pwd|secret|token|credential)"
)
# The secret word is the key's last part: after a separator or a camelCase step, never inside a
# word, so ``authToken`` and ``DB_PASSWORD`` are secret keys and ``tokenizer``, ``max_tokens``
# and ``secretName`` are not.
_SECRET_NAME = rf"(?<![\w$.-])(?:[\w$.-]*?(?:(?<=[_.$-])|(?<=[a-z0-9])(?=[A-Z])))?{_SECRET_WORD}(?![\w$])"
_SEPARATOR = r"[\"']?[ \t]*[:=](?![:=>])[ \t]*"


def _keyed(value: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(rf"(?P<key>{_SECRET_NAME}){_SEPARATOR}{value}", flags)


_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----|\Z)", re.S
)
_KEY_MARKER_LINE = re.compile(r"^.*-----(?:BEGIN|END) [A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----.*$", re.M)
_TOKEN_SHAPES = (
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+(?P<value>[A-Za-z0-9._~+/=-]{16,})")
_SHELL_WORD = (
    r"""(?:"(?:\\[\s\S]|\\\Z|[^"\\])*(?:"|\Z)|'(?:\\[\s\S]|\\\Z|[^'\\])*(?:'|\Z)|\\[\s\S]|[^\s"'\\])+"""
)
_SHELL_ASSIGNMENT = re.compile(
    rf"^[ \t]*(?:export[ \t]+)?(?P<key>{_SECRET_NAME})=(?P<value>(?![{{\[]){_SHELL_WORD})"
    r"(?=[ \t]*(?:$|#|;|&&|\|\||[A-Za-z_]\w*=))",
    re.M,
)
_QUOTED_VALUE = _keyed(
    r"(?P<quote>[\"'`])(?P<value>(?:\\[\s\S]|\\\Z|(?!(?P=quote))[^\\])*)(?:(?P=quote)(?P<tail>[\w\"'][^\s,;})\]]*)?|\Z)"
)
_FLOW_VALUE = _keyed(r"(?P<value>(?=[\[{]))")
_YAML_HEADER = re.compile(
    rf"^(?P<leader>[ \t]*(?:-[ \t]+)?)[\"']?(?P<key>{_SECRET_NAME})[\"']?[ \t]*:(?![:=])[ \t]*"
    r"(?P<value>[^\r\n]*)$",
    re.M,
)
_YAML_PROPERTIES = re.compile(r"(?:(?:&[^\s|>]+|!(?:<[^>\r\n]+>|[^\s|>]*))[ \t]+)+")
_YAML_BLOCK_SCALAR = re.compile(r"(?:(?:&[^\s|>]+|!(?:<[^>\r\n]+>|[^\s|>]*))[ \t]+)*[|>][-+0-9]*")
_PLAIN_VALUE = re.compile(
    rf"^[ \t]*(?:-[ \t]+)?[\"']?(?P<key>{_SECRET_NAME}){_SEPARATOR}"
    r"(?P<value>[^\s\"'`{\[(#][^\n#]*?)(?=[ \t]*(?:#|//|$))",
    re.M,
)
_PLAIN_WORD = re.compile(r"[^\s\"'`(){}\[\]=<>!&|?+*;,:]+")
_CODE_WORDS = frozenset(
    {
        "and",
        "as",
        "async",
        "await",
        "else",
        "for",
        "function",
        "if",
        "in",
        "instanceof",
        "is",
        "lambda",
        "new",
        "not",
        "of",
        "or",
        "return",
        "typeof",
        "void",
        "yield",
    }
)
_BARE_VALUE = _keyed(r"(?P<value>[\w.$@%+/~-][\w.$@%+/~=-]*)(?=[ \t]*(?:$|[,;})\]]|#|//))", re.M)
_LITERAL_FALLBACK = _keyed(
    r"[^\n,;]*?(?:\|\||\?\?|\bor\b)\s*(?P<quote>[\"'`])(?P<value>[^\"'`\n]+)(?P=quote)"
)
_SECRET_NAMED_CALL = re.compile(
    r"(?:(?i:\b[\w.$]*(?:secret|token|password|passwd|credential|api_?key|hmac)\w*)|\bsign)"
    r"\((?P<arguments>[^(){}\[\]\n]*)\)"
)
_QUOTED_LITERAL = re.compile(r"(?P<quote>[\"'`])(?P<value>[^\"'`\n]+)(?P=quote)")
_QUOTED_ASSIGNMENT = re.compile(r"""[:=]\s*["'](?P<value>[A-Za-z0-9+/=_\-]{20,})["']""")
_CODE_REFERENCE = re.compile(
    r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*|-?(?:0[xob][\da-fA-F_]+|\d[\d_]*(?:\.\d+)*[A-Za-z%]{0,4})"
)
_NESTED_COMMENT = re.compile(r"(?://|(?:^|(?<=\s))#)[^\n]*")
_NESTED_QUOTED = re.compile(r"(?P<quote>[\"'`])(?:\\.|(?!(?P=quote)).)*(?P=quote)", re.S)
_NESTED_LEAF = re.compile(r"(?P<key>[\w$.-]+)[\"']?[ \t]*:(?![:=])[ \t]*(?P<value>[^,;}\]\n]*)")
_KEY_BEFORE = re.compile(r"(?P<key>[\w$.-]+)[\"']?[ \t]*:[ \t]*$")
# A nested key that names or describes something holds metadata, not a secret value.
_NAMING_KEY = re.compile(
    r"(?i)(?:^|[_.-]|(?<=[a-z0-9])(?=[A-Z]))(?:name|path|dir|directory|file|header|ref|url|uri|type|kind"
    r"|field|label|annotation|mount|env|key|items|mode|description|in|enabled|required|optional)$"
)
_CALL_OR_INDEX = re.compile(r"[A-Za-z_$][\w$.]*\s*[(\[]")
_DOTTED_PATH = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_NAME_LITERAL = re.compile(r"[A-Z][A-Z0-9_]*|[a-z]+(?:[-_./][a-z]+)*")
_IDENTIFIER_WORDS = re.compile(r"[A-Za-z]+(?:_[A-Za-z]+)*")


def hide_secrets(text: str) -> tuple[str, list[str]]:
    """The text with every secret value masked, and the values that were masked, in rule order."""
    hidden: list[str] = []
    for rule in _RULES:
        spans = rule(text)
        hidden += [text[start:end] for start, end in spans]
        text = _masked_spans(text, spans)
    return text, [value for value in hidden if value and value != MASK]


def _masked_spans(text: str, spans: list[Span]) -> str:
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + MASK + text[end:]
    return text


def _matches(pattern: re.Pattern[str], hides: Callable[[re.Match[str]], bool] = bool):
    """Spans of the pattern's ``value`` group (the whole match when it has none) that ``hides`` accepts."""
    group = "value" if "value" in pattern.groupindex else 0

    def spans(text: str) -> list[Span]:
        return [match.span(group) for match in pattern.finditer(text) if hides(match)]

    return spans


def _quoted_spans(text: str) -> list[Span]:
    """A quoted value runs to its closing quote across escapes and lines, or to the end of the text
    when it never closes; letters glued to the closing quote belong to it."""
    return [
        (match.start("value"), match.end("tail") if match["tail"] else match.end("value"))
        for match in _QUOTED_VALUE.finditer(text)
        if match["value"] and _is_literal(match["value"])
    ]


def _flow_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    for match in _FLOW_VALUE.finditer(text):
        end = _balanced_flow_value_end(text, match.start("value"))
        if end is not None and _holds_literal_leaf(text[match.start("value") + 1 : end - 1]):
            spans.append((match.start("value"), end))
    return spans


def _holds_literal_leaf(content: str) -> bool:
    """Whether nested content under a secret key holds a literal: a quoted string or a scalar that is
    not a reference, under a key that does not name something. ``{ type: String, required: true }``
    and a Kubernetes ``secret:`` volume (``secretName``, ``items``, ``key``, ``path``) hold none."""
    content = _NESTED_COMMENT.sub("", content)
    quoted = any(
        not _under_naming_key(content[: match.start()]) for match in _NESTED_QUOTED.finditer(content)
    )
    unquoted = _NESTED_QUOTED.sub("", content)
    return quoted or any(_is_literal_leaf(leaf) for leaf in _NESTED_LEAF.finditer(unquoted))


def _under_naming_key(before: str) -> bool:
    key = _KEY_BEFORE.search(before)
    return key is not None and bool(_NAMING_KEY.search(key["key"]))


def _is_literal_leaf(leaf: re.Match[str]) -> bool:
    value = leaf["value"].strip()
    return (
        bool(value)
        and not _NAMING_KEY.search(leaf["key"])
        and not _CODE_REFERENCE.fullmatch(value)
        and not value.startswith(("(", "[", "{", "=", "&", "!", "|", "<"))
        and "=>" not in value
        and not _CALL_OR_INDEX.match(value)
    )


def _yaml_block_spans(text: str) -> list[Span]:
    """A secret key's YAML block scalar, empty value with indented lines, or plain value continued
    on deeper lines: the value and its continuation lines, never the sibling keys."""
    spans: list[Span] = []
    for header in _YAML_HEADER.finditer(text):
        continuation_end = _continuation_end(text, header)
        if continuation_end is not None and _yaml_value_continues(text, header, continuation_end):
            start = _yaml_value_start(text, header)
            end = continuation_end - 1 if text[continuation_end - 1] == "," else continuation_end
            spans.append((start, end))
    return spans


def _continuation_end(text: str, header: re.Match[str]) -> int | None:
    """The end of the last line indented deeper than the key, or None when no such line follows."""
    key_column = len(header["leader"].expandtabs(8))
    end = None
    for line in _following_lines(text, header.end()):
        content = text[line[0] : line[1]]
        if not content.strip():
            continue
        if len(content[: len(content) - len(content.lstrip(" \t"))].expandtabs(8)) <= key_column:
            break
        end = line[1]
    return end


def _following_lines(text: str, position: int) -> Iterator[Span]:
    """Each line after ``position`` (which ends a line), without its line ending."""
    while position < len(text):
        line_start = position + 1
        line_end = text.find("\n", line_start)
        line_end = len(text) if line_end == -1 else line_end
        yield line_start, line_end - 1 if text[line_end - 1 : line_end] == "\r" else line_end
        position = line_end


def _yaml_value_continues(text: str, header: re.Match[str], continuation_end: int) -> bool:
    """A block scalar, YAML properties or plain words continue as a value; an empty value continues
    as nested content, which is a value only when it holds a literal."""
    value = header["value"].strip()
    if not value:
        return _holds_literal_leaf(text[header.end() : continuation_end])
    return bool(
        _YAML_BLOCK_SCALAR.fullmatch(value) or _YAML_PROPERTIES.match(value + " ")
    ) or _is_plain_words(value)


def _yaml_value_start(text: str, header: re.Match[str]) -> int:
    """Where the hidden value begins: at the inline value, or at the first indented line when it is empty."""
    if not header["value"].strip():
        return _first_content(text, header.end())
    return header.start("value")


def _first_content(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _is_plain_words(value: str) -> bool:
    """Two or more words of a plain YAML or ini scalar (``correct horse battery``), not an expression."""
    words = value.split()
    return (
        len(words) >= 2
        and all(_PLAIN_WORD.fullmatch(word) for word in words)
        and not _CODE_WORDS.intersection(words)
    )


def _call_literal_spans(text: str) -> list[Span]:
    return [
        (call.start("arguments") + literal.start("value"), call.start("arguments") + literal.end("value"))
        for call in _SECRET_NAMED_CALL.finditer(text)
        for literal in _QUOTED_LITERAL.finditer(call.group("arguments"))
        if _is_literal(literal["value"]) and not _NAME_LITERAL.fullmatch(literal["value"])
    ]


def _shell_literal(match: re.Match[str]) -> bool:
    """A shell word, unless it is code: a reference, call or index, an interpolation, or a keyword
    argument ending in a comma or bracket."""
    value = match["value"]
    unquoted = value.strip("\"'")
    return (
        not value.endswith((",", ")", ";"))
        and _is_literal(unquoted)
        and not unquoted.startswith("$")
        and not _DOTTED_PATH.fullmatch(value)
        and not _CALL_OR_INDEX.match(value)
    )


def _plain_literal(match: re.Match[str]) -> bool:
    return _is_plain_words(match["value"])


def _bare_literal(match: re.Match[str]) -> bool:
    value = match["value"]
    return any(c.isalnum() for c in value) and not _CODE_REFERENCE.fullmatch(value)


def _quoted_literal(match: re.Match[str]) -> bool:
    return _is_literal(match["value"])


def _is_literal(value: str) -> bool:
    return not value.startswith("$") and "${" not in value and "$(" not in value


def _bearer_value(match: re.Match[str]) -> bool:
    return any(character.isdigit() for character in match["value"])


def _high_entropy_value(match: re.Match[str]) -> bool:
    """Identifier words (``RunAttemptConflictError``, ``max_items``) are code, whatever their entropy."""
    value = match["value"]
    return not _IDENTIFIER_WORDS.fullmatch(value) and _is_high_entropy(value)


def _is_high_entropy(value: str) -> bool:
    counts = Counter(value)
    bits = -sum(count / len(value) * math.log2(count / len(value)) for count in counts.values())
    return bits >= HIGH_ENTROPY_BITS_PER_CHAR


def _balanced_flow_value_end(text: str, start: int) -> int | None:
    """Return the end of a quote-aware balanced object or array value (the engine's flow grammar)."""
    expected_closers = {"{": "}", "[": "]"}
    stack = [expected_closers[text[start]]]
    scalar_started = [False]
    last_scalar_was_quoted = [False]
    quote: str | None = None
    index = start + 1
    while index < len(text):
        character = text[index]
        if quote is not None:
            if character == "\\" and quote == '"':
                index += 2
                continue
            if character == quote:
                if quote == "'" and index + 1 < len(text) and text[index + 1] == "'":
                    index += 2
                    continue
                quote = None
                scalar_started[-1] = True
                last_scalar_was_quoted[-1] = True
            index += 1
            continue

        if not scalar_started[-1] and character in {"&", "!"}:
            if character == "!" and index + 1 < len(text) and text[index + 1] == "<":
                tag_end = text.find(">", index + 2)
                if tag_end < 0:
                    return None
                index = tag_end + 1
                continue
            index += 1
            while index < len(text) and (not text[index].isspace() and text[index] not in "[]{},"):
                index += 1
            continue

        if (
            not scalar_started[-1]
            and character == "?"
            and index + 1 < len(text)
            and text[index + 1].isspace()
        ):
            index += 1
            continue

        if character == "#" and (not scalar_started[-1] or text[index - 1].isspace()):
            line_end = text.find("\n", index + 1)
            if line_end < 0:
                return None
            index = line_end + 1
            continue

        if character in {'"', "'"} and not scalar_started[-1]:
            quote = character
        elif character in expected_closers:
            stack.append(expected_closers[character])
            scalar_started.append(False)
            last_scalar_was_quoted.append(False)
        elif character in {"}", "]"}:
            if character != stack[-1]:
                return None
            stack.pop()
            scalar_started.pop()
            last_scalar_was_quoted.pop()
            if not stack:
                return index + 1
            scalar_started[-1] = True
            last_scalar_was_quoted[-1] = False
        elif character == ",":
            scalar_started[-1] = False
            last_scalar_was_quoted[-1] = False
        elif character == ":":
            next_character = text[index + 1] if index + 1 < len(text) else ""
            is_separator = (
                last_scalar_was_quoted[-1]
                or not next_character
                or next_character.isspace()
                or next_character in "[]{},"
            )
            scalar_started[-1] = not is_separator
            last_scalar_was_quoted[-1] = False
        elif not character.isspace():
            scalar_started[-1] = True
            last_scalar_was_quoted[-1] = False
        index += 1
    return None


_RULES: tuple[Callable[[str], list[Span]], ...] = (
    _matches(_PRIVATE_KEY_BLOCK),
    _matches(_KEY_MARKER_LINE),
    *(_matches(shape) for shape in _TOKEN_SHAPES),
    _matches(_BEARER_VALUE, _bearer_value),
    _matches(_SHELL_ASSIGNMENT, _shell_literal),
    _quoted_spans,
    _flow_spans,
    _yaml_block_spans,
    _matches(_PLAIN_VALUE, _plain_literal),
    _matches(_BARE_VALUE, _bare_literal),
    _matches(_LITERAL_FALLBACK, _quoted_literal),
    _call_literal_spans,
    _matches(_QUOTED_ASSIGNMENT, _high_entropy_value),
)
