"""Prisma schema blocks: the models, views, enums and composite types a ``.prisma`` file declares.

No grammar JVN depends on reads Prisma, and ast-grep has none, so a small scanner finds the blocks.
A block starts at a line holding a block keyword, a name and an opening brace, and ends at the brace
that closes it; braces inside strings and after a ``//`` comment never count. Every top-level block
is passed over whole, but only ``model``, ``view``, ``enum`` and ``type`` blocks are reported: a
``generator`` or ``datasource`` holds settings, not a shape the code reads. A header whose brace
never closes is no block.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

SCHEMA_SUFFIX = ".prisma"
SCHEMA_LANGUAGE = "prisma"
SHAPE_KEYWORDS = frozenset({"model", "view", "enum", "type"})
QUERIED_KEYWORDS = frozenset({"model", "view"})

_BLOCK_HEADER = re.compile(r"^\s*([A-Za-z]+)\s+([A-Za-z][A-Za-z0-9_]*)\s*\{")
_STRING_COMMENT_OR_BRACE = re.compile(r'"(?:[^"\\]|\\.)*"|//.*|([{}])')


@dataclass(frozen=True)
class SchemaBlock:
    """One block, from its header line ``start`` to its closing brace's line ``end``."""

    keyword: str
    name: str
    start: int
    end: int

    @property
    def client_accessor(self) -> str | None:
        """The property Prisma Client queries a model or view through, as its generator spells it:
        the name with only its first character lower-cased (``model WebsiteEvent`` is
        ``prisma.websiteEvent``). None for an enum or a composite type, which are never queried."""
        if self.keyword not in QUERIED_KEYWORDS:
            return None
        return self.name[0].lower() + self.name[1:]


def is_schema_file(path: str) -> bool:
    return PurePosixPath(path).suffix == SCHEMA_SUFFIX


def schema_blocks(lines: Sequence[str]) -> tuple[SchemaBlock, ...]:
    """The model, view, enum and type blocks of a schema's ``lines``, in file order."""
    blocks = []
    number = 1
    while number <= len(lines):
        header = _BLOCK_HEADER.match(lines[number - 1])
        end = _closing_line(lines, number, header.end()) if header else None
        if end is None:
            number += 1
            continue
        if header[1] in SHAPE_KEYWORDS:
            blocks.append(SchemaBlock(header[1], header[2], number, end))
        number = end + 1
    return tuple(blocks)


def _closing_line(lines: Sequence[str], header: int, column: int) -> int | None:
    """The line whose brace closes the block opened on line ``header`` just before ``column``."""
    depth = 1
    for number in range(header, len(lines) + 1):
        text = lines[number - 1][column:] if number == header else lines[number - 1]
        for brace in _braces(text):
            depth += 1 if brace == "{" else -1
            if depth == 0:
                return number
    return None


def _braces(text: str) -> list[str]:
    """The braces of a line's code, outside its strings and its ``//`` comment."""
    return [match[1] for match in _STRING_COMMENT_OR_BRACE.finditer(text) if match[1]]
