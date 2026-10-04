"""What a file's bytes say about the cost of parsing it, measured without parsing it.

ast-grep's memory follows the number of syntax nodes on a line, and a node is roughly one punctuation
character (``{ } ( ) ; , [ ]``): a 120 KB bundle on one line peaked at 681 MB, the same bundle cut into
lines of about 1,000 characters at 35 MB, and one 2.5 MB image string, which is a single node, at 58 MB.
So the estimate sums, over lines, the square of each line's punctuation count. It fits every real file
measured on 03.10.2026 and 04.10.2026 (the table is in the ``census/parse-peak`` folder of the
evaluation, ``fit-table-punctuation-5.5.md``): the 13 files that really peak above 250 MB stay refused,
and a long string of data is parsed. Ordinary code of short lines costs by how much of it there is:
on 04.10.2026 files of 27,000 to 185,000 lines of Heedvane TypeScript and saleor Python peaked at 2.7
to 3.8 MB per 1,000 lines over the base, so the estimate adds a term per line.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

BASE_PEAK_MB = 25.0
PEAK_MB_PER_SQUARED_THOUSAND_PUNCTUATION = 5.5
"""The fitted constant is 5.0 (the real peaks of eight bundles give 3.5 to 5.0 after the base, and it
under-counts by at most 1%); 5.5 is the margin."""
PEAK_MB_PER_THOUSAND_LINES = 4.0
"""The largest measured cost is 3.8 MB per 1,000 lines (27,504 lines of Heedvane API code); 4.0 is the
margin."""
PUNCTUATION = frozenset(b"{}();,[]")
_NOT_PUNCTUATION = bytes(set(range(256)) - PUNCTUATION)
MAX_PARSE_PEAK_MB = 250.0
"""A file whose estimated parse peak exceeds this is never parsed. The largest parse measured under it
peaked at 122 MB, and ast-grep scans files in parallel, so several can be in memory at once."""


def _safe_size_bytes() -> int:
    """The largest size whose worst case stays under the bound: every byte a line (lines cannot exceed
    bytes plus one) and every byte punctuation on one line (the sum of the squared line counts is at most
    the square of the file's size). Solves base + lines x + punctuation x^2 = bound, x in thousands."""
    lines, punctuation = PEAK_MB_PER_THOUSAND_LINES, PEAK_MB_PER_SQUARED_THOUSAND_PUNCTUATION
    room = MAX_PARSE_PEAK_MB - BASE_PEAK_MB - lines / 1000
    return int(1000 * (-lines + (lines * lines + 4 * punctuation * room) ** 0.5) / (2 * punctuation))


PARSEABLE_UP_TO_BYTES = _safe_size_bytes()
"""A file this small can never be over the bound, whatever its lines. The size alone clears it, without
a read."""


LONG_LINE_CHARS = 10_000
DENSE_AVERAGE_LINE_CHARS = 110
DENSE_MINIMUM_BYTES = 4_096
LARGE_FILE_BYTES = 500_000


class Trigger(StrEnum):
    """A reason to ask whether a file is generated. A trigger never refuses a parse: only the memory
    bound does that. Starting values from the census of 12,976 files (37 flagged)."""

    LONG_LINE = "long_line"
    """Some line is longer than ``LONG_LINE_CHARS`` characters."""
    DENSE_LINES = "dense_lines"
    """More than ``DENSE_AVERAGE_LINE_CHARS`` characters per line (the average GitHub Linguist uses for
    minified files) on a file of at least ``DENSE_MINIMUM_BYTES``."""
    LARGE_FILE = "large_file"
    """More than ``LARGE_FILE_BYTES`` bytes."""


@dataclass(frozen=True)
class FileShape:
    size_bytes: int
    line_count: int
    longest_line: int
    squared_thousands_of_punctuation: float

    @property
    def chars_per_line(self) -> float:
        return 0.0 if self.line_count == 0 else self.size_bytes / self.line_count

    @property
    def parse_peak_mb(self) -> float:
        return (
            BASE_PEAK_MB
            + PEAK_MB_PER_THOUSAND_LINES * self.line_count / 1000
            + PEAK_MB_PER_SQUARED_THOUSAND_PUNCTUATION * self.squared_thousands_of_punctuation
        )

    @property
    def triggers(self) -> tuple[Trigger, ...]:
        fired = (
            (Trigger.LONG_LINE, self.longest_line > LONG_LINE_CHARS),
            (
                Trigger.DENSE_LINES,
                self.size_bytes >= DENSE_MINIMUM_BYTES and self.chars_per_line > DENSE_AVERAGE_LINE_CHARS,
            ),
            (Trigger.LARGE_FILE, self.size_bytes > LARGE_FILE_BYTES),
        )
        return tuple(trigger for trigger, tripped in fired if tripped)

    @property
    def too_large_to_parse(self) -> bool:
        return self.parse_peak_mb > MAX_PARSE_PEAK_MB

    @property
    def refusal(self) -> str | None:
        if not self.too_large_to_parse:
            return None
        return (
            f"too large to parse: estimated parse peak {_peak_text(self.parse_peak_mb)}, "
            f"{_counted(self.line_count, 'line')}, longest line {self.longest_line:,} bytes"
        )


def shape_of(root: Path, path: str) -> FileShape:
    """The measured facts and fired triggers of one file, given the repository folder and the path."""
    return measure((root / path).read_bytes())


def refusal_of(root: Path, path: str) -> str | None:
    """Why the file must not be parsed, or None. A file small enough to be safe by size is not read."""
    file = root / path
    if file.stat().st_size <= PARSEABLE_UP_TO_BYTES:
        return None
    return measure(file.read_bytes()).refusal


def measure(content: bytes) -> FileShape:
    """Line lengths are counted in bytes, which over-counts multibyte text and so errs on the safe side."""
    lines = content.split(b"\n")
    return FileShape(
        size_bytes=len(content),
        line_count=len(lines) - 1 if lines[-1] == b"" else len(lines),
        longest_line=max(len(line) for line in lines),
        squared_thousands_of_punctuation=sum(_punctuation_in(line) ** 2 for line in lines) / 1_000_000,
    )


def _punctuation_in(line: bytes) -> int:
    return len(line.translate(None, _NOT_PUNCTUATION))


def _counted(count: int, noun: str) -> str:
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"


def _peak_text(megabytes: float) -> str:
    return f"{megabytes:,.0f} MB" if megabytes < 1000 else f"{megabytes / 1000:,.0f} GB"
