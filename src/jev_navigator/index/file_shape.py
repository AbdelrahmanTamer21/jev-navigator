"""What a file's bytes say about the cost of parsing it, measured without parsing it.

ast-grep's memory follows the length of each line of code, roughly with its square: a 120 KB bundle on
one line peaked at 681 MB, the same bundle cut into lines of about 1,000 characters at 35 MB. Size
alone does not drive it (703 KB of short lines peaked at 28 MB). The estimate below fits every real
file measured on 03.10.2026 within about 10%; a line of data such as one image string holds few syntax
nodes and the estimate over-counts it, which errs on the safe side.
"""

from __future__ import annotations

from dataclasses import dataclass

BASE_PEAK_MB = 25.0
PEAK_MB_PER_SQUARED_THOUSAND_CHARACTERS = 0.05
MAX_PARSE_PEAK_MB = 250.0
"""A file whose estimated parse peak exceeds this is never parsed. For a one-line file that is a line
of about 70,000 characters. The largest parse measured under it peaked at 122 MB, and ast-grep scans
files in parallel, so several can be in memory at once."""


@dataclass(frozen=True)
class FileShape:
    size_bytes: int
    line_count: int
    longest_line: int
    squared_thousands: float

    @property
    def chars_per_line(self) -> float:
        return 0.0 if self.line_count == 0 else self.size_bytes / self.line_count

    @property
    def parse_peak_mb(self) -> float:
        return BASE_PEAK_MB + PEAK_MB_PER_SQUARED_THOUSAND_CHARACTERS * self.squared_thousands

    @property
    def too_large_to_parse(self) -> bool:
        return self.parse_peak_mb > MAX_PARSE_PEAK_MB

    @property
    def refusal(self) -> str | None:
        if not self.too_large_to_parse:
            return None
        return (
            f"too large to parse: estimated parse peak {_peak_text(self.parse_peak_mb)}, "
            f"longest line {self.longest_line:,} characters"
        )


def measure(content: bytes) -> FileShape:
    lines = content.decode(errors="replace").split("\n")
    lengths = [len(line) for line in lines]
    return FileShape(
        size_bytes=len(content),
        line_count=len(lines) - 1 if lines[-1] == "" else len(lines),
        longest_line=max(lengths),
        squared_thousands=sum((length / 1000) ** 2 for length in lengths),
    )


def _peak_text(megabytes: float) -> str:
    return f"{megabytes:,.0f} MB" if megabytes < 1000 else f"{megabytes / 1000:,.0f} GB"
