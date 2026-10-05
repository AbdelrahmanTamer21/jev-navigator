"""The code a request shows: long lines cut, and a whole slice cut at a line boundary, so one line or
one long function can never push a request past a route's state limit. Every cut stays visible."""

from __future__ import annotations

from dataclasses import replace

from ..index.spans import CodeSlice

MAX_LINE_CHARS = 240
MAX_SLICE_CHARS = 12_000
LINE_CUT_MARK = " [line cut]"


def cut_long_line(line: str, max_chars: int = MAX_LINE_CHARS) -> str:
    return line if len(line) <= max_chars else f"{line[:max_chars]}{LINE_CUT_MARK}"


def shown_slice(
    code: CodeSlice, max_chars: int = MAX_SLICE_CHARS, max_line_chars: int = MAX_LINE_CHARS
) -> CodeSlice | None:
    """The lines of ``code`` that fit ``max_chars``, each cut at ``max_line_chars``. After a cut the
    span ends at the last shown line and a note names what was left out. Returns None when
    not even the first line fits, so the caller retains the source as uninspected."""
    lines = [cut_long_line(line, max_line_chars) for line in code.text.split("\n")]
    kept = _lines_that_fit(lines, max_chars)
    if not kept:
        return None
    text = "\n".join(kept)
    if len(kept) < len(lines):
        text += f"\n[cut after {len(kept)} of {len(lines)} lines at {max_chars} characters]"
    return replace(code, span=replace(code.span, end=code.span.start + len(kept) - 1), text=text)


def _lines_that_fit(lines: list[str], max_chars: int) -> list[str]:
    kept: list[str] = []
    used = 0
    for line in lines:
        used += len(line) + 1
        if used > max_chars + 1:
            break
        kept.append(line)
    return kept
