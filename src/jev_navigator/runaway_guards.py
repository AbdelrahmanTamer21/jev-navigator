"""The only time limits JVN applies: runaway guards, never budgets.

André ruled on 06.10.2026 at 10:28 that a search is not limited by time: "if search finds the right
goddamn thing and takes a minute thats fine. I do not want arbitrary guards at this point killing our
results. I want to prevent hanging infinite spend infinite search but nothing else." At 11:08 the same day
he set the values: 30 s for one file's parse and 7 minutes for one search. So each value sits
far above a slow but good run, and what a guard cuts is always counted, never dropped silently:

- ``PARSE_GUARD_SECONDS``: one file's parse. Normal files parse in well under a second; a file over
  the guard is refused with ``parse_guard_reason`` and read as text units instead.
- ``SEARCH_CEILING_SECONDS``: one ``find_code`` search. A search over the ceiling ends ``runaway``,
  and every place it had not opened stays listed, so Resume can continue it.
"""

from __future__ import annotations

PARSE_GUARD_SECONDS = 30.0
SEARCH_CEILING_SECONDS = 420.0


def parse_guard_reason(guard_seconds: float) -> str:
    """Why a file was refused after its parse ran past ``guard_seconds``."""
    return f"not parsed: its parse ran past the {guard_seconds:g} s runaway guard"
