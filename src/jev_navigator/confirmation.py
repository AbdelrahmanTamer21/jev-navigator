"""When a cache entry was last confirmed: a run confirms an entry whenever it finds a real file whose
content key the entry matches, or reuses an answer. Stamps count whole UTC days, so a store writes at
most one stamp per entry per day and a same-day warm run writes none."""

from __future__ import annotations

import time

SECONDS_PER_DAY = 86_400


def today() -> int:
    """The current UTC day, counted from the Unix epoch."""
    return day_of(time.time())


def day_of(timestamp: float) -> int:
    """The UTC day of a Unix timestamp, such as a file's modification time."""
    return int(timestamp // SECONDS_PER_DAY)
