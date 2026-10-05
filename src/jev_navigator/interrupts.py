"""Holding Ctrl-C on the main thread while a small step must not be split."""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def defer_keyboard_interrupts(*, re_raise: bool = True) -> Iterator[None]:
    """Run the block with Ctrl-C held on the main thread; one that arrived is raised once the block
    ends, unless ``re_raise`` is false, as once cancellation has begun and later interrupts only
    repeat it. Other threads never receive SIGINT, so there the block runs unchanged."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    interrupted = False

    def defer(signum, frame) -> None:
        del signum, frame
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, defer)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
    if interrupted and re_raise:
        raise KeyboardInterrupt
