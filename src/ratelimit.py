"""Per-client-key rate limiting.

A fixed-window counter, not a token bucket: the goal is to stop one client from
starving the rest, not to smooth traffic to the millisecond. Windows are cheap,
predictable, and easy to reason about when a client asks "why am I getting 429".

State is per key and guarded by a lock, because a request can arrive on any
worker thread while another is mid-update.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    """The outcome of one check, with what the client needs to know."""

    allowed: bool
    #: Requests still available in the current window.
    remaining: int
    #: Seconds until the window resets. Zero when allowed.
    retry_after: float


@dataclass
class _Window:
    started_at: float
    used: int


class RateLimiter:
    """Fixed-window limits, per key.

    ``clock`` is injectable so tests can move time without sleeping.
    """

    def __init__(self, *, limit: int, window_seconds: float,
                 clock: Callable[[], float] = time.monotonic):
        if limit < 1:
            raise ValueError("rate limit must be at least 1")
        if window_seconds <= 0:
            raise ValueError("rate limit window must be positive")
        self.limit = limit
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: dict[str, _Window] = {}

    def check(self, key: str) -> Decision:
        """Consume one unit for ``key``, or refuse and say when to retry."""
        now = self._clock()
        with self._lock:
            window = self._windows.get(key)
            if window is None or now - window.started_at >= self.window_seconds:
                self._windows[key] = _Window(started_at=now, used=1)
                return Decision(allowed=True, remaining=self.limit - 1, retry_after=0.0)
            if window.used >= self.limit:
                # Report what is left of *this* window: the caller waits exactly
                # that long, not a fixed guess.
                elapsed = now - window.started_at
                return Decision(
                    allowed=False,
                    remaining=0,
                    retry_after=max(0.0, self.window_seconds - elapsed),
                )
            window.used += 1
            return Decision(allowed=True, remaining=self.limit - window.used, retry_after=0.0)
