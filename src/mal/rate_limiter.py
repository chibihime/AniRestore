"""Token-bucket rate limiter, thread-safe, usable as a callable.

Ported from AniOntology's utils/rate_limiter.py so the MAL pipeline stays
standalone (no cross-project import dependency), while keeping the same
throttling behavior across both projects.
"""

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, calls_per_period: float, period_seconds: float = 60.0) -> None:
        self._calls_per_period = calls_per_period
        self._period = period_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            cutoff = now - self._period
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) >= self._calls_per_period:
                sleep_until = self._timestamps[0] + self._period
                sleep_for = sleep_until - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                now = time.monotonic()
                cutoff = now - self._period
                while self._timestamps and self._timestamps[0] < cutoff:
                    self._timestamps.popleft()

            self._timestamps.append(time.monotonic())
