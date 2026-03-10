"""Shared rate-limit gate for map-worker coordination."""

from __future__ import annotations

import threading
import time


class SharedRateLimitGate:
    """Coordinate temporary global pauses across worker threads."""

    def __init__(self) -> None:
        self._blocked_until_monotonic = 0.0
        self._lock = threading.Lock()

    def wait_until_ready(self) -> None:
        """Block until global backoff window expires."""
        while True:
            with self._lock:
                delay = self._blocked_until_monotonic - time.monotonic()
            if delay <= 0:
                return
            time.sleep(min(delay, 0.5))

    def block_for(self, delay_s: float) -> None:
        """Set/extend global backoff window."""
        if delay_s <= 0:
            return
        blocked_until = time.monotonic() + delay_s
        with self._lock:
            if blocked_until > self._blocked_until_monotonic:
                self._blocked_until_monotonic = blocked_until
