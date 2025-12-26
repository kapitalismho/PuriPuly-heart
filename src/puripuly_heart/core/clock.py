from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Return monotonic seconds."""


class SystemClock:
    def now(self) -> float:
        return time.monotonic()


@dataclass(slots=True)
class FakeClock:
    _now: float = 0.0

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("seconds must be >= 0")
        self._now += seconds
