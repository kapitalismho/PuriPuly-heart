from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum


class DesktopOverlayStartupPhase(StrEnum):
    LAUNCHED = "launched"
    PAGE_CONFIGURED = "page_configured"
    NATIVE_READY = "native_ready"
    BOUNDS_CONFIRMED = "bounds_confirmed"
    VISIBLE_CONFIRMED = "visible_confirmed"
    READY = "ready"


type DesktopOverlayStartupTraceSink = Callable[[str, dict[str, object]], None]


class DesktopOverlayStartupCoordinator:
    _SEQUENCE = (
        DesktopOverlayStartupPhase.LAUNCHED,
        DesktopOverlayStartupPhase.PAGE_CONFIGURED,
        DesktopOverlayStartupPhase.NATIVE_READY,
        DesktopOverlayStartupPhase.BOUNDS_CONFIRMED,
        DesktopOverlayStartupPhase.VISIBLE_CONFIRMED,
        DesktopOverlayStartupPhase.READY,
    )

    def __init__(
        self,
        generation: int,
        *,
        trace_sink: DesktopOverlayStartupTraceSink | None = None,
    ) -> None:
        if generation <= 0:
            raise ValueError("desktop overlay startup generation must be positive")
        self.generation = generation
        self.phase = DesktopOverlayStartupPhase.LAUNCHED
        self._trace_sink = trace_sink
        self._started_at = time.monotonic()
        self._retired = False
        self._emit(self.phase.value, accepted=True)

    @property
    def ready(self) -> bool:
        return not self._retired and self.phase is DesktopOverlayStartupPhase.READY

    @property
    def retired(self) -> bool:
        return self._retired

    def advance(self, phase: DesktopOverlayStartupPhase, **fields: object) -> None:
        if self._retired:
            raise RuntimeError("retired desktop overlay startup generation cannot advance")
        current_index = self._SEQUENCE.index(self.phase)
        if (
            current_index + 1 >= len(self._SEQUENCE)
            or self._SEQUENCE[current_index + 1] is not phase
        ):
            raise RuntimeError(
                f"illegal desktop overlay startup transition: {self.phase.value} -> {phase.value}"
            )
        self.phase = phase
        self._emit(phase.value, accepted=True, **fields)

    def accepts(self, generation: int) -> bool:
        return not self._retired and generation == self.generation

    def reject(self, event: str, generation: int) -> None:
        self._emit(event, accepted=False, event_generation=generation)

    def record(self, event: str, **fields: object) -> None:
        if self._retired:
            return
        self._emit(event, accepted=True, **fields)

    def retire(self) -> None:
        self._retired = True

    def _emit(self, event: str, **fields: object) -> None:
        sink = self._trace_sink
        if sink is None:
            return
        sink(
            event,
            {
                "generation": self.generation,
                "phase": self.phase.value,
                "monotonic_ms": round((time.monotonic() - self._started_at) * 1000, 3),
                **fields,
            },
        )


__all__ = [
    "DesktopOverlayStartupCoordinator",
    "DesktopOverlayStartupPhase",
    "DesktopOverlayStartupTraceSink",
]
