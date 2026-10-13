from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

_FrameResultT = TypeVar("_FrameResultT")
MicrophoneTestMeterCallback = Callable[[float], object]


@dataclass(frozen=True, slots=True)
class MicrophoneTestCaptureRequest:
    saved_host_api: str
    requested_device: str
    internal_channels: int
    generation: int | None = None
    ready_callback: Callable[[int], None] | None = field(default=None, repr=False)
    failure_callback: Callable[[int, str, str | None], None] | None = field(
        default=None,
        repr=False,
    )
    meter_callback: MicrophoneTestMeterCallback | None = field(
        default=None,
        repr=False,
    )
    level_log_interval_s: float = 1.0
    frame_callback: Callable[[object], bool] | None = field(default=None, repr=False)
    opened_callback: Callable[[dict], None] | None = field(default=None, repr=False)


class MicrophoneTestRuntimePort(Protocol):
    def begin_direct_capture(self) -> int: ...

    def end_direct_capture(self, generation: int) -> None: ...

    def is_current_generation(self, generation: int) -> bool: ...

    def attach_source(self, source: object, *, generation: int) -> bool: ...

    async def close_source(self, source: object | None) -> None: ...

    def create_frame_task(
        self,
        coroutine: Coroutine[Any, Any, _FrameResultT],
        *,
        generation: int,
    ) -> asyncio.Task[_FrameResultT]: ...

    async def cancel_frame_task(self, task: asyncio.Task[Any] | None = None) -> None: ...


class MicrophoneTestCapturePort(Protocol):
    async def capture(
        self,
        request: MicrophoneTestCaptureRequest,
        *,
        runtime: MicrophoneTestRuntimePort,
    ) -> None: ...


__all__ = [
    "MicrophoneTestCapturePort",
    "MicrophoneTestCaptureRequest",
    "MicrophoneTestMeterCallback",
    "MicrophoneTestRuntimePort",
]
