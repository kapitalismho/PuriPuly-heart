from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

_FrameResultT = TypeVar("_FrameResultT")
AudioTestChannel = Literal["self", "peer"]
AUDIO_TEST_CHANNELS: tuple[AudioTestChannel, ...] = ("self", "peer")


class AudioTestSourceUnavailable(RuntimeError):
    def __init__(self, reason: str, cause: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cause = cause


@dataclass(frozen=True, slots=True)
class AudioTestOpenedSource:
    source: object
    device: Mapping[str, object] = field(default_factory=dict)


class AudioTestSourceOpener(Protocol):
    async def open(self, *, on_acquired: Callable[[object], None]) -> AudioTestOpenedSource: ...


@dataclass(frozen=True, slots=True)
class AudioTestCaptureRequest:
    channel: AudioTestChannel
    generation: int
    opener: AudioTestSourceOpener
    ready_on_open: bool = False
    frame_timeout_s: float | None = None
    opened_callback: Callable[[int, Mapping[str, object]], None] | None = field(
        default=None,
        repr=False,
    )
    ready_callback: Callable[[int], None] | None = field(default=None, repr=False)
    failure_callback: Callable[[int, str, str | None], None] | None = field(
        default=None,
        repr=False,
    )
    frame_callback: Callable[[int, object], None] | None = field(default=None, repr=False)


class AudioTestRuntimePort(Protocol):
    @property
    def source(self) -> object | None: ...

    def retain_source(self, source: object) -> None: ...

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


class AudioTestCapturePort(Protocol):
    async def capture(
        self,
        request: AudioTestCaptureRequest,
        *,
        runtime: AudioTestRuntimePort,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class AudioTestCaptureState:
    enabled: bool
    busy: bool
    source_open: bool
    token: object = None
    close_exception: BaseException | None = field(default=None, repr=False)


class AudioTestCaptureControlPort(Protocol):
    def state(self, channel: AudioTestChannel) -> AudioTestCaptureState: ...

    async def set_enabled(self, channel: AudioTestChannel, enabled: bool) -> None: ...


__all__ = [
    "AUDIO_TEST_CHANNELS",
    "AudioTestCaptureControlPort",
    "AudioTestCaptureState",
    "AudioTestCapturePort",
    "AudioTestCaptureRequest",
    "AudioTestChannel",
    "AudioTestOpenedSource",
    "AudioTestRuntimePort",
    "AudioTestSourceOpener",
    "AudioTestSourceUnavailable",
]
