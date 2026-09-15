from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from puripuly_heart.app.ports.microphone_test import (
    MicrophoneTestCaptureRequest,
    MicrophoneTestMeterCallback,
    MicrophoneTestRuntimePort,
)
from puripuly_heart.core.audio.source import (
    MicrophoneTestRouteObservation,
    SelfMicCaptureChannelDecision,
)

MicrophoneTestLogSink = Callable[[str], None]
MicrophoneTestMeterSink = Callable[
    [float, MicrophoneTestMeterCallback | None, int | None],
    Awaitable[None],
]
MicrophoneTestRouteObserver = Callable[..., MicrophoneTestRouteObservation]
MicrophoneTestChannelDecision = Callable[..., SelfMicCaptureChannelDecision]
MicrophoneTestSourceFactory = Callable[..., object]


class MicrophoneTestClock(Protocol):
    def now(self) -> float: ...


@dataclass(frozen=True, slots=True)
class MicrophoneTestCaptureAdapter:
    clock: MicrophoneTestClock
    log_sink: MicrophoneTestLogSink
    meter_sink: MicrophoneTestMeterSink
    route_observer: MicrophoneTestRouteObserver
    channel_decision: MicrophoneTestChannelDecision
    source_factory: MicrophoneTestSourceFactory

    async def capture(
        self,
        request: MicrophoneTestCaptureRequest,
        *,
        runtime: MicrophoneTestRuntimePort,
    ) -> None:
        direct_generation = request.generation is None
        capture_generation = (
            request.generation if request.generation is not None else runtime.begin_direct_capture()
        )
        try:
            source: object | None = None
            opened = False
            end_exception: BaseException | None = None
            pending_frame: asyncio.Task[object] | None = None
            frame_count = 0

            await self.meter_sink(
                0.0,
                request.meter_callback,
                capture_generation,
            )
            observation = self.route_observer(
                saved_host_api=request.saved_host_api,
                requested_device=request.requested_device,
            )

            try:
                if not observation.should_attempt_open:
                    cause = observation.resolution_exception_class or "unavailable"
                    self.log_sink(f"[MicTest] failed cause={cause}")
                    return

                decision = self.channel_decision(
                    device_idx=observation.resolved_device_idx,
                    internal_channels=request.internal_channels,
                )
                requested_channels = decision.preferred_capture_channels
                try:
                    source = self.source_factory(
                        sample_rate_hz=None,
                        channels=requested_channels,
                        device=observation.resolved_device_idx,
                        wasapi_auto_convert=observation.wasapi_auto_convert,
                        wasapi_exclusive=observation.wasapi_exclusive,
                    )
                    if not runtime.attach_source(
                        source,
                        generation=capture_generation,
                    ):
                        await self._close_unattached_source(source)
                        return
                except Exception as exc:
                    end_exception = exc
                    self.log_sink(f"[MicTest] failed cause={type(exc).__name__}")
                    return

                opened = True

                frame_iterator = source.frames()
                pending_frame = runtime.create_frame_task(
                    anext(frame_iterator),
                    generation=capture_generation,
                )
                while True:
                    await asyncio.wait({pending_frame})
                    try:
                        frame = pending_frame.result()
                    except StopAsyncIteration:
                        pending_frame = None
                        break

                    frame_count += 1
                    await self.meter_sink(
                        self._meter_level_from_frame(frame),
                        request.meter_callback,
                        capture_generation,
                    )
                    pending_frame = runtime.create_frame_task(
                        anext(frame_iterator),
                        generation=capture_generation,
                    )

                    continue
            except asyncio.CancelledError as exc:
                end_exception = exc
                raise
            except Exception as exc:
                end_exception = exc
            finally:
                cleanup_failures: list[Exception] = []
                if pending_frame is not None and not pending_frame.done():
                    try:
                        await runtime.cancel_frame_task(pending_frame)
                    except Exception as exc:
                        cleanup_failures.append(exc)

                if source is not None:
                    try:
                        await runtime.close_source(source)
                    except Exception as exc:
                        cleanup_failures.append(exc)

                if opened:
                    if end_exception is None:
                        self.log_sink(f"[MicTest] completed frames={frame_count}")
                    else:
                        self.log_sink(f"[MicTest] failed cause={type(end_exception).__name__}")
                await self.meter_sink(
                    0.0,
                    request.meter_callback,
                    capture_generation,
                )
                self._raise_cleanup_failures(cleanup_failures)
        finally:
            if direct_generation:
                runtime.end_direct_capture(capture_generation)

    @staticmethod
    async def _close_unattached_source(source: object) -> None:
        with contextlib.suppress(Exception):
            close = getattr(source, "close", None)
            if callable(close):
                outcome = close()
                if inspect.isawaitable(outcome):
                    await outcome

    @staticmethod
    def _meter_level_from_frame(frame: object) -> float:
        samples = np.asarray(getattr(frame, "samples"), dtype=np.float32)
        if samples.size == 0:
            return 0.0
        peak_abs = float(np.max(np.abs(samples)))
        if peak_abs <= 1e-6:
            return 0.0
        return min(1.0, peak_abs)

    @staticmethod
    def _raise_cleanup_failures(failures: list[Exception]) -> None:
        if not failures:
            return
        if len(failures) == 1:
            raise failures[0]
        raise ExceptionGroup("Microphone test capture cleanup failed", failures)


__all__ = ["MicrophoneTestCaptureAdapter"]
