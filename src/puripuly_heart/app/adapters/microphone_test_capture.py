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
from puripuly_heart.core.feature_diagnostics import log_feature

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
                    if request.failure_callback is not None:
                        request.failure_callback(
                            capture_generation,
                            "input_route_unavailable",
                            cause,
                        )
                    return

                decision = self.channel_decision(
                    device_idx=observation.resolved_device_idx,
                    internal_channels=request.internal_channels,
                )
                requested_channels = decision.preferred_capture_channels
                try:
                    source = self._open_selected_source(observation, request, requested_channels)
                    if not runtime.attach_source(
                        source,
                        generation=capture_generation,
                    ):
                        await self._close_unattached_source(source)
                        return
                except Exception as exc:
                    end_exception = exc
                    log_feature("AudioTest", "device_open", "source_open_failed", error=exc)
                    if request.failure_callback is not None:
                        request.failure_callback(
                            capture_generation,
                            "source_open_failed",
                            type(exc).__name__,
                        )
                    self.log_sink(f"[MicTest] failed cause={type(exc).__name__}")
                    return

                opened = True
                if request.opened_callback is not None:
                    metadata = getattr(source, "opened_input_metadata", None)
                    try:
                        actual = metadata() if callable(metadata) else {}
                    except Exception as error:
                        import logging

                        log_feature(
                            "AudioTest",
                            "device_metadata",
                            "metadata_unavailable",
                            error=error,
                            level=logging.WARNING,
                        )
                        actual = {}
                    request.opened_callback(actual)

                frame_iterator = source.frames()
                pending_frame = runtime.create_frame_task(
                    anext(frame_iterator),
                    generation=capture_generation,
                )
                while True:
                    if request.frame_callback is not None:
                        done, _ = await asyncio.wait({pending_frame}, timeout=5)
                        if not done:
                            raise TimeoutError("no_audio_frames")
                    else:
                        await asyncio.wait({pending_frame})
                    try:
                        frame = pending_frame.result()
                    except StopAsyncIteration:
                        pending_frame = None
                        if (
                            request.frame_callback is not None
                            and request.failure_callback is not None
                        ):
                            request.failure_callback(
                                capture_generation, "capture_ended_early", None
                            )
                        break

                    frame_count += 1
                    await self.meter_sink(
                        self._meter_level_from_frame(frame),
                        request.meter_callback,
                        capture_generation,
                    )
                    if frame_count == 1 and request.ready_callback is not None:
                        request.ready_callback(capture_generation)
                    if request.frame_callback is not None and not request.frame_callback(frame):
                        pending_frame = None
                        break
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
                log_feature(
                    "AudioTest",
                    "capture",
                    "capture_timeout" if isinstance(exc, TimeoutError) else "capture_failed",
                    error=exc,
                )
                if request.failure_callback is not None:
                    request.failure_callback(
                        capture_generation,
                        "capture_failed",
                        type(exc).__name__,
                    )
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
                        if request.failure_callback is not None:
                            request.failure_callback(
                                capture_generation,
                                "source_close_failed",
                                type(exc).__name__,
                            )

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

    def _open_selected_source(self, observation, request, channels):
        def open_channels(count):
            return self.source_factory(
                sample_rate_hz=None,
                channels=count,
                device=observation.resolved_device_idx,
                wasapi_auto_convert=observation.wasapi_auto_convert,
                wasapi_exclusive=observation.wasapi_exclusive,
            )

        try:
            return open_channels(channels)
        except Exception as error:
            if channels <= request.internal_channels:
                raise
            import logging

            log_feature(
                "AudioTest",
                "device_open",
                "capture_channels_fallback",
                error=error,
                level=logging.WARNING,
            )
            return open_channels(request.internal_channels)

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
