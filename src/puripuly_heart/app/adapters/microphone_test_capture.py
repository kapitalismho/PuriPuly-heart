from __future__ import annotations

import asyncio
import contextlib
import inspect
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from puripuly_heart.app.ports.microphone_test import (
    MicrophoneTestCaptureRequest,
    MicrophoneTestMeterCallback,
    MicrophoneTestRuntimePort,
)
from puripuly_heart.core.audio.diagnostics import compute_audio_frame_metrics
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


def _log_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, float):
        return str(value)
    return str(value)


def _db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return round(float(20.0 * math.log10(max(value, 1e-6))), 1)


@dataclass(slots=True)
class _LevelStats:
    frames: int = 0
    audio_ms: float = 0.0
    sample_count: int = 0
    square_sum: float = 0.0
    peak_abs: float = 0.0
    zero_count: int = 0

    def add_frame(self, frame: object) -> None:
        metrics = compute_audio_frame_metrics(frame)
        samples = np.asarray(getattr(frame, "samples"), dtype=np.float32)
        self.frames += 1
        self.audio_ms += metrics.audio_ms
        self.sample_count += int(samples.size)
        if samples.size == 0:
            return
        abs_samples = np.abs(samples)
        self.square_sum += float(np.sum(np.square(samples, dtype=np.float32)))
        self.peak_abs = max(self.peak_abs, float(np.max(abs_samples)))
        self.zero_count += int(np.count_nonzero(abs_samples < 1e-6))

    @property
    def rms_db(self) -> float:
        if self.sample_count <= 0:
            return -120.0
        return _db(math.sqrt(self.square_sum / float(self.sample_count)))

    @property
    def peak_db(self) -> float:
        return _db(self.peak_abs)

    @property
    def zero_ratio(self) -> float:
        if self.sample_count <= 0:
            return 1.0
        return round(float(self.zero_count) / float(self.sample_count), 3)

    def reset(self) -> None:
        self.frames = 0
        self.audio_ms = 0.0
        self.sample_count = 0
        self.square_sum = 0.0
        self.peak_abs = 0.0
        self.zero_count = 0


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
            level_log_interval_s = max(0.0, float(request.level_log_interval_s))
            source: object | None = None
            opened = False
            end_exception: BaseException | None = None
            level_logged = False
            pending_frame: asyncio.Task[object] | None = None
            interval_stats = _LevelStats()
            total_stats = _LevelStats()

            await self.meter_sink(
                0.0,
                request.meter_callback,
                capture_generation,
            )
            observation = self.route_observer(
                saved_host_api=request.saved_host_api,
                requested_device=request.requested_device,
            )
            self.log_sink(self._format_route_log(observation))

            try:
                if not observation.should_attempt_open:
                    self._log_open(
                        attempted=False,
                        opened=False,
                        requested_channels=None,
                        source=None,
                        observation=observation,
                    )
                    self._log_level(interval_stats, source=None)
                    level_logged = True
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
                    self._log_open(
                        attempted=True,
                        opened=False,
                        requested_channels=requested_channels,
                        source=None,
                        observation=observation,
                        exception=exc,
                    )
                    self._log_level(interval_stats, source=None)
                    level_logged = True
                    return

                opened = True
                self._log_open(
                    attempted=True,
                    opened=True,
                    requested_channels=requested_channels,
                    source=source,
                    observation=observation,
                )

                frame_iterator = source.frames()
                pending_frame = runtime.create_frame_task(
                    anext(frame_iterator),
                    generation=capture_generation,
                )
                last_level_log_s = self.clock.now()
                while True:
                    if level_log_interval_s > 0.0:
                        elapsed_s = max(0.0, self.clock.now() - last_level_log_s)
                        timeout_s = max(0.0, level_log_interval_s - elapsed_s)
                        done, _pending = await asyncio.wait(
                            {pending_frame},
                            timeout=timeout_s,
                        )
                        if not done:
                            self._log_level(interval_stats, source=source)
                            level_logged = True
                            interval_stats.reset()
                            last_level_log_s = self.clock.now()
                            continue
                    else:
                        await asyncio.wait({pending_frame})

                    try:
                        frame = pending_frame.result()
                    except StopAsyncIteration:
                        pending_frame = None
                        break

                    interval_stats.add_frame(frame)
                    total_stats.add_frame(frame)
                    await self.meter_sink(
                        self._meter_level_from_frame(frame),
                        request.meter_callback,
                        capture_generation,
                    )
                    pending_frame = runtime.create_frame_task(
                        anext(frame_iterator),
                        generation=capture_generation,
                    )

                    if level_log_interval_s <= 0.0 or (
                        self.clock.now() - last_level_log_s >= level_log_interval_s
                    ):
                        self._log_level(interval_stats, source=source)
                        level_logged = True
                        interval_stats.reset()
                        last_level_log_s = self.clock.now()
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

                if source is not None and interval_stats.frames > 0:
                    self._log_level(interval_stats, source=source)
                    level_logged = True
                elif source is not None and not level_logged:
                    self._log_level(interval_stats, source=source)

                self._log_end(
                    opened=opened,
                    stats=total_stats,
                    source=source,
                    exception=end_exception,
                )
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
    def _format_route_log(observation: MicrophoneTestRouteObservation) -> str:
        return (
            "[MicTest] route "
            f"saved_host_api={_log_value(observation.saved_host_api)} "
            f"actual_host_api={_log_value(observation.actual_host_api)} "
            f"requested_device={_log_value(observation.requested_device)} "
            f"hostapi_index={_log_value(observation.hostapi_index)} "
            f"resolved_device_idx={_log_value(observation.resolved_device_idx)} "
            f"resolved_device_name={_log_value(observation.resolved_device_name)} "
            "resolution_exception_class="
            f"{_log_value(observation.resolution_exception_class)} "
            "resolution_exception_message="
            f"{_log_value(observation.resolution_exception_message)}"
        )

    @staticmethod
    def _source_value(
        source: object | None,
        attr: str,
        fallback: object,
    ) -> object:
        if source is None:
            return fallback
        try:
            return getattr(source, attr, fallback)
        except Exception:
            return fallback

    @staticmethod
    def _source_int(
        source: object | None,
        attr: str,
        fallback: int | None,
    ) -> int | None:
        if source is None:
            return fallback
        try:
            value = getattr(source, attr, fallback)
            if value is None:
                return None
            return int(value)
        except Exception:
            return fallback

    def _log_open(
        self,
        *,
        attempted: bool,
        opened: bool,
        requested_channels: int | None,
        source: object | None,
        observation: MicrophoneTestRouteObservation,
        exception: BaseException | None = None,
    ) -> None:
        opened_channels = self._source_int(source, "opened_channels", None)
        frame_channels = self._source_int(
            source,
            "frame_channels",
            opened_channels,
        )
        actual_sample_rate_hz = self._source_value(
            source,
            "actual_sample_rate_hz",
            None,
        )
        self.log_sink(
            "[MicTest] open "
            f"attempted={attempted} "
            f"opened={opened} "
            f"requested_channels={_log_value(requested_channels)} "
            f"opened_channels={_log_value(opened_channels)} "
            f"frame_channels={_log_value(frame_channels)} "
            "requested_sample_rate_hz=None "
            f"actual_sample_rate_hz={_log_value(actual_sample_rate_hz)} "
            f"wasapi_auto_convert={observation.wasapi_auto_convert} "
            f"wasapi_exclusive={observation.wasapi_exclusive} "
            "exception_class="
            f"{_log_value(type(exception).__name__ if exception else None)} "
            "exception_message="
            f"{_log_value(str(exception) if exception else None)}"
        )

    def _log_level(
        self,
        stats: _LevelStats,
        *,
        source: object | None,
    ) -> None:
        self.log_sink(
            "[MicTest] level "
            f"rms_db={stats.rms_db:.1f} "
            f"peak_db={stats.peak_db:.1f} "
            f"zero_ratio={stats.zero_ratio:.3f} "
            f"frames={stats.frames} "
            f"audio_ms={stats.audio_ms:.1f} "
            f"queue_drops={self._source_int(source, 'queue_drop_count', 0)} "
            "callback_statuses="
            f"{self._source_int(source, 'callback_status_count', 0)}"
        )

    def _log_end(
        self,
        *,
        opened: bool,
        stats: _LevelStats,
        source: object | None,
        exception: BaseException | None,
    ) -> None:
        self.log_sink(
            "[MicTest] end "
            f"opened={opened} "
            f"frames_total={stats.frames} "
            f"audio_ms_total={stats.audio_ms:.1f} "
            f"rms_db_total={stats.rms_db:.1f} "
            f"peak_db_max={stats.peak_db:.1f} "
            f"zero_ratio_total={stats.zero_ratio:.3f} "
            f"queue_drops={self._source_int(source, 'queue_drop_count', 0)} "
            "callback_statuses="
            f"{self._source_int(source, 'callback_status_count', 0)} "
            "exception_class="
            f"{_log_value(type(exception).__name__ if exception else None)} "
            "exception_message="
            f"{_log_value(str(exception) if exception else None)}"
        )

    @staticmethod
    def _raise_cleanup_failures(failures: list[Exception]) -> None:
        if not failures:
            return
        if len(failures) == 1:
            raise failures[0]
        raise ExceptionGroup("Microphone test capture cleanup failed", failures)


__all__ = ["MicrophoneTestCaptureAdapter"]
