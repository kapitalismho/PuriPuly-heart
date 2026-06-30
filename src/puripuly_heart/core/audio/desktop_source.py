from __future__ import annotations

import contextlib
import logging
import platform
import queue
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

import janus
import numpy as np

from puripuly_heart.core.audio.format import AudioFrameF32

logger = logging.getLogger(__name__)

_CALLBACK_WARNING_MIN_INTERVAL_S = 1.0


@dataclass(frozen=True, slots=True)
class DesktopLoopbackDevice:
    index: int
    name: str
    channels: int
    sample_rate_hz: int


@dataclass(frozen=True, slots=True)
class DesktopLoopbackDeviceResolution:
    device: DesktopLoopbackDevice | None
    used_default_fallback: bool


@dataclass(slots=True)
class DesktopLoopbackDeviceResolver:
    devices: Sequence[DesktopLoopbackDevice | str]
    default_device: DesktopLoopbackDevice | str | None = None

    def resolve(self, *, saved_device_name: str) -> DesktopLoopbackDevice | str | None:
        return self.resolve_with_metadata(saved_device_name=saved_device_name).device

    def resolve_with_metadata(self, *, saved_device_name: str) -> DesktopLoopbackDeviceResolution:
        if saved_device_name:
            for device in self.devices:
                if self._device_name(device) == saved_device_name:
                    return DesktopLoopbackDeviceResolution(
                        device=device, used_default_fallback=False
                    )

        return DesktopLoopbackDeviceResolution(
            device=self.default_device,
            used_default_fallback=bool(saved_device_name and self.default_device is not None),
        )

    @staticmethod
    def _device_name(device: DesktopLoopbackDevice | str) -> str:
        if isinstance(device, DesktopLoopbackDevice):
            return device.name
        return str(device)


@dataclass(slots=True)
class DesktopLoopbackAudioSource:
    device_name: str = ""
    frames_per_buffer: int = 1024
    max_queue_frames: int = 64

    _queue: janus.Queue[np.ndarray | None] = field(init=False, repr=False)
    _stream: object = field(init=False, repr=False)
    _manager: object = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False)
    _actual_sample_rate_hz: int = field(init=False, repr=False)
    _resolved_device: DesktopLoopbackDevice = field(init=False, repr=False)
    _used_default_fallback: bool = field(init=False, default=False, repr=False)
    _callback_status_count: int = field(init=False, default=0, repr=False)
    _queue_drop_count: int = field(init=False, default=0, repr=False)
    _last_callback_status: object | None = field(init=False, default=None, repr=False)
    _last_reported_callback_status_count: int = field(init=False, default=0, repr=False)
    _last_reported_queue_drop_count: int = field(init=False, default=0, repr=False)
    _last_callback_warning_monotonic_s: float = field(init=False, default=float("-inf"), repr=False)

    def __post_init__(self) -> None:
        if self.frames_per_buffer <= 0:
            raise ValueError("frames_per_buffer must be > 0")
        if self.max_queue_frames <= 0:
            raise ValueError("max_queue_frames must be > 0")

        pyaudio = _import_pyaudiowpatch()
        self._queue = janus.Queue(maxsize=self.max_queue_frames)

        manager = pyaudio.PyAudio()
        try:
            devices = _enumerate_loopback_devices(manager)
            default_device = _get_default_loopback_device(manager)
            resolution = DesktopLoopbackDeviceResolver(
                devices=devices, default_device=default_device
            ).resolve_with_metadata(saved_device_name=self.device_name)
            resolved = resolution.device
            if not isinstance(resolved, DesktopLoopbackDevice):
                raise RuntimeError("No Windows loopback output device is available")
            if resolution.used_default_fallback:
                logger.warning(
                    "Saved desktop loopback device unavailable, falling back to default output "
                    "loopback (saved=%r, resolved=%r)",
                    self.device_name,
                    resolved.name,
                )

            self._resolved_device = resolved
            self._actual_sample_rate_hz = resolved.sample_rate_hz
            self._used_default_fallback = resolution.used_default_fallback
            self._manager = manager

            continue_flag = getattr(pyaudio, "paContinue", 0)
            float32_format = getattr(pyaudio, "paFloat32")

            def _callback(in_data, _frame_count, _time_info, status_flags):
                if self._closed:
                    return (None, continue_flag)
                if status_flags:
                    self._callback_status_count += 1
                    self._last_callback_status = status_flags
                if in_data:
                    try:
                        samples = np.frombuffer(in_data, dtype=np.float32).copy()
                        self._queue.sync_q.put_nowait(samples)
                    except queue.Full:
                        self._queue_drop_count += 1
                        return (None, continue_flag)
                return (None, continue_flag)

            stream = manager.open(
                format=float32_format,
                channels=resolved.channels,
                rate=resolved.sample_rate_hz,
                input=True,
                input_device_index=resolved.index,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=_callback,
            )
            stream.start_stream()
            self._stream = stream
        except Exception:
            with contextlib.suppress(Exception):
                manager.terminate()
            raise

    @property
    def resolved_device_name(self) -> str:
        return self._resolved_device.name

    @property
    def resolved_device_index(self) -> int:
        return self._resolved_device.index

    @property
    def resolved_channels(self) -> int:
        return self._resolved_device.channels

    @property
    def actual_sample_rate_hz(self) -> int:
        return self._actual_sample_rate_hz

    @property
    def used_default_fallback(self) -> bool:
        return self._used_default_fallback

    @property
    def callback_status_count(self) -> int:
        return self._callback_status_count

    @property
    def queue_drop_count(self) -> int:
        return self._queue_drop_count

    @property
    def last_callback_status(self) -> object | None:
        return self._last_callback_status

    async def frames(self) -> AsyncIterator[AudioFrameF32]:
        while True:
            item = await self._queue.async_q.get()
            if item is None:
                return
            self._report_callback_warnings_from_consumer()
            yield AudioFrameF32(
                samples=item,
                sample_rate_hz=self._actual_sample_rate_hz,
                channels=self._resolved_device.channels,
            )

    def _report_callback_warnings_from_consumer(self) -> None:
        callback_status_count = self._callback_status_count
        queue_drop_count = self._queue_drop_count
        status_new_count = callback_status_count - self._last_reported_callback_status_count
        drop_new_count = queue_drop_count - self._last_reported_queue_drop_count
        if status_new_count <= 0 and drop_new_count <= 0:
            return

        now = time.monotonic()
        if now - self._last_callback_warning_monotonic_s < _CALLBACK_WARNING_MIN_INTERVAL_S:
            return

        self._last_callback_warning_monotonic_s = now
        self._last_reported_callback_status_count = callback_status_count
        self._last_reported_queue_drop_count = queue_drop_count
        with contextlib.suppress(Exception):
            logger.warning(
                "[AudioDiag][Drop][peer] source=desktop_loopback "
                "callback_status_total=%s callback_status_new=%s "
                "last_status=%s queue_drop_total=%s queue_drop_new=%s",
                callback_status_count,
                max(0, status_new_count),
                self._last_callback_status,
                queue_drop_count,
                max(0, drop_new_count),
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        with contextlib.suppress(Exception):
            self._stream.stop_stream()
        with contextlib.suppress(Exception):
            self._stream.close()
        with contextlib.suppress(Exception):
            self._manager.terminate()

        try:
            self._queue.sync_q.put_nowait(None)
        except Exception:
            pass

        self._queue.close()
        with contextlib.suppress(Exception):
            await self._queue.wait_closed()


def _import_pyaudiowpatch() -> Any:
    if platform.system() != "Windows":
        raise RuntimeError("Desktop loopback capture requires Windows")

    import pyaudiowpatch as pyaudio  # type: ignore

    return pyaudio


def _enumerate_loopback_devices(manager: Any) -> list[DesktopLoopbackDevice]:
    return [_coerce_device_info(info) for info in manager.get_loopback_device_info_generator()]


def _get_default_loopback_device(manager: Any) -> DesktopLoopbackDevice | None:
    with contextlib.suppress(Exception):
        return _coerce_device_info(manager.get_default_wasapi_loopback())

    with contextlib.suppress(Exception):
        default_output = manager.get_default_wasapi_device(deviceType="output")
        if default_output is None:
            return None
        analogue = manager.get_wasapi_loopback_analogue_by_dict(default_output)
        return _coerce_device_info(analogue)

    return None


def _coerce_device_info(info: Any) -> DesktopLoopbackDevice:
    if not isinstance(info, dict):
        raise TypeError("loopback device info must be a dictionary")

    index = int(info.get("index", -1))
    name = str(info.get("name", "") or "")
    channels = int(
        info.get(
            "maxInputChannels",
            info.get(
                "max_input_channels",
                info.get("maxOutputChannels", info.get("max_output_channels", 0)),
            ),
        )
        or 0
    )
    channels = max(channels, 1)
    sample_rate_raw = info.get("defaultSampleRate", info.get("default_sample_rate", 48000.0))
    sample_rate_hz = int(round(float(sample_rate_raw or 48000.0)))

    if index < 0 or not name:
        raise ValueError("loopback device info is missing a usable index or name")

    return DesktopLoopbackDevice(
        index=index,
        name=name,
        channels=channels,
        sample_rate_hz=sample_rate_hz,
    )
