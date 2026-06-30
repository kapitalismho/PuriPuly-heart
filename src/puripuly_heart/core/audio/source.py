from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

import janus
import numpy as np

from puripuly_heart.config.audio_host_api import normalize_input_host_api
from puripuly_heart.core.audio.format import AudioFrameF32

logger = logging.getLogger(__name__)

_CALLBACK_WARNING_MIN_INTERVAL_S = 1.0


@dataclass(frozen=True, slots=True)
class SoundDeviceInputMetadata:
    device_idx: int | None
    name: str | None
    max_input_channels: int | None
    default_samplerate: float | None
    metadata_status: str
    metadata_error: str | None = None


@dataclass(frozen=True, slots=True)
class SelfMicCaptureChannelDecision:
    device_idx: int | None
    internal_channels: int
    preferred_capture_channels: int
    metadata: SoundDeviceInputMetadata


@dataclass(frozen=True, slots=True)
class MicrophoneTestRouteObservation:
    saved_host_api: str
    actual_host_api: str
    requested_device: str
    hostapi_index: int | None
    resolved_device_idx: int | None
    resolved_device_name: str | None
    resolution_exception_class: str | None
    resolution_exception_message: str | None
    should_attempt_open: bool
    wasapi_auto_convert: bool = False
    wasapi_exclusive: bool = False


def _input_metadata_from_info(
    *,
    device_idx: int | None,
    info: object,
    ok_status: str,
    invalid_status: str,
) -> SoundDeviceInputMetadata:
    if not hasattr(info, "get"):
        return SoundDeviceInputMetadata(
            device_idx=device_idx,
            name=None,
            max_input_channels=None,
            default_samplerate=None,
            metadata_status=invalid_status,
            metadata_error="device info is not mapping-like",
        )

    get_value = info.get  # type: ignore[attr-defined]
    name_value = get_value("name", None)
    name = str(name_value) if name_value is not None else None

    try:
        max_input_channels = int(get_value("max_input_channels", 0) or 0)
    except Exception as exc:
        return SoundDeviceInputMetadata(
            device_idx=device_idx,
            name=name,
            max_input_channels=None,
            default_samplerate=None,
            metadata_status=invalid_status,
            metadata_error=str(exc),
        )

    samplerate_value = get_value("default_samplerate", None)
    if samplerate_value is None:
        default_samplerate = None
    else:
        try:
            default_samplerate = float(samplerate_value)
        except Exception:
            default_samplerate = None

    if max_input_channels <= 0:
        return SoundDeviceInputMetadata(
            device_idx=device_idx,
            name=name,
            max_input_channels=max_input_channels,
            default_samplerate=default_samplerate,
            metadata_status=invalid_status,
            metadata_error="max_input_channels is not positive",
        )

    return SoundDeviceInputMetadata(
        device_idx=device_idx,
        name=name,
        max_input_channels=max_input_channels,
        default_samplerate=default_samplerate,
        metadata_status=ok_status,
    )


def query_sounddevice_input_metadata(device_idx: int | None) -> SoundDeviceInputMetadata:
    import sounddevice as sd  # type: ignore

    if device_idx is None:
        try:
            info = sd.query_devices(kind="input")
        except Exception as exc:
            return SoundDeviceInputMetadata(
                device_idx=None,
                name=None,
                max_input_channels=None,
                default_samplerate=None,
                metadata_status="query_failed",
                metadata_error=str(exc),
            )
        return _input_metadata_from_info(
            device_idx=None,
            info=info,
            ok_status="default_resolved",
            invalid_status="unavailable",
        )

    try:
        devices = sd.query_devices()
    except Exception as exc:
        return SoundDeviceInputMetadata(
            device_idx=device_idx,
            name=None,
            max_input_channels=None,
            default_samplerate=None,
            metadata_status="query_failed",
            metadata_error=str(exc),
        )

    if device_idx < 0 or device_idx >= len(devices):
        return SoundDeviceInputMetadata(
            device_idx=device_idx,
            name=None,
            max_input_channels=None,
            default_samplerate=None,
            metadata_status="invalid",
            metadata_error=f"device index {device_idx} is out of range",
        )

    return _input_metadata_from_info(
        device_idx=device_idx,
        info=devices[device_idx],
        ok_status="ok",
        invalid_status="invalid",
    )


def determine_self_mic_capture_channels(
    *,
    device_idx: int | None,
    internal_channels: int,
    metadata: SoundDeviceInputMetadata | None = None,
) -> SelfMicCaptureChannelDecision:
    if internal_channels <= 0:
        raise ValueError("internal_channels must be > 0")

    resolved_metadata = metadata or query_sounddevice_input_metadata(device_idx)
    max_input_channels = resolved_metadata.max_input_channels
    if (
        resolved_metadata.metadata_status in {"ok", "default_resolved"}
        and max_input_channels is not None
        and max_input_channels > 0
    ):
        preferred_capture_channels = min(max_input_channels, 2)
    else:
        preferred_capture_channels = internal_channels

    return SelfMicCaptureChannelDecision(
        device_idx=device_idx,
        internal_channels=internal_channels,
        preferred_capture_channels=preferred_capture_channels,
        metadata=resolved_metadata,
    )


def _sounddevice_mapping_get(item: object, key: str, default: object = None) -> object:
    getter = getattr(item, "get", None)
    if callable(getter):
        return getter(key, default)
    return default


def _sounddevice_device_name(item: object) -> str | None:
    value = _sounddevice_mapping_get(item, "name", None)
    return str(value) if value is not None else None


def _sounddevice_input_channels(item: object) -> int:
    try:
        return int(_sounddevice_mapping_get(item, "max_input_channels", 0) or 0)
    except Exception:
        return 0


def _sounddevice_device_hostapi_index(item: object) -> int | None:
    value = _sounddevice_mapping_get(item, "hostapi", None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _microphone_test_observation(
    *,
    saved_host_api: str,
    actual_host_api: str,
    requested_device: str,
    hostapi_index: int | None,
    resolved_device_idx: int | None,
    resolved_device_name: str | None,
    should_attempt_open: bool,
    wasapi_auto_convert: bool,
    wasapi_exclusive: bool,
    resolution_exception: BaseException | None = None,
) -> MicrophoneTestRouteObservation:
    return MicrophoneTestRouteObservation(
        saved_host_api=saved_host_api,
        actual_host_api=actual_host_api,
        requested_device=requested_device,
        hostapi_index=hostapi_index,
        resolved_device_idx=resolved_device_idx,
        resolved_device_name=resolved_device_name,
        resolution_exception_class=(
            type(resolution_exception).__name__ if resolution_exception is not None else None
        ),
        resolution_exception_message=(
            str(resolution_exception) if resolution_exception is not None else None
        ),
        should_attempt_open=should_attempt_open,
        wasapi_auto_convert=wasapi_auto_convert,
        wasapi_exclusive=wasapi_exclusive,
    )


def _microphone_test_device_idx_resolves(
    devices: list[object],
    device_idx: int,
    *,
    hostapi_index: int | None,
) -> bool:
    if device_idx < 0 or device_idx >= len(devices):
        return False
    info = devices[device_idx]
    if _sounddevice_input_channels(info) <= 0:
        return False
    if hostapi_index is None:
        return True
    return _sounddevice_device_hostapi_index(info) == hostapi_index


def _resolve_microphone_test_requested_device_idx(
    devices: list[object],
    *,
    requested_device: str,
    hostapi_index: int | None,
) -> int | None:
    with contextlib.suppress(ValueError):
        idx = int(requested_device)
        if _microphone_test_device_idx_resolves(
            devices,
            idx,
            hostapi_index=hostapi_index,
        ):
            return idx

    requested_device_lower = requested_device.lower()
    for idx, info in enumerate(devices):
        if not _microphone_test_device_idx_resolves(
            devices,
            idx,
            hostapi_index=hostapi_index,
        ):
            continue
        name = _sounddevice_device_name(info) or ""
        if name.lower() == requested_device_lower:
            return idx
    return None


def observe_microphone_test_route(
    *,
    saved_host_api: str | None = "",
    requested_device: str | None = "",
) -> MicrophoneTestRouteObservation:
    """Resolve the manual microphone-test route without hidden fallback.

    ``device_idx=None`` remains an openable route only for true Auto/system-default
    testing, where both the normalized Host API and requested device are blank.
    Any explicit Host API or device miss returns a neutral observation and sets
    ``should_attempt_open`` to ``False`` for the downstream capture/logging layer.
    """

    profile = normalize_input_host_api(saved_host_api)
    saved_value = profile.saved_value
    actual_host_api = profile.actual_host_api
    requested_value = str(requested_device or "").strip()

    if not actual_host_api and not requested_value:
        return _microphone_test_observation(
            saved_host_api=saved_value,
            actual_host_api=actual_host_api,
            requested_device=requested_value,
            hostapi_index=None,
            resolved_device_idx=None,
            resolved_device_name=None,
            should_attempt_open=True,
            wasapi_auto_convert=profile.wasapi_auto_convert,
            wasapi_exclusive=profile.wasapi_exclusive,
        )

    import sounddevice as sd  # type: ignore

    hostapi_index: int | None = None
    hostapis: list[object] = []
    if actual_host_api:
        try:
            hostapis = list(sd.query_hostapis())
        except Exception as exc:
            return _microphone_test_observation(
                saved_host_api=saved_value,
                actual_host_api=actual_host_api,
                requested_device=requested_value,
                hostapi_index=None,
                resolved_device_idx=None,
                resolved_device_name=None,
                should_attempt_open=False,
                wasapi_auto_convert=profile.wasapi_auto_convert,
                wasapi_exclusive=profile.wasapi_exclusive,
                resolution_exception=exc,
            )

        for idx, item in enumerate(hostapis):
            name = str(_sounddevice_mapping_get(item, "name", "") or "")
            if name.lower() == actual_host_api.lower():
                hostapi_index = idx
                break
        if hostapi_index is None:
            return _microphone_test_observation(
                saved_host_api=saved_value,
                actual_host_api=actual_host_api,
                requested_device=requested_value,
                hostapi_index=None,
                resolved_device_idx=None,
                resolved_device_name=None,
                should_attempt_open=False,
                wasapi_auto_convert=profile.wasapi_auto_convert,
                wasapi_exclusive=profile.wasapi_exclusive,
            )

    try:
        devices = list(sd.query_devices())
    except Exception as exc:
        return _microphone_test_observation(
            saved_host_api=saved_value,
            actual_host_api=actual_host_api,
            requested_device=requested_value,
            hostapi_index=hostapi_index,
            resolved_device_idx=None,
            resolved_device_name=None,
            should_attempt_open=False,
            wasapi_auto_convert=profile.wasapi_auto_convert,
            wasapi_exclusive=profile.wasapi_exclusive,
            resolution_exception=exc,
        )

    resolved_device_idx: int | None = None
    if requested_value:
        resolved_device_idx = _resolve_microphone_test_requested_device_idx(
            devices,
            requested_device=requested_value,
            hostapi_index=hostapi_index,
        )
    elif hostapi_index is not None:
        default_input = _sounddevice_mapping_get(
            hostapis[hostapi_index],
            "default_input_device",
            None,
        )
        if type(default_input) is int and _microphone_test_device_idx_resolves(
            devices,
            default_input,
            hostapi_index=hostapi_index,
        ):
            resolved_device_idx = default_input

    resolved_device_name = (
        _sounddevice_device_name(devices[resolved_device_idx])
        if resolved_device_idx is not None
        else None
    )
    return _microphone_test_observation(
        saved_host_api=saved_value,
        actual_host_api=actual_host_api,
        requested_device=requested_value,
        hostapi_index=hostapi_index,
        resolved_device_idx=resolved_device_idx,
        resolved_device_name=resolved_device_name,
        should_attempt_open=resolved_device_idx is not None,
        wasapi_auto_convert=profile.wasapi_auto_convert,
        wasapi_exclusive=profile.wasapi_exclusive,
    )


class AudioSource(Protocol):
    async def frames(self) -> AsyncIterator[AudioFrameF32]: ...
    async def close(self) -> None: ...


@dataclass(slots=True)
class SoundDeviceAudioSource(AudioSource):
    """Audio source using sounddevice/PortAudio.

    If sample_rate_hz is None, the device's default sample rate is used.
    This is important for WASAPI which may not support arbitrary sample rates.
    """

    sample_rate_hz: int | None = None
    channels: int = 1
    device: int | str | None = None
    blocksize: int | None = None
    wasapi_auto_convert: bool = False
    wasapi_exclusive: bool = False
    max_queue_frames: int = 64

    _queue: janus.Queue[np.ndarray | None] = field(init=False, repr=False)
    _stream: object = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False)
    _actual_sample_rate_hz: int = field(init=False, repr=False)
    _opened_channels: int = field(init=False, repr=False)
    _frame_channels: int = field(init=False, repr=False)
    _callback_status_count: int = field(init=False, default=0, repr=False)
    _queue_drop_count: int = field(init=False, default=0, repr=False)
    _last_callback_status: object | None = field(init=False, default=None, repr=False)
    _last_reported_callback_status_count: int = field(init=False, default=0, repr=False)
    _last_reported_queue_drop_count: int = field(init=False, default=0, repr=False)
    _last_callback_warning_monotonic_s: float = field(init=False, default=float("-inf"), repr=False)

    def __post_init__(self) -> None:
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0 or None")
        if self.channels <= 0:
            raise ValueError("channels must be > 0")
        if self.max_queue_frames <= 0:
            raise ValueError("max_queue_frames must be > 0")

        import sounddevice as sd  # type: ignore

        needs_wasapi_settings = self.wasapi_auto_convert or self.wasapi_exclusive
        wasapi_settings = getattr(sd, "WasapiSettings", None)
        if needs_wasapi_settings and wasapi_settings is None:
            raise RuntimeError("WASAPI settings support is unavailable in sounddevice")

        self._queue = janus.Queue(maxsize=self.max_queue_frames)
        self._opened_channels = self.channels
        self._frame_channels = self.channels

        def _callback(indata, _frames, _time, status):  # called from PortAudio thread
            if self._closed:
                return
            if status:
                self._callback_status_count += 1
                self._last_callback_status = status

            try:
                samples = np.asarray(indata, dtype=np.float32).copy()
                if samples.ndim == 2 and samples.shape[-1] > 0:
                    self._frame_channels = int(samples.shape[-1])
                else:
                    self._frame_channels = self._opened_channels
                self._queue.sync_q.put_nowait(samples)
            except queue.Full:
                # Drop if the asyncio consumer is too slow; better than blocking audio thread.
                self._queue_drop_count += 1
                return

        stream_kwargs = {
            "samplerate": self.sample_rate_hz,  # None = use device default
            "channels": self.channels,
            "dtype": "float32",
            "callback": _callback,
            "device": self.device,
            "blocksize": self.blocksize or 0,
        }
        if needs_wasapi_settings:
            stream_kwargs["extra_settings"] = wasapi_settings(
                exclusive=self.wasapi_exclusive,
                auto_convert=self.wasapi_auto_convert,
            )

        stream = sd.InputStream(**stream_kwargs)
        try:
            stream.start()
            actual_sample_rate_hz = int(stream.samplerate)
        except Exception:
            with contextlib.suppress(Exception):
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()
            raise

        self._stream = stream
        self._opened_channels = self.channels
        self._actual_sample_rate_hz = actual_sample_rate_hz

    @property
    def actual_sample_rate_hz(self) -> int:
        return self._actual_sample_rate_hz

    @property
    def requested_channels(self) -> int:
        return self.channels

    @property
    def opened_channels(self) -> int:
        return self._opened_channels

    @property
    def frame_channels(self) -> int:
        return self._frame_channels

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
            frame_channels = self._opened_channels
            if item.ndim == 2 and item.shape[-1] > 0:
                frame_channels = int(item.shape[-1])
            self._frame_channels = frame_channels
            yield AudioFrameF32(
                samples=item,
                sample_rate_hz=self._actual_sample_rate_hz,
                channels=frame_channels,
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
                "[AudioDiag][Drop][self] source=sounddevice "
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

        stream = self._stream
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()

        try:
            self._queue.sync_q.put_nowait(None)
        except Exception:
            pass

        self._queue.close()
        with contextlib.suppress(Exception):
            await self._queue.wait_closed()


def resolve_sounddevice_input_device(*, host_api: str = "", device: str = "") -> int | None:
    host_api = (host_api or "").strip()
    device = (device or "").strip()
    if not host_api and not device:
        return None

    import sounddevice as sd  # type: ignore

    hostapis = sd.query_hostapis()
    devices = sd.query_devices()

    hostapi_index: int | None = None
    if host_api:
        for idx, item in enumerate(hostapis):
            name = str(item.get("name", "") or "")
            if name.lower() == host_api.lower():
                hostapi_index = idx
                break

    if device:
        with contextlib.suppress(ValueError):
            idx = int(device)
            if 0 <= idx < len(devices) and int(devices[idx].get("max_input_channels", 0) or 0) > 0:
                hostapi_value = devices[idx].get("hostapi")
                if hostapi_value is None:
                    hostapi_value = -1
                if hostapi_index is None or int(hostapi_value) == hostapi_index:
                    return idx

    if hostapi_index is not None and not device:
        default_input = hostapis[hostapi_index].get("default_input_device")
        if isinstance(default_input, int) and default_input >= 0:
            return default_input

    for idx, info in enumerate(devices):
        if int(info.get("max_input_channels", 0) or 0) <= 0:
            continue
        hostapi_value = info.get("hostapi")
        if hostapi_value is None:
            hostapi_value = -1
        if hostapi_index is not None and int(hostapi_value) != hostapi_index:
            continue
        if device:
            name = str(info.get("name", "") or "")
            if name.lower() != device.lower():
                continue
        return idx

    return None


import contextlib  # keep main logic compact
