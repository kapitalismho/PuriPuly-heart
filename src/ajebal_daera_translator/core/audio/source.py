from __future__ import annotations

import asyncio
import logging
import queue
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

import janus
import numpy as np

from ajebal_daera_translator.core.audio.format import AudioFrameF32

logger = logging.getLogger(__name__)


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
    max_queue_frames: int = 64

    _queue: janus.Queue[np.ndarray | None] = field(init=False, repr=False)
    _stream: object = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False)
    _actual_sample_rate_hz: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0 or None")
        if self.channels <= 0:
            raise ValueError("channels must be > 0")
        if self.max_queue_frames <= 0:
            raise ValueError("max_queue_frames must be > 0")

        import sounddevice as sd  # type: ignore

        self._queue = janus.Queue(maxsize=self.max_queue_frames)

        def _callback(indata, _frames, _time, status):  # called from PortAudio thread
            if self._closed:
                return
            if status:
                logger.warning("sounddevice input status: %s", status)

            try:
                self._queue.sync_q.put_nowait(np.asarray(indata, dtype=np.float32).copy())
            except queue.Full:
                # Drop if the asyncio consumer is too slow; better than blocking audio thread.
                return

        stream = sd.InputStream(
            samplerate=self.sample_rate_hz,  # None = use device default
            channels=self.channels,
            dtype="float32",
            callback=_callback,
            device=self.device,
            blocksize=self.blocksize or 0,
        )
        stream.start()
        self._stream = stream
        self._actual_sample_rate_hz = int(stream.samplerate)

    async def frames(self) -> AsyncIterator[AudioFrameF32]:
        while True:
            item = await self._queue.async_q.get()
            if item is None:
                return
            yield AudioFrameF32(samples=item, sample_rate_hz=self._actual_sample_rate_hz)

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
                if hostapi_index is None or int(devices[idx].get("hostapi", -1) or -1) == hostapi_index:
                    return idx

    if hostapi_index is not None and not device:
        default_input = hostapis[hostapi_index].get("default_input_device")
        if isinstance(default_input, int) and default_input >= 0:
            return default_input

    for idx, info in enumerate(devices):
        if int(info.get("max_input_channels", 0) or 0) <= 0:
            continue
        if hostapi_index is not None and int(info.get("hostapi", -1) or -1) != hostapi_index:
            continue
        if device:
            name = str(info.get("name", "") or "")
            if name.lower() != device.lower():
                continue
        return idx

    return None


import contextlib  # keep main logic compact
