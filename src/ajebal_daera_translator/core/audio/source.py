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
    sample_rate_hz: int
    channels: int = 1
    device: int | str | None = None
    blocksize: int | None = None
    max_queue_frames: int = 64

    _queue: janus.Queue[np.ndarray | None] = field(init=False, repr=False)
    _stream: object = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
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
            samplerate=self.sample_rate_hz,
            channels=self.channels,
            dtype="float32",
            callback=_callback,
            device=self.device,
            blocksize=self.blocksize or 0,
        )
        stream.start()
        self._stream = stream

    async def frames(self) -> AsyncIterator[AudioFrameF32]:
        while True:
            item = await self._queue.async_q.get()
            if item is None:
                return
            yield AudioFrameF32(samples=item, sample_rate_hz=self.sample_rate_hz)

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


import contextlib  # keep main logic compact
