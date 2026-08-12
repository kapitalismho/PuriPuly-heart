from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from puripuly_heart.core.audio.format import pcm16le_bytes_to_float32
from puripuly_heart.core.runtime.gpu_asr import (
    GpuASRChannel,
    GpuASRDecodeDropped,
    SharedGpuASRRuntime,
)
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import STTBackend, STTBackendSession, STTBackendTranscriptEvent
from puripuly_heart.domain.models import FinalLanguageRun


@dataclass(slots=True)
class LocalGpuSTTBackend(STTBackend):
    runtime: SharedGpuASRRuntime
    channel: GpuASRChannel
    model_path: Path
    model_id: str
    device_id: str
    sample_rate_hz: int = 16_000
    source_mode: str = "manual"
    language_hint: str | None = None
    speech_end_clock: Callable[[], float] = field(default_factory=lambda: time.monotonic)
    _closed: bool = field(init=False, default=False, repr=False)
    _active: bool = field(init=False, default=False, repr=False)
    _lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 16_000:
            raise ValueError("sample_rate_hz must be 16000")
        self._lock = asyncio.Lock()

    async def open_session(self) -> STTBackendSession:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Local GPU STT backend is closed")
            await self.runtime.activate_channel(
                self.channel,
                model_path=self.model_path,
                model_id=self.model_id,
                device_id=self.device_id,
            )
            self._active = True
        return _LocalGpuSTTSession(backend=self)

    async def reconfigure_session_options(self, options: LocalASRSessionOptions) -> None:
        self.source_mode = options.source_mode
        self.language_hint = options.language_hint

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            active = self._active or self.channel in self.runtime.active_channels
            if active:
                await self.runtime.deactivate_channel(self.channel)
            self._active = False
            self._closed = True


@dataclass(slots=True)
class _LocalGpuSTTSession(STTBackendSession):
    backend: LocalGpuSTTBackend
    _buffer: list[np.ndarray] = field(init=False, default_factory=list, repr=False)
    _events: asyncio.Queue[STTBackendTranscriptEvent | BaseException | None] = field(
        init=False,
        default_factory=asyncio.Queue,
        repr=False,
    )
    _tasks: set[asyncio.Task[None]] = field(init=False, default_factory=set, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _stopping: bool = field(init=False, default=False, repr=False)

    async def send_audio(self, pcm16le: bytes) -> None:
        await self.send_audio_f32(pcm16le_bytes_to_float32(pcm16le))

    async def send_audio_f32(self, samples_f32: np.ndarray) -> None:
        if self._closed or self._stopping:
            return
        samples = np.asarray(samples_f32, dtype=np.float32).reshape(-1)
        if samples.size:
            self._buffer.append(samples.copy())

    async def on_speech_end(
        self,
        *,
        trailing_silence_ms: int | None = None,
        reason: SpeechBoundaryReason | None = None,
    ) -> None:
        _ = (trailing_silence_ms, reason)
        if self._closed or self._stopping:
            return
        samples = np.concatenate(self._buffer) if self._buffer else np.empty((0,), dtype=np.float32)
        self._buffer.clear()
        if not samples.size:
            await self._events.put(STTBackendTranscriptEvent(text="", is_final=True))
            return
        task = asyncio.create_task(
            self._transcribe(samples, self.backend.speech_end_clock()),
            name=f"gpu-asr-{self.backend.channel}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _transcribe(self, samples: np.ndarray, speech_end_at: float) -> None:
        try:
            result = await self.backend.runtime.submit(
                self.backend.channel,
                samples,
                speech_end_at=speech_end_at,
                language_hint=self.backend.language_hint,
            )
        except asyncio.CancelledError:
            raise
        except GpuASRDecodeDropped:
            await self._events.put(STTBackendTranscriptEvent(text="", is_final=True))
            return
        except BaseException as exc:
            await self._events.put(STTBackendTranscriptEvent(text="", is_final=True))
            await self._events.put(exc)
            return
        text = result.text.strip()
        detected_language = (result.detected_language or "").strip()
        final_language_runs = (
            (FinalLanguageRun(text=text, language=detected_language),)
            if self.backend.channel == "peer"
            and self.backend.source_mode == "auto"
            and text
            and detected_language
            else ()
        )
        await self._events.put(
            STTBackendTranscriptEvent(
                text=text,
                is_final=True,
                final_language_runs=final_language_runs,
            )
        )

    async def stop(self) -> None:
        self._stopping = True
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        await self.close()

    async def abort_for_toggle_off(self) -> None:
        self._stopping = True
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer.clear()
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._events.put_nowait(None)

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                break
            if isinstance(event, BaseException):
                raise event
            yield event


__all__ = ["LocalGpuSTTBackend"]
