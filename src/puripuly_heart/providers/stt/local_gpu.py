from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan, pcm16le_bytes_to_float32
from puripuly_heart.core.audio.ownership import SegmentTerminalOutcome
from puripuly_heart.core.runtime.gpu_asr import (
    GpuASRChannel,
    GpuASRDecodeDropped,
    GpuASRWorkExpired,
    SharedGpuASRRuntime,
)
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.speech_boundary import SpeechBoundaryReason
from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendSession,
    STTBackendTranscriptEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
)
from puripuly_heart.core.stt.local_qwen_hallucination import (
    is_known_local_qwen_hallucination,
)
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
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
    active_decode_timeout_s: float = 30.0
    _closed: bool = field(init=False, default=False, repr=False)
    _active: bool = field(init=False, default=False, repr=False)
    _lock: asyncio.Lock = field(init=False, repr=False)
    _closing: bool = field(init=False, default=False, repr=False)
    _quarantined_session: _LocalGpuSTTSession | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 16_000:
            raise ValueError("sample_rate_hz must be 16000")
        if self.active_decode_timeout_s <= 0:
            raise ValueError("active_decode_timeout_s must be > 0")
        self._lock = asyncio.Lock()

    async def open_session(self) -> STTBackendSession:
        async with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("Local GPU STT backend is closed")
            if self._quarantined_session is not None:
                raise RuntimeError("Local GPU STT resource is awaiting decode cleanup")
            if not self._active:
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
            self._closing = True
            quarantined = self._quarantined_session
        if quarantined is not None:
            await quarantined.close()
        async with self._lock:
            try:
                active = self._active or self.channel in self.runtime.active_channels
                if active:
                    await self.runtime.deactivate_channel(self.channel)
            except BaseException:
                self._closing = False
                raise
            self._active = False
            self._closed = True
            self._closing = False

    async def _quarantine(self, session: _LocalGpuSTTSession) -> None:
        async with self._lock:
            self._quarantined_session = session

    async def _release_quarantine(self, session: _LocalGpuSTTSession) -> None:
        async with self._lock:
            if self._quarantined_session is session:
                self._quarantined_session = None


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
    _scoped_events: STTProviderEventBuffer = field(
        init=False, default_factory=STTProviderEventBuffer, repr=False
    )
    _scoped_identity: STTProviderTurnIdentity | None = field(init=False, default=None, repr=False)
    _scoped_payload_sequence: int = field(init=False, default=0, repr=False)
    _scoped_sealed: bool = field(init=False, default=False, repr=False)
    _scoped_epoch_retired: bool = field(init=False, default=False, repr=False)

    async def send_audio(self, pcm16le: bytes) -> None:
        await self.send_audio_f32(pcm16le_bytes_to_float32(pcm16le))

    async def send_audio_f32(self, samples_f32: np.ndarray) -> None:
        if self._closed or self._stopping:
            return
        samples = np.asarray(samples_f32, dtype=np.float32).reshape(-1)
        if samples.size:
            self._buffer.append(samples.copy())

    async def begin_turn(self, request: STTProviderTurnRequest) -> None:
        if self._closed or self._stopping or self._scoped_epoch_retired:
            raise RuntimeError("local GPU STT session is unavailable")
        if self._scoped_identity is not None:
            raise RuntimeError("local GPU STT session already has an unresolved turn")
        self._scoped_identity = request.identity
        self._scoped_payload_sequence = 0
        self._scoped_sealed = False
        self._buffer.clear()

    async def send_turn_audio(
        self,
        identity: STTProviderTurnIdentity,
        pcm16le: bytes,
        *,
        payload_sequence: int,
        source_ranges: tuple[AudioCaptureSpan, ...],
        context_only: bool,
    ) -> None:
        _ = source_ranges, context_only
        self._require_scoped_identity(identity)
        if self._scoped_sealed:
            raise RuntimeError("local GPU STT turn is already sealed")
        if payload_sequence <= self._scoped_payload_sequence:
            raise ValueError("payload_sequence must increase")
        self._scoped_payload_sequence = payload_sequence
        await self.send_audio(pcm16le)

    async def seal_turn(
        self,
        identity: STTProviderTurnIdentity,
        *,
        sealed_content_ranges: tuple[AudioCaptureSpan, ...],
        seal_reason: str,
        observed_trailing_silence_ms: int | None,
    ) -> None:
        _ = sealed_content_ranges, seal_reason, observed_trailing_silence_ms
        self._require_scoped_identity(identity)
        if self._scoped_sealed:
            raise RuntimeError("local GPU STT turn is already sealed")
        self._scoped_sealed = True
        samples = np.concatenate(self._buffer) if self._buffer else np.empty((0,), dtype=np.float32)
        self._buffer.clear()
        if not samples.size:
            self._events.put_nowait(STTBackendTranscriptEvent(text="", is_final=True))
            self._terminalize_scoped(identity, outcome="empty")
            return
        task = asyncio.create_task(
            self._transcribe(samples, self.backend.speech_end_clock(), identity),
            name=f"gpu-asr-{self.backend.channel}-scoped",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def abort_turn(self, identity: STTProviderTurnIdentity, *, reason: str) -> None:
        self._require_scoped_identity(identity)
        self._buffer.clear()
        self._terminalize_scoped(
            identity,
            outcome="cancelled",
            failure_reason=reason,
            retire=True,
        )

    async def turn_events(self):
        async for event in self._scoped_events.events():
            yield event

    def _require_scoped_identity(self, identity: STTProviderTurnIdentity) -> None:
        if self._scoped_identity != identity:
            raise RuntimeError("unknown or retired local GPU STT turn")

    def _terminalize_scoped(
        self,
        identity: STTProviderTurnIdentity,
        *,
        outcome: SegmentTerminalOutcome,
        text: str = "",
        final_language_runs: tuple[FinalLanguageRun, ...] = (),
        failure_reason: str | None = None,
        retire: bool = False,
    ) -> None:
        if self._scoped_identity != identity:
            return
        self._scoped_events.put(
            STTProviderTurnTerminal(
                identity=identity,
                outcome=outcome,
                text=text,
                final_language_runs=final_language_runs,
                text_authority="authoritative" if outcome in ("final", "empty") else "none",
                failure_reason=failure_reason,
                epoch_disposition="retire" if retire else "reuse",
            )
        )
        self._scoped_identity = None
        self._scoped_sealed = False
        self._scoped_epoch_retired = retire

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

    async def _transcribe(
        self,
        samples: np.ndarray,
        speech_end_at: float,
        scoped_identity: STTProviderTurnIdentity | None = None,
    ) -> None:
        try:
            if scoped_identity is None:
                result = await self.backend.runtime.submit(
                    self.backend.channel,
                    samples,
                    speech_end_at=speech_end_at,
                    language_hint=self.backend.language_hint,
                )
            else:
                submit_task = asyncio.create_task(
                    self.backend.runtime.submit(
                        self.backend.channel,
                        samples,
                        speech_end_at=speech_end_at,
                        language_hint=self.backend.language_hint,
                    ),
                    name=f"gpu-asr-submit-{self.backend.channel}",
                )
                self._tasks.add(submit_task)
                submit_task.add_done_callback(self._tasks.discard)
                done, _ = await asyncio.wait(
                    {submit_task},
                    timeout=self.backend.active_decode_timeout_s,
                )
                if not done:
                    await self.backend._quarantine(self)
                    self._terminalize_scoped(
                        scoped_identity,
                        outcome="failed",
                        failure_reason="local_decode_timeout",
                        retire=True,
                    )
                    return
                result = submit_task.result()
        except asyncio.CancelledError:
            raise
        except (GpuASRDecodeDropped, GpuASRWorkExpired) as exc:
            await self._events.put(STTBackendTranscriptEvent(text="", is_final=True))
            if scoped_identity is not None:
                outcome = "expired" if isinstance(exc, GpuASRWorkExpired) else "failed"
                self._terminalize_scoped(
                    scoped_identity,
                    outcome=outcome,
                    failure_reason=type(exc).__name__,
                )
            return
        except BaseException as exc:
            await self._events.put(STTBackendTranscriptEvent(text="", is_final=True))
            await self._events.put(exc)
            if scoped_identity is not None:
                self._terminalize_scoped(
                    scoped_identity,
                    outcome="failed",
                    failure_reason=type(exc).__name__,
                    retire=True,
                )
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
        if scoped_identity is not None:
            if text and is_known_local_qwen_hallucination(text):
                self._terminalize_scoped(scoped_identity, outcome="suppressed")
            else:
                self._terminalize_scoped(
                    scoped_identity,
                    outcome="final" if text else "empty",
                    text=text,
                    final_language_runs=final_language_runs,
                )

    async def stop(self) -> None:
        self._stopping = True
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        await self.close()

    async def abort_for_toggle_off(self) -> None:
        self._stopping = True
        identity = self._scoped_identity
        if identity is not None:
            self._terminalize_scoped(
                identity,
                outcome="cancelled",
                failure_reason="toggle_off",
                retire=True,
            )
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer.clear()
        identity = self._scoped_identity
        if identity is not None:
            self._terminalize_scoped(
                identity,
                outcome="failed",
                failure_reason="session_closed",
                retire=True,
            )
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._events.put_nowait(None)
        self._scoped_events.close()
        await self.backend._release_quarantine(self)

    async def events(self) -> AsyncIterator[STTBackendTranscriptEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                break
            if isinstance(event, BaseException):
                raise event
            yield event


__all__ = ["LocalGpuSTTBackend"]
