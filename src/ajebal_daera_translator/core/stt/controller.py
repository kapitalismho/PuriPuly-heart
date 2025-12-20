from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)

from ajebal_daera_translator.core.audio.format import float32_to_pcm16le_bytes
from ajebal_daera_translator.core.audio.ring_buffer import RingBufferF32
from ajebal_daera_translator.core.clock import Clock, SystemClock
from ajebal_daera_translator.core.stt.backend import STTBackend, STTBackendSession
from ajebal_daera_translator.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, VadEvent
from ajebal_daera_translator.domain.events import (
    STTErrorEvent,
    STTFinalEvent,
    STTPartialEvent,
    STTSessionState,
    STTSessionStateEvent,
)
from ajebal_daera_translator.domain.models import Transcript


@dataclass(slots=True)
class ManagedSTTProvider:
    backend: STTBackend
    sample_rate_hz: int
    clock: Clock = SystemClock()
    reset_deadline_s: float = 90.0
    drain_timeout_s: float = 2.0
    bridging_ms: int = 500

    _state: STTSessionState = STTSessionState.DISCONNECTED
    _active_session: STTBackendSession | None = None
    _session_started_at: float | None = None
    _consumer_task: asyncio.Task[None] | None = None
    _draining: set[asyncio.Task[None]] = field(default_factory=set)
    _events: asyncio.Queue = field(default_factory=asyncio.Queue)

    _active_utterance_id: UUID | None = None
    _pending_final_utterance_id: UUID | None = None
    _audio_ring: RingBufferF32 | None = None

    def __post_init__(self) -> None:
        if self.sample_rate_hz not in (8000, 16000):
            raise ValueError("sample_rate_hz must be 8000 or 16000")
        if self.reset_deadline_s <= 0:
            raise ValueError("reset_deadline_s must be > 0")
        if self.drain_timeout_s <= 0:
            raise ValueError("drain_timeout_s must be > 0")
        if self.bridging_ms <= 0:
            raise ValueError("bridging_ms must be > 0")

        capacity_samples = int(self.sample_rate_hz * (self.bridging_ms / 1000.0))
        self._audio_ring = RingBufferF32(capacity_samples=capacity_samples)

    @property
    def state(self) -> STTSessionState:
        return self._state

    async def close(self) -> None:
        await self._set_state(STTSessionState.DRAINING if self._active_session else STTSessionState.DISCONNECTED)

        if self._consumer_task:
            self._consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._consumer_task
            self._consumer_task = None

        if self._active_session:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._active_session.close()
            self._active_session = None

        if self._draining:
            for task in list(self._draining):
                task.cancel()
            await asyncio.gather(*self._draining, return_exceptions=True)
            self._draining.clear()

        self._session_started_at = None
        await self._set_state(STTSessionState.DISCONNECTED)

    async def handle_vad_event(self, event: VadEvent) -> None:
        if isinstance(event, SpeechStart):
            await self._on_speech_start(event)
        elif isinstance(event, SpeechChunk):
            await self._on_speech_chunk(event)
        elif isinstance(event, SpeechEnd):
            await self._on_speech_end(event)
        else:
            raise TypeError(f"Unknown VadEvent: {type(event)}")

    async def events(self) -> AsyncIterator[object]:
        while True:
            item = await self._events.get()
            yield item

    async def _on_speech_start(self, event: SpeechStart) -> None:
        self._active_utterance_id = event.utterance_id
        self._pending_final_utterance_id = None

        await self._ensure_session()
        await self._maybe_reset(is_speaking=True)

        await self._send_audio(event.pre_roll)
        await self._send_audio(event.chunk)

    async def _on_speech_chunk(self, event: SpeechChunk) -> None:
        self._active_utterance_id = event.utterance_id
        await self._ensure_session()
        await self._maybe_reset(is_speaking=True)
        await self._send_audio(event.chunk)

    async def _on_speech_end(self, event: SpeechEnd) -> None:
        if self._active_utterance_id == event.utterance_id:
            self._active_utterance_id = None
        self._pending_final_utterance_id = event.utterance_id

        # Send 800ms of silence to trigger server endpointing (600ms + 200ms safety margin)
        if self._active_session is not None:
            silence_samples = int(self.sample_rate_hz * 0.8)
            silence = np.zeros(silence_samples, dtype=np.float32)
            pcm = float32_to_pcm16le_bytes(silence)
            await self._active_session.send_audio(pcm)
            logger.info(f"[STT] Trailing silence sent ({silence_samples} samples) for id={str(event.utterance_id)[:8]}")

        await self._maybe_reset(is_speaking=False)

    async def _send_audio(self, samples_f32: np.ndarray) -> None:
        samples_f32 = np.asarray(samples_f32, dtype=np.float32).reshape(-1)
        self._audio_ring.append(samples_f32)  # type: ignore[union-attr]
        pcm = float32_to_pcm16le_bytes(samples_f32)
        if self._active_session is None:
            raise RuntimeError("STT session is not active")
        await self._active_session.send_audio(pcm)

    async def _ensure_session(self) -> None:
        if self._active_session is not None:
            return

        logger.info("[STT] Opening new session...")
        try:
            session = await self.backend.open_session()
        except Exception as exc:
            logger.error(f"[STT] Failed to open session: {exc}")
            await self._events.put(STTErrorEvent(f"Failed to open STT session: {exc}"))
            raise

        self._active_session = session
        self._session_started_at = self.clock.now()
        self._consumer_task = asyncio.create_task(self._consume_session_events(session))
        await self._set_state(STTSessionState.STREAMING)
        logger.info(f"[STT] Session opened (reset_deadline={self.reset_deadline_s}s)")

    async def _maybe_reset(self, *, is_speaking: bool) -> None:
        if self._active_session is None or self._session_started_at is None:
            return

        elapsed = self.clock.now() - self._session_started_at
        if elapsed < self.reset_deadline_s:
            return

        logger.warning(f"[STT] Session exceeded {self.reset_deadline_s}s (elapsed={elapsed:.1f}s, speaking={is_speaking})")
        if is_speaking:
            await self._reset_with_bridging()
        else:
            await self._reset_on_silence()

    async def _reset_with_bridging(self) -> None:
        logger.info("[STT] BRIDGING: Resetting session while speaking...")
        old_session = self._active_session
        old_consumer = self._consumer_task

        bridging_audio = self._audio_ring.get_last_samples(self._audio_ring.capacity_samples)  # type: ignore[union-attr]
        bridging_ms = len(bridging_audio) / self.sample_rate_hz * 1000

        logger.info(f"[STT] BRIDGING: Opening new session with {bridging_ms:.0f}ms audio buffer")
        new_session = await self.backend.open_session()
        self._active_session = new_session
        self._session_started_at = self.clock.now()
        self._consumer_task = asyncio.create_task(self._consume_session_events(new_session))

        await self._set_state(STTSessionState.STREAMING)

        await new_session.send_audio(float32_to_pcm16le_bytes(bridging_audio))
        logger.info("[STT] BRIDGING: New session ready, bridging audio sent")

        if old_session and old_consumer:
            logger.info("[STT] BRIDGING: Starting drain of old session in background")
            self._draining.add(asyncio.create_task(self._drain_and_close(old_session, old_consumer)))

    async def _reset_on_silence(self) -> None:
        if self._active_session is None or self._consumer_task is None:
            return

        logger.info("[STT] SILENCE RESET: Closing session during silence...")
        old_session = self._active_session
        old_consumer = self._consumer_task
        self._active_session = None
        self._consumer_task = None
        self._session_started_at = None

        await self._set_state(STTSessionState.DRAINING)
        await self._drain_and_close(old_session, old_consumer)
        await self._set_state(STTSessionState.DISCONNECTED)
        logger.info("[STT] SILENCE RESET: Session closed, will reconnect on next speech")

    async def _drain_and_close(self, session: STTBackendSession, consumer_task: asyncio.Task[None]) -> None:
        logger.debug(f"[STT] DRAIN: Starting drain (timeout={self.drain_timeout_s}s)...")
        with contextlib.suppress(Exception):
            await session.stop()

        try:
            await asyncio.wait_for(consumer_task, timeout=self.drain_timeout_s)
            logger.debug("[STT] DRAIN: Consumer task completed normally")
        except asyncio.TimeoutError:
            logger.warning(f"[STT] DRAIN: Timeout after {self.drain_timeout_s}s, cancelling consumer task")
            consumer_task.cancel()
            with contextlib.suppress(Exception):
                await consumer_task

        with contextlib.suppress(Exception):
            await session.close()
        logger.debug("[STT] DRAIN: Session closed")

    async def _consume_session_events(self, session: STTBackendSession) -> None:
        try:
            async for ev in session.events():
                utterance_id = self._active_utterance_id or self._pending_final_utterance_id
                if utterance_id is None:
                    continue

                transcript = Transcript(
                    utterance_id=utterance_id,
                    text=ev.text,
                    is_final=ev.is_final,
                    created_at=self.clock.now(),
                )
                if ev.is_final:
                    await self._events.put(STTFinalEvent(utterance_id, transcript))
                    if self._pending_final_utterance_id == utterance_id and self._active_utterance_id is None:
                        self._pending_final_utterance_id = None
                else:
                    await self._events.put(STTPartialEvent(utterance_id, transcript))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._events.put(STTErrorEvent(f"STT session error: {exc}"))

    async def _set_state(self, state: STTSessionState) -> None:
        if self._state == state:
            return
        old_state = self._state
        self._state = state
        logger.info(f"[STT] State: {old_state.name} -> {state.name}")
        await self._events.put(STTSessionStateEvent(state))


import contextlib  # placed at bottom to keep the main logic compact
