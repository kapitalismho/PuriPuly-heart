from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable
from uuid import UUID

import numpy as np

from puripuly_heart.config.settings import STTProviderName

logger = logging.getLogger(__name__)
MANAGED_STT_SAMPLE_RATE_HZ = 16000
PENDING_FINAL_QUEUE_WARN_SIZE = 8
STT_FINALIZATION_LAG_AGE_MS = 1500
STT_FINALIZATION_LAG_QUEUE_SIZE = 2

from puripuly_heart.core.audio.diagnostics import AudioFaultProfile, normalize_audio_fault_profile
from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes
from puripuly_heart.core.audio.ring_buffer import RingBufferF32
from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.runtime_logging import SessionLoggingMode, SessionRuntimeLoggingService
from puripuly_heart.core.stt.backend import (
    STTBackend,
    STTBackendFloat32Session,
    STTBackendSession,
)
from puripuly_heart.core.stt.local_qwen_hallucination import (
    is_known_local_qwen_hallucination,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart, VadEvent
from puripuly_heart.domain.events import (
    STTErrorEvent,
    STTFinalEvent,
    STTPartialEvent,
    STTSessionState,
    STTSessionStateEvent,
)
from puripuly_heart.domain.models import ChannelId, Transcript


@dataclass(frozen=True, slots=True)
class FinalTranscriptSuppressedNotification:
    utterance_id: UUID
    channel: ChannelId
    stt_provider_name: STTProviderName


@dataclass(slots=True)
class ManagedSTTProvider:
    backend: STTBackend
    sample_rate_hz: int
    stt_provider_name: STTProviderName | None = None
    channel: ChannelId = "self"
    clock: Clock = SystemClock()
    reset_deadline_s: float = 180.0
    drain_timeout_s: float = 1.5
    bridging_ms: int = 500
    finalize_grace_s: float = 0.2
    connect_attempts: int = 3
    connect_retry_base_s: float = 0.8
    connect_retry_max_s: float = 6.0
    reconnect_window_s: float = 20.0
    on_terminal_failure: Callable[[Exception], Awaitable[None] | None] | None = None
    on_final_transcript_suppressed: (
        Callable[[FinalTranscriptSuppressedNotification], Awaitable[None] | None] | None
    ) = None
    runtime_logging: SessionRuntimeLoggingService | None = None
    stt_input_fault_profile_provider: Callable[[], AudioFaultProfile | str | None] | None = None

    _state: STTSessionState = STTSessionState.DISCONNECTED
    _active_session: STTBackendSession | None = None
    _session_started_at: float | None = None
    _consumer_task: asyncio.Task[None] | None = None
    _draining: set[asyncio.Task[None]] = field(default_factory=set)
    _events: asyncio.Queue = field(default_factory=asyncio.Queue)

    _active_utterance_id: UUID | None = None
    _pending_final_utterance_ids: deque[UUID] = field(default_factory=deque)
    _pending_final_utterance_times: dict[UUID, float] = field(default_factory=dict)
    _audio_ring: RingBufferF32 | None = None
    _reset_timer: asyncio.Task[None] | None = None
    _last_speech_end_time: float | None = None
    _diagnostic_chunk_count: int = 0
    _diagnostic_sample_count: int = 0
    _diagnostic_sum_squares: float = 0.0
    _diagnostic_peak: float = 0.0
    _diagnostic_zero_count: int = 0
    _stt_fault_logged_for_utterance: bool = False

    def __post_init__(self) -> None:
        if self.channel not in ("self", "peer"):
            raise ValueError("channel must be 'self' or 'peer'")
        if self.stt_provider_name is not None and not isinstance(
            self.stt_provider_name,
            STTProviderName,
        ):
            self.stt_provider_name = STTProviderName(self.stt_provider_name)
        if self.sample_rate_hz != MANAGED_STT_SAMPLE_RATE_HZ:
            raise ValueError(f"sample_rate_hz must be {MANAGED_STT_SAMPLE_RATE_HZ}")
        if self.reset_deadline_s <= 0:
            raise ValueError("reset_deadline_s must be > 0")
        if self.drain_timeout_s <= 0:
            raise ValueError("drain_timeout_s must be > 0")
        if self.bridging_ms <= 0:
            raise ValueError("bridging_ms must be > 0")
        if self.connect_attempts <= 0:
            raise ValueError("connect_attempts must be > 0")
        if self.connect_retry_base_s <= 0:
            raise ValueError("connect_retry_base_s must be > 0")
        if self.connect_retry_max_s <= 0:
            raise ValueError("connect_retry_max_s must be > 0")

        capacity_samples = int(self.sample_rate_hz * (self.bridging_ms / 1000.0))
        self._audio_ring = RingBufferF32(capacity_samples=capacity_samples)

    @property
    def state(self) -> STTSessionState:
        return self._state

    @staticmethod
    def _format_log_message(message: str, *args: object) -> str:
        return message % args if args else message

    def _emit_basic(
        self,
        message: str,
        *args: object,
        level: int = logging.INFO,
        fallback_level: int | None = None,
    ) -> None:
        formatted = self._format_log_message(message, *args)
        if self.runtime_logging is not None:
            self.runtime_logging.emit_basic(formatted, level=level)
            return
        logger.log(level if fallback_level is None else fallback_level, formatted)

    def _emit_detailed(
        self,
        message: str,
        *args: object,
        level: int = logging.INFO,
        fallback_level: int | None = None,
    ) -> None:
        formatted = self._format_log_message(message, *args)
        if self.runtime_logging is not None:
            self.runtime_logging.emit_detailed(formatted, level=level)
        _ = fallback_level

    def _emit_audio_diag_detailed(
        self,
        message: str,
        *args: object,
        level: int = logging.INFO,
        fallback_level: int | None = None,
    ) -> None:
        with contextlib.suppress(Exception):
            self._emit_detailed(
                message,
                *args,
                level=level,
                fallback_level=fallback_level,
            )

    def _log_session_connected(self, *, attempts: int) -> None:
        retries = max(0, attempts - 1)
        if retries == 0:
            self._emit_basic("[STT] Session connected")
            return
        suffix = "retry" if retries == 1 else "retries"
        self._emit_basic(f"[STT] Session connected after {retries} {suffix}")

    async def close(self) -> None:
        await self._set_state(
            STTSessionState.DRAINING if self._active_session else STTSessionState.DISCONNECTED
        )

        if self._reset_timer:
            self._reset_timer.cancel()
            self._reset_timer = None

        if self._active_session and self._consumer_task:
            await self._drain_and_close(
                self._active_session, self._consumer_task, allow_finalize=True
            )
        elif self._consumer_task:
            self._consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._consumer_task
        elif self._active_session:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._active_session.close()

        self._consumer_task = None
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

    async def warmup(self) -> None:
        """Pre-establish STT session for faster first response."""
        if await self._ensure_session():
            self._emit_detailed("[STT] Session pre-warmed", fallback_level=logging.INFO)

    async def _on_speech_start(self, event: SpeechStart) -> None:
        self._active_utterance_id = event.utterance_id
        self._diagnostic_chunk_count = 0
        self._diagnostic_sample_count = 0
        self._diagnostic_sum_squares = 0.0
        self._diagnostic_peak = 0.0
        self._diagnostic_zero_count = 0
        self._stt_fault_logged_for_utterance = False

        if not await self._ensure_session():
            return

        await self._send_audio(event.pre_roll)
        await self._send_audio(event.chunk)

    async def _on_speech_chunk(self, event: SpeechChunk) -> None:
        self._active_utterance_id = event.utterance_id
        if not await self._ensure_session():
            return
        await self._send_audio(event.chunk)

    async def _on_speech_end(self, event: SpeechEnd) -> None:
        if self._active_utterance_id == event.utterance_id:
            self._active_utterance_id = None
        self._last_speech_end_time = self.clock.now()

        # Delegate end-of-speech handling to the backend (silence + finalize etc.)
        if self._active_session is not None:
            ended_at = self.clock.now()
            self._pending_final_utterance_ids.append(event.utterance_id)
            self._pending_final_utterance_times[event.utterance_id] = ended_at
            if len(self._pending_final_utterance_ids) > PENDING_FINAL_QUEUE_WARN_SIZE:
                self._emit_basic(
                    "[STT] Pending final queue size is unexpectedly high: %s",
                    len(self._pending_final_utterance_ids),
                    level=logging.WARNING,
                    fallback_level=logging.WARNING,
                )
            self._emit_detailed(
                "[STT] Speech end handling for id=%s (trailing_silence_ms=%s)",
                str(event.utterance_id)[:8],
                event.trailing_silence_ms,
                fallback_level=logging.INFO,
            )
            self._emit_stt_input_diagnostics(event.utterance_id, finalize=True)
            await self._active_session.on_speech_end(trailing_silence_ms=event.trailing_silence_ms)

    async def _send_audio(self, samples_f32: np.ndarray) -> None:
        samples_f32 = np.asarray(samples_f32, dtype=np.float32).reshape(-1)
        if samples_f32.size == 0:
            return
        samples_f32 = self._apply_stt_input_fault(samples_f32)
        self._record_stt_input_diagnostics(samples_f32)
        self._audio_ring.append(samples_f32)  # type: ignore[union-attr]
        if self._active_session is None:
            raise RuntimeError("STT session is not active")
        await self._send_audio_to_session(self._active_session, samples_f32)

    def _current_stt_fault_profile(self) -> AudioFaultProfile:
        if self.stt_input_fault_profile_provider is None:
            return AudioFaultProfile.NONE
        with contextlib.suppress(Exception):
            return normalize_audio_fault_profile(self.stt_input_fault_profile_provider())
        return AudioFaultProfile.NONE

    def _apply_stt_input_fault(self, samples_f32: np.ndarray) -> np.ndarray:
        profile = self._current_stt_fault_profile()
        if profile is not AudioFaultProfile.STT_INPUT_LOW_SNR_VAD_PASS:
            return samples_f32
        with contextlib.suppress(Exception):
            flat = np.arange(samples_f32.size, dtype=np.float32)
            noise = np.sin(flat * np.float32(12.9898)) * np.float32(0.003)
            transformed = (samples_f32 * np.float32(0.01)) + noise.astype(np.float32)
            if not self._stt_fault_logged_for_utterance:
                self._stt_fault_logged_for_utterance = True
                self._emit_audio_diag_detailed(
                    "[AudioDiag][STTFault][%s] profile=%s applies_after_vad=True",
                    self.channel,
                    profile.value,
                )
            return transformed.astype(np.float32)
        return samples_f32

    def _record_stt_input_diagnostics(self, samples_f32: np.ndarray) -> None:
        if (
            self.runtime_logging is None
            or self.runtime_logging.mode is not SessionLoggingMode.DETAILED
        ):
            return
        with contextlib.suppress(Exception):
            samples = np.asarray(samples_f32, dtype=np.float32).reshape(-1)
            if samples.size == 0:
                return
            sample_count = int(samples.size)
            sum_squares = float(np.sum(np.square(samples)))
            peak = float(np.max(np.abs(samples)))
            zero_count = int(np.count_nonzero(np.abs(samples) < 1e-6))
            self._diagnostic_chunk_count += 1
            self._diagnostic_sample_count += sample_count
            self._diagnostic_sum_squares += sum_squares
            self._diagnostic_peak = max(self._diagnostic_peak, peak)
            self._diagnostic_zero_count += zero_count

    def _emit_stt_input_diagnostics(self, utterance_id: UUID, *, finalize: bool) -> None:
        if (
            self.runtime_logging is None
            or self.runtime_logging.mode is not SessionLoggingMode.DETAILED
        ):
            return
        with contextlib.suppress(Exception):
            if self._diagnostic_sample_count <= 0:
                return
            audio_ms = self._diagnostic_sample_count * 1000.0 / float(self.sample_rate_hz)
            rms = float(np.sqrt(self._diagnostic_sum_squares / self._diagnostic_sample_count))
            rms_db = -120.0 if rms <= 0.0 else round(float(20.0 * np.log10(max(rms, 1e-6))), 1)
            peak_db = (
                -120.0
                if self._diagnostic_peak <= 0.0
                else round(float(20.0 * np.log10(max(self._diagnostic_peak, 1e-6))), 1)
            )
            zero_ratio = self._diagnostic_zero_count / float(self._diagnostic_sample_count)
            self._emit_audio_diag_detailed(
                "[AudioDiag][STTInput][%s] utterance_id=%s chunk_count=%s audio_ms=%.1f "
                "rms_db=%.1f peak_db=%.1f zero_ratio=%.3f finalize=%s",
                self.channel,
                str(utterance_id)[:8],
                self._diagnostic_chunk_count,
                audio_ms,
                rms_db,
                peak_db,
                zero_ratio,
                finalize,
            )

    async def _send_audio_to_session(
        self, session: STTBackendSession, samples_f32: np.ndarray
    ) -> None:
        if samples_f32.size == 0:
            return
        if isinstance(session, STTBackendFloat32Session):
            await session.send_audio_f32(samples_f32)
            return

        pcm = float32_to_pcm16le_bytes(samples_f32)
        if not pcm:
            return
        await session.send_audio(pcm)

    async def _ensure_session(self) -> bool:
        if self._active_session is not None:
            return True

        await self._set_state(STTSessionState.CONNECTING)
        last_exc: Exception | None = None

        for attempt in range(1, self.connect_attempts + 1):
            self._emit_detailed(
                "[STT] Opening new session (attempt %s/%s)...",
                attempt,
                self.connect_attempts,
                fallback_level=logging.INFO,
            )
            try:
                session = await self.backend.open_session()
            except Exception as exc:
                last_exc = exc
                self._emit_detailed(
                    "[STT] Failed to open session (attempt %s/%s): %s",
                    attempt,
                    self.connect_attempts,
                    exc,
                    level=logging.WARNING,
                    fallback_level=logging.WARNING,
                )
                if attempt < self.connect_attempts:
                    delay = min(
                        self.connect_retry_base_s * (2 ** (attempt - 1)),
                        self.connect_retry_max_s,
                    )
                    self._emit_detailed(
                        "[STT] Retrying session in %.1fs",
                        delay,
                        fallback_level=logging.INFO,
                    )
                    await asyncio.sleep(delay)
                    continue
                break
            else:
                self._active_session = session
                self._session_started_at = self.clock.now()
                self._consumer_task = asyncio.create_task(self._consume_session_events(session))
                self._schedule_reset_timer()
                await self._set_state(STTSessionState.STREAMING)
                self._log_session_connected(attempts=attempt)
                self._emit_detailed(
                    "[STT] Session ready (reset_deadline=%ss)",
                    self.reset_deadline_s,
                    fallback_level=logging.INFO,
                )
                return True

        reason = str(last_exc) if last_exc is not None else "unknown error"
        self._emit_basic(
            "[STT] Failed to open session after %s attempts: %s",
            self.connect_attempts,
            reason,
            level=logging.ERROR,
            fallback_level=logging.ERROR,
        )
        await self._set_state(STTSessionState.DISCONNECTED)
        await self._events.put(
            STTErrorEvent(
                f"Failed to open STT session after {self.connect_attempts} attempts: {reason}",
                channel=self.channel,
                runtime_log_handled=True,
            )
        )
        return False

    async def _reset_with_bridging(self) -> None:
        old_session = self._active_session
        old_consumer = self._consumer_task

        bridging_audio = self._audio_ring.get_last_samples(self._audio_ring.capacity_samples)  # type: ignore[union-attr]
        bridging_ms = len(bridging_audio) / self.sample_rate_hz * 1000

        self._emit_detailed(
            "[STT] Bridging buffered audio: %.0fms",
            bridging_ms,
            fallback_level=logging.INFO,
        )
        new_session = await self.backend.open_session()
        self._active_session = new_session
        self._session_started_at = self.clock.now()
        self._consumer_task = asyncio.create_task(
            self._consume_session_events(
                new_session,
            )
        )
        self._schedule_reset_timer()

        await self._set_state(STTSessionState.STREAMING)

        await self._send_audio_to_session(new_session, bridging_audio)
        self._emit_basic("[STT] Session reset while speaking; bridged to a new session")

        if old_session and old_consumer:
            self._emit_detailed(
                "[STT] Draining replaced session in background",
                fallback_level=logging.INFO,
            )
            self._draining.add(
                asyncio.create_task(
                    self._drain_and_close(old_session, old_consumer, allow_finalize=False)
                )
            )

    async def _reset_with_reconnect(self) -> None:
        """Close current session and immediately open a new one.

        Used when the session limit is reached during silence but there was
        recent speech activity. Unlike bridging, no audio buffer is sent.
        """
        if self._active_session is None or self._consumer_task is None:
            return

        elapsed = self.clock.now() - (self._last_speech_end_time or 0)
        self._emit_detailed(
            f"[STT] RECONNECT: Session limit during silence, "
            f"last speech {elapsed:.1f}s ago, reconnecting...",
            fallback_level=logging.INFO,
        )

        old_session = self._active_session
        old_consumer = self._consumer_task

        # Open new session
        try:
            new_session = await self.backend.open_session()
        except Exception as e:
            self._emit_basic(
                f"[STT] Reconnect failed; closing until next speech: {e}",
                level=logging.ERROR,
                fallback_level=logging.ERROR,
            )
            await self._reset_on_silence()
            return

        self._active_session = new_session
        self._session_started_at = self.clock.now()
        self._consumer_task = asyncio.create_task(
            self._consume_session_events(
                new_session,
            )
        )
        self._schedule_reset_timer()

        await self._set_state(STTSessionState.STREAMING)
        self._emit_basic("[STT] Session reconnected after recent speech")

        # Drain old session with finalize (unlike bridging)
        self._draining.add(
            asyncio.create_task(
                self._drain_and_close(old_session, old_consumer, allow_finalize=True)
            )
        )

    async def _reset_on_silence(self) -> None:
        if self._active_session is None or self._consumer_task is None:
            return

        old_session = self._active_session
        old_consumer = self._consumer_task
        self._active_session = None
        self._consumer_task = None
        self._session_started_at = None

        await self._set_state(STTSessionState.DRAINING)
        await self._drain_and_close(old_session, old_consumer, allow_finalize=True)
        await self._set_state(STTSessionState.DISCONNECTED)
        self._emit_basic("[STT] Session closed after silence")

    async def _drain_and_close(
        self,
        session: STTBackendSession,
        consumer_task: asyncio.Task[None],
        *,
        allow_finalize: bool,
    ) -> None:
        self._emit_detailed(
            f"[STT] DRAIN: Starting drain (timeout={self.drain_timeout_s}s)...",
            fallback_level=logging.DEBUG,
        )
        if allow_finalize and self._should_finalize_before_stop():
            await self._finalize_before_stop(session)
        with contextlib.suppress(Exception):
            await session.stop()

        try:
            await asyncio.wait_for(consumer_task, timeout=self.drain_timeout_s)
            self._emit_detailed(
                "[STT] DRAIN: Consumer task completed normally",
                fallback_level=logging.DEBUG,
            )
        except asyncio.TimeoutError:
            self._emit_detailed(
                f"[STT] DRAIN: Timeout after {self.drain_timeout_s}s, cancelling consumer task",
                level=logging.WARNING,
                fallback_level=logging.WARNING,
            )
            consumer_task.cancel()
            with contextlib.suppress(Exception):
                await consumer_task

        with contextlib.suppress(Exception):
            await session.close()
        self._emit_detailed("[STT] DRAIN: Session closed", fallback_level=logging.DEBUG)

    def _should_finalize_before_stop(self) -> bool:
        return self._active_utterance_id is not None or bool(self._pending_final_utterance_ids)

    async def _finalize_before_stop(self, session: STTBackendSession) -> None:
        if self._active_utterance_id is not None:
            with contextlib.suppress(Exception):
                await session.on_speech_end()
        if self.finalize_grace_s <= 0:
            return
        await asyncio.sleep(self.finalize_grace_s)

    def _build_transcript(
        self,
        *,
        utterance_id: UUID,
        text: str,
        is_final: bool,
        created_at: float,
    ) -> Transcript:
        return Transcript(
            utterance_id=utterance_id,
            text=text,
            is_final=is_final,
            created_at=created_at,
            channel=self.channel,
        )

    def _drop_stale_pending_final_utterance_ids(self) -> None:
        stale_after_s = max(0.0, float(self.reconnect_window_s))
        now = self.clock.now()

        while self._pending_final_utterance_ids:
            if len(self._pending_final_utterance_ids) <= 1 and self._active_utterance_id is None:
                return

            utterance_id = self._pending_final_utterance_ids[0]
            ended_at = self._pending_final_utterance_times.get(utterance_id)
            if ended_at is None:
                return

            age_s = now - ended_at
            if age_s <= stale_after_s:
                return

            self._pending_final_utterance_ids.popleft()
            self._pending_final_utterance_times.pop(utterance_id, None)
            self._emit_detailed(
                "[STT] Dropped stale pending final id=%s age_s=%.1f",
                str(utterance_id)[:8],
                age_s,
                level=logging.WARNING,
                fallback_level=logging.WARNING,
            )

    def _should_suppress_final_transcript(self, text: str) -> bool:
        return (
            self.stt_provider_name is STTProviderName.LOCAL_QWEN
            and is_known_local_qwen_hallucination(text)
        )

    def _maybe_emit_finalization_lag(
        self,
        *,
        utterance_id: UUID | None,
        pending_queue_size_before: int,
        text_len: int,
    ) -> None:
        if utterance_id is None:
            return
        ended_at = self._pending_final_utterance_times.get(utterance_id)
        if ended_at is None:
            if pending_queue_size_before < STT_FINALIZATION_LAG_QUEUE_SIZE:
                return
            pending_age_ms = 0
        else:
            pending_age_ms = max(0, int(round((self.clock.now() - ended_at) * 1000.0)))
            if (
                pending_age_ms < STT_FINALIZATION_LAG_AGE_MS
                and pending_queue_size_before < STT_FINALIZATION_LAG_QUEUE_SIZE
            ):
                return
        provider_name = (
            self.stt_provider_name.value if self.stt_provider_name is not None else "unknown"
        )
        active_utterance_id = (
            str(self._active_utterance_id)[:8] if self._active_utterance_id is not None else "none"
        )
        self._emit_detailed(
            "[STT][FinalizationLag] channel=%s provider=%s utterance_id=%s "
            "pending_age_ms=%s pending_queue_size=%s active_utterance_id=%s text_len=%s",
            self.channel,
            provider_name,
            str(utterance_id)[:8],
            pending_age_ms,
            pending_queue_size_before,
            active_utterance_id,
            text_len,
            fallback_level=logging.INFO,
        )

    async def _handle_suppressed_final_transcript(
        self,
        *,
        utterance_id: UUID,
    ) -> None:
        provider_name = self.stt_provider_name
        if provider_name is not STTProviderName.LOCAL_QWEN:
            return

        notification_status = "not_configured"
        if self.on_final_transcript_suppressed is not None:
            notification = FinalTranscriptSuppressedNotification(
                utterance_id=utterance_id,
                channel=self.channel,
                stt_provider_name=provider_name,
            )
            try:
                maybe_awaitable = self.on_final_transcript_suppressed(notification)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            except Exception as exc:
                notification_status = "failed"
                self._emit_detailed(
                    "[STT][%s][%s] Suppressed-final notification callback failed: %s",
                    provider_name.value,
                    self.channel,
                    exc,
                    level=logging.WARNING,
                    fallback_level=logging.WARNING,
                )
            else:
                notification_status = "emitted"

        self._emit_basic(
            "[STT][%s][%s] Known hallucination suppressed: utterance_id=%s notification=%s",
            provider_name.value,
            self.channel,
            str(utterance_id)[:8],
            notification_status,
            fallback_level=logging.INFO,
        )

    async def _consume_session_events(
        self,
        session: STTBackendSession,
    ) -> None:
        try:
            async for ev in session.events():
                if ev.is_final:
                    self._drop_stale_pending_final_utterance_ids()
                    pending_queue_size_before = len(self._pending_final_utterance_ids)
                    assigned_utterance_id = (
                        self._pending_final_utterance_ids[0]
                        if self._pending_final_utterance_ids
                        else self._active_utterance_id
                    )
                    self._maybe_emit_finalization_lag(
                        utterance_id=assigned_utterance_id,
                        pending_queue_size_before=pending_queue_size_before,
                        text_len=len(ev.text),
                    )
                    utterance_id = (
                        self._pending_final_utterance_ids.popleft()
                        if self._pending_final_utterance_ids
                        else self._active_utterance_id
                    )
                    if utterance_id is not None:
                        self._pending_final_utterance_times.pop(utterance_id, None)
                else:
                    utterance_id = self._active_utterance_id or (
                        self._pending_final_utterance_ids[0]
                        if self._pending_final_utterance_ids
                        else None
                    )
                if utterance_id is None:
                    continue
                if ev.is_final and not ev.text.strip():
                    continue
                if ev.is_final and self._should_suppress_final_transcript(ev.text):
                    await self._handle_suppressed_final_transcript(
                        utterance_id=utterance_id,
                    )
                    continue
                created_at = self.clock.now()
                transcript = self._build_transcript(
                    utterance_id=utterance_id,
                    text=ev.text,
                    is_final=ev.is_final,
                    created_at=created_at,
                )
                if ev.is_final:
                    await self._events.put(STTFinalEvent(utterance_id, transcript))
                else:
                    await self._events.put(STTPartialEvent(utterance_id, transcript))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._handle_terminal_session_failure(session, exc)

    async def _handle_terminal_session_failure(
        self,
        session: STTBackendSession,
        exc: Exception,
    ) -> None:
        is_active_session = session is self._active_session
        if is_active_session:
            self._active_session = None
            self._consumer_task = None
            self._session_started_at = None
            self._active_utterance_id = None
            self._pending_final_utterance_ids.clear()
            self._pending_final_utterance_times.clear()
            self._last_speech_end_time = None
            if self._reset_timer is not None:
                self._reset_timer.cancel()
                self._reset_timer = None
            await self._set_state(STTSessionState.DISCONNECTED)
            if self.on_terminal_failure is not None:
                maybe_awaitable = self.on_terminal_failure(exc)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable

        with contextlib.suppress(Exception):
            await session.stop()
        with contextlib.suppress(Exception):
            await session.close()

        self._emit_basic(
            "[STT] Session failed: %s",
            exc,
            level=logging.ERROR,
            fallback_level=logging.ERROR,
        )
        await self._events.put(
            STTErrorEvent(
                f"STT session error: {exc}",
                channel=self.channel,
                runtime_log_handled=True,
            )
        )

    async def _set_state(self, state: STTSessionState) -> None:
        if self._state == state:
            return
        old_state = self._state
        self._state = state
        self._emit_detailed(
            f"[STT] State: {old_state.name} -> {state.name}",
            fallback_level=logging.INFO,
        )
        await self._events.put(STTSessionStateEvent(state, channel=self.channel))

    def _has_recent_speech(self) -> bool:
        """Check if speech ended recently within the reconnect window."""
        if self._last_speech_end_time is None:
            return False
        elapsed = self.clock.now() - self._last_speech_end_time
        return elapsed < self.reconnect_window_s

    def _schedule_reset_timer(self) -> None:
        """Schedule a timer to reset the session after reset_deadline_s."""
        if self._reset_timer:
            self._reset_timer.cancel()
        self._reset_timer = asyncio.create_task(self._reset_timer_task())

    async def _reset_timer_task(self) -> None:
        """Background task that resets the session when the deadline expires."""
        try:
            await asyncio.sleep(self.reset_deadline_s)
            if self._active_session is None:
                return
            self._emit_detailed(
                f"[STT] Timer expired after {self.reset_deadline_s}s",
                fallback_level=logging.INFO,
            )
            if self._active_utterance_id is not None:
                # Speaking: reset with bridging
                await self._reset_with_bridging()
            elif self._has_recent_speech():
                # Recent speech: reconnect immediately
                await self._reset_with_reconnect()
            else:
                # Silence: close session
                await self._reset_on_silence()
        except asyncio.CancelledError:
            pass


import contextlib  # placed at bottom to keep the main logic compact
import inspect
