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

from puripuly_heart.core.audio.diagnostics import AudioFaultProfile, normalize_audio_fault_profile
from puripuly_heart.core.audio.format import float32_to_pcm16le_bytes
from puripuly_heart.core.audio.ring_buffer import RingBufferF32
from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.error_messages import format_error_report_for_log, stt_failure_report
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.runtime_logging import SessionLoggingMode, SessionRuntimeLoggingService
from puripuly_heart.core.stt.backend import (
    LocalASRReconfigurableBackend,
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
from puripuly_heart.domain.models import ChannelId, FinalLanguageRun, Transcript


@dataclass(frozen=True, slots=True)
class FinalTranscriptSuppressedNotification:
    utterance_id: UUID
    channel: ChannelId
    stt_provider_name: STTProviderName


@dataclass(frozen=True, slots=True)
class _EventIngressBarrier:
    reached: asyncio.Event


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
    event_ingress_observer: Callable[..., object] | None = None

    _state: STTSessionState = STTSessionState.DISCONNECTED
    _active_session: STTBackendSession | None = None
    _session_started_at: float | None = None
    _consumer_task: asyncio.Task[None] | None = None
    _draining: set[asyncio.Task[None]] = field(default_factory=set)
    _events: asyncio.Queue = field(default_factory=asyncio.Queue)
    _event_enqueued_at: deque[float] = field(default_factory=deque)
    _session_open_lock: asyncio.Lock = field(init=False, repr=False)

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
    _backend_closed: bool = False
    _closing: bool = False
    _pending_session_options: LocalASRSessionOptions | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _publication_generation: int = field(init=False, default=0, repr=False)

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

        self._session_open_lock = asyncio.Lock()
        capacity_samples = int(self.sample_rate_hz * (self.bridging_ms / 1000.0))
        self._audio_ring = RingBufferF32(capacity_samples=capacity_samples)

    def _provider_label(self) -> str:
        if self.stt_provider_name is None:
            return "stt"
        return self.stt_provider_name.value

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
        self._closing = True
        try:
            await self._set_state(
                STTSessionState.DRAINING if self._active_session else STTSessionState.DISCONNECTED
            )

            if self._reset_timer:
                self._reset_timer.cancel()

            if self._active_session and self._consumer_task:
                await self._drain_and_close(
                    self._active_session, self._consumer_task, allow_finalize=True
                )
            elif self._consumer_task:
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling():
                        raise
                except Exception:
                    pass
            elif self._active_session:
                await self._close_session_for_cleanup(self._active_session)
        finally:
            await self._complete_close_cleanup()

    async def abort_for_toggle_off(self) -> None:
        self._closing = True
        self._publication_generation += 1
        discarded_audio_samples = (
            int(self._audio_ring.get_last_samples(self._audio_ring.capacity_samples).size)
            if self._audio_ring is not None
            else 0
        )
        active_decode = self._active_session is not None
        self._emit_detailed(
            "[STT][Abort] channel=%s provider=%s discarded_audio_samples=%s "
            "pending_finals=%s active_decode=%s",
            self.channel,
            self._provider_label(),
            discarded_audio_samples,
            len(self._pending_final_utterance_ids),
            active_decode,
            fallback_level=logging.INFO,
        )
        reset_timer = self._reset_timer
        self._reset_timer = None
        if reset_timer is not None:
            reset_timer.cancel()
            await asyncio.gather(reset_timer, return_exceptions=True)

        session = self._active_session
        consumer_task = self._consumer_task
        self._active_session = None
        self._consumer_task = None
        self._session_started_at = None
        self._active_utterance_id = None
        self._pending_final_utterance_ids.clear()
        self._pending_final_utterance_times.clear()
        self._last_speech_end_time = None
        if self._audio_ring is not None:
            self._audio_ring.clear()

        if consumer_task is not None and not consumer_task.done():
            consumer_task.cancel()
        if session is not None:
            abort = getattr(session, "abort_for_toggle_off", None)
            if callable(abort):
                result = abort()
                if inspect.isawaitable(result):
                    await result
            else:
                await self._close_session_for_cleanup(session)
        if consumer_task is not None:
            await asyncio.gather(consumer_task, return_exceptions=True)

        draining = tuple(self._draining)
        for task in draining:
            if not task.done():
                task.cancel()
        if draining:
            await asyncio.gather(*draining, return_exceptions=True)
            self._draining.difference_update(draining)
        await self.discard_pending_events()
        await self._set_state(STTSessionState.DISCONNECTED)
        await self.discard_pending_events()
        self._closing = False

    async def _complete_close_cleanup(self) -> None:
        current_task = asyncio.current_task()
        cleanup_cancelled = False
        while True:
            try:
                await self._complete_close_cleanup_once()
                break
            except asyncio.CancelledError:
                if current_task is None or not current_task.cancelling():
                    raise
                cleanup_cancelled = True

        if cleanup_cancelled:
            raise asyncio.CancelledError

    async def _close_session_for_cleanup(self, session: STTBackendSession) -> None:
        try:
            await session.close()
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
        except Exception:
            pass

    async def _complete_close_cleanup_once(self) -> None:
        if self._reset_timer:
            reset_timer = self._reset_timer
            reset_timer.cancel()
            try:
                await reset_timer
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
            except Exception:
                pass
            self._reset_timer = None

        if self._active_session:
            await self._close_session_for_cleanup(self._active_session)
        if self._consumer_task and not self._consumer_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._consumer_task),
                    timeout=self.drain_timeout_s,
                )
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
            except Exception:
                pass
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
            except Exception:
                pass
        self._consumer_task = None
        self._active_session = None

        if self._draining:
            draining = tuple(self._draining)
            for task in draining:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*draining, return_exceptions=True)
            self._draining.difference_update(draining)

        self._session_started_at = None
        self._active_utterance_id = None
        self._pending_final_utterance_ids.clear()
        self._pending_final_utterance_times.clear()
        self._last_speech_end_time = None
        if self._audio_ring is not None:
            self._audio_ring.clear()
        await self._set_state(STTSessionState.DISCONNECTED)
        self._closing = False

    async def close_backend(self) -> None:
        """Close active STT session and backend-level resources once.

        ``close()`` is the toggle-off/session drain policy and intentionally leaves
        backend-level caches usable for a later toggle-on. Runtime owner shutdown
        and provider replacement call this method to propagate close to providers
        that expose backend-level resources.
        """

        await self.close()
        if self._backend_closed:
            return
        backend_close = getattr(self.backend, "close", None)
        if not callable(backend_close):
            self._backend_closed = True
            return
        result = backend_close()
        if inspect.isawaitable(result):
            await result
        self._backend_closed = True

    async def discard_pending_events(self) -> None:
        while True:
            try:
                self._events.get_nowait()
            except asyncio.QueueEmpty:
                self._event_enqueued_at.clear()
                return
            if self._event_enqueued_at:
                self._event_enqueued_at.popleft()

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
            item = await self._take_event()
            if isinstance(item, _EventIngressBarrier):
                item.reached.set()
                continue
            self._observe_event_ingress("stt_handler_start", item)
            yield item

    def event_ingress_snapshot(self) -> dict[str, object]:
        now = self.clock.now()
        oldest = self._event_enqueued_at[0] if self._event_enqueued_at else None
        return {
            "queue_depth": self._events.qsize(),
            "oldest_age_s": None if oldest is None else round(max(0.0, now - oldest), 3),
        }

    async def _publish_event(self, item: object) -> None:
        await self._events.put(item)
        if not isinstance(item, _EventIngressBarrier):
            self._event_enqueued_at.append(self.clock.now())
            self._observe_event_ingress("stt_enqueue", item)

    async def _take_event(self) -> object:
        item = await self._events.get()
        if not isinstance(item, _EventIngressBarrier) and self._event_enqueued_at:
            self._event_enqueued_at.popleft()
        return item

    def _observe_event_ingress(self, event: str, item: object) -> None:
        if self.event_ingress_observer is None:
            return
        utterance_id = getattr(item, "utterance_id", None)
        self.event_ingress_observer(
            event,
            channel=self.channel,
            event_type=type(item).__name__,
            utterance_id=None if utterance_id is None else str(utterance_id),
            **self.event_ingress_snapshot(),
        )

    @property
    def is_at_utterance_boundary(self) -> bool:
        return self._active_utterance_id is None

    async def wait_for_event_ingress_drain(self) -> None:
        reached = asyncio.Event()
        await self._events.put(_EventIngressBarrier(reached))
        await reached.wait()

    async def reconfigure_session_options(self, options: LocalASRSessionOptions) -> None:
        self._pending_session_options = options

    async def warmup(self) -> None:
        """Pre-establish STT session for faster first response."""
        if await self._ensure_session():
            self._emit_detailed("[STT] Session pre-warmed", fallback_level=logging.INFO)

    async def _on_speech_start(self, event: SpeechStart) -> None:
        await self._apply_pending_session_options()
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

    async def _apply_pending_session_options(self) -> None:
        options = self._pending_session_options
        if options is None:
            return
        backend = self.backend
        if isinstance(backend, LocalASRReconfigurableBackend):
            await backend.reconfigure_session_options(options)
        self._pending_session_options = None

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
                "[STT] Speech end handling for id=%s " "(reason=%s, trailing_silence_ms=%s)",
                str(event.utterance_id)[:8],
                event.reason,
                event.trailing_silence_ms,
                fallback_level=logging.INFO,
            )
            self._emit_stt_input_diagnostics(event.utterance_id, finalize=True)
            await self._active_session.on_speech_end(
                trailing_silence_ms=event.trailing_silence_ms,
                reason=event.reason,
            )

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

        async with self._session_open_lock:
            if self._active_session is not None:
                return True

            await self._set_state(STTSessionState.CONNECTING)
            last_exc: Exception | None = None

            try:
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
                        attempt_report = stt_failure_report(
                            exc,
                            provider=self._provider_label(),
                            operation="open_session",
                            channel=self.channel,
                            attempts=attempt,
                        )
                        self._emit_detailed(
                            "[STT] Failed to open session (attempt %s/%s): %s",
                            attempt,
                            self.connect_attempts,
                            format_error_report_for_log(attempt_report),
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
                        self._consumer_task = asyncio.create_task(
                            self._consume_session_events(
                                session,
                                publication_generation=self._publication_generation,
                            )
                        )
                        self._schedule_reset_timer()
                        await self._set_state(STTSessionState.STREAMING)
                        self._log_session_connected(attempts=attempt)
                        self._emit_detailed(
                            "[STT] Session ready (reset_deadline=%ss)",
                            self.reset_deadline_s,
                            fallback_level=logging.INFO,
                        )
                        return True
            except asyncio.CancelledError:
                if self._active_session is None:
                    await self._set_state(STTSessionState.DISCONNECTED)
                raise

            report = stt_failure_report(
                last_exc,
                provider=self._provider_label(),
                operation="open_session",
                channel=self.channel,
                attempts=self.connect_attempts,
            )
            self._emit_basic(
                "[STT] Failed to open session after %s attempts: %s",
                self.connect_attempts,
                format_error_report_for_log(report),
                level=logging.ERROR,
                fallback_level=logging.ERROR,
            )
            await self._set_state(STTSessionState.DISCONNECTED)
            await self._publish_event(
                STTErrorEvent(
                    message=report.message,
                    diagnostics=report.diagnostics,
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
                publication_generation=self._publication_generation,
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
            self._track_draining_task(
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
            report = stt_failure_report(
                e,
                provider=self._provider_label(),
                operation="reconnect",
                channel=self.channel,
            )
            self._emit_basic(
                "[STT] Reconnect failed; closing until next speech: %s",
                format_error_report_for_log(report),
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
                publication_generation=self._publication_generation,
            )
        )
        self._schedule_reset_timer()

        await self._set_state(STTSessionState.STREAMING)
        self._emit_basic("[STT] Session reconnected after recent speech")

        # Drain old session with finalize (unlike bridging)
        self._track_draining_task(
            asyncio.create_task(
                self._drain_and_close(old_session, old_consumer, allow_finalize=True)
            )
        )

    def _track_draining_task(self, task: asyncio.Task[None]) -> None:
        self._draining.add(task)
        task.add_done_callback(self._on_draining_task_done)

    def _on_draining_task_done(self, task: asyncio.Task[None]) -> None:
        self._draining.discard(task)
        if task.cancelled():
            return
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.exception()

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
        stop_timed_out = False
        try:
            if allow_finalize and self._should_finalize_before_stop():
                await self._finalize_before_stop(session)
            try:
                await asyncio.wait_for(session.stop(), timeout=self.drain_timeout_s)
            except asyncio.TimeoutError:
                stop_timed_out = True
                self._emit_detailed(
                    f"[STT] DRAIN: Stop timeout after {self.drain_timeout_s}s",
                    level=logging.WARNING,
                    fallback_level=logging.WARNING,
                )
            except asyncio.CancelledError:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise
            except Exception:
                pass

            if stop_timed_out:
                await self._close_session_for_cleanup(session)

            if not consumer_task.done():
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
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await consumer_task
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling():
                        raise
        finally:
            await self._close_session_for_cleanup(session)
            if not consumer_task.done():
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    await asyncio.wait_for(
                        asyncio.shield(consumer_task),
                        timeout=self.drain_timeout_s,
                    )
            if not consumer_task.done():
                consumer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await consumer_task
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
        final_language_runs: tuple[FinalLanguageRun, ...] = (),
    ) -> Transcript:
        return Transcript(
            utterance_id=utterance_id,
            text=text,
            is_final=is_final,
            created_at=created_at,
            channel=self.channel,
            final_language_runs=final_language_runs,
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
            self._emit_finalization_lag_diagnostic(
                utterance_id=utterance_id,
                pending_duration_s=age_s,
                threshold_s=stale_after_s,
                outcome="stale_drop",
            )
            self._emit_detailed(
                "[STT] Dropped stale pending final id=%s age_s=%.1f",
                str(utterance_id)[:8],
                age_s,
                level=logging.WARNING,
                fallback_level=logging.WARNING,
            )

    def _emit_finalization_lag_diagnostic(
        self,
        *,
        utterance_id: UUID,
        pending_duration_s: float,
        threshold_s: float,
        outcome: str,
    ) -> None:
        pending_ms = max(0, int(round(pending_duration_s * 1000)))
        threshold_ms = max(0, int(round(threshold_s * 1000)))
        if pending_ms <= threshold_ms:
            return
        self._emit_detailed(
            "[STT][FinalizationLag] channel=%s provider=%s utterance_id=%s "
            "pending_ms=%s threshold_ms=%s dominant_stage=stt_finalization_pending "
            "speech_end_to_stt_final_ms=%s outcome=%s",
            self.channel,
            self._provider_label(),
            str(utterance_id)[:8],
            pending_ms,
            threshold_ms,
            pending_ms,
            outcome,
            level=logging.WARNING,
            fallback_level=logging.WARNING,
        )

    def _should_suppress_final_transcript(self, text: str) -> bool:
        return (
            self.stt_provider_name is STTProviderName.LOCAL_QWEN
            and is_known_local_qwen_hallucination(text)
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
        *,
        publication_generation: int | None = None,
    ) -> None:
        if publication_generation is None:
            publication_generation = self._publication_generation
        try:
            async for ev in session.events():
                if publication_generation != self._publication_generation:
                    continue
                if ev.is_final:
                    self._drop_stale_pending_final_utterance_ids()
                    utterance_id = (
                        self._pending_final_utterance_ids.popleft()
                        if self._pending_final_utterance_ids
                        else self._active_utterance_id
                    )
                    if utterance_id is not None:
                        ended_at = self._pending_final_utterance_times.pop(utterance_id, None)
                        if ended_at is not None:
                            self._emit_finalization_lag_diagnostic(
                                utterance_id=utterance_id,
                                pending_duration_s=self.clock.now() - ended_at,
                                threshold_s=max(0.0, float(self.reconnect_window_s)),
                                outcome="final_received",
                            )
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
                    final_language_runs=ev.final_language_runs,
                )
                if ev.is_final:
                    await self._publish_event(STTFinalEvent(utterance_id, transcript))
                else:
                    await self._publish_event(STTPartialEvent(utterance_id, transcript))
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

        report = stt_failure_report(
            exc,
            provider=self._provider_label(),
            operation="stream",
            channel=self.channel,
        )
        self._emit_basic(
            "[STT] Session failed: %s",
            format_error_report_for_log(report),
            level=logging.ERROR,
            fallback_level=logging.ERROR,
        )
        if not self._closing:
            await self._publish_event(
                STTErrorEvent(
                    message=report.message,
                    diagnostics=report.diagnostics,
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
        await self._publish_event(STTSessionStateEvent(state, channel=self.channel))

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
