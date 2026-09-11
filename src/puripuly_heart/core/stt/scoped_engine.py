from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from puripuly_heart.core.audio.format import AudioCaptureSpan, float32_to_pcm16le_bytes
from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    OwnedVadEvent,
)
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderTurnEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTScopedTurnSession,
)
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.scoped_normalizer import (
    STTNormalizationDiagnostic,
    STTNormalizationError,
    STTScopedTurnNormalizer,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart

STTScopedSessionFactory = Callable[
    [AudioSegmentSettingsSnapshot, str],
    Awaitable[STTScopedTurnSession],
]
STTScopedTurnEventSink = Callable[[STTProviderTurnEvent], Awaitable[None] | None]
STTScopedDiagnosticSink = Callable[[object], Awaitable[None] | None]
STTWatchdogResolver = Callable[[AudioSegmentSettingsSnapshot], "STTRecognitionWatchdogs"]


class PermanentSTTScopedSessionError(RuntimeError):
    __slots__ = ()


@dataclass(frozen=True, slots=True)
class STTRecognitionWatchdogs:
    readiness_timeout_s: float = 30.0
    write_timeout_s: float = 5.0
    final_timeout_s: float = 20.0
    drain_timeout_s: float = 1.5
    healthy_reset_age_s: float = 180.0
    connect_attempts: int = 3
    connect_retry_base_s: float = 0.8
    connect_retry_max_s: float = 1.6

    def __post_init__(self) -> None:
        values = (
            self.readiness_timeout_s,
            self.write_timeout_s,
            self.final_timeout_s,
            self.drain_timeout_s,
            self.healthy_reset_age_s,
            self.connect_retry_base_s,
            self.connect_retry_max_s,
        )
        if any(value <= 0 for value in values):
            raise ValueError("recognition watchdog values must be positive")
        if self.connect_attempts != 3:
            raise ValueError("scoped recognition recovery requires exactly three attempts")


@dataclass(frozen=True, slots=True)
class STTRetentionProfile:
    max_retained_samples: int
    max_retained_bytes: int
    release_after_write: bool
    retained_bytes_per_sample: int = 2

    def __post_init__(self) -> None:
        if self.max_retained_samples < 1 or self.max_retained_bytes < 1:
            raise ValueError("recognition retention limits must be positive")
        if self.retained_bytes_per_sample < 1:
            raise ValueError("retained_bytes_per_sample must be positive")


@dataclass(frozen=True, slots=True)
class STTRetentionSnapshot:
    retained_samples: int
    retained_bytes: int
    high_water_samples: int
    high_water_bytes: int


@dataclass(slots=True)
class _ActiveTurn:
    identity: STTProviderTurnIdentity
    settings: AudioSegmentSettingsSnapshot
    normalizer: STTScopedTurnNormalizer
    watchdogs: STTRecognitionWatchdogs
    terminal_ready: asyncio.Future[STTProviderTurnTerminal]
    retention_profile: STTRetentionProfile | None
    payload_sequence: int = 0
    local_sealed: bool = False
    terminal_emitted: bool = False
    write_failed: bool = False
    retained_samples: int = 0
    retained_bytes: int = 0


@dataclass(slots=True)
class ScopedRecognitionEngine:
    session_factory: STTScopedSessionFactory
    channel: Literal["self", "peer"] = "peer"
    event_sink: STTScopedTurnEventSink | None = None
    watchdog_resolver: STTWatchdogResolver = lambda _settings: STTRecognitionWatchdogs()
    diagnostic_sink: STTScopedDiagnosticSink | None = None
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    monotonic_clock: Callable[[], float] = time.monotonic
    accepted_settings_scope: tuple[object, ...] | None = None
    backend_close: Callable[[], Awaitable[None] | None] | None = None
    retention_profile_resolver: (
        Callable[[AudioSegmentSettingsSnapshot], STTRetentionProfile] | None
    ) = None
    event_drain_timeout_s: float = 1.5
    terminal_failure_sink: Callable[[Exception], Awaitable[None] | None] | None = None
    exclusive_provider_ids: frozenset[str] = frozenset(
        {
            "local_cpu_auto",
            "local_parakeet_v3",
            "local_parakeet_ja",
            "local_qwen",
            "local_qwen_gpu",
        }
    )
    _session: STTScopedTurnSession | None = field(init=False, default=None, repr=False)
    _session_consumer: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _session_opened_at_s: float | None = field(init=False, default=None, repr=False)
    _session_scope: tuple[object, ...] | None = field(init=False, default=None, repr=False)
    _provider_epoch_id: str | None = field(init=False, default=None, repr=False)
    _session_watchdogs: STTRecognitionWatchdogs | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _ended_provider_epoch_id: str | None = field(init=False, default=None, repr=False)
    _turn: _ActiveTurn | None = field(init=False, default=None, repr=False)
    _input_lock: asyncio.Lock = field(init=False, repr=False)
    _abort_lock: asyncio.Lock = field(init=False, repr=False)
    _cleanup_tasks: set[asyncio.Task[None]] = field(init=False, default_factory=set, repr=False)
    _factory_tasks: set[asyncio.Task[STTScopedTurnSession]] = field(
        init=False,
        default_factory=set,
        repr=False,
    )
    _operation_tasks: dict[int, set[asyncio.Task[object]]] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _notification_tasks: set[asyncio.Task[None]] = field(
        init=False,
        default_factory=set,
        repr=False,
    )
    _terminal_turn_ids: set[tuple[str, str]] = field(init=False, default_factory=set, repr=False)
    _terminal_turn_order: deque[tuple[str, str]] = field(
        init=False,
        default_factory=deque,
        repr=False,
    )
    _terminal_failure_notified: bool = field(init=False, default=False, repr=False)
    _episode_failures: int = field(init=False, default=0, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _deferred_event_sink: STTScopedTurnEventSink | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _event_buffer: STTProviderEventBuffer = field(init=False, repr=False)
    _event_dispatch_task: asyncio.Task[None] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _event_dispatching: bool = field(init=False, default=False, repr=False)
    _event_drained: asyncio.Event = field(init=False, repr=False)
    _backend_closed: bool = field(init=False, default=False, repr=False)
    _backend_close_task: asyncio.Task[None] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _retained_high_water_samples: int = field(init=False, default=0, repr=False)
    _retained_high_water_bytes: int = field(init=False, default=0, repr=False)
    _authority_generation: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self._input_lock = asyncio.Lock()
        self._abort_lock = asyncio.Lock()
        self._event_buffer = STTProviderEventBuffer()
        self._event_drained = asyncio.Event()
        self._event_drained.set()

    @property
    def is_at_turn_boundary(self) -> bool:
        return self._turn is None

    @property
    def cleanup_debt(self) -> int:
        session_debt = sum(not task.done() for task in self._cleanup_tasks) + sum(
            not task.done() for task in self._factory_tasks
        )
        if session_debt:
            return session_debt
        return int(self._backend_close_task is not None and not self._backend_close_task.done())

    @property
    def is_at_utterance_boundary(self) -> bool:
        return self.is_at_turn_boundary

    @property
    def scoped_settings_scope(self) -> tuple[object, ...] | None:
        return self.accepted_settings_scope

    @property
    def retain_for_scoped_dispatch(self) -> bool:
        return not self._closed

    @property
    def retention_snapshot(self) -> STTRetentionSnapshot:
        turn = self._turn
        return STTRetentionSnapshot(
            retained_samples=turn.retained_samples if turn is not None else 0,
            retained_bytes=turn.retained_bytes if turn is not None else 0,
            high_water_samples=self._retained_high_water_samples,
            high_water_bytes=self._retained_high_water_bytes,
        )

    def bind_event_sink(self, sink: STTScopedTurnEventSink) -> None:
        self.event_sink = None
        self._deferred_event_sink = sink
        task = self._event_dispatch_task
        if task is None or task.done():
            self._event_dispatch_task = asyncio.create_task(
                self._dispatch_events(),
                name="scoped-stt-output",
            )

    async def wait_for_event_ingress_drain(self) -> None:
        await self._event_drained.wait()

    async def handle_owned_vad_event(self, owned: object) -> None:
        if not isinstance(owned, OwnedVadEvent):
            raise TypeError("scoped recognition requires OwnedVadEvent")
        event = owned.event
        authority_generation = self._authority_generation
        if isinstance(event, SpeechStart):
            await asyncio.sleep(0)
        async with self._input_lock:
            if self._closed:
                return
            if isinstance(event, SpeechStart):
                await self._handle_start(owned, event, authority_generation)
            elif isinstance(event, SpeechChunk):
                await self._handle_chunk(owned, event)
            elif isinstance(event, SpeechEnd):
                await self._handle_end(owned, event)
            else:
                raise TypeError(f"unknown owned VAD event: {type(event)!r}")

    async def reject_owned_segment(
        self,
        owned: OwnedVadEvent,
        *,
        reason: str,
        outcome: Literal["expired", "failed"] = "expired",
    ) -> None:
        identity = STTProviderTurnIdentity(
            segment=owned.segment.identity,
            provider_epoch_id="admission",
            provider_turn_id=uuid4().hex,
            settings_scope=self._settings_scope(owned.segment.settings),
        )
        await self._emit(
            STTProviderTurnTerminal(
                identity=identity,
                outcome=outcome,
                text_authority="none",
                failure_reason=reason,
                epoch_disposition="retire",
            )
        )

    async def fail_owned_segment(
        self,
        owned: OwnedVadEvent,
        *,
        reason: str,
    ) -> None:
        turn = self._matching_turn(owned)
        if turn is None:
            await self.reject_owned_segment(owned, reason=reason, outcome="failed")
            return
        self._set_turn_failure(turn, reason, allow_provisional=True)
        turn.write_failed = True
        await self._finish_failed_turn_immediately(turn)

    async def abort(self, *, reason: str = "cancelled") -> None:
        # Invalidate provider authority before waiting for any in-flight
        # open/write/seal/final operation. Native work may continue under
        # cleanup ownership, but it can no longer publish into this engine.
        self._authority_generation += 1
        async with self._abort_lock:
            turn = self._turn
            session = self._session
            if turn is not None:
                turn.local_sealed = True
                terminal = STTProviderTurnTerminal(
                    identity=turn.identity,
                    outcome="cancelled",
                    text_authority="none",
                    failure_reason=reason,
                    epoch_disposition="retire",
                )
                if not turn.terminal_ready.done():
                    turn.terminal_ready.set_result(terminal)
                if session is not None:
                    task = asyncio.create_task(
                        session.abort_turn(turn.identity, reason=reason),
                        name=f"scoped-stt-abort:{turn.identity.provider_turn_id}",
                    )
                    self._operation_tasks.setdefault(id(session), set()).add(task)
                await self._finish_turn(turn, terminal)
            else:
                self._retire_current_session()

    async def stop(self) -> None:
        await self.abort(reason="stopped")

    async def abort_for_toggle_off(self) -> None:
        await self.abort(reason="toggle_off")

    async def close(self) -> None:
        if self._closed:
            return
        await self.abort(reason="closed")
        self._closed = True
        await self._await_event_drain(self.event_drain_timeout_s)
        self._event_buffer.close()
        event_task = self._event_dispatch_task
        if event_task is not None:
            done, _pending = await asyncio.wait(
                {event_task},
                timeout=self.event_drain_timeout_s,
            )
            if event_task not in done:
                event_task.cancel()
        pending = tuple(self._cleanup_tasks) + tuple(self._factory_tasks)
        if pending:
            done, _pending = await asyncio.wait(
                pending,
                timeout=self.event_drain_timeout_s,
            )
            for task in done:
                self._consume_task_result(task)

    async def close_backend(self) -> None:
        await self.close()
        if self._backend_closed:
            return
        self._backend_closed = True
        if self.backend_close is None:
            return
        task = asyncio.create_task(
            self._close_backend_after_cleanup(),
            name="scoped-stt-backend-close",
        )
        self._backend_close_task = task
        task.add_done_callback(self._backend_close_done)
        done, _pending = await asyncio.wait({task}, timeout=self.event_drain_timeout_s)
        if task in done:
            self._consume_task_result(task)

    async def _handle_start(
        self,
        owned: OwnedVadEvent,
        event: SpeechStart,
        authority_generation: int,
    ) -> None:
        if authority_generation != self._authority_generation:
            return
        if self._turn is not None:
            raise RuntimeError("one unresolved provider turn is allowed per provider epoch")
        settings = owned.segment.settings
        watchdogs = self.watchdog_resolver(settings)
        open_failure: BaseException | None = None
        try:
            await self._ensure_session(settings, watchdogs)
        except Exception as exc:
            open_failure = exc
            self._notify_terminal_failure(exc)
        if authority_generation != self._authority_generation:
            self._retire_current_session(watchdogs)
            return
        session = self._session
        epoch_id = self._provider_epoch_id or uuid4().hex
        identity = STTProviderTurnIdentity(
            segment=owned.segment.identity,
            provider_epoch_id=epoch_id,
            provider_turn_id=uuid4().hex,
            settings_scope=self._settings_scope(settings),
        )
        loop = asyncio.get_running_loop()
        turn = _ActiveTurn(
            identity=identity,
            settings=settings,
            normalizer=STTScopedTurnNormalizer(
                identity,
                diagnostic_sink=self._normalization_diagnostic,
            ),
            retention_profile=(
                self.retention_profile_resolver(settings)
                if self.retention_profile_resolver is not None
                else None
            ),
            watchdogs=watchdogs,
            terminal_ready=loop.create_future(),
        )
        self._turn = turn
        if open_failure is not None or session is None or self._provider_epoch_id is None:
            self._set_turn_failure(
                turn,
                "provider_not_ready:" + type(open_failure).__name__,
            )
            turn.write_failed = True
            await self._finish_failed_turn_immediately(turn)
            return
        if not await self._run_write(
            session,
            turn,
            "begin",
            session.begin_turn(
                STTProviderTurnRequest(
                    identity=identity,
                    settings=settings,
                    channel=self.channel,
                )
            ),
        ):
            await self._finish_failed_turn_immediately(turn)
            return
        if event.pre_roll.size:
            await self._send_payload(
                turn,
                session,
                event.pre_roll,
                event.pre_roll_capture,
                context_only=True,
            )
        if not turn.write_failed and event.chunk.size:
            await self._send_payload(
                turn,
                session,
                event.chunk,
                event.chunk_capture,
                context_only=False,
            )

    async def _handle_chunk(self, owned: OwnedVadEvent, event: SpeechChunk) -> None:
        turn = self._matching_turn(owned)
        if turn is None or turn.write_failed:
            return
        session = self._session
        if session is None:
            self._set_turn_failure(turn, "provider_session_unavailable")
            turn.write_failed = True
            await self._finish_failed_turn_immediately(turn)
            return
        await self._send_payload(
            turn,
            session,
            event.chunk,
            event.chunk_capture,
            context_only=False,
        )

    async def _handle_end(self, owned: OwnedVadEvent, event: SpeechEnd) -> None:
        turn = self._matching_turn(owned)
        if turn is None:
            return
        if owned.segment.state != "sealed" or owned.segment.sealed_at_monotonic_s is None:
            raise RuntimeError("recognition terminality requires a locally sealed segment")
        turn.local_sealed = True
        session = self._session
        if not turn.write_failed and session is not None:
            sent = await self._run_write(
                session,
                turn,
                "seal",
                session.seal_turn(
                    turn.identity,
                    sealed_content_ranges=owned.segment.content_ranges,
                    seal_reason=str(event.reason),
                    observed_trailing_silence_ms=event.trailing_silence_ms,
                ),
            )
            if sent:
                await self._await_terminal(turn)
        if not turn.terminal_ready.done():
            self._set_turn_failure(turn, "provider_turn_failed_before_terminal")
        terminal = turn.terminal_ready.result()
        await self._finish_turn(turn, terminal)

    async def _send_payload(
        self,
        turn: _ActiveTurn,
        session: STTScopedTurnSession,
        samples: object,
        source_ranges: tuple[AudioCaptureSpan, ...],
        *,
        context_only: bool,
    ) -> None:
        pcm = float32_to_pcm16le_bytes(samples)
        if not pcm:
            return
        sample_count = len(pcm) // 2
        profile = turn.retention_profile
        retained_bytes = (
            sample_count * profile.retained_bytes_per_sample if profile is not None else len(pcm)
        )
        if profile is not None and (
            turn.retained_samples + sample_count > profile.max_retained_samples
            or turn.retained_bytes + retained_bytes > profile.max_retained_bytes
        ):
            self._set_turn_failure(
                turn,
                "buffer_exhausted",
                allow_provisional=True,
            )
            turn.write_failed = True
            await self._finish_failed_turn_immediately(turn)
            return
        turn.retained_samples += sample_count
        turn.retained_bytes += retained_bytes
        self._retained_high_water_samples = max(
            self._retained_high_water_samples,
            turn.retained_samples,
        )
        self._retained_high_water_bytes = max(
            self._retained_high_water_bytes,
            turn.retained_bytes,
        )
        turn.payload_sequence += 1
        written = await self._run_write(
            session,
            turn,
            "send",
            session.send_turn_audio(
                turn.identity,
                pcm,
                payload_sequence=turn.payload_sequence,
                source_ranges=source_ranges,
                context_only=context_only,
            ),
        )
        if written and profile is not None and profile.release_after_write:
            turn.retained_samples -= sample_count
            turn.retained_bytes -= retained_bytes
        if not written:
            await self._finish_failed_turn_immediately(turn)

    async def _finish_failed_turn_immediately(self, turn: _ActiveTurn) -> None:
        if turn.terminal_emitted or not turn.terminal_ready.done():
            return
        if self.channel != "self" and not turn.local_sealed:
            return
        turn.local_sealed = True
        await self._finish_turn(turn, turn.terminal_ready.result())

    async def _ensure_session(
        self,
        settings: AudioSegmentSettingsSnapshot,
        watchdogs: STTRecognitionWatchdogs,
    ) -> None:
        scope = self._settings_scope(settings)
        if self.accepted_settings_scope is not None and scope != self.accepted_settings_scope:
            raise PermanentSTTScopedSessionError("provider_configuration_scope_mismatch")
        if self._session is not None:
            if self._ended_provider_epoch_id == self._provider_epoch_id:
                self._retire_current_session(watchdogs)
            else:
                opened_at = self._session_opened_at_s
                age = 0.0 if opened_at is None else self.monotonic_clock() - opened_at
                if self._session_scope == scope and age < watchdogs.healthy_reset_age_s:
                    return
                self._retire_current_session(watchdogs)
        if self.cleanup_debt:
            await self._await_cleanup_debt(watchdogs.readiness_timeout_s)
            if self.cleanup_debt:
                raise RuntimeError("provider_resource_quarantined")
        last_error: BaseException | None = None
        while self._episode_failures < watchdogs.connect_attempts:
            epoch_id = uuid4().hex
            task = asyncio.create_task(
                self.session_factory(settings, epoch_id),
                name=f"scoped-stt-open:{epoch_id}",
            )
            self._factory_tasks.add(task)
            done, _pending = await asyncio.wait({task}, timeout=watchdogs.readiness_timeout_s)
            if task not in done:
                self._episode_failures += 1
                self._schedule_late_factory_reclaim(task, watchdogs)
                last_error = TimeoutError("provider_readiness_timeout")
                if settings.provider_id in self.exclusive_provider_ids:
                    break
            else:
                self._factory_tasks.discard(task)
                try:
                    session = task.result()
                except PermanentSTTScopedSessionError:
                    self._episode_failures = watchdogs.connect_attempts
                    raise
                except Exception as exc:
                    self._episode_failures += 1
                    last_error = exc
                else:
                    self._session = session
                    self._provider_epoch_id = epoch_id
                    self._session_scope = scope
                    self._session_opened_at_s = self.monotonic_clock()
                    self._session_watchdogs = watchdogs
                    self._ended_provider_epoch_id = None
                    self._session_consumer = asyncio.create_task(
                        self._consume_session_events(session, epoch_id),
                        name=f"scoped-stt-events:{epoch_id}",
                    )
                    return
            if self._episode_failures < watchdogs.connect_attempts:
                delay = min(
                    watchdogs.connect_retry_base_s * (2 ** (self._episode_failures - 1)),
                    watchdogs.connect_retry_max_s,
                )
                await self.sleep(delay)
        raise RuntimeError("provider_recovery_exhausted") from last_error

    async def _consume_session_events(
        self,
        session: STTScopedTurnSession,
        epoch_id: str,
    ) -> None:
        try:
            async for event in session.turn_events():
                if epoch_id != self._provider_epoch_id:
                    continue
                if isinstance(event, STTProviderEpochEnded):
                    if event.provider_epoch_id != epoch_id:
                        continue
                    self._ended_provider_epoch_id = epoch_id
                    turn = self._turn
                    if turn is not None and not turn.terminal_ready.done():
                        self._set_turn_failure(turn, event.reason or "provider_epoch_ended")
                    await self._emit(event)
                    if turn is None:
                        async with self._input_lock:
                            if epoch_id == self._provider_epoch_id and self._turn is None:
                                self._retire_current_session()
                    return
                turn = self._turn
                if turn is None or event.identity != turn.identity:
                    continue
                if isinstance(event, STTProviderTurnUpdate):
                    try:
                        normalized = turn.normalizer.apply_update(event)
                    except STTNormalizationError as exc:
                        self._set_turn_failure(turn, exc.reason)
                    else:
                        if normalized is not None:
                            await self._emit(normalized)
                    continue
                if isinstance(event, STTProviderTurnTerminal):
                    if turn.terminal_ready.done():
                        continue
                    try:
                        terminal = turn.normalizer.apply_terminal(event)
                    except STTNormalizationError as exc:
                        terminal = STTProviderTurnTerminal(
                            identity=turn.identity,
                            outcome="failed",
                            text_authority="none",
                            failure_reason=exc.reason,
                            epoch_disposition="retire",
                        )
                    turn.terminal_ready.set_result(terminal)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if epoch_id == self._provider_epoch_id:
                turn = self._turn
                if turn is not None:
                    self._set_turn_failure(
                        turn, f"provider_event_stream_failed:{type(exc).__name__}"
                    )

    async def _await_terminal(self, turn: _ActiveTurn) -> None:
        done, _pending = await asyncio.wait(
            {turn.terminal_ready},
            timeout=turn.watchdogs.final_timeout_s,
        )
        if turn.terminal_ready in done:
            return
        session = self._session
        allow_interim = bool(getattr(session, "allows_interim_timeout_fallback", False))
        allow_interim = allow_interim and turn.settings.provider_id == "gemini_transcribe"
        self._set_turn_failure(
            turn,
            "provider_final_timeout",
            allow_provisional=allow_interim,
        )

    async def _run_write(
        self,
        session: STTScopedTurnSession,
        turn: _ActiveTurn,
        operation: Literal["begin", "send", "seal", "abort"],
        awaitable: Awaitable[None],
    ) -> bool:
        task = asyncio.create_task(
            awaitable,
            name=f"scoped-stt-{operation}:{turn.identity.provider_turn_id}",
        )
        operations = self._operation_tasks.setdefault(id(session), set())
        operations.add(task)
        done, _pending = await asyncio.wait({task}, timeout=turn.watchdogs.write_timeout_s)
        if task not in done:
            self._set_turn_failure(turn, f"provider_{operation}_timeout")
            turn.write_failed = True
            self._retire_current_session(turn.watchdogs)
            return False
        operations.discard(task)
        if not operations:
            self._operation_tasks.pop(id(session), None)
        try:
            task.result()
        except BaseException as exc:
            self._set_turn_failure(turn, f"provider_{operation}_failed:{type(exc).__name__}")
            turn.write_failed = True
            self._retire_current_session(turn.watchdogs)
            return False
        return True

    def _set_turn_failure(
        self,
        turn: _ActiveTurn,
        reason: str,
        *,
        allow_provisional: bool = False,
    ) -> None:
        if turn.terminal_ready.done():
            return
        try:
            terminal = turn.normalizer.failure_terminal(
                reason=reason,
                allow_provisional=allow_provisional,
            )
        except STTNormalizationError:
            terminal = STTProviderTurnTerminal(
                identity=turn.identity,
                outcome="failed",
                text_authority="none",
                failure_reason=reason,
                epoch_disposition="retire",
            )
        turn.terminal_ready.set_result(terminal)

    async def _finish_turn(
        self,
        turn: _ActiveTurn,
        terminal: STTProviderTurnTerminal,
    ) -> None:
        if turn is not self._turn or turn.terminal_emitted:
            return
        if not turn.local_sealed:
            return
        turn.terminal_emitted = True
        key = (turn.identity.provider_epoch_id, turn.identity.provider_turn_id)
        should_emit = key not in self._terminal_turn_ids
        if should_emit:
            self._terminal_turn_ids.add(key)
            self._terminal_turn_order.append(key)
            while len(self._terminal_turn_order) > 4096:
                self._terminal_turn_ids.discard(self._terminal_turn_order.popleft())
        turn.retained_samples = 0
        turn.retained_bytes = 0
        self._turn = None
        if terminal.outcome in ("final", "empty"):
            self._episode_failures = 0
            self._terminal_failure_notified = False
        else:
            self._episode_failures += 1
        if (
            terminal.epoch_disposition == "retire"
            or terminal.outcome in ("failed", "expired", "cancelled")
            or self._ended_provider_epoch_id == turn.identity.provider_epoch_id
        ):
            self._retire_current_session(turn.watchdogs)
        if should_emit:
            await self._emit(terminal)

    def _matching_turn(self, owned: OwnedVadEvent) -> _ActiveTurn | None:
        turn = self._turn
        if turn is None or turn.identity.segment != owned.segment.identity:
            return None
        if turn.settings != owned.segment.settings:
            raise RuntimeError("owned segment settings changed during provider turn")
        return turn

    def _retire_current_session(
        self,
        watchdogs: STTRecognitionWatchdogs | None = None,
    ) -> None:
        session = self._session
        if session is None:
            return
        consumer = self._session_consumer
        if watchdogs is None:
            watchdogs = self._session_watchdogs
        if watchdogs is None:
            watchdogs = (
                self._turn.watchdogs if self._turn is not None else STTRecognitionWatchdogs()
            )
        self._session = None
        self._session_consumer = None
        self._session_opened_at_s = None
        self._session_scope = None
        self._session_watchdogs = None
        self._provider_epoch_id = None
        cleanup_consumer = consumer
        if cleanup_consumer is asyncio.current_task():
            cleanup_consumer = None
        task = asyncio.create_task(
            self._cleanup_session(session, cleanup_consumer, watchdogs),
            name="scoped-stt-cleanup",
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_done)

    async def _emit(self, event: STTProviderTurnEvent) -> None:
        sink = self.event_sink
        if sink is not None:
            result = sink(event)
            if inspect.isawaitable(result):
                await result
            return
        self._event_drained.clear()
        self._event_buffer.put(event)

    async def _dispatch_events(self) -> None:
        async for event in self._event_buffer.events():
            sink = self._deferred_event_sink
            if sink is None:
                raise RuntimeError("scoped STT event sink is not bound")
            self._event_dispatching = True
            try:
                result = sink(event)
                if inspect.isawaitable(result):
                    await result
            finally:
                self._event_dispatching = False
                if self._event_buffer.depth == 0:
                    self._event_drained.set()

    async def _await_event_drain(self, timeout: float) -> None:
        if self.event_sink is not None:
            return
        try:
            await asyncio.wait_for(self._event_drained.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return

    async def _cleanup_session(
        self,
        session: STTScopedTurnSession,
        consumer: asyncio.Task[None] | None,
        watchdogs: STTRecognitionWatchdogs,
    ) -> None:
        operations = tuple(self._operation_tasks.pop(id(session), ()))
        if operations:
            await asyncio.gather(*operations, return_exceptions=True)
        await self._bounded_cleanup_call(session.stop(), watchdogs.drain_timeout_s)
        await self._bounded_cleanup_call(session.close(), watchdogs.drain_timeout_s)
        if consumer is not None and not consumer.done():
            consumer.cancel()
        if consumer is not None:
            await asyncio.gather(consumer, return_exceptions=True)

    async def _bounded_cleanup_call(self, awaitable: Awaitable[None], timeout: float) -> None:
        task = asyncio.create_task(awaitable)
        done, _pending = await asyncio.wait({task}, timeout=timeout)
        if task in done:
            self._consume_task_result(task)
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _schedule_late_factory_reclaim(
        self,
        task: asyncio.Task[STTScopedTurnSession],
        watchdogs: STTRecognitionWatchdogs,
    ) -> None:
        async def reclaim() -> None:
            try:
                session = await task
            except BaseException:
                return
            finally:
                self._factory_tasks.discard(task)
            await self._cleanup_session(session, None, watchdogs)

        reclaim_task = asyncio.create_task(reclaim(), name="scoped-stt-late-open-reclaim")
        self._cleanup_tasks.add(reclaim_task)
        self._factory_tasks.discard(task)
        reclaim_task.add_done_callback(self._cleanup_done)

    async def _close_backend_after_cleanup(self) -> None:
        pending = tuple(self._cleanup_tasks) + tuple(self._factory_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        close_backend = self.backend_close
        if close_backend is None:
            return
        close_result = close_backend()
        if inspect.isawaitable(close_result):
            await close_result

    def _backend_close_done(self, task: asyncio.Task[None]) -> None:
        if self._backend_close_task is task:
            self._backend_close_task = None
        self._consume_task_result(task)

    async def _await_cleanup_debt(self, timeout: float) -> None:
        pending = {
            *self._cleanup_tasks,
            *self._factory_tasks,
        }
        pending = {task for task in pending if not task.done()}
        if not pending:
            return
        done, _pending = await asyncio.wait(pending, timeout=timeout)
        for task in done:
            self._consume_task_result(task)

    def _cleanup_done(self, task: asyncio.Task[None]) -> None:
        self._cleanup_tasks.discard(task)
        self._consume_task_result(task)

    def _notify_terminal_failure(self, failure: Exception) -> None:
        if self._terminal_failure_notified:
            return
        self._terminal_failure_notified = True
        sink = self.terminal_failure_sink
        if sink is None:
            return
        try:
            result = sink(failure)
        except Exception:
            return
        if inspect.isawaitable(result):
            task = asyncio.create_task(result, name="scoped-stt-terminal-failure")
            self._notification_tasks.add(task)
            task.add_done_callback(self._notification_done)

    def _notification_done(self, task: asyncio.Task[None]) -> None:
        self._notification_tasks.discard(task)
        self._consume_task_result(task)

    def _normalization_diagnostic(self, diagnostic: STTNormalizationDiagnostic) -> None:
        if self.diagnostic_sink is None:
            return
        result = self.diagnostic_sink(diagnostic)
        if inspect.isawaitable(result):
            task = asyncio.create_task(result)
            task.add_done_callback(self._consume_task_result)

    @staticmethod
    def _settings_scope(settings: AudioSegmentSettingsSnapshot) -> tuple[object, ...]:
        return (
            settings.provider_id,
            settings.provider_signature,
            settings.runtime_signature,
        )

    @staticmethod
    def _consume_task_result(task: asyncio.Future[object]) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            return


__all__ = [
    "PermanentSTTScopedSessionError",
    "STTRecognitionWatchdogs",
    "STTScopedDiagnosticSink",
    "STTScopedSessionFactory",
    "STTScopedTurnEventSink",
    "STTWatchdogResolver",
    "STTRetentionProfile",
    "STTRetentionSnapshot",
    "ScopedRecognitionEngine",
]
