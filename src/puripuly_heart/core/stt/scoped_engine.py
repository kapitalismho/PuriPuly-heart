from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Literal
from uuid import uuid4

import numpy as np

from puripuly_heart.core.audio.format import AudioCaptureSpan, float32_to_pcm16le_bytes
from puripuly_heart.core.audio.ownership import (
    AudioRetentionBudget,
    AudioSegmentSettingsSnapshot,
    OwnedStreamInput,
    OwnedVadEvent,
)
from puripuly_heart.core.stt.backend import (
    PermanentSTTScopedSessionError,
    STTIndependentRecognitionSession,
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnEvent,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
    STTScopedTurnSession,
)
from puripuly_heart.core.stt.diagnostics import recognition_cause
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.stt.scoped_normalizer import (
    STTNormalizationDiagnostic,
    STTNormalizationError,
    STTScopedTurnNormalizer,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.domain.recognition import RecognitionStreamIdentity

logger = logging.getLogger(__name__)


def _log_recognition_terminal(
    identity: STTProviderTurnIdentity,
    channel: Literal["self", "peer"],
    provider_id: str,
    terminal: STTProviderTurnTerminal | STTProviderInputTerminal,
    payloads: int,
    samples: int,
    byte_count: int,
    content_bytes: int,
    context_bytes: int,
    final_wait_ms: int | None,
    final_timeout_ms: int | None,
) -> None:
    with contextlib.suppress(Exception):
        logger.info(
            "[Recognition] terminal channel=%s utterance_id=%s provider=%s epoch=%s turn=%s "
            "outcome=%s cause=%s text_authority=%s successful_payloads=%d "
            "successful_samples=%d successful_bytes=%d content_bytes=%d context_bytes=%d "
            "activation_generation=%d final_wait_ms=%s final_timeout_ms=%s "
            "failure_retryable=%d recovery_pending=%d",
            channel,
            identity.segment.segment_id,
            provider_id,
            identity.provider_epoch_id,
            identity.provider_turn_id,
            terminal.outcome,
            recognition_cause(terminal.failure_reason),
            getattr(terminal, "text_authority", "none"),
            payloads,
            samples,
            byte_count,
            content_bytes,
            context_bytes,
            identity.segment.activation_generation,
            final_wait_ms if final_wait_ms is not None else "none",
            final_timeout_ms if final_timeout_ms is not None else "none",
            int(getattr(terminal, "failure_retryable", False)),
            int(terminal.recovery_pending),
        )


STTScopedSessionFactory = Callable[
    [AudioSegmentSettingsSnapshot, str],
    Awaitable[STTScopedTurnSession],
]
STTScopedTurnEventSink = Callable[[STTProviderTurnEvent], Awaitable[None] | None]
STTScopedDiagnosticSink = Callable[[object], Awaitable[None] | None]
STTWatchdogResolver = Callable[[AudioSegmentSettingsSnapshot], "STTRecognitionWatchdogs"]


@dataclass(frozen=True, slots=True)
class STTRecognitionWatchdogs:
    readiness_timeout_s: float = 30.0
    write_timeout_s: float = 5.0
    final_timeout_s: float = 20.0
    drain_timeout_s: float = 1.5
    idle_timeout_s: float = 60.0
    max_session_age_s: float | None = None
    connect_attempts: int = 3
    connect_retry_base_s: float = 0.8
    connect_retry_max_s: float = 1.6

    def __post_init__(self) -> None:
        values = (
            self.readiness_timeout_s,
            self.write_timeout_s,
            self.final_timeout_s,
            self.drain_timeout_s,
            self.idle_timeout_s,
            self.connect_retry_base_s,
            self.connect_retry_max_s,
        )
        if self.max_session_age_s is not None and self.max_session_age_s <= 0:
            raise ValueError("recognition session age limit must be positive")
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
    authority_generation: int
    terminal_ready: asyncio.Future[STTProviderTurnTerminal | STTProviderInputTerminal]
    retention_profile: STTRetentionProfile | None
    payload_sequence: int = 0
    successful_payloads: int = 0
    successful_samples: int = 0
    successful_bytes: int = 0
    successful_content_bytes: int = 0
    successful_context_bytes: int = 0
    local_sealed: bool = False
    terminal_emitted: bool = False
    write_failed: bool = False
    retained_samples: int = 0
    retained_bytes: int = 0
    retention_budget: AudioRetentionBudget | None = None
    retention_allocations: list[object] = field(default_factory=list)
    final_wait_started_at_s: float | None = None
    independent_recognition: bool = False


@dataclass(slots=True)
class _StreamRecovery:
    stream: RecognitionStreamIdentity
    settings: AudioSegmentSettingsSnapshot
    authority_generation: int
    source_frontier: int
    pending: bool = True


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
    session_lifetime_enabled: bool = True
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
    _recognition_stream: RecognitionStreamIdentity | None = field(
        init=False, default=None, repr=False
    )
    _last_receipt_sequence: int = field(init=False, default=0, repr=False)
    _estimated_speech_scope: tuple[object, ...] | None = field(init=False, default=None, repr=False)
    _estimated_last_speech_at: float | None = field(init=False, default=None, repr=False)
    _recognition_authorities: OrderedDict[RecognitionStreamIdentity, int] = field(
        init=False, default_factory=OrderedDict, repr=False
    )
    _session_watchdogs: STTRecognitionWatchdogs | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _ended_provider_epoch_id: str | None = field(init=False, default=None, repr=False)
    _retiring_provider_epoch_ids: set[str] = field(init=False, default_factory=set, repr=False)
    _turn: _ActiveTurn | None = field(init=False, default=None, repr=False)
    _turns: dict[STTProviderTurnIdentity, _ActiveTurn] = field(
        init=False, default_factory=dict, repr=False
    )
    _turn_order: deque[STTProviderTurnIdentity] = field(
        init=False, default_factory=deque, repr=False
    )
    _terminal_wait_tasks: set[asyncio.Task[None]] = field(
        init=False, default_factory=set, repr=False
    )
    _terminal_drain_lock: asyncio.Lock = field(init=False, repr=False)
    _session_retirement_requested: bool = field(init=False, default=False, repr=False)
    _source_speech_active: bool = field(init=False, default=False, repr=False)
    _last_source_speech_at_s: float | None = field(init=False, default=None, repr=False)
    _source_work_pending: bool = field(init=False, default=False, repr=False)
    _lifetime_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _lifetime_deadline_s: float | None = field(init=False, default=None, repr=False)
    _session_max_age_s: float | None = field(init=False, default=None, repr=False)
    _input_lock: asyncio.Lock = field(init=False, repr=False)
    _turn_resolved: asyncio.Event = field(init=False, repr=False)
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
    _failed_terminal_turn_ids: set[tuple[str, str]] = field(
        init=False, default_factory=set, repr=False
    )
    _terminal_turn_order: deque[tuple[str, str]] = field(
        init=False,
        default_factory=deque,
        repr=False,
    )
    _terminal_failure_notified: bool = field(init=False, default=False, repr=False)
    _episode_failures: int = field(init=False, default=0, repr=False)
    _recovery_backoff_pending: bool = field(init=False, default=False, repr=False)
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
    _stream_recovery: _StreamRecovery | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._input_lock = asyncio.Lock()
        self._terminal_drain_lock = asyncio.Lock()
        self._abort_lock = asyncio.Lock()
        self._turn_resolved = asyncio.Event()
        self._turn_resolved.set()
        self._event_buffer = STTProviderEventBuffer()
        self._event_drained = asyncio.Event()
        self._event_drained.set()

    @property
    def is_at_turn_boundary(self) -> bool:
        return not self._turns

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
    def is_live(self) -> bool:
        return not self._closed

    def is_current_recognition_stream(self, stream: RecognitionStreamIdentity) -> bool:
        return (
            not self._closed
            and self._recognition_authorities.get(stream) == self._authority_generation
        )

    @property
    def retention_snapshot(self) -> STTRetentionSnapshot:
        return STTRetentionSnapshot(
            retained_samples=sum(turn.retained_samples for turn in self._turns.values()),
            retained_bytes=sum(turn.retained_bytes for turn in self._turns.values()),
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

    async def handle_stream_input(self, owned: OwnedStreamInput) -> None:
        authority_generation = self._authority_generation
        async with self._input_lock:
            if (
                self._closed
                or authority_generation != self._authority_generation
                or (owned.is_current is not None and not owned.is_current())
            ):
                return
            if owned.event.boundary_reason not in (None, "source_eof"):
                await self.abort(reason=owned.event.boundary_reason)
                return
            recovery = self._stream_recovery
            if recovery is not None:
                if not self._recovery_matches(recovery, owned):
                    await self.abort(reason="stream_input_ownership_changed")
                    return
                if recovery.pending:
                    if not owned.event.chunk.size or not owned.event.capture:
                        self._stream_recovery = None
                        return
                    if all(
                        span.normalized_end_sample is not None
                        and span.normalized_end_sample <= recovery.source_frontier
                        for span in owned.event.capture
                    ):
                        return
                    recovery.pending = False
                    watchdogs = self.watchdog_resolver(owned.settings)
                    try:
                        await self._ensure_session(owned.settings, watchdogs, authority_generation)
                    except asyncio.CancelledError:
                        await self.abort(reason="cancelled")
                        raise
                    except Exception as exc:
                        if authority_generation == self._authority_generation:
                            self._stream_recovery = None
                            self._notify_terminal_failure(exc)
                        return
                    session = self._session
                    if (
                        authority_generation != self._authority_generation
                        or not self._recovery_matches(recovery, owned)
                        or not isinstance(session, STTIndependentRecognitionSession)
                        or not session.independent_recognition_units
                        or not session.accepts_stream_input
                        or self._provider_epoch_id is None
                    ):
                        self._retire_current_session(watchdogs)
                        self._stream_recovery = None
                        return
                    stream = replace(recovery.stream, provider_epoch_id=self._provider_epoch_id)
                    if not await self._run_stream_write(
                        session, session.begin_stream(stream), watchdogs.write_timeout_s
                    ):
                        return
                    if (
                        authority_generation != self._authority_generation
                        or not self._recovery_matches(recovery, owned)
                    ):
                        self._retire_current_session(watchdogs)
                        return
                    self._register_recognition_stream(stream)
            if (
                owned.ledger.activation_generation != owned.activation_generation
                or owned.ledger.settings != owned.settings
                or (owned.is_current is not None and not owned.is_current())
            ):
                return
            capture = owned.event.capture
            if owned.event.chunk.size and capture:
                capture_epoch = capture[0].capture_epoch
                if any(span.capture_epoch != capture_epoch for span in capture):
                    return
                scope = (
                    owned.activation_generation,
                    capture_epoch,
                    self._settings_scope(owned.settings),
                )
                if self._estimated_speech_scope != scope or any(
                    span.discontinuity_before is not None for span in capture
                ):
                    self._estimated_last_speech_at = None
                self._estimated_speech_scope = scope
                if owned.event.speech_observed:
                    self._estimated_last_speech_at = max(
                        span.source_end_monotonic_s for span in capture
                    )
            session = self._session
            stream = self._recognition_stream
            watchdogs = self._session_watchdogs
            if (
                not isinstance(session, STTIndependentRecognitionSession)
                or not session.independent_recognition_units
                or not session.accepts_stream_input
                or stream is None
                or watchdogs is None
                or owned.activation_generation != stream.activation_generation
                or self._settings_scope(owned.settings) != stream.settings_scope
                or owned.ledger.activation_generation != owned.activation_generation
                or (owned.is_current is not None and not owned.is_current())
                or owned.ledger.settings != owned.settings
                or any(span.capture_epoch != stream.capture_epoch for span in owned.event.capture)
            ):
                return
            samples, ranges = self._after_stream_failure(owned.event.chunk, owned.event.capture)
            if getattr(samples, "size", 0) and not session.recognition_source_covers(ranges):
                budget = owned.retention.budget if owned.retention is not None else None
                allocation = object()
                count = int(getattr(samples, "size", 0))
                if budget is not None and not budget.try_reserve(
                    allocation, count * 2, sample_equivalents=count
                ):
                    await self.abort(reason="stream_input_capacity_exhausted")
                    return
                candidate = _StreamRecovery(
                    stream,
                    owned.settings,
                    authority_generation,
                    max(span.normalized_end_sample or 0 for span in ranges),
                )
                try:
                    written = await self._run_stream_write(
                        session,
                        session.send_stream_audio(
                            float32_to_pcm16le_bytes(samples), source_ranges=ranges
                        ),
                        watchdogs.write_timeout_s,
                        recovery=candidate,
                    )
                    if written:
                        self._episode_failures = 0
                        self._terminal_failure_notified = False
                finally:
                    if budget is not None:
                        budget.release(allocation)
            if owned.event.boundary_reason is not None and session is self._session:
                self._stream_recovery = None
                await self._run_stream_write(
                    session,
                    session.end_stream(reason=owned.event.boundary_reason),
                    watchdogs.write_timeout_s + watchdogs.drain_timeout_s,
                )
                await self._await_event_drain(self.event_drain_timeout_s)

    def _recovery_matches(self, recovery: _StreamRecovery, owned: OwnedStreamInput) -> bool:
        return (
            (owned.is_current is None or owned.is_current())
            and recovery.authority_generation == self._authority_generation
            and recovery.stream.activation_generation == owned.activation_generation
            and recovery.settings == owned.settings == owned.ledger.settings
            and owned.ledger.activation_generation == owned.activation_generation
            and all(
                span.capture_epoch == recovery.stream.capture_epoch
                and span.discontinuity_before is None
                for span in owned.event.capture
            )
        )

    def _after_stream_failure(
        self, samples: np.ndarray, ranges: tuple[AudioCaptureSpan, ...]
    ) -> tuple[np.ndarray, tuple[AudioCaptureSpan, ...]]:
        recovery = self._stream_recovery
        if recovery is None or not ranges:
            return samples, ranges
        frontier = recovery.source_frontier
        skipped = 0
        remaining: list[AudioCaptureSpan] = []
        for span in ranges:
            start, end = span.normalized_start_sample, span.normalized_end_sample
            if start is None or end is None:
                return samples, ranges
            left = min(end, max(start, frontier))
            skipped += left - start
            if left < end:
                remaining.append(span if left == start else span.slice_normalized(left, end))
        return samples[skipped:], tuple(remaining)

    def _register_recognition_stream(self, stream: RecognitionStreamIdentity) -> None:
        if self._recognition_stream != stream:
            self._estimated_speech_scope = None
            self._estimated_last_speech_at = None
            self._recognition_stream = stream
            self._last_receipt_sequence = 0
        self._recognition_authorities[stream] = self._authority_generation
        self._recognition_authorities.move_to_end(stream)
        while len(self._recognition_authorities) > 4096:
            self._recognition_authorities.popitem(last=False)

    async def _run_stream_write(
        self,
        session: STTIndependentRecognitionSession,
        awaitable: Awaitable[None],
        timeout: float,
        *,
        recovery: _StreamRecovery | None = None,
    ) -> bool:
        deadline = (
            self._session_opened_at_s + self._session_max_age_s
            if self._session_opened_at_s is not None and self._session_max_age_s is not None
            else None
        )
        if deadline is not None:
            timeout = min(timeout, max(0.0, deadline - self.monotonic_clock()))
        task = asyncio.create_task(awaitable)
        operations = self._operation_tasks.setdefault(id(session), set())
        operations.add(task)
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            task.cancel()
            if session is self._session:
                await self.abort(reason="cancelled")
            raise
        if deadline is not None and self.monotonic_clock() >= deadline:
            task.cancel()
            await self._expire_current_session()
            return False
        failure = None
        if task not in done:
            task.cancel()
            failure = "provider_stream_write_timeout"
        else:
            operations.discard(task)
            if not operations:
                self._operation_tasks.pop(id(session), None)
            try:
                task.result()
            except Exception:
                failure = "provider_stream_write_failed"
        if failure is None:
            return session is self._session and not self._closed
        generation = self._authority_generation
        failures = self._episode_failures
        await self.abort(reason=failure)
        if (
            recovery is not None
            and not self._closed
            and recovery.authority_generation == generation
            and self._authority_generation == generation + 1
        ):
            self._episode_failures = max(self._episode_failures, failures + 1)
            if self._episode_failures < (
                self.watchdog_resolver(recovery.settings).connect_attempts
            ):
                recovery.authority_generation = self._authority_generation
                self._stream_recovery = recovery
                self._recovery_backoff_pending = True
            else:
                self._notify_terminal_failure(RuntimeError("provider_recovery_exhausted"))
        return False

    async def handle_owned_vad_event(self, owned: object) -> None:
        if not isinstance(owned, OwnedVadEvent):
            raise TypeError("scoped recognition requires OwnedVadEvent")
        event = owned.event
        if isinstance(event, SpeechEnd):
            if not self._closed:
                await self._handle_end(owned, event)
            return
        authority_generation = self._authority_generation
        if isinstance(event, SpeechStart):
            await asyncio.sleep(0)
            while True:
                async with self._input_lock:
                    if self._closed or authority_generation != self._authority_generation:
                        return
                    if self._can_start_turn(owned.segment.settings):
                        await self._handle_start(owned, event, authority_generation)
                        return
                    resolved = self._turn_resolved
                await resolved.wait()
        async with self._input_lock:
            if self._closed:
                return
            if isinstance(event, SpeechChunk):
                await self._handle_chunk(owned, event)
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
        terminal = STTProviderTurnTerminal(
            identity=identity,
            outcome=outcome,
            text_authority="none",
            failure_reason=reason,
            epoch_disposition="retire",
        )
        _log_recognition_terminal(
            identity,
            self.channel,
            owned.segment.settings.provider_id,
            terminal,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
        )
        await self._emit(terminal)

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

    async def observe_source_activity(
        self,
        *,
        speech_observed: bool,
        observed_at_monotonic_s: float | None = None,
    ) -> None:
        if self._closed or not self.session_lifetime_enabled:
            return
        now = self.monotonic_clock() if observed_at_monotonic_s is None else observed_at_monotonic_s
        if speech_observed or self._source_speech_active:
            self._last_source_speech_at_s = now
        self._source_speech_active = speech_observed
        self._schedule_lifetime_check()

    async def observe_pending_source_work(self, *, pending: bool) -> None:
        if self._closed or not self.session_lifetime_enabled:
            return
        self._source_work_pending = pending
        self._schedule_lifetime_check()

    async def abort(self, *, reason: str = "cancelled") -> None:
        self._authority_generation += 1
        self._stream_recovery = None
        self._recognition_authorities.clear()
        self._estimated_speech_scope = None
        self._estimated_last_speech_at = None
        async with self._abort_lock:
            turns = tuple(self._turns.values())
            session = self._session
            for turn in turns:
                turn.local_sealed = True
                terminal = STTProviderTurnTerminal(
                    identity=turn.identity,
                    outcome="cancelled",
                    text_authority="none",
                    failure_reason=reason,
                    epoch_disposition="retire",
                )
                if turn.terminal_ready.done():
                    turn.terminal_ready = asyncio.get_running_loop().create_future()
                turn.terminal_ready.set_result(terminal)
                if session is not None:
                    task = asyncio.create_task(
                        session.abort_turn(turn.identity, reason=reason),
                        name=f"scoped-stt-abort:{turn.identity.provider_turn_id}",
                    )
                    self._operation_tasks.setdefault(id(session), set()).add(task)
            await self._drain_completed_turns()
            if not turns:
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
        self._cancel_lifetime_check()
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
        pending = (
            tuple(self._cleanup_tasks)
            + tuple(self._factory_tasks)
            + tuple(self._terminal_wait_tasks)
        )
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
            raise RuntimeError("one capturing provider turn is allowed per provider epoch")
        settings = owned.segment.settings
        watchdogs = self.watchdog_resolver(settings)
        stream = self._recognition_stream
        speech_scope = (
            owned.segment.identity.activation_generation,
            owned.segment.identity.capture_epoch,
            self._settings_scope(settings),
        )
        startup_last_speech_at = (
            self._estimated_last_speech_at
            if self._estimated_speech_scope == speech_scope
            and (
                stream is None
                or (stream.activation_generation, stream.capture_epoch, stream.settings_scope)
                != speech_scope
            )
            else None
        )
        if stream is not None and (
            stream.activation_generation != owned.segment.identity.activation_generation
            or stream.capture_epoch != owned.segment.identity.capture_epoch
        ):
            self._retire_current_session(watchdogs)
        open_failure: BaseException | None = None
        try:
            await self._ensure_session(settings, watchdogs, authority_generation)
        except Exception as exc:
            if authority_generation != self._authority_generation:
                return
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
        independent = isinstance(session, STTIndependentRecognitionSession) and (
            session.independent_recognition_units
        )
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
            authority_generation=authority_generation,
            retention_budget=(owned.retention.budget if owned.retention is not None else None),
            independent_recognition=independent,
        )
        self._turn = turn
        self._turns[identity] = turn
        self._turn_order.append(identity)
        self._turn_resolved.clear()
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
        turn.independent_recognition = isinstance(session, STTIndependentRecognitionSession) and (
            session.independent_recognition_units
        )
        if turn.independent_recognition:
            stream = RecognitionStreamIdentity(
                self.channel,
                identity.segment.activation_generation,
                identity.segment.capture_epoch,
                identity.provider_epoch_id,
                identity.settings_scope,
            )
            self._register_recognition_stream(stream)
            if startup_last_speech_at is not None:
                self._estimated_speech_scope = speech_scope
                self._estimated_last_speech_at = startup_last_speech_at
        if event.pre_roll.size:
            await self._send_payload(
                turn,
                session,
                event.pre_roll,
                event.pre_roll_capture,
                context_only=True,
            )
        if self._has_write_authority(session, turn) and event.chunk.size:
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
        if session is None or not self._has_write_authority(session, turn):
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
        if (
            not turn.write_failed
            and session is not None
            and self._has_write_authority(session, turn)
        ):
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
            if turn.independent_recognition:
                if sent and not turn.terminal_ready.done():
                    turn.terminal_ready.set_result(
                        STTProviderInputTerminal(turn.identity, "submitted", self.channel)
                    )
                if not turn.terminal_ready.done():
                    self._set_turn_failure(turn, "provider_input_submission_failed")
                await self._drain_completed_turns()
                return
            if sent and self._has_write_authority(session, turn):
                if self._session_allows_sealed_turn_overlap(session):
                    self._turn = None
                    self._turn_resolved.set()
                    task = asyncio.create_task(
                        self._await_and_finish_turn(turn),
                        name=f"scoped-stt-terminal:{turn.identity.provider_turn_id}",
                    )
                    self._terminal_wait_tasks.add(task)
                    task.add_done_callback(self._terminal_wait_done)
                    return
                await self._await_terminal(turn)
        if not turn.terminal_ready.done():
            self._set_turn_failure(turn, "provider_turn_failed_before_terminal")
        await self._drain_completed_turns()

    async def _send_payload(
        self,
        turn: _ActiveTurn,
        session: STTScopedTurnSession,
        samples: np.ndarray,
        source_ranges: tuple[AudioCaptureSpan, ...],
        *,
        context_only: bool,
    ) -> None:
        if not self._has_write_authority(session, turn):
            return
        if turn.independent_recognition and self._stream_recovery is not None:
            recovery = self._stream_recovery
            if (
                turn.identity.segment.activation_generation != recovery.stream.activation_generation
                or turn.identity.segment.capture_epoch != recovery.stream.capture_epoch
                or turn.settings != recovery.settings
            ):
                await self.abort(reason="stream_input_ownership_changed")
                return
            samples, source_ranges = self._after_stream_failure(samples, source_ranges)
        if (
            turn.independent_recognition
            and source_ranges
            and isinstance(session, STTIndependentRecognitionSession)
            and session.recognition_source_covers(source_ranges)
        ):
            return
        sample_count = int(getattr(samples, "size", 0))
        if sample_count <= 0:
            return
        pcm_bytes = sample_count * 2
        profile = turn.retention_profile
        retained_bytes = (
            sample_count * profile.retained_bytes_per_sample if profile is not None else pcm_bytes
        )
        if profile is not None and (
            turn.retained_samples + sample_count > profile.max_retained_samples
            or turn.retained_bytes + retained_bytes > profile.max_retained_bytes
        ):
            await self._fail_for_retention(turn)
            return
        transient_owner = object()
        native_owner = object()
        budget = turn.retention_budget
        if budget is not None:
            if not budget.try_reserve(
                transient_owner,
                pcm_bytes,
                sample_equivalents=sample_count,
            ):
                await self._fail_for_retention(turn)
                return
            if not budget.try_reserve(
                native_owner,
                retained_bytes,
                sample_equivalents=sample_count,
            ):
                budget.release(transient_owner)
                await self._fail_for_retention(turn)
                return
            turn.retention_allocations.append(native_owner)
        pcm = float32_to_pcm16le_bytes(samples)
        if not pcm:
            if budget is not None:
                budget.release(transient_owner)
                budget.release(native_owner)
                turn.retention_allocations.remove(native_owner)
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
        try:
            operation = session.send_turn_audio(
                turn.identity,
                pcm,
                payload_sequence=turn.payload_sequence,
                source_ranges=source_ranges,
                context_only=context_only,
            )
            if (
                turn.independent_recognition
                and isinstance(session, STTIndependentRecognitionSession)
                and session.accepts_stream_input
                and self._recognition_stream is not None
                and source_ranges
            ):
                written = await self._run_stream_write(
                    session,
                    operation,
                    turn.watchdogs.write_timeout_s,
                    recovery=_StreamRecovery(
                        self._recognition_stream,
                        turn.settings,
                        turn.authority_generation,
                        max(span.normalized_end_sample or 0 for span in source_ranges),
                    ),
                )
            else:
                written = await self._run_write(session, turn, "send", operation)
        finally:
            if budget is not None:
                budget.release(transient_owner)
        if written:
            turn.successful_payloads += 1
            turn.successful_samples += sample_count
            turn.successful_bytes += len(pcm)
            if context_only:
                turn.successful_context_bytes += len(pcm)
            else:
                turn.successful_content_bytes += len(pcm)
            if self.channel == "peer" and turn.successful_payloads == 1:
                with contextlib.suppress(Exception):
                    logger.info(
                        "[Recognition] payload_received channel=peer utterance_id=%s "
                        "provider=%s epoch=%s turn=%s successful_payloads=1 "
                        "successful_samples=%d successful_bytes=%d context_only=%d",
                        turn.identity.segment.segment_id,
                        turn.settings.provider_id,
                        turn.identity.provider_epoch_id,
                        turn.identity.provider_turn_id,
                        turn.successful_samples,
                        turn.successful_bytes,
                        int(context_only),
                    )
        if written and (profile is None or profile.release_after_write):
            turn.retained_samples -= sample_count
            turn.retained_bytes -= retained_bytes
            if budget is not None:
                budget.release(native_owner)
                turn.retention_allocations.remove(native_owner)
        if not written:
            await self._finish_failed_turn_immediately(turn)

    async def _fail_for_retention(self, turn: _ActiveTurn) -> None:
        self._set_turn_failure(
            turn,
            "buffer_exhausted",
            allow_provisional=True,
        )
        turn.write_failed = True
        await self._finish_failed_turn_immediately(turn)

    async def _finish_failed_turn_immediately(self, turn: _ActiveTurn) -> None:
        if turn.terminal_emitted or not turn.terminal_ready.done():
            return
        if self.channel != "self" and not turn.local_sealed:
            return
        turn.local_sealed = True
        if turn is self._turn:
            self._turn = None
            self._turn_resolved.set()
        await self._drain_completed_turns()

    async def _ensure_session(
        self,
        settings: AudioSegmentSettingsSnapshot,
        watchdogs: STTRecognitionWatchdogs,
        authority_generation: int,
    ) -> None:
        scope = self._settings_scope(settings)
        if self.accepted_settings_scope is not None and scope != self.accepted_settings_scope:
            raise PermanentSTTScopedSessionError("provider_configuration_scope_mismatch")
        if self._session is not None:
            if self._ended_provider_epoch_id == self._provider_epoch_id:
                self._retire_current_session(watchdogs)
            elif self._session_scope == scope:
                if not self._session_near_ceiling():
                    return
                self._retire_current_session()
            else:
                self._retire_current_session(watchdogs)
        if self.cleanup_debt:
            await self._await_cleanup_debt(watchdogs.readiness_timeout_s)
            if self.cleanup_debt:
                raise RuntimeError("provider_resource_quarantined")
        if self._closed or authority_generation != self._authority_generation:
            return
        if self._recovery_backoff_pending:
            self._recovery_backoff_pending = False
            if self._episode_failures < watchdogs.connect_attempts:
                await self.sleep(
                    min(
                        watchdogs.connect_retry_base_s * (2 ** (self._episode_failures - 1)),
                        watchdogs.connect_retry_max_s,
                    )
                )
        last_error: BaseException | None = None
        if self._closed or authority_generation != self._authority_generation:
            return
        while self._episode_failures < watchdogs.connect_attempts:
            if self._closed or authority_generation != self._authority_generation:
                return
            epoch_id = uuid4().hex
            task = asyncio.create_task(
                self.session_factory(settings, epoch_id),
                name=f"scoped-stt-open:{epoch_id}",
            )
            self._factory_tasks.add(task)
            try:
                done, _pending = await asyncio.wait({task}, timeout=watchdogs.readiness_timeout_s)
            except asyncio.CancelledError:
                self._schedule_late_factory_reclaim(task, watchdogs)
                raise
            if self._closed or authority_generation != self._authority_generation:
                self._schedule_late_factory_reclaim(task, watchdogs)
                return
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
                    provider_max_age = getattr(session, "max_session_age_s", None)
                    if provider_max_age is not None and provider_max_age <= 0:
                        self._retire_current_session(watchdogs)
                        raise ValueError("provider session age limit must be positive")
                    limits = tuple(
                        limit
                        for limit in (watchdogs.max_session_age_s, provider_max_age)
                        if limit is not None
                    )
                    self._session_max_age_s = min(limits) if limits else None
                    self._ended_provider_epoch_id = None
                    self._session_retirement_requested = False
                    self._session_consumer = asyncio.create_task(
                        self._consume_session_events(session, epoch_id),
                        name=f"scoped-stt-events:{epoch_id}",
                    )
                    self._schedule_lifetime_check()
                    return
            if self._episode_failures < watchdogs.connect_attempts:
                delay = min(
                    watchdogs.connect_retry_base_s * (2 ** (self._episode_failures - 1)),
                    watchdogs.connect_retry_max_s,
                )
                await self.sleep(delay)
                if self._closed or authority_generation != self._authority_generation:
                    return
        raise RuntimeError("provider_recovery_exhausted") from last_error

    async def _consume_session_events(
        self,
        session: STTScopedTurnSession,
        epoch_id: str,
    ) -> None:
        authority_generation = self._authority_generation
        try:
            async for event in session.turn_events():
                if authority_generation != self._authority_generation or self._closed:
                    continue
                if (
                    epoch_id != self._provider_epoch_id
                    and epoch_id not in self._retiring_provider_epoch_ids
                ):
                    continue
                if isinstance(event, STTRecognitionUnit):
                    if (
                        epoch_id != self._provider_epoch_id
                        or event.identity.stream != self._recognition_stream
                        or event.identity.receipt_sequence <= self._last_receipt_sequence
                    ):
                        continue
                    self._last_receipt_sequence = event.identity.receipt_sequence
                    stream = event.identity.stream
                    event = replace(
                        event,
                        estimated_last_speech_at=(
                            self._estimated_last_speech_at
                            if self._estimated_speech_scope
                            == (
                                stream.activation_generation,
                                stream.capture_epoch,
                                stream.settings_scope,
                            )
                            else None
                        ),
                    )
                    await self._emit(
                        STTRecognitionUnitTerminal(event, "final" if event.text else "empty")
                    )
                    continue
                if isinstance(event, STTProviderEpochEnded):
                    if event.provider_epoch_id != epoch_id:
                        continue
                    idle_failure = (
                        epoch_id == self._provider_epoch_id
                        and not event.orderly
                        and event.reason not in ("cancelled", "closed", "stopped", "toggle_off")
                        and not any(
                            turn.identity.provider_epoch_id == epoch_id
                            for turn in self._turns.values()
                        )
                        and (
                            event.provider_turn_id is None
                            or (epoch_id, event.provider_turn_id)
                            not in self._failed_terminal_turn_ids
                        )
                    )
                    if (
                        idle_failure
                        and self._session_watchdogs is not None
                        and self._session_scope is not None
                        and self._session_scope[0] == "soniox"
                    ):
                        if event.failure_retryable:
                            self._episode_failures += 1
                            if self._episode_failures < self._session_watchdogs.connect_attempts:
                                self._recovery_backoff_pending = True
                            else:
                                self._notify_terminal_failure(RuntimeError(event.reason))
                        else:
                            self._episode_failures = self._session_watchdogs.connect_attempts
                            self._notify_terminal_failure(RuntimeError(event.reason))
                    self._ended_provider_epoch_id = epoch_id
                    self._session_retirement_requested = True
                    for turn in tuple(self._turns.values()):
                        if (
                            turn.identity.provider_epoch_id == epoch_id
                            and not turn.terminal_ready.done()
                        ):
                            self._set_turn_failure(
                                turn,
                                event.reason or "provider_epoch_ended",
                                failure_retryable=event.failure_retryable,
                            )
                    await self._drain_completed_turns()
                    await self._emit(event)
                    if not self._turns:
                        async with self._input_lock:
                            if epoch_id == self._provider_epoch_id and not self._turns:
                                self._retire_current_session()
                    return
                turn = self._turns.get(event.identity)
                if turn is None:
                    continue
                if isinstance(event, STTProviderInputTerminal):
                    if not turn.independent_recognition or turn.terminal_ready.done():
                        continue
                    turn.terminal_ready.set_result(event)
                    await self._drain_completed_turns()
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
                    if terminal.epoch_disposition == "retire":
                        self._session_retirement_requested = True
                    turn.terminal_ready.set_result(terminal)
                    await self._drain_completed_turns()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if epoch_id == self._provider_epoch_id or epoch_id in self._retiring_provider_epoch_ids:
                self._session_retirement_requested = True
                active = tuple(
                    turn
                    for turn in self._turns.values()
                    if turn.identity.provider_epoch_id == epoch_id
                )
                if (
                    not active
                    and epoch_id == self._provider_epoch_id
                    and authority_generation == self._authority_generation
                    and self._session_scope is not None
                    and self._session_scope[0] == "soniox"
                    and self._session_watchdogs is not None
                ):
                    if isinstance(exc, OSError):
                        self._episode_failures += 1
                        if self._episode_failures < self._session_watchdogs.connect_attempts:
                            self._recovery_backoff_pending = True
                        else:
                            self._notify_terminal_failure(
                                RuntimeError(f"provider_event_stream_failed:{type(exc).__name__}")
                            )
                    else:
                        self._episode_failures = self._session_watchdogs.connect_attempts
                        self._notify_terminal_failure(
                            RuntimeError(f"provider_event_stream_failed:{type(exc).__name__}")
                        )
                for turn in active:
                    self._set_turn_failure(
                        turn,
                        f"provider_event_stream_failed:{type(exc).__name__}",
                        failure_retryable=isinstance(exc, OSError),
                    )
                await self._drain_completed_turns()
                if (
                    not self._turns
                    and self._session_scope is not None
                    and self._session_scope[0] == "soniox"
                ):
                    async with self._input_lock:
                        if epoch_id == self._provider_epoch_id and not self._turns:
                            self._retire_current_session()

    async def _await_terminal(self, turn: _ActiveTurn) -> None:
        turn.final_wait_started_at_s = self.monotonic_clock()
        done, _pending = await asyncio.wait(
            {turn.terminal_ready},
            timeout=turn.watchdogs.final_timeout_s,
        )
        if turn.terminal_ready in done:
            return
        self._set_turn_failure(
            turn,
            "provider_final_timeout",
            failure_retryable=True,
        )

    async def _await_and_finish_turn(self, turn: _ActiveTurn) -> None:
        await self._await_terminal(turn)
        await self._drain_completed_turns()

    def _terminal_wait_done(self, task: asyncio.Task[None]) -> None:
        self._terminal_wait_tasks.discard(task)
        self._consume_task_result(task)

    async def _run_write(
        self,
        session: STTScopedTurnSession,
        turn: _ActiveTurn,
        operation: Literal["begin", "send", "seal", "abort"],
        awaitable: Awaitable[None],
    ) -> bool:
        deadline_s = (
            self._session_opened_at_s + self._session_max_age_s
            if session is self._session
            and self._session_opened_at_s is not None
            and self._session_max_age_s is not None
            else None
        )
        timeout_s = turn.watchdogs.write_timeout_s
        if deadline_s is not None:
            timeout_s = min(timeout_s, max(0.0, deadline_s - self.monotonic_clock()))
        task = asyncio.create_task(
            awaitable,
            name=f"scoped-stt-{operation}:{turn.identity.provider_turn_id}",
        )
        operations = self._operation_tasks.setdefault(id(session), set())
        operations.add(task)
        done, _pending = await asyncio.wait({task}, timeout=timeout_s)
        if (
            deadline_s is not None
            and self.monotonic_clock() >= deadline_s
            and self._has_write_authority(session, turn)
        ):
            await self._expire_current_session()
            return False
        if task not in done:
            if self._has_write_authority(session, turn):
                self._set_turn_failure(
                    turn, f"provider_{operation}_timeout", failure_retryable=True
                )
                if turn.settings.provider_id == "soniox" or turn.independent_recognition:
                    task.cancel()
                turn.write_failed = True
                self._retire_current_session(turn.watchdogs)
            return False
        operations.discard(task)
        if not operations:
            self._operation_tasks.pop(id(session), None)
        try:
            task.result()
        except BaseException as exc:
            if self._has_write_authority(session, turn):
                self._set_turn_failure(
                    turn,
                    f"provider_{operation}_failed:{type(exc).__name__}",
                    failure_retryable=isinstance(exc, OSError),
                )
                turn.write_failed = True
                self._retire_current_session(turn.watchdogs)
            return False
        return self._has_write_authority(session, turn)

    def _has_write_authority(
        self,
        session: STTScopedTurnSession,
        turn: _ActiveTurn,
    ) -> bool:
        return (
            not self._closed
            and not turn.terminal_emitted
            and turn.authority_generation == self._authority_generation
            and turn is self._turn
            and session is self._session
            and turn.identity.provider_epoch_id == self._provider_epoch_id
        )

    def _set_turn_failure(
        self,
        turn: _ActiveTurn,
        reason: str,
        *,
        allow_provisional: bool = False,
        failure_retryable: bool = False,
    ) -> None:
        if turn.terminal_ready.done():
            return
        if turn.independent_recognition:
            turn.terminal_ready.set_result(
                STTProviderInputTerminal(turn.identity, "failed", self.channel, reason, "retire")
            )
            return
        try:
            terminal = turn.normalizer.failure_terminal(
                reason=reason,
                allow_provisional=allow_provisional,
                failure_retryable=failure_retryable and turn.settings.provider_id == "soniox",
            )
        except STTNormalizationError:
            terminal = STTProviderTurnTerminal(
                identity=turn.identity,
                outcome="failed",
                text_authority="none",
                failure_reason=reason,
                epoch_disposition="retire",
                failure_retryable=failure_retryable and turn.settings.provider_id == "soniox",
            )
        turn.terminal_ready.set_result(terminal)

    async def _drain_completed_turns(self) -> None:
        async with self._terminal_drain_lock:
            while self._turn_order:
                identity = self._turn_order[0]
                turn = self._turns.get(identity)
                if turn is None:
                    self._turn_order.popleft()
                    continue
                if not turn.local_sealed or not turn.terminal_ready.done():
                    break
                self._turn_order.popleft()
                await self._finish_turn(turn, turn.terminal_ready.result())

    async def _finish_turn(
        self,
        turn: _ActiveTurn,
        terminal: STTProviderTurnTerminal | STTProviderInputTerminal,
    ) -> None:
        if self._turns.get(turn.identity) is not turn or turn.terminal_emitted:
            return
        if not turn.local_sealed:
            return
        if turn.independent_recognition and isinstance(terminal, STTProviderTurnTerminal):
            terminal = STTProviderInputTerminal(
                terminal.identity,
                (
                    terminal.outcome
                    if terminal.outcome in {"failed", "expired", "cancelled"}
                    else "failed"
                ),
                self.channel,
                terminal.failure_reason,
                terminal.epoch_disposition,
                terminal.recovery_pending,
            )
        turn.terminal_emitted = True
        key = (turn.identity.provider_epoch_id, turn.identity.provider_turn_id)
        if turn.retention_budget is not None:
            for allocation in turn.retention_allocations:
                turn.retention_budget.release(allocation)
            turn.retention_allocations.clear()
        should_emit = key not in self._terminal_turn_ids
        if should_emit:
            self._terminal_turn_ids.add(key)
            self._terminal_turn_order.append(key)
            while len(self._terminal_turn_order) > 4096:
                expired_key = self._terminal_turn_order.popleft()
                self._terminal_turn_ids.discard(expired_key)
                self._failed_terminal_turn_ids.discard(expired_key)
        turn.retained_samples = 0
        turn.retained_bytes = 0
        self._turns.pop(turn.identity, None)
        if turn is self._turn:
            self._turn = None
        self._turn_resolved.set()
        if (
            turn.authority_generation != self._authority_generation
            and terminal.outcome != "cancelled"
        ):
            terminal = (
                STTProviderInputTerminal(
                    turn.identity, "cancelled", self.channel, "cancelled", "retire"
                )
                if turn.independent_recognition
                else STTProviderTurnTerminal(
                    identity=turn.identity,
                    outcome="cancelled",
                    failure_reason="cancelled",
                    epoch_disposition="retire",
                )
            )
        if terminal.outcome in ("final", "empty", "submitted"):
            self._episode_failures = 0
            self._recovery_backoff_pending = False
            self._terminal_failure_notified = False
        else:
            self._episode_failures += 1
            if turn.settings.provider_id == "soniox" and terminal.outcome in ("failed", "degraded"):
                if not terminal.failure_retryable:
                    self._episode_failures = turn.watchdogs.connect_attempts
                elif (
                    self._episode_failures < turn.watchdogs.connect_attempts
                    and turn.authority_generation == self._authority_generation
                    and not self._closed
                ):
                    terminal = replace(terminal, recovery_pending=True)
                    self._recovery_backoff_pending = True
        if should_emit and terminal.outcome in ("failed", "degraded"):
            self._failed_terminal_turn_ids.add(key)
        if (
            terminal.epoch_disposition == "retire"
            or terminal.outcome in ("failed", "expired", "cancelled")
            or self._ended_provider_epoch_id == turn.identity.provider_epoch_id
        ):
            self._session_retirement_requested = True
        if self._session_near_ceiling():
            self._session_retirement_requested = True
        if should_emit:
            started_at = turn.final_wait_started_at_s
            _log_recognition_terminal(
                turn.identity,
                self.channel,
                turn.settings.provider_id,
                terminal,
                turn.successful_payloads,
                turn.successful_samples,
                turn.successful_bytes,
                turn.successful_content_bytes,
                turn.successful_context_bytes,
                (
                    max(0, int((self.monotonic_clock() - started_at) * 1000))
                    if started_at is not None
                    else None
                ),
                int(turn.watchdogs.final_timeout_s * 1000),
            )
            await self._emit(terminal)
            if (
                turn.settings.provider_id == "soniox"
                and terminal.outcome in ("failed", "degraded")
                and not terminal.recovery_pending
                and turn.authority_generation == self._authority_generation
            ):
                self._notify_terminal_failure(
                    RuntimeError(terminal.failure_reason or "provider_recovery_exhausted")
                )
        if self._session_retirement_requested and not self._turns:
            self._retire_current_session(turn.watchdogs)
        self._schedule_lifetime_check()

    def _can_start_turn(self, settings: AudioSegmentSettingsSnapshot) -> bool:
        if self._turn is not None:
            self._turn_resolved.clear()
            return False
        if not self._turns:
            return True
        if self._session_near_ceiling():
            self._session_retirement_requested = True
        session = self._session
        allowed = (
            session is not None
            and self._session_allows_sealed_turn_overlap(session)
            and not self._session_retirement_requested
            and self._ended_provider_epoch_id != self._provider_epoch_id
            and self._session_scope == self._settings_scope(settings)
        )
        if not allowed:
            self._turn_resolved.clear()
        return allowed

    def _session_near_ceiling(self) -> bool:
        opened_at = self._session_opened_at_s
        ceiling = self._session_max_age_s
        watchdogs = self._session_watchdogs
        if opened_at is None or ceiling is None or watchdogs is None:
            return False
        guard = min(
            ceiling / 2,
            watchdogs.write_timeout_s + watchdogs.final_timeout_s + watchdogs.drain_timeout_s,
        )
        return self.monotonic_clock() >= opened_at + ceiling - guard

    @staticmethod
    def _session_allows_sealed_turn_overlap(session: STTScopedTurnSession) -> bool:
        return bool(getattr(session, "allows_sealed_turn_overlap", False))

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
        self._cancel_lifetime_check()
        self._estimated_speech_scope = None
        self._estimated_last_speech_at = None
        session = self._session
        if session is None:
            return
        consumer = self._session_consumer
        if watchdogs is None:
            watchdogs = self._session_watchdogs
        if watchdogs is None:
            pending_turn = next(iter(self._turns.values()), None)
            watchdogs = (
                pending_turn.watchdogs if pending_turn is not None else STTRecognitionWatchdogs()
            )
        epoch_id = self._provider_epoch_id
        if epoch_id is not None:
            self._retiring_provider_epoch_ids.add(epoch_id)
        self._session = None
        self._recognition_stream = None
        self._last_receipt_sequence = 0
        self._session_consumer = None
        self._session_opened_at_s = None
        self._session_scope = None
        self._session_watchdogs = None
        self._session_max_age_s = None
        self._provider_epoch_id = None
        cleanup_consumer = consumer
        if cleanup_consumer is asyncio.current_task():
            cleanup_consumer = None
        task = asyncio.create_task(
            self._cleanup_session(
                session,
                cleanup_consumer,
                watchdogs,
                retiring_epoch_id=epoch_id,
            ),
            name="scoped-stt-cleanup",
        )
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_done)

    def _schedule_lifetime_check(self) -> None:
        opened_at = self._session_opened_at_s
        watchdogs = self._session_watchdogs
        if self._closed or self._session is None or opened_at is None or watchdogs is None:
            return
        now = self.monotonic_clock()
        cap_due_at = (
            opened_at + self._session_max_age_s if self._session_max_age_s is not None else None
        )
        idle_due_at = (
            max(opened_at, self._last_source_speech_at_s or opened_at) + watchdogs.idle_timeout_s
            if self.session_lifetime_enabled and not self._source_speech_active
            else None
        )
        if idle_due_at is not None and now >= idle_due_at:
            if self._source_work_pending or self._turns:
                idle_due_at = None
        deadline = cap_due_at
        if idle_due_at is not None and (deadline is None or idle_due_at < deadline):
            deadline = idle_due_at
        task = self._lifetime_task
        if (
            task is not None
            and self._lifetime_deadline_s == deadline
            and not task.done()
            and task is not asyncio.current_task()
        ):
            return
        self._cancel_lifetime_check()
        if deadline is None:
            return
        self._lifetime_deadline_s = deadline
        self._lifetime_task = asyncio.create_task(
            self._run_lifetime_check(deadline),
            name="scoped-stt-session-lifetime",
        )

    async def _expire_current_session(self) -> None:
        session = self._session
        if session is None:
            return
        turns = tuple(
            turn
            for turn in self._turns.values()
            if turn.identity.provider_epoch_id == self._provider_epoch_id
        )
        self._session_retirement_requested = True
        for turn in turns:
            if not turn.local_sealed and turn.terminal_ready.done():
                turn.terminal_ready = asyncio.get_running_loop().create_future()
                turn.terminal_ready.set_result(
                    STTProviderTurnTerminal(
                        identity=turn.identity,
                        outcome="failed",
                        text_authority="none",
                        failure_reason="provider_session_lifetime_exceeded",
                        epoch_disposition="retire",
                        failure_retryable=turn.settings.provider_id == "soniox",
                    )
                )
            else:
                self._set_turn_failure(
                    turn, "provider_session_lifetime_exceeded", failure_retryable=True
                )
            turn.write_failed = True
        for task in self._operation_tasks.get(id(session), ()):
            task.cancel()
        self._retire_current_session()
        for turn in turns:
            await self._finish_failed_turn_immediately(turn)

    async def _run_lifetime_check(self, deadline_s: float) -> None:
        try:
            await self.sleep(max(0.0, deadline_s - self.monotonic_clock()))
            async with self._input_lock:
                if self._closed or self._session is None:
                    return
                watchdogs = self._session_watchdogs
                opened_at = self._session_opened_at_s
                if watchdogs is None or opened_at is None:
                    return
                now = self.monotonic_clock()
                if (
                    self._session_max_age_s is not None
                    and now >= opened_at + self._session_max_age_s
                ):
                    await self._expire_current_session()
                    return
                if (
                    self.session_lifetime_enabled
                    and not self._source_speech_active
                    and not self._source_work_pending
                    and not self._turns
                    and now
                    >= (
                        max(opened_at, self._last_source_speech_at_s or opened_at)
                        + watchdogs.idle_timeout_s
                    )
                ):
                    self._retire_current_session(watchdogs)
                    return
                self._schedule_lifetime_check()
        except asyncio.CancelledError:
            return
        finally:
            if self._lifetime_task is asyncio.current_task():
                self._lifetime_task = None
                self._lifetime_deadline_s = None

    def _cancel_lifetime_check(self) -> None:
        task = self._lifetime_task
        self._lifetime_task = None
        self._lifetime_deadline_s = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _emit(self, event: STTProviderTurnEvent) -> None:
        sink = self.event_sink
        if sink is None and isinstance(event, STTProviderInputTerminal):
            sink = self._deferred_event_sink
        if sink is not None:
            result = sink(event)
            if inspect.isawaitable(result):
                await result
            return
        self._event_drained.clear()
        accepted = self._event_buffer.put(event)
        if not accepted and isinstance(
            event, (STTRecognitionUnitTerminal, STTProviderInputTerminal)
        ):
            epoch_id = (
                event.unit.identity.stream.provider_epoch_id
                if isinstance(event, STTRecognitionUnitTerminal)
                else event.identity.provider_epoch_id
            )
            if epoch_id == self._provider_epoch_id:
                for turn in self._turns.values():
                    if turn.identity.provider_epoch_id == epoch_id:
                        self._set_turn_failure(turn, "provider_event_buffer_overflow")
                        turn.write_failed = True
                self._retire_current_session()

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
        *,
        retiring_epoch_id: str | None = None,
    ) -> None:
        operations = tuple(self._operation_tasks.pop(id(session), ()))
        if operations:
            await asyncio.gather(*operations, return_exceptions=True)
        await self._bounded_cleanup_call(session.stop(), watchdogs.drain_timeout_s)
        if retiring_epoch_id is not None:
            drained = await self._await_retiring_epoch_terminals(
                retiring_epoch_id,
                watchdogs.drain_timeout_s,
            )
            if not drained:
                for turn in tuple(self._turns.values()):
                    if turn.identity.provider_epoch_id == retiring_epoch_id:
                        self._set_turn_failure(turn, "provider_retirement_drain_timeout")
                await self._drain_completed_turns()
        await self._bounded_cleanup_call(session.close(), watchdogs.drain_timeout_s)
        if consumer is not None and not consumer.done():
            consumer.cancel()
        if consumer is not None:
            await asyncio.gather(consumer, return_exceptions=True)
        if retiring_epoch_id is not None:
            self._retiring_provider_epoch_ids.discard(retiring_epoch_id)

    async def _await_retiring_epoch_terminals(
        self,
        epoch_id: str,
        timeout: float,
    ) -> bool:
        pending = {
            turn.terminal_ready
            for turn in self._turns.values()
            if turn.identity.provider_epoch_id == epoch_id and not turn.terminal_ready.done()
        }
        if not pending:
            return True
        _done, unresolved = await asyncio.wait(pending, timeout=timeout)
        return not unresolved

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
        except asyncio.CancelledError, Exception:
            return


__all__ = [
    "STTRecognitionWatchdogs",
    "STTScopedDiagnosticSink",
    "STTScopedSessionFactory",
    "STTScopedTurnEventSink",
    "STTWatchdogResolver",
    "STTRetentionProfile",
    "STTRetentionSnapshot",
    "ScopedRecognitionEngine",
]
