from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast
from uuid import UUID

from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    OwnedVadEvent,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmissionPort,
    SelfCaptureAdmissionStatus,
    SelfCaptureDiagnostic,
    SelfCaptureDiagnosticEvent,
    SelfCaptureFailureReason,
    SelfCaptureProviderMutationStatus,
    SelfCaptureProviderPort,
    SelfCaptureProviderStatus,
    SelfCaptureSessionConfig,
    SelfCaptureSessionSnapshot,
    SelfCaptureSessionState,
    SelfCaptureTerminalFailureHandler,
)
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart

SelfCaptureProviderRequestFactory = Callable[[SelfCaptureSessionConfig, bool], object]
SelfCaptureSourceFactory = Callable[[SelfCaptureSessionConfig], Awaitable[object] | object]
SelfCaptureVadFactory = Callable[[SelfCaptureSessionConfig], object]
SelfCaptureAudioLoop = Callable[..., Awaitable[None]]
SelfCaptureStateChanged = Callable[[SelfCaptureSessionSnapshot], object]
SelfCaptureDiagnosticSink = Callable[[SelfCaptureDiagnostic], object]


class _VadSink(Protocol):
    async def handle_vad_event(self, event: object) -> None: ...

    async def reject_owned_segment(
        self,
        event: OwnedVadEvent,
        *,
        reason: str,
        outcome: str,
    ) -> None: ...

    async def fail_owned_segment(
        self,
        event: OwnedVadEvent,
        *,
        reason: str,
    ) -> None: ...


@dataclass(slots=True)
class _CaptureGeneration:
    value: int


@dataclass(slots=True)
class _UnsentRecognitionSegment:
    start: OwnedVadEvent
    admitted_at_monotonic_s: float
    expiry_task: asyncio.Task[None] | None = None



class _GenerationGuardedVadSink:
    max_retained_samples = 2_880_000
    max_control_events = 32
    max_wholly_unsent_segments = 8
    wholly_unsent_ttl_s = 12.0

    def __init__(
        self,
        *,
        sink: object,
        owner: "SelfCaptureSessionOwner",
        capture_generation: _CaptureGeneration,
        ledger: PeerAudioSegmentLedger,
    ) -> None:
        self.sink = sink
        self.owner = owner
        self.capture_generation = capture_generation
        self.ledger = ledger
        self._queue: deque[OwnedVadEvent] = deque()
        self._retained_samples = 0
        self._control_events = 0
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._closing = False
        self._rejected_segment_ids: set[UUID] = set()
        self._unsent_segments: dict[UUID, _UnsentRecognitionSegment] = {}

    def __getattr__(self, name: str) -> object:
        return getattr(self.sink, name)

    @property
    def retained_samples(self) -> int:
        return self._retained_samples

    async def handle_vad_event(self, event: object) -> None:
        if not self.owner.is_current_generation(self.capture_generation.value):
            return
        if not hasattr(event, "utterance_id"):
            await cast(_VadSink, self.sink).handle_vad_event(event)
            return
        event_segment_id = cast(UUID, getattr(event, "utterance_id"))
        if event_segment_id in self._rejected_segment_ids:
            if isinstance(event, SpeechEnd):
                self._rejected_segment_ids.discard(event_segment_id)
            return
        owned = self.ledger.observe_vad_event(
            event,
            now_monotonic_s=asyncio.get_running_loop().time(),
        )
        pcm_samples = int(getattr(getattr(event, "chunk", None), "size", 0))
        if isinstance(event, SpeechStart):
            await self._admit_unsent_segment(owned)
        if pcm_samples:
            await self._reclaim_unsent_for_samples(pcm_samples)
            if self._retained_samples + pcm_samples > self.max_retained_samples:
                await self._fail_current_recognition(owned, "buffer_exhausted")
                return
            self._retained_samples += pcm_samples
        else:
            self._control_events += 1
            if self._control_events > self.max_control_events:
                self._control_events -= 1
                await self._fail_current_recognition(
                    owned,
                    "self_recognition_control_capacity_exhausted",
                )
                return
        self._queue.append(owned)
        worker = self._worker
        if worker is None:
            self._worker = asyncio.create_task(self._run(), name="self-vad-dispatch")
        elif worker.done():
            await worker
            raise RuntimeError("self VAD dispatch worker stopped")
        self._wake.set()
        await asyncio.sleep(0)

    async def finish(self) -> None:
        self._closing = True
        self._wake.set()
        worker = self._worker
        if worker is not None:
            await worker

    async def abort(self) -> None:
        worker = self._worker
        if worker is not None and not worker.done():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        for pending in tuple(self._unsent_segments.values()):
            task = pending.expiry_task
            if task is not None and not task.done():
                task.cancel()
        self._unsent_segments.clear()
        self._queue.clear()
        self._retained_samples = 0
        self._rejected_segment_ids.clear()
        self._control_events = 0
        self.ledger.cancel_unfinished(now_monotonic_s=asyncio.get_running_loop().time())

    async def _admit_unsent_segment(self, owned: OwnedVadEvent) -> None:
        while len(self._unsent_segments) >= self.max_wholly_unsent_segments:
            oldest_id = next(iter(self._unsent_segments))
            await self._reject_unsent_segment(
                oldest_id,
                reason="recognition_admission_overload",
            )
        segment_id = owned.segment.identity.segment_id
        pending = _UnsentRecognitionSegment(
            start=owned,
            admitted_at_monotonic_s=owned.segment.opened_at_monotonic_s,
        )
        self._unsent_segments[segment_id] = pending
        pending.expiry_task = asyncio.create_task(
            self._expire_unsent_segment(
                segment_id,
                pending.admitted_at_monotonic_s,
            ),
            name=f"self-recognition-expiry:{segment_id}",
        )

    async def _expire_unsent_segment(
        self,
        segment_id: UUID,
        admitted_at_monotonic_s: float,
    ) -> None:
        deadline = admitted_at_monotonic_s + self.wholly_unsent_ttl_s
        await asyncio.sleep(max(0.0, deadline - asyncio.get_running_loop().time()))
        pending = self._unsent_segments.get(segment_id)
        if pending is None or pending.admitted_at_monotonic_s != admitted_at_monotonic_s:
            return
        await self._reject_unsent_segment(
            segment_id,
            reason="recognition_admission_timeout",
        )

    async def _reclaim_unsent_for_samples(self, sample_count: int) -> None:
        while (
            self._unsent_segments
            and self._retained_samples + sample_count > self.max_retained_samples
        ):
            await self._reject_unsent_segment(
                next(iter(self._unsent_segments)),
                reason="recognition_admission_overload",
            )

    async def _reject_unsent_segment(self, segment_id: UUID, *, reason: str) -> None:
        pending = self._unsent_segments.pop(segment_id, None)
        self._rejected_segment_ids.add(segment_id)
        if pending is None:
            return
        expiry_task = pending.expiry_task
        if (
            expiry_task is not None
            and expiry_task is not asyncio.current_task()
            and not expiry_task.done()
        ):
            expiry_task.cancel()
        retained: deque[OwnedVadEvent] = deque()
        while self._queue:
            queued = self._queue.popleft()
            if queued.segment.identity.segment_id == segment_id:
                self._release_queue_charge(queued)
            else:
                retained.append(queued)
        self._queue = retained
        self.ledger.terminalize_for_failure(
            segment_id,
            now_monotonic_s=asyncio.get_running_loop().time(),
            failure_reason=reason,
            outcome="expired",
        )
        await cast(_VadSink, self.sink).reject_owned_segment(
            pending.start,
            reason=reason,
            outcome="expired",
        )

    async def _fail_current_recognition(
        self,
        owned: OwnedVadEvent,
        reason: str,
    ) -> None:
        pending = self._unsent_segments.pop(owned.segment.identity.segment_id, None)
        if pending is not None:
            expiry_task = pending.expiry_task
            if expiry_task is not None and not expiry_task.done():
                expiry_task.cancel()
        await cast(_VadSink, self.sink).fail_owned_segment(owned, reason=reason)
        self.ledger.terminalize_for_failure(
            owned.segment.identity.segment_id,
            now_monotonic_s=asyncio.get_running_loop().time(),
            failure_reason=reason,
        )
        self.owner.note_recognition_failure(reason)

    def _release_queue_charge(self, owned: OwnedVadEvent) -> None:
        event = owned.event
        pcm_samples = int(getattr(getattr(event, "chunk", None), "size", 0))
        if pcm_samples:
            self._retained_samples -= pcm_samples
        else:
            self._control_events -= 1

    async def _run(self) -> None:
        while True:
            if not self._queue:
                if self._closing:
                    return
                self._wake.clear()
                await self._wake.wait()
                continue
            owned = self._queue.popleft()
            event = owned.event
            if isinstance(event, SpeechStart):
                segment_id = owned.segment.identity.segment_id
                pending = self._unsent_segments.pop(segment_id, None)
                if pending is not None:
                    expiry_task = pending.expiry_task
                    if expiry_task is not None and not expiry_task.done():
                        expiry_task.cancel()
            try:
                if self.owner.is_current_generation(self.capture_generation.value):
                    await cast(_VadSink, self.sink).handle_vad_event(owned)
            finally:
                self._release_queue_charge(owned)


class SelfCaptureSessionOwner:
    resource_fields = (
        "_source",
        "_vad",
        "_loop_task",
        "_transition_task",
        "_vad_dispatch",
        "_fault_tasks",
        "_retired_sources",
        "_generation",
    )
    stop_ingress = "invalidate the generation, cancel the Self loop, and close the source"
    toggle_off_policy = "stop capture ingress before the channel-specific provider release request"
    shutdown_policy = (
        "cancel admission and capture, abort the provider channel, and retry cleanup debt"
    )
    late_callback_rule = "only the current running generation may publish Self VAD events"

    def __init__(
        self,
        *,
        admission: SelfCaptureAdmissionPort,
        provider: SelfCaptureProviderPort,
        provider_request_factory: SelfCaptureProviderRequestFactory,
        source_factory: SelfCaptureSourceFactory,
        vad_factory: SelfCaptureVadFactory,
        run_audio_loop: SelfCaptureAudioLoop,
        vad_sink: object,
        state_changed: SelfCaptureStateChanged | None = None,
        diagnostic_sink: SelfCaptureDiagnosticSink | None = None,
        audio_gate_reset: Callable[[], object] | None = None,
    ) -> None:
        self._admission = admission
        self._provider = provider
        self._provider_request_factory = provider_request_factory
        self._source_factory = source_factory
        self._vad_factory = vad_factory
        self._run_audio_loop = run_audio_loop
        self._vad_sink = vad_sink
        self._state_changed = state_changed
        self._diagnostic_sink = diagnostic_sink
        self._audio_gate_reset = audio_gate_reset
        self._state = SelfCaptureSessionState.STOPPED
        self._provider_status = SelfCaptureProviderStatus.DETACHED
        self._desired_active = False
        self._generation = 0
        self._config: SelfCaptureSessionConfig | None = None
        self._provider_signature: tuple[object, ...] | None = None
        self._provider_attachment_token: object | None = None
        self._pending_provider_recoveries: dict[
            SelfCaptureTerminalFailureHandler,
            tuple[tuple[object, ...], object, Exception | None],
        ] = {}
        self._source: object | None = None
        self._vad: object | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._capture_generation: _CaptureGeneration | None = None
        self._vad_dispatch: _GenerationGuardedVadSink | None = None
        self._segment_ledgers: deque[PeerAudioSegmentLedger] = deque(maxlen=16)
        self._transition_task: asyncio.Task[None] | None = None
        self._fault_tasks: set[asyncio.Task[None]] = set()
        self._retired_sources: list[object] = []
        self._failure_reason: SelfCaptureFailureReason | None = None
        self._admission_reason: str | None = None
        self._last_cleanup_exception: Exception | None = None
        self._closed = False
        self._state_lock = asyncio.Lock()
        self._activation_lock = asyncio.Lock()

    @property
    def snapshot(self) -> SelfCaptureSessionSnapshot:
        return SelfCaptureSessionSnapshot(
            state=self._state,
            provider_status=self._provider_status,
            desired_active=self._desired_active,
            effective_active=(
                self._state is SelfCaptureSessionState.RUNNING
                and self._loop_task is not None
                and not self._loop_task.done()
            ),
            generation=self._generation,
            provider_id=self._config.provider_id if self._config is not None else None,
            runtime_signature=(
                self._config.runtime_signature if self._config is not None else None
            ),
            failure_reason=self._failure_reason,
            admission_reason=self._admission_reason,
            has_source=self._source is not None,
            has_vad=self._vad is not None,
            has_loop_task=self._loop_task is not None,
            cleanup_debt=len(self._retired_sources),
            closed=self._closed,
        )

    @property
    def source(self) -> object | None:
        return self._source

    @property
    def cleanup_source(self) -> object | None:
        return self._retired_sources[0] if self._retired_sources else None

    @property
    def vad(self) -> object | None:
        return self._vad

    @property
    def loop_task(self) -> asyncio.Task[None] | None:
        return self._loop_task

    @property
    def current_config(self) -> SelfCaptureSessionConfig | None:
        return self._config

    @property
    def last_cleanup_exception(self) -> Exception | None:
        return self._last_cleanup_exception

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": "SelfCaptureSessionOwner",
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "toggle_off_policy": self.toggle_off_policy,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    def is_current_generation(self, generation: int) -> bool:
        return (
            generation == self._generation
            and self._desired_active
            and self._state is SelfCaptureSessionState.RUNNING
            and self._loop_task is not None
        )

    def invalidate_intent(self) -> SelfCaptureSessionSnapshot:
        self._desired_active = False
        self._generation += 1
        self._notify_state_changed()
        return self.snapshot

    def guard_vad_sink(self, generation: int | None = None) -> object:
        capture_generation = _CaptureGeneration(
            self._generation if generation is None else generation
        )
        config = self._config
        if config is None:
            raise RuntimeError("Self capture configuration is unavailable")
        ledger = PeerAudioSegmentLedger(
            activation_generation=capture_generation.value,
            settings=self._segment_settings(config),
        )
        self._segment_ledgers.append(ledger)
        return _GenerationGuardedVadSink(
            sink=self._vad_sink,
            owner=self,
            capture_generation=capture_generation,
            ledger=ledger,
        )

    def note_recognition_terminal(self, terminal: STTProviderTurnTerminal) -> None:
        segment_id = terminal.identity.segment.segment_id
        for ledger in reversed(self._segment_ledgers):
            if not ledger.contains_segment(segment_id):
                continue
            snapshot = next(
                (
                    item
                    for item in ledger.snapshots
                    if item.identity.segment_id == segment_id
                ),
                None,
            )
            if snapshot is not None and snapshot.state == "open":
                ledger.terminalize_for_failure(
                    segment_id,
                    now_monotonic_s=asyncio.get_running_loop().time(),
                    failure_reason=terminal.failure_reason or terminal.outcome,
                    provider_epoch_id=terminal.identity.provider_epoch_id,
                    provider_turn_id=terminal.identity.provider_turn_id,
                    text_authority=terminal.text_authority,
                    outcome=terminal.outcome,
                )
            else:
                ledger.terminalize(
                    segment_id,
                    outcome=terminal.outcome,
                    now_monotonic_s=asyncio.get_running_loop().time(),
                    provider_epoch_id=terminal.identity.provider_epoch_id,
                    provider_turn_id=terminal.identity.provider_turn_id,
                    text_authority=terminal.text_authority,
                    failure_reason=terminal.failure_reason,
                )
            break
        reason = terminal.failure_reason
        if (
            reason
            and not reason.startswith("recognition_admission_")
            and terminal.outcome not in ("final", "empty", "suppressed", "cancelled")
        ):
            self.note_recognition_failure(reason)

    def note_recognition_failure(self, reason: str) -> None:
        if (
            self._closed
            or not self._desired_active
            or self._state is not SelfCaptureSessionState.RUNNING
        ):
            return
        generation = self._generation
        failure_reason = (
            SelfCaptureFailureReason.PROVIDER_FAILED
            if reason.startswith("provider_")
            else SelfCaptureFailureReason.SESSION_FAILED
        )
        task = asyncio.create_task(
            self._fault_generation(
                generation,
                failure_reason,
                RuntimeError(reason),
            ),
            name=f"SelfCaptureSessionOwner:recognition-fault:{generation}",
        )
        self._fault_tasks.add(task)
        task.add_done_callback(self._fault_tasks.discard)

    async def apply_intent(
        self,
        config: SelfCaptureSessionConfig,
        *,
        enabled: bool,
        restart: bool = False,
        force_immediate: bool = False,
        explicit_toggle_off: bool = True,
    ) -> SelfCaptureSessionSnapshot:
        async with self._state_lock:
            if self._closed:
                raise RuntimeError("SelfCaptureSessionOwner is closed")
            self._generation += 1
            generation = self._generation
            self._desired_active = enabled
            if (
                enabled
                and not restart
                and self._state is SelfCaptureSessionState.RUNNING
                and self._config is not None
                and self._config.capture_signature == config.capture_signature
            ):
                self._rebind_capture_generation(generation)
            previous_task = self._transition_task
            transition_task = asyncio.create_task(
                self._apply_generation(
                    generation,
                    config,
                    enabled=enabled,
                    restart=restart,
                    force_immediate=force_immediate,
                    explicit_toggle_off=explicit_toggle_off,
                ),
                name="SelfCaptureSessionOwner:transition",
            )
            self._transition_task = transition_task
            self._emit(
                SelfCaptureDiagnosticEvent.INTENT_CHANGED,
                generation=generation,
            )
            self._notify_state_changed()
        if previous_task is not None and previous_task is not asyncio.current_task():
            if not previous_task.done():
                previous_task.cancel()
            await asyncio.gather(previous_task, return_exceptions=True)
        try:
            await transition_task
        except asyncio.CancelledError:
            if generation == self._generation and not self._closed:
                raise
        finally:
            async with self._state_lock:
                if self._transition_task is transition_task:
                    self._transition_task = None
        return self.snapshot

    async def prepare_provider(
        self,
        config: SelfCaptureSessionConfig,
    ) -> SelfCaptureSessionSnapshot:
        if self._desired_active or self._source is not None or self._loop_task is not None:
            return await self.apply_intent(config, enabled=True)
        async with self._state_lock:
            if self._closed:
                raise RuntimeError("SelfCaptureSessionOwner is closed")
            self._generation += 1
            generation = self._generation
            self._config = config
            self._state = SelfCaptureSessionState.STARTING
            self._provider_status = SelfCaptureProviderStatus.PENDING
            self._failure_reason = None
            self._notify_state_changed()
        async with self._activation_lock:
            if self._is_superseded(generation):
                return self.snapshot
            attachment_token = self._provider_attachment_token
            if self._attached_provider_matches(config):
                result_status = SelfCaptureProviderMutationStatus.APPLIED
                failure_reason = None
            else:
                attachment_token = object()
                try:
                    result = await self._provider.replace(
                        self._provider_request_factory(config, False),
                        start=False,
                        on_terminal_failure=lambda exc: self._on_terminal_provider_failure(
                            exc,
                            attachment_token=attachment_token,
                        ),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    result_status = SelfCaptureProviderMutationStatus.FAILED
                    failure_reason = type(exc).__name__
                else:
                    result_status = result.status
                    failure_reason = result.reason
            if self._is_superseded(generation):
                return self.snapshot
            if result_status is SelfCaptureProviderMutationStatus.APPLIED:
                self._provider_signature = config.provider_signature
                self._commit_provider_attachment(attachment_token)
                self._provider_status = SelfCaptureProviderStatus.READY
                self._state = SelfCaptureSessionState.STOPPED
                self._emit(SelfCaptureDiagnosticEvent.PROVIDER_CHANGED, generation=generation)
            elif result_status is SelfCaptureProviderMutationStatus.PENDING:
                self._provider_status = SelfCaptureProviderStatus.PENDING
                self._state = SelfCaptureSessionState.ADMISSION_PENDING
                self._admission_reason = failure_reason
            elif result_status is SelfCaptureProviderMutationStatus.SUPERSEDED:
                self._provider_status = SelfCaptureProviderStatus.DETACHED
                self._state = SelfCaptureSessionState.STOPPED
            else:
                self._provider_status = SelfCaptureProviderStatus.FAILED
                self._state = SelfCaptureSessionState.FAULTED
                self._failure_reason = SelfCaptureFailureReason.PROVIDER_FAILED
                self._emit(
                    SelfCaptureDiagnosticEvent.FAILURE,
                    generation=generation,
                    reason=SelfCaptureFailureReason.PROVIDER_FAILED,
                    detail=failure_reason,
                )
            self._notify_state_changed()
            return self.snapshot

    async def cancel(self) -> SelfCaptureSessionSnapshot:
        config = self._config
        if config is None:
            async with self._state_lock:
                self._generation += 1
                self._desired_active = False
                self._state = SelfCaptureSessionState.STOPPED
                self._notify_state_changed()
            return self.snapshot
        return await self.apply_intent(
            config,
            enabled=False,
            force_immediate=True,
            explicit_toggle_off=True,
        )

    async def release_for_microphone_test(self) -> SelfCaptureSessionSnapshot:
        return await self.cancel()

    async def suspend_provider_consumer(self) -> SelfCaptureSessionSnapshot:
        async with self._state_lock:
            if self._closed:
                raise RuntimeError("SelfCaptureSessionOwner is closed")
            self._generation += 1
            generation = self._generation
            transition_task = self._transition_task
            self._transition_task = None
            self._state = SelfCaptureSessionState.STOPPING
            self._notify_state_changed()
        if transition_task is not None and transition_task is not asyncio.current_task():
            if not transition_task.done():
                transition_task.cancel()
            await asyncio.gather(transition_task, return_exceptions=True)
        async with self._activation_lock:
            await self._teardown(
                generation=generation,
                target_state=SelfCaptureSessionState.STOPPED,
                release_mode="drain",
                preserve_intent=True,
                release_provider=False,
            )
        return self.snapshot

    def prepare_provider_recovery(
        self,
        config: SelfCaptureSessionConfig,
    ) -> SelfCaptureTerminalFailureHandler:
        if self._closed:
            raise RuntimeError("SelfCaptureSessionOwner is closed")
        attachment_token = object()

        async def on_terminal_failure(exc: Exception) -> None:
            await self._on_terminal_provider_failure(
                exc,
                attachment_token=attachment_token,
            )

        self._pending_provider_recoveries[on_terminal_failure] = (
            config.provider_signature,
            attachment_token,
            None,
        )
        return on_terminal_failure

    def abort_provider_recovery(
        self,
        on_terminal_failure: SelfCaptureTerminalFailureHandler,
    ) -> bool:
        return self._pending_provider_recoveries.pop(on_terminal_failure, None) is not None

    async def adopt_recovered_provider(
        self,
        config: SelfCaptureSessionConfig,
        *,
        on_terminal_failure: SelfCaptureTerminalFailureHandler,
    ) -> SelfCaptureSessionSnapshot:
        async with self._activation_lock:
            pending = self._pending_provider_recoveries.get(on_terminal_failure)
            if pending is None or pending[0] != config.provider_signature:
                if self._provider.is_ready(config):
                    await self._release_provider(mode="abort")
                raise RuntimeError("recovered Self provider has no matching owner callback")
            if not self._provider.is_ready(config):
                self.abort_provider_recovery(on_terminal_failure)
                raise RuntimeError("recovered Self provider is not attached")
            attachment_token = pending[1]
            pending_failure = pending[2]
            async with self._state_lock:
                if self._closed:
                    raise RuntimeError("SelfCaptureSessionOwner is closed")
                if self._source is not None or self._loop_task is not None:
                    raise RuntimeError("Self provider recovery requires suspended capture")
                self._config = config
                self._provider_signature = config.provider_signature
                self._commit_provider_attachment(
                    attachment_token,
                    recovery_handler=on_terminal_failure,
                )
                self._provider_status = SelfCaptureProviderStatus.READY
                self._state = SelfCaptureSessionState.STOPPED
                self._notify_state_changed()
            if pending_failure is not None:
                if self._desired_active:
                    await self._fault_generation_locked(
                        self._generation,
                        SelfCaptureFailureReason.PROVIDER_FAILED,
                        pending_failure,
                    )
                else:
                    self._provider_status = SelfCaptureProviderStatus.FAILED
                    self._state = SelfCaptureSessionState.FAULTED
                    self._failure_reason = SelfCaptureFailureReason.PROVIDER_FAILED
                    self._emit(
                        SelfCaptureDiagnosticEvent.FAILURE,
                        generation=self._generation,
                        reason=SelfCaptureFailureReason.PROVIDER_FAILED,
                        detail=type(pending_failure).__name__,
                    )
                    self._notify_state_changed()
        return self.snapshot

    async def close(self) -> None:
        async with self._state_lock:
            self._closed = True
            self._desired_active = False
            self._generation += 1
            generation = self._generation
            transition_task = self._transition_task
            self._transition_task = None
            self._state = SelfCaptureSessionState.STOPPING
            self._notify_state_changed()
        if transition_task is not None and transition_task is not asyncio.current_task():
            if not transition_task.done():
                transition_task.cancel()
            await asyncio.gather(transition_task, return_exceptions=True)
        async with self._activation_lock:
            await self._teardown(
                generation=generation,
                target_state=SelfCaptureSessionState.STOPPED,
                release_mode="abort",
            )
        fault_tasks = tuple(self._fault_tasks)
        for task in fault_tasks:
            if not task.done():
                task.cancel()
        if fault_tasks:
            await asyncio.gather(*fault_tasks, return_exceptions=True)

    async def _apply_generation(
        self,
        generation: int,
        config: SelfCaptureSessionConfig,
        *,
        enabled: bool,
        restart: bool,
        force_immediate: bool,
        explicit_toggle_off: bool,
    ) -> None:
        async with self._activation_lock:
            if self._is_superseded(generation):
                return
            if not enabled:
                self._state = SelfCaptureSessionState.STOPPING
                if self._provider_status is SelfCaptureProviderStatus.DETACHED:
                    self._provider_status = SelfCaptureProviderStatus.READY
                self._notify_state_changed()
                mode, release_backend_after = self._release_plan(
                    config,
                    force_immediate=force_immediate,
                    explicit_toggle_off=explicit_toggle_off,
                )
                await self._teardown(
                    generation=generation,
                    target_state=SelfCaptureSessionState.STOPPED,
                    release_mode=mode,
                    release_backend_after=release_backend_after,
                )
                return
            if (
                not restart
                and self._state is SelfCaptureSessionState.RUNNING
                and self._config is not None
            ):
                self._rebind_capture_generation(generation)
                if self._config.runtime_signature == config.runtime_signature:
                    self._config = config
                    self._failure_reason = None
                    self._notify_state_changed()
                    return
                if self._config.capture_signature == config.capture_signature:
                    await self._transition_provider(generation, config)
                    return
            if self._source is not None or self._loop_task is not None or self._vad is not None:
                mode, release_backend_after = self._release_plan(
                    self._config or config,
                    force_immediate=force_immediate,
                    explicit_toggle_off=False,
                )
                await self._teardown(
                    generation=generation,
                    target_state=SelfCaptureSessionState.STOPPED,
                    release_mode=mode,
                    release_backend_after=release_backend_after,
                    preserve_intent=True,
                )
                if self._is_stale(generation):
                    return
            await self._start_generation(generation, config)

    async def _start_generation(
        self,
        generation: int,
        config: SelfCaptureSessionConfig,
    ) -> None:
        self._state = SelfCaptureSessionState.STARTING
        self._failure_reason = None
        self._admission_reason = None
        self._config = config
        self._notify_state_changed()
        try:
            admission = await self._admission.admit(config)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_start(
                generation,
                config,
                SelfCaptureFailureReason.ADMISSION_REJECTED,
                exc,
            )
            return
        if self._is_stale(generation):
            return
        self._admission_reason = admission.reason
        self._emit(SelfCaptureDiagnosticEvent.ADMISSION_CHANGED, generation=generation)
        if admission.status is SelfCaptureAdmissionStatus.PENDING:
            self._state = SelfCaptureSessionState.ADMISSION_PENDING
            self._provider_status = SelfCaptureProviderStatus.PENDING
            self._notify_state_changed()
            return
        if admission.status is SelfCaptureAdmissionStatus.REJECTED:
            self._desired_active = admission.retain_intent
            await self._fail_start(
                generation,
                config,
                SelfCaptureFailureReason.ADMISSION_REJECTED,
            )
            return

        attachment_token = self._provider_attachment_token
        provider_was_ready = self._attached_provider_matches(config)
        if not provider_was_ready:
            attachment_token = object()
            self._provider_status = SelfCaptureProviderStatus.PENDING
            self._notify_state_changed()
            try:
                result = await self._provider.replace(
                    self._provider_request_factory(config, config.local_cpu or config.local_gpu),
                    start=False,
                    on_terminal_failure=lambda exc: self._on_terminal_provider_failure(
                        exc,
                        attachment_token=attachment_token,
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._fail_start(
                    generation,
                    config,
                    SelfCaptureFailureReason.PROVIDER_FAILED,
                    exc,
                )
                return
            if self._is_stale(generation):
                await self._release_provider(mode="abort")
                return
            if result.status is SelfCaptureProviderMutationStatus.PENDING:
                self._state = SelfCaptureSessionState.ADMISSION_PENDING
                self._provider_status = SelfCaptureProviderStatus.PENDING
                self._admission_reason = result.reason
                self._notify_state_changed()
                return
            if result.status is SelfCaptureProviderMutationStatus.SUPERSEDED:
                self._state = SelfCaptureSessionState.STOPPED
                self._provider_status = SelfCaptureProviderStatus.DETACHED
                self._notify_state_changed()
                return
            if result.status is not SelfCaptureProviderMutationStatus.APPLIED:
                await self._fail_start(
                    generation,
                    config,
                    SelfCaptureFailureReason.PROVIDER_FAILED,
                )
                return
        self._provider_status = SelfCaptureProviderStatus.READY
        self._provider_signature = config.provider_signature
        if not provider_was_ready:
            self._commit_provider_attachment(attachment_token)
        self._emit(SelfCaptureDiagnosticEvent.PROVIDER_CHANGED, generation=generation)

        try:
            vad = self._vad_factory(config)
        except Exception as exc:
            await self._fail_start(
                generation,
                config,
                SelfCaptureFailureReason.VAD_FAILED,
                exc,
                release_provider=True,
            )
            return
        try:
            source = self._source_factory(config)
            if inspect.isawaitable(source):
                source = await source
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail_start(
                generation,
                config,
                SelfCaptureFailureReason.SOURCE_OPEN_FAILED,
                exc,
                release_provider=True,
            )
            return
        if self._is_stale(generation):
            await self._close_source(source)
            await self._release_provider(mode="abort")
            return

        self._source = source
        self._vad = vad
        capture_generation = _CaptureGeneration(generation)
        self._capture_generation = capture_generation
        loop_task = asyncio.create_task(
            self._run_loop_guarded(
                source=source,
                vad=vad,
                config=config,
                capture_generation=capture_generation,
            ),
            name="SelfCaptureSessionOwner:session-loop",
        )
        self._loop_task = loop_task
        loop_task.add_done_callback(
            lambda task: self._on_loop_task_done(
                task,
                generation=capture_generation.value,
            )
        )
        try:
            await self._provider.start_ingress()
        except Exception as exc:
            await self._fault_generation_locked(
                generation,
                SelfCaptureFailureReason.PROVIDER_FAILED,
                exc,
            )
            return
        self._state = SelfCaptureSessionState.RUNNING
        self._notify_state_changed()
        if config.warmup:
            try:
                await self._provider.warmup()
            except Exception as exc:
                self._emit(
                    SelfCaptureDiagnosticEvent.FAILURE,
                    generation=generation,
                    reason=SelfCaptureFailureReason.PROVIDER_FAILED,
                    detail=type(exc).__name__,
                )

    async def _transition_provider(
        self,
        generation: int,
        config: SelfCaptureSessionConfig,
    ) -> None:
        previous_config = self._config
        attachment_token = self._provider_attachment_token
        self._provider_status = SelfCaptureProviderStatus.PENDING
        self._notify_state_changed()
        try:
            if self._provider_signature == config.provider_signature:
                if config.session_options is not None:
                    await self._provider.reconfigure(config.session_options)
                result_status = SelfCaptureProviderMutationStatus.APPLIED
            else:
                attachment_token = object()
                result = await self._provider.handoff(
                    self._provider_request_factory(config, True),
                    start=True,
                    on_terminal_failure=lambda exc: self._on_terminal_provider_failure(
                        exc,
                        attachment_token=attachment_token,
                    ),
                )
                result_status = result.status
        except asyncio.CancelledError:
            await self._provider.cancel_handoff()
            raise
        except Exception as exc:
            await self._provider.cancel_handoff()
            self._provider_status = SelfCaptureProviderStatus.READY
            self._failure_reason = SelfCaptureFailureReason.PROVIDER_FAILED
            self._config = previous_config
            self._emit(
                SelfCaptureDiagnosticEvent.FAILURE,
                generation=generation,
                reason=SelfCaptureFailureReason.PROVIDER_FAILED,
                detail=type(exc).__name__,
            )
            self._notify_state_changed()
            return
        if self._is_stale(generation):
            await self._provider.cancel_handoff()
            return
        if result_status is SelfCaptureProviderMutationStatus.APPLIED:
            self._config = config
            dispatch = self._vad_dispatch
            if dispatch is not None:
                dispatch.ledger.rebind(
                    activation_generation=generation,
                    settings=self._segment_settings(config),
                )
            self._provider_signature = config.provider_signature
            self._commit_provider_attachment(attachment_token)
            self._provider_status = SelfCaptureProviderStatus.READY
            self._failure_reason = None
            self._emit(SelfCaptureDiagnosticEvent.PROVIDER_CHANGED, generation=generation)
            self._notify_state_changed()
            return
        if result_status is SelfCaptureProviderMutationStatus.PENDING:
            self._provider_status = SelfCaptureProviderStatus.PENDING
            self._config = previous_config
            self._notify_state_changed()
            return
        await self._provider.cancel_handoff()
        self._provider_status = SelfCaptureProviderStatus.READY
        self._config = previous_config
        if result_status is SelfCaptureProviderMutationStatus.FAILED:
            self._failure_reason = SelfCaptureFailureReason.PROVIDER_FAILED
            self._emit(
                SelfCaptureDiagnosticEvent.FAILURE,
                generation=generation,
                reason=SelfCaptureFailureReason.PROVIDER_FAILED,
            )
        self._notify_state_changed()

    async def _run_loop_guarded(
        self,
        *,
        source: object,
        vad: object,
        config: SelfCaptureSessionConfig,
        capture_generation: _CaptureGeneration,
    ) -> None:
        ledger = PeerAudioSegmentLedger(
            activation_generation=capture_generation.value,
            settings=self._segment_settings(config),
        )
        self._segment_ledgers.append(ledger)
        dispatch = _GenerationGuardedVadSink(
            sink=self._vad_sink,
            owner=self,
            capture_generation=capture_generation,
            ledger=ledger,
        )
        self._vad_dispatch = dispatch
        try:
            await self._run_audio_loop(
                source=source,
                vad=vad,
                sink=dispatch,
                target_sample_rate_hz=config.target_sample_rate_hz,
            )
            await dispatch.finish()
        finally:
            if self._vad_dispatch is dispatch:
                self._vad_dispatch = None

    def _rebind_capture_generation(self, generation: int) -> None:
        capture_generation = self._capture_generation
        if capture_generation is not None:
            capture_generation.value = generation

    def _on_loop_task_done(
        self,
        task: asyncio.Task[None],
        *,
        generation: int,
    ) -> None:
        if task.cancelled() or generation != self._generation or self._closed:
            return
        exception: BaseException | None
        try:
            exception = task.exception()
        except asyncio.CancelledError:
            return
        fault_task = asyncio.create_task(
            self._fault_generation(
                generation,
                SelfCaptureFailureReason.SESSION_FAILED,
                exception if isinstance(exception, Exception) else None,
                completed_task=task,
            ),
            name="SelfCaptureSessionOwner:session-fault",
        )
        self._fault_tasks.add(fault_task)
        fault_task.add_done_callback(self._fault_tasks.discard)

    async def _on_terminal_provider_failure(
        self,
        exc: Exception,
        *,
        attachment_token: object,
    ) -> None:
        async with self._activation_lock:
            for handler, pending in tuple(self._pending_provider_recoveries.items()):
                if attachment_token is pending[1]:
                    self._pending_provider_recoveries[handler] = (
                        pending[0],
                        pending[1],
                        exc,
                    )
                    return
            if (
                self._closed
                or attachment_token is not self._provider_attachment_token
                or self._provider_status
                in {
                    SelfCaptureProviderStatus.DETACHED,
                    SelfCaptureProviderStatus.RELEASING,
                }
            ):
                return
            generation = self._generation
            if self._desired_active:
                await self._fault_generation_locked(
                    generation,
                    SelfCaptureFailureReason.PROVIDER_FAILED,
                    exc,
                )
                return
            self._provider_status = SelfCaptureProviderStatus.FAILED
            self._state = SelfCaptureSessionState.FAULTED
            self._failure_reason = SelfCaptureFailureReason.PROVIDER_FAILED
            self._emit(
                SelfCaptureDiagnosticEvent.FAILURE,
                generation=generation,
                reason=SelfCaptureFailureReason.PROVIDER_FAILED,
                detail=type(exc).__name__,
            )
            self._notify_state_changed()

    async def _fault_generation(
        self,
        generation: int,
        reason: SelfCaptureFailureReason,
        exc: Exception | None = None,
        *,
        completed_task: asyncio.Task[None] | None = None,
    ) -> None:
        async with self._activation_lock:
            await self._fault_generation_locked(
                generation,
                reason,
                exc,
                completed_task=completed_task,
            )

    async def _fault_generation_locked(
        self,
        generation: int,
        reason: SelfCaptureFailureReason,
        exc: Exception | None = None,
        *,
        completed_task: asyncio.Task[None] | None = None,
    ) -> None:
        if self._is_stale(generation):
            return
        self._desired_active = False
        self._generation += 1
        teardown_generation = self._generation
        self._failure_reason = reason
        self._emit(
            SelfCaptureDiagnosticEvent.FAILURE,
            generation=generation,
            reason=reason,
            detail=type(exc).__name__ if exc is not None else None,
        )
        await self._teardown(
            generation=teardown_generation,
            target_state=SelfCaptureSessionState.FAULTED,
            release_mode="abort",
            completed_task=completed_task,
        )

    async def _fail_start(
        self,
        generation: int,
        config: SelfCaptureSessionConfig,
        reason: SelfCaptureFailureReason,
        exc: Exception | None = None,
        *,
        release_provider: bool = False,
    ) -> None:
        if self._is_superseded(generation):
            return
        self._failure_reason = reason
        self._state = SelfCaptureSessionState.FAULTED
        if reason is not SelfCaptureFailureReason.ADMISSION_REJECTED:
            self._desired_active = False
        if release_provider:
            await self._release_provider(mode="abort")
        self._provider_status = (
            SelfCaptureProviderStatus.FAILED
            if reason is SelfCaptureFailureReason.PROVIDER_FAILED
            else SelfCaptureProviderStatus.DETACHED
        )
        self._emit(
            SelfCaptureDiagnosticEvent.FAILURE,
            generation=generation,
            reason=reason,
            detail=type(exc).__name__ if exc is not None else None,
        )
        self._notify_state_changed()

    async def _teardown(
        self,
        *,
        generation: int,
        target_state: SelfCaptureSessionState,
        release_mode: Literal["drain", "abort"],
        release_backend_after: float | None = None,
        preserve_intent: bool = False,
        completed_task: asyncio.Task[None] | None = None,
        release_provider: bool = True,
    ) -> None:
        if generation != self._generation:
            return
        loop_task = self._loop_task
        source = self._source
        self._loop_task = None
        self._source = None
        self._vad = None
        self._capture_generation = None
        vad_dispatch = self._vad_dispatch
        self._vad_dispatch = None
        self._state = SelfCaptureSessionState.STOPPING
        self._notify_state_changed()
        failures: list[Exception] = []
        prior_cleanup_debt = tuple(self._retired_sources)
        if loop_task is not None and loop_task is not completed_task:
            if not loop_task.done():
                loop_task.cancel()
            await asyncio.gather(loop_task, return_exceptions=True)
        if vad_dispatch is not None:
            await vad_dispatch.abort()
        if source is not None:
            try:
                await self._close_source(source)
            except Exception as exc:
                self._retain_source(source)
                failures.append(exc)
        await self._retry_cleanup_debt(failures, prior_cleanup_debt)
        if self._audio_gate_reset is not None:
            try:
                self._audio_gate_reset()
            except Exception as exc:
                failures.append(exc)
        if release_provider and self._provider_status is not SelfCaptureProviderStatus.DETACHED:
            self._provider_status = SelfCaptureProviderStatus.RELEASING
            self._notify_state_changed()
            try:
                await self._provider.release(
                    mode=release_mode,
                    release_backend_after=release_backend_after,
                )
            except Exception as exc:
                failures.append(exc)
            else:
                self._provider_status = SelfCaptureProviderStatus.DETACHED
                self._retire_provider_attachment()
                if release_mode == "abort":
                    self._provider_signature = None
        if not preserve_intent:
            self._desired_active = False
        if failures:
            self._last_cleanup_exception = failures[0]
            self._failure_reason = SelfCaptureFailureReason.CLEANUP_FAILED
            self._state = SelfCaptureSessionState.FAULTED
            self._emit(
                SelfCaptureDiagnosticEvent.FAILURE,
                generation=generation,
                reason=SelfCaptureFailureReason.CLEANUP_FAILED,
                detail=type(failures[0]).__name__,
            )
            self._notify_state_changed()
            raise failures[0]
        self._last_cleanup_exception = None
        self._state = target_state
        if target_state is not SelfCaptureSessionState.FAULTED:
            self._failure_reason = None
        self._emit(SelfCaptureDiagnosticEvent.SESSION_CHANGED, generation=generation)
        self._notify_state_changed()

    @staticmethod
    def _segment_settings(config: SelfCaptureSessionConfig) -> AudioSegmentSettingsSnapshot:
        return AudioSegmentSettingsSnapshot(
            provider_id=config.provider_id,
            provider_signature=config.provider_signature,
            runtime_signature=config.runtime_signature,
            source_mode=config.source_mode,
            source_language=config.source_language,
            expected_languages=config.expected_languages,
            target_sample_rate_hz=config.target_sample_rate_hz,
            vad_speech_threshold=config.vad_speech_threshold,
            vad_hangover_ms=config.vad_hangover_ms,
            vad_pre_roll_ms=config.ring_buffer_ms,
        )

    async def _release_provider(
        self,
        *,
        mode: Literal["drain", "abort"],
        release_backend_after: float | None = None,
    ) -> None:
        self._provider_status = SelfCaptureProviderStatus.RELEASING
        try:
            await self._provider.release(
                mode=mode,
                release_backend_after=release_backend_after,
            )
        finally:
            self._provider_status = SelfCaptureProviderStatus.DETACHED
            self._retire_provider_attachment()
            if mode == "abort":
                self._provider_signature = None

    def _commit_provider_attachment(
        self,
        attachment_token: object | None,
        *,
        recovery_handler: SelfCaptureTerminalFailureHandler | None = None,
    ) -> None:
        self._provider_attachment_token = attachment_token
        if recovery_handler is None:
            self._pending_provider_recoveries.clear()
        else:
            self._pending_provider_recoveries.pop(recovery_handler, None)

    def _retire_provider_attachment(self) -> None:
        self._commit_provider_attachment(None)

    async def _close_source(self, source: object) -> None:
        close = getattr(source, "close", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    async def _retry_cleanup_debt(
        self,
        failures: list[Exception],
        sources: tuple[object, ...],
    ) -> None:
        for source in sources:
            try:
                await self._close_source(source)
            except Exception as exc:
                failures.append(exc)
            else:
                self._retired_sources = [
                    item for item in self._retired_sources if item is not source
                ]

    def _retain_source(self, source: object) -> None:
        if not any(item is source for item in self._retired_sources):
            self._retired_sources.append(source)

    def _attached_provider_matches(self, config: SelfCaptureSessionConfig) -> bool:
        return (
            self._provider.is_ready(config)
            and self._provider_signature == config.provider_signature
        )

    def _release_plan(
        self,
        config: SelfCaptureSessionConfig,
        *,
        force_immediate: bool,
        explicit_toggle_off: bool,
    ) -> tuple[Literal["drain", "abort"], float | None]:
        if force_immediate or config.local_gpu or explicit_toggle_off:
            return "abort", None
        return "drain", config.release_backend_after if config.local_cpu else None

    def _is_stale(self, generation: int) -> bool:
        return self._is_superseded(generation) or not self._desired_active

    def _is_superseded(self, generation: int) -> bool:
        return generation != self._generation or self._closed

    def _notify_state_changed(self) -> None:
        if self._state_changed is not None:
            self._state_changed(self.snapshot)

    def _emit(
        self,
        event: SelfCaptureDiagnosticEvent,
        *,
        generation: int,
        reason: SelfCaptureFailureReason | None = None,
        detail: str | None = None,
    ) -> None:
        if self._diagnostic_sink is None:
            return
        self._diagnostic_sink(
            SelfCaptureDiagnostic(
                event=event,
                generation=generation,
                state=self._state,
                provider_id=self._config.provider_id if self._config is not None else None,
                reason=reason,
                detail=detail,
            )
        )


__all__ = ["SelfCaptureSessionOwner"]
