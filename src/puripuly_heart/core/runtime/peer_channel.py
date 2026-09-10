from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Protocol, cast
from uuid import UUID

from puripuly_heart.config.resolved import ResolvedDesktopAudioCaptureTarget, ResolvedSTTConfig
from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    AudioSegmentTerminalReceipt,
    PeerAudioSegmentLedger,
    SegmentTerminalOutcome,
)
from puripuly_heart.core.audio.process_source import (
    ProcessAudioCaptureSetupError,
    ProcessAudioCaptureUnavailableError,
)
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmissionPort,
    PeerCaptureAdmissionStatus,
    PeerCaptureDiagnostic,
    PeerCaptureDiagnosticEvent,
    PeerCaptureFailureReason,
    PeerCaptureProviderMutationStatus,
    PeerCaptureProviderPort,
    PeerCaptureProviderStatus,
    PeerCaptureResolvedTarget,
    PeerCaptureSessionConfig,
    PeerCaptureSessionSnapshot,
    PeerCaptureSessionState,
    PeerCaptureTargetResolverPort,
    PeerCaptureTargetStatus,
    PeerCaptureTerminalFailureHandler,
)
from puripuly_heart.core.runtime.local_asr_transition import (
    LocalASRSessionOptions,
    LocalASRTransitionCoordinator,
    LocalASRTransitionDiagnosticSink,
    LocalASRTransitionRequest,
    PreparedLocalASRTransition,
)
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart

_LOCAL_ASR_PROVIDERS = frozenset(
    {
        "local_cpu_auto",
        "local_parakeet_v3",
        "local_parakeet_ja",
        "local_qwen",
        "local_qwen_gpu",
    }
)

PeerChannelRuntimeState = PeerCaptureSessionState
PeerRuntimeFailureReason = PeerCaptureFailureReason


class PeerLocalASRTransitionSuperseded(RuntimeError):
    pass


class _PeerCaptureTargetUnavailable(RuntimeError):
    pass


class _PeerCaptureAdmissionRejected(RuntimeError):
    pass


class _PeerCaptureVadFailed(RuntimeError):
    pass


class _PeerCaptureSourceOpenFailed(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PeerRuntimeDiagnostic:
    reason: PeerRuntimeFailureReason
    capture_kind: str
    process_unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PeerRuntimeConfig:
    backend: ResolvedSTTConfig
    output_device: str
    vad_threshold: float
    vad_hangover_ms: int
    vad_pre_roll_ms: int
    provider_signature: tuple[object, ...]
    runtime_signature: tuple[object, ...]
    capture_target: ResolvedDesktopAudioCaptureTarget = ResolvedDesktopAudioCaptureTarget(
        kind="default_output_device"
    )
    model_id: str | None = None
    session_options: LocalASRSessionOptions | None = None
    capture_vad_signature: tuple[object, ...] = ()


class SpeechChannelRuntime(Protocol):
    @property
    def state(self) -> PeerChannelRuntimeState: ...

    @property
    def current_signature(self) -> object | None: ...

    async def apply_policy(
        self,
        *,
        config: PeerRuntimeConfig,
        desired_active: bool,
        stop_mode: Literal["retain", "release"] = "retain",
    ) -> None: ...

    async def warmup(self) -> None: ...

    async def suspend_provider_consumer(self) -> None: ...

    async def adopt_recovered_provider(
        self,
        config: PeerRuntimeConfig,
        *,
        on_terminal_failure: PeerCaptureTerminalFailureHandler | None = None,
    ) -> None: ...

    async def close(self) -> None: ...


class _VadSink(Protocol):
    async def handle_vad_event(self, event: object) -> None: ...
    async def handle_owned_vad_event(self, event: object) -> None: ...



@dataclass(slots=True)
class _CaptureGeneration:
    value: int

@dataclass(frozen=True, slots=True)
class _QueuedVadEvent:
    owned: bool
    event: object
    pcm_samples: int
    segment_id: UUID | None
    segment_order: int | None
    content_pcm_samples: int
    context_pcm_samples: int
    opens_segment: bool
    closes_segment: bool
    sealed_at_dispatch_s: float | None


class _GenerationGuardedVadSink:
    _MAX_WHOLE_UNSENT_SEGMENTS = 8
    _MAX_RESERVED_CONTROL_EVENTS = 32
    _SEALED_SEGMENT_TTL_S = 12.0

    def __init__(
        self,
        *,
        sink: object,
        runtime: "PeerCaptureSessionOwner",
        capture_generation: _CaptureGeneration,
        provider_ingress_ready: asyncio.Event,
    ) -> None:
        self.sink = sink
        self.runtime = runtime
        self.capture_generation = capture_generation
        self.provider_ingress_ready = provider_ingress_ready
        self._queue: deque[_QueuedVadEvent] = deque()
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._expiry_task: asyncio.Task[None] | None = None
        self._expiry_deadline_s: float | None = None
        self._closing = False
        self._queued_pcm_samples = 0
        self._queued_content_samples = 0
        self._queued_context_samples = 0
        self._queued_control_events = 0
        self._started_segment_ids: set[UUID] = set()

    async def handle_vad_event(self, event: object) -> None:
        await self._submit(False, event)

    async def handle_owned_vad_event(self, event: object) -> None:
        await self._submit(True, event)

    async def finish(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._closing = True
        self._wake.set()
        await worker
        await self._cancel_expiry()

    async def abort(self) -> None:
        worker = self._worker
        if worker is not None:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        await self._cancel_expiry()

    async def _submit(self, owned: bool, event: object) -> None:
        if not self.runtime.is_current_generation(self.capture_generation.value):
            return
        worker = self._worker
        if worker is None:
            worker = asyncio.create_task(self._run(), name="peer-vad-dispatch")
            self._worker = worker
        elif worker.done():
            await worker
            raise RuntimeError("peer VAD dispatch worker stopped")

        queued = self._describe_event(owned, event)
        self._queue.append(queued)
        self._queued_pcm_samples += queued.pcm_samples
        self._queued_content_samples += queued.content_pcm_samples
        self._queued_context_samples += queued.context_pcm_samples
        if queued.pcm_samples == 0:
            self._queued_control_events += 1

        now = asyncio.get_running_loop().time()
        self._retire_expired_whole_segments(now)
        self._enforce_segment_budget()
        self._validate_pcm_accounting()
        if self._queued_control_events > self._MAX_RESERVED_CONTROL_EVENTS:
            raise RuntimeError("peer VAD dispatch exceeded the control event budget")
        self._arm_expiry_timer()
        self._wake.set()
        await asyncio.sleep(0)

    async def _run(self) -> None:
        while True:
            if not self.provider_ingress_ready.is_set():
                await self.provider_ingress_ready.wait()
                continue
            if not self._queue:
                if self._closing:
                    return
                self._wake.clear()
                await self._wake.wait()
                continue
            queued = self._queue.popleft()
            if queued.opens_segment and queued.segment_id is not None:
                self._started_segment_ids.add(queued.segment_id)
            try:
                if not self.runtime.is_current_generation(self.capture_generation.value):
                    continue
                event = queued.event
                if queued.owned:
                    handler = getattr(self.sink, "handle_owned_vad_event", None)
                    if callable(handler):
                        await handler(event)
                        continue
                    event = getattr(event, "event")
                await cast(_VadSink, self.sink).handle_vad_event(event)
            finally:
                self._release_event_accounting(queued)
                if queued.closes_segment and queued.segment_id is not None:
                    self._started_segment_ids.discard(queued.segment_id)

    def _enforce_segment_budget(self) -> None:
        candidates = self._whole_unsent_sealed_segments()
        while len(candidates) > self._MAX_WHOLE_UNSENT_SEGMENTS:
            self._retire_segment(candidates[0][2], failure_reason="overload")
            candidates = self._whole_unsent_sealed_segments()

    def _validate_pcm_accounting(self) -> None:
        if (
            self._queued_pcm_samples
            > self._queued_content_samples + self._queued_context_samples
        ):
            raise RuntimeError("peer VAD dispatch received PCM without owned range accounting")

    def _whole_unsent_sealed_segments(self) -> list[tuple[float, int, UUID]]:
        opened: dict[UUID, int] = {}
        sealed: dict[UUID, tuple[float, int]] = {}
        for queued in self._queue:
            segment_id = queued.segment_id
            segment_order = queued.segment_order
            if segment_id is None or segment_order is None:
                continue
            if queued.opens_segment:
                opened[segment_id] = segment_order
            if queued.closes_segment and queued.sealed_at_dispatch_s is not None:
                sealed[segment_id] = (queued.sealed_at_dispatch_s, segment_order)
        return sorted(
            (
                sealed_at,
                segment_order,
                segment_id,
            )
            for segment_id, segment_order in opened.items()
            if segment_id not in self._started_segment_ids
            and (sealed_entry := sealed.get(segment_id)) is not None
            for sealed_at, _ in (sealed_entry,)
        )

    def _retire_expired_whole_segments(self, now: float) -> None:
        for sealed_at, _segment_order, segment_id in self._whole_unsent_sealed_segments():
            if now - sealed_at >= self._SEALED_SEGMENT_TTL_S:
                self._retire_segment(
                    segment_id,
                    failure_reason="expired_before_recognition",
                )

    def _retire_segment(self, segment_id: UUID, *, failure_reason: str) -> None:
        retained: deque[_QueuedVadEvent] = deque()
        removed = False
        for queued in self._queue:
            if queued.segment_id == segment_id:
                removed = True
                self._release_event_accounting(queued)
            else:
                retained.append(queued)
        if not removed:
            return
        self._queue = retained
        self.runtime.record_segment_terminal(
            segment_id,
            outcome="expired",
            text_authority="none",
            failure_reason=failure_reason,
        )

    def _arm_expiry_timer(self) -> None:
        candidates = self._whole_unsent_sealed_segments()
        deadline = (
            min(item[0] for item in candidates) + self._SEALED_SEGMENT_TTL_S
            if candidates
            else None
        )
        expiry_task = self._expiry_task
        if (
            deadline is not None
            and expiry_task is not None
            and not expiry_task.done()
            and self._expiry_deadline_s == deadline
        ):
            return
        if expiry_task is not None:
            expiry_task.cancel()
            self._expiry_task = None
        self._expiry_deadline_s = deadline
        if deadline is None:
            return
        self._expiry_task = asyncio.create_task(
            self._expire_at(deadline),
            name="peer-vad-expiry",
        )

    async def _expire_at(self, deadline: float) -> None:
        try:
            loop = asyncio.get_running_loop()
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            self._expiry_task = None
            self._expiry_deadline_s = None
            self._retire_expired_whole_segments(loop.time())
            self._arm_expiry_timer()
            self._wake.set()
        except asyncio.CancelledError:
            raise

    async def _cancel_expiry(self) -> None:
        expiry_task = self._expiry_task
        self._expiry_task = None
        self._expiry_deadline_s = None
        if expiry_task is None:
            return
        expiry_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await expiry_task

    def _release_event_accounting(self, queued: _QueuedVadEvent) -> None:
        self._queued_pcm_samples -= queued.pcm_samples
        self._queued_content_samples -= queued.content_pcm_samples
        self._queued_context_samples -= queued.context_pcm_samples
        if queued.pcm_samples == 0:
            self._queued_control_events -= 1

    @staticmethod
    def _describe_event(owned: bool, event: object) -> _QueuedVadEvent:
        raw_event = getattr(event, "event", event) if owned else event
        segment = getattr(event, "segment", None) if owned else None
        identity = getattr(segment, "identity", None)
        segment_id = getattr(raw_event, "utterance_id", None)
        if not isinstance(segment_id, UUID):
            segment_id = None
        if isinstance(raw_event, SpeechStart):
            pcm_samples = int(raw_event.pre_roll.size + raw_event.chunk.size)
            content_ranges = raw_event.chunk_capture
            opens_segment = True
        elif isinstance(raw_event, SpeechChunk):
            pcm_samples = int(raw_event.chunk.size)
            content_ranges = raw_event.chunk_capture
            opens_segment = False
        else:
            pcm_samples = 0
            content_ranges = ()
            opens_segment = False
        content_pcm_samples = min(
            pcm_samples,
            sum(item.normalized_sample_count for item in content_ranges),
        )
        context_pcm_samples = pcm_samples - content_pcm_samples
        closes_segment = isinstance(raw_event, SpeechEnd)
        return _QueuedVadEvent(
            owned=owned,
            event=event,
            pcm_samples=pcm_samples,
            segment_id=segment_id,
            segment_order=getattr(identity, "segment_order", None),
            content_pcm_samples=content_pcm_samples,
            context_pcm_samples=context_pcm_samples,
            opens_segment=opens_segment,
            closes_segment=closes_segment,
            sealed_at_dispatch_s=(
                asyncio.get_running_loop().time() if closes_segment else None
            ),
        )


PeerCaptureSourceFactory = Callable[
    [PeerCaptureSessionConfig, PeerCaptureResolvedTarget],
    Awaitable[object] | object,
]
PeerCaptureVadFactory = Callable[[PeerCaptureSessionConfig], object]
PeerCaptureAudioLoop = Callable[..., Awaitable[None]]


class PeerCaptureSessionOwner:
    resource_fields = (
        "_audio_source",
        "_vad",
        "_loop_task",
        "_generation",
        "_provider_ingress_ready",
        "_provider_setup_task",
        "_segment_ledger",
        "_desired_active",
        "_lock",
        "_activation_lock",
    )
    stop_ingress = "invalidate generation and desired-active state"
    shutdown_policy = "cancel capture loop, close source, release owner channel"
    late_callback_rule = "late peer callbacks cannot mutate current runtime or output to chatbox"

    def __init__(
        self,
        *,
        admission: PeerCaptureAdmissionPort,
        target_resolver: PeerCaptureTargetResolverPort,
        provider: PeerCaptureProviderPort,
        clock: Clock,
        provider_request_factory: Callable[
            [PeerCaptureSessionConfig, bool],
            object,
        ],
        source_factory: PeerCaptureSourceFactory,
        vad_factory: PeerCaptureVadFactory,
        run_audio_loop: PeerCaptureAudioLoop,
        vad_sink: object,
        state_changed: Callable[[PeerCaptureSessionSnapshot], object] | None = None,
        diagnostic_sink: Callable[[PeerCaptureDiagnostic], object] | None = None,
        local_asr_diagnostic_sink: LocalASRTransitionDiagnosticSink | None = None,
    ) -> None:
        self._admission = admission
        self._target_resolver = target_resolver
        self._provider = provider
        self.clock = clock
        self._provider_request_factory = provider_request_factory
        self._source_factory = source_factory
        self._vad_factory = vad_factory
        self._run_audio_loop = run_audio_loop
        self._vad_sink = vad_sink
        self._state_changed = state_changed
        self._diagnostic_sink = diagnostic_sink
        self._local_asr_diagnostic_sink = local_asr_diagnostic_sink
        self._requested_config: PeerCaptureSessionConfig | None = None
        self._config: PeerCaptureSessionConfig | None = None
        self._resolved_target: PeerCaptureResolvedTarget | None = None
        self._audio_source: object | None = None
        self._vad: object | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._signature: tuple[object, ...] | None = None
        self._provider_signature: tuple[object, ...] | None = None
        self._provider_attachment_token: object | None = None
        self._provider_ingress_ready: asyncio.Event | None = None
        self._provider_setup_task: asyncio.Task[object] | None = None
        self._pending_provider_failures: dict[object, Exception | None] = {}
        self._pending_provider_recoveries: dict[
            PeerCaptureTerminalFailureHandler,
            tuple[tuple[object, ...], object],
        ] = {}
        self._provider_status = PeerCaptureProviderStatus.DETACHED
        self._target_status: PeerCaptureTargetStatus | None = None
        self._state = PeerCaptureSessionState.STOPPED
        self._generation = 0
        self._desired_active = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._activation_lock = asyncio.Lock()
        self._retired_sources: list[object] = []
        self._last_failure: PeerCaptureDiagnostic | None = None
        self._last_failure_unavailable_reason: str | None = None
        self._admission_reason: str | None = None
        self._retry_required_capture_target = None
        self._capture_generation: _CaptureGeneration | None = None
        self._deferred_loop_diagnostics: dict[asyncio.Task[None], PeerCaptureDiagnostic] = {}
        self._segment_ledgers: deque[PeerAudioSegmentLedger] = deque(maxlen=4096)
        self._segment_ledger: PeerAudioSegmentLedger | None = None
        self._transition_coordinator = LocalASRTransitionCoordinator(
            channel="peer",
            clock=clock.now,
            diagnostic_sink=local_asr_diagnostic_sink,
        )
        self._last_local_asr_transition_status = "idle"

    @property
    def state(self) -> PeerChannelRuntimeState:
        return self._state

    @property
    def snapshot(self) -> PeerCaptureSessionSnapshot:
        return PeerCaptureSessionSnapshot(
            state=self._state,
            provider_status=self._provider_status,
            target_status=self._target_status,
            desired_active=self._desired_active,
            effective_active=(
                self._desired_active
                and self._loop_task is not None
                and not self._loop_task.done()
            ),
            generation=self._generation,
            provider_id=self._config.provider_id if self._config is not None else None,
            runtime_signature=self._signature,
            capture_target=(self._config.capture_target if self._config is not None else None),
            resolved_target=self._resolved_target,
            language=self._config.language if self._config is not None else None,
            failure_reason=(self._last_failure.reason if self._last_failure is not None else None),
            admission_reason=self._admission_reason,
            target_reason=self._last_failure_unavailable_reason,
            retry_available=self._retry_required_capture_target is not None,
            has_source=self._audio_source is not None,
            has_vad=self._vad is not None,
            has_loop_task=self._loop_task is not None,
            requested_delivery_profile=(
                "off" if self._requested_config is not None else "off"
            ),
            effective_delivery_profile=(
                "off" if self._segment_ledger is not None else None
            ),
            requested_vad_hangover_ms=(
                self._requested_config.vad_hangover_ms
                if self._requested_config is not None
                else None
            ),
            effective_vad_hangover_ms=(
                self._config.vad_hangover_ms
                if self._config is not None and self._segment_ledger is not None
                else None
            ),
            cleanup_debt=len(self._retired_sources),
            closed=self._closed,
        )

    @property
    def current_signature(self) -> object | None:
        return self._signature

    @property
    def loop_task(self) -> asyncio.Task[None] | None:
        return self._loop_task

    @property
    def source(self) -> object | None:
        return self._audio_source

    @property
    def cleanup_source(self) -> object | None:
        return self._retired_sources[0] if self._retired_sources else None

    @property
    def vad(self) -> object | None:
        return self._vad

    @property
    def current_config(self) -> PeerCaptureSessionConfig | None:
        return self._config

    @property
    def last_failure(self) -> PeerCaptureDiagnostic | None:
        return self._last_failure

    @property
    def last_local_asr_transition_status(self) -> str:
        return self._last_local_asr_transition_status

    @property
    def segment_ledger(self) -> PeerAudioSegmentLedger | None:
        return self._segment_ledger

    @property
    def segment_ledgers(self) -> tuple[PeerAudioSegmentLedger, ...]:
        return tuple(self._segment_ledgers)

    def record_segment_terminal(
        self,
        segment_id: UUID,
        *,
        outcome: SegmentTerminalOutcome,
        provider_epoch_id: str | None = None,
        provider_turn_id: str | None = None,
        native_request_id: str | None = None,
        text_authority: Literal["authoritative", "degraded", "none"] = "none",
        failure_reason: str | None = None,
    ) -> AudioSegmentTerminalReceipt:
        for ledger in reversed(self._segment_ledgers):
            if ledger.contains_segment(segment_id):
                return ledger.terminalize(
                    segment_id,
                    outcome=outcome,
                    now_monotonic_s=self.clock.now(),
                    provider_epoch_id=provider_epoch_id,
                    provider_turn_id=provider_turn_id,
                    native_request_id=native_request_id,
                    text_authority=text_authority,
                    failure_reason=failure_reason,
                )
        raise KeyError(f"unknown peer audio segment: {segment_id}")


    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": "PeerCaptureSessionOwner",
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
            "local_asr_transition": self._transition_coordinator.lifecycle_snapshot(),
        }

    async def apply_intent(
        self,
        config: PeerCaptureSessionConfig,
        *,
        enabled: bool,
        restart: bool = False,
        stop_mode: Literal["retain", "release"] = "retain",
    ) -> PeerCaptureSessionSnapshot:
        if stop_mode not in {"retain", "release"}:
            raise ValueError("stop_mode must be 'retain' or 'release'")
        transition_only = False
        setup_to_cancel: asyncio.Task[object] | None = None
        current_task = asyncio.current_task()
        async with self._lock:
            if self._closed:
                raise RuntimeError("PeerCaptureSessionOwner is closed")
            self._requested_config = config
            if (
                enabled
                and self._desired_active
                and self._state is PeerCaptureSessionState.RUNNING
                and not restart
                and self._signature == config.runtime_signature
            ):
                self._config = config
                self._rebind_capture_generation(self._generation)
                self._notify_state_changed()
                return self.snapshot
            current_config = self._config
            if (
                enabled
                and self._desired_active
                and self._state is PeerCaptureSessionState.RUNNING
                and not restart
                and current_config is not None
                and current_config.capture_signature == config.capture_signature
            ):
                transition_only = True
                if current_config.provider_signature == config.provider_signature:
                    generation = self._generation
                else:
                    self._generation += 1
                    generation = self._generation
                    if self._capture_generation is not None:
                        self._capture_generation.value = generation
                setup_to_cancel = self._provider_setup_task
                self._provider_setup_task = current_task
            else:
                self._generation += 1
                generation = self._generation
                self._config = config
                self._desired_active = enabled
                self._state = (
                    PeerCaptureSessionState.STARTING
                    if enabled
                    else PeerCaptureSessionState.STOPPING
                )
                if enabled:
                    setup_to_cancel = self._provider_setup_task
                    self._provider_setup_task = current_task
                else:
                    setup_to_cancel = self._provider_setup_task
            self._notify_state_changed()
        if setup_to_cancel is not None and setup_to_cancel is not current_task:
            setup_to_cancel.cancel()
        if setup_to_cancel is not None:
            await self._transition_coordinator.cancel_current()
        try:
            async with self._activation_lock:
                if transition_only:
                    await self._transition_running_provider(config, generation=generation)
                    return self.snapshot
                if not enabled:
                    release_mode: Literal["dormant", "abort"] = (
                        "dormant"
                        if stop_mode == "retain"
                        and config.provider_id == "local_qwen"
                        and self._provider_signature == config.provider_signature
                        else "abort"
                    )
                    await self._teardown_resources(
                        target_state=PeerCaptureSessionState.STOPPED,
                        generation=generation,
                        release_mode=release_mode,
                    )
                    return self.snapshot
                await self._start_generation(generation, config)
                return self.snapshot
        except asyncio.CancelledError:
            if self._is_superseded(generation):
                return self.snapshot
            raise
        finally:
            if current_task is not None:
                async with self._lock:
                    if self._provider_setup_task is current_task:
                        self._provider_setup_task = None

    async def apply_policy(
        self,
        *,
        config: PeerCaptureSessionConfig,
        desired_active: bool,
        stop_mode: Literal["retain", "release"] = "retain",
    ) -> None:
        await self.apply_intent(
            config,
            enabled=desired_active,
            stop_mode=stop_mode,
        )

    async def prepare_provider(
        self,
        config: PeerCaptureSessionConfig,
    ) -> PeerCaptureSessionSnapshot:
        async with self._lock:
            if self._closed:
                raise RuntimeError("PeerCaptureSessionOwner is closed")
            if (
                self._desired_active
                or self._audio_source is not None
                or self._loop_task is not None
            ):
                return self.snapshot
            self._generation += 1
            generation = self._generation
            self._config = config
            self._state = PeerCaptureSessionState.PROVIDER_PENDING
            self._provider_status = PeerCaptureProviderStatus.PENDING
            self._last_failure = None
            self._admission_reason = None
            self._notify_state_changed()
        async with self._activation_lock:
            if generation != self._generation or self._closed:
                return self.snapshot
            reusable = (
                self._provider.is_ready(config) and self._provider_attachment_token is not None
            )
            attachment_token = self._provider_attachment_token
            if reusable:
                result_status = PeerCaptureProviderMutationStatus.APPLIED
                failure_reason = None
                pending_failure = None
            else:
                attachment_token = object()
                self._pending_provider_failures[attachment_token] = None
                try:
                    result = await self._provider.replace(
                        self._provider_request_factory(config, config.local_provider),
                        start=False,
                        on_terminal_failure=lambda exc: self._on_terminal_stt_failure(
                            exc,
                            attachment_token=attachment_token,
                        ),
                    )
                except asyncio.CancelledError:
                    self._pending_provider_failures.pop(attachment_token, None)
                    raise
                except Exception as exc:
                    self._pending_provider_failures.pop(attachment_token, None)
                    result_status = PeerCaptureProviderMutationStatus.FAILED
                    failure_reason = type(exc).__name__
                    pending_failure = None
                else:
                    result_status = result.status
                    failure_reason = result.reason
                    pending_failure = self._pending_provider_failures.pop(
                        attachment_token,
                        None,
                    )
            if generation != self._generation or self._closed:
                if not reusable and result_status is PeerCaptureProviderMutationStatus.APPLIED:
                    await self._provider.release(mode="abort")
                return self.snapshot
            if pending_failure is not None:
                await self._fault_current_generation_locked(
                    generation,
                    config=config,
                    reason=PeerCaptureFailureReason.PROVIDER_FAILED,
                )
                return self.snapshot
            if result_status is PeerCaptureProviderMutationStatus.APPLIED:
                self._provider_signature = config.provider_signature
                self._signature = config.runtime_signature
                if not reusable:
                    self._commit_provider_attachment(attachment_token)
                self._provider_status = PeerCaptureProviderStatus.READY
                self._state = PeerCaptureSessionState.STOPPED
                self._emit_event(PeerCaptureDiagnosticEvent.PROVIDER_CHANGED)
            elif result_status is PeerCaptureProviderMutationStatus.PENDING:
                self._provider_status = PeerCaptureProviderStatus.PENDING
                self._state = PeerCaptureSessionState.PROVIDER_PENDING
                self._admission_reason = failure_reason
            elif result_status is PeerCaptureProviderMutationStatus.SUPERSEDED:
                self._provider_status = PeerCaptureProviderStatus.DETACHED
                self._state = PeerCaptureSessionState.STOPPED
            else:
                await self._fault_current_generation_locked(
                    generation,
                    config=config,
                    reason=PeerCaptureFailureReason.PROVIDER_FAILED,
                )
                return self.snapshot
            self._notify_state_changed()
            return self.snapshot

    async def retry_process_capture(
        self,
        *,
        config: PeerCaptureSessionConfig | None = None,
    ) -> bool:
        target_config = config or self._config
        if target_config is None:
            return False
        async with self._lock:
            if (
                target_config.capture_target.kind != "process"
                or self._retry_required_capture_target != target_config.capture_target
                or self._state is not PeerCaptureSessionState.FAULTED
            ):
                return False
            self._generation += 1
            generation = self._generation
            self._config = target_config
            self._desired_active = True
            self._retry_required_capture_target = None
            self._state = PeerCaptureSessionState.STARTING
            self._notify_state_changed()
        async with self._activation_lock:
            await self._start_generation(generation, target_config)
        return self._state is PeerCaptureSessionState.RUNNING

    async def warmup(self) -> None:
        if self._desired_active and self._state is PeerCaptureSessionState.RUNNING:
            await self._provider.warmup()

    async def suspend_provider_consumer(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._generation += 1
            generation = self._generation
            self._desired_active = False
            self._state = PeerCaptureSessionState.STOPPING
            self._notify_state_changed()
        setup_task = self._provider_setup_task
        if setup_task is not None and setup_task is not asyncio.current_task():
            setup_task.cancel()
        await self._transition_coordinator.cancel_current()
        async with self._activation_lock:
            await self._teardown_resources(
                target_state=PeerCaptureSessionState.STOPPED,
                generation=generation,
                release_mode="drain",
                release_provider=False,
            )

    def prepare_provider_recovery(
        self,
        config: PeerCaptureSessionConfig,
    ) -> PeerCaptureTerminalFailureHandler:
        if self._closed:
            raise RuntimeError("PeerCaptureSessionOwner is closed")
        attachment_token = object()
        self._pending_provider_failures[attachment_token] = None

        async def on_terminal_failure(exc: Exception) -> None:
            await self._on_terminal_stt_failure(
                exc,
                attachment_token=attachment_token,
            )

        self._pending_provider_recoveries[on_terminal_failure] = (
            config.provider_signature,
            attachment_token,
        )
        return on_terminal_failure

    def abort_provider_recovery(
        self,
        on_terminal_failure: PeerCaptureTerminalFailureHandler,
    ) -> bool:
        pending = self._pending_provider_recoveries.pop(on_terminal_failure, None)
        if pending is None:
            return False
        self._pending_provider_failures.pop(pending[1], None)
        return True

    async def adopt_recovered_provider(
        self,
        config: PeerCaptureSessionConfig,
        *,
        on_terminal_failure: PeerCaptureTerminalFailureHandler | None = None,
    ) -> None:
        async with self._activation_lock:
            pending = (
                self._pending_provider_recoveries.get(on_terminal_failure)
                if on_terminal_failure is not None
                else None
            )
            if on_terminal_failure is not None and (
                pending is None or pending[0] != config.provider_signature
            ):
                if self._provider.is_ready(config):
                    await self._provider.release(mode="abort")
                raise RuntimeError("recovered Peer provider has no matching owner callback")
            if not self._provider.is_ready(config):
                if on_terminal_failure is not None:
                    self.abort_provider_recovery(on_terminal_failure)
                raise RuntimeError("recovered Peer provider is not attached")
            attachment_token = pending[1] if pending is not None else object()
            pending_failure = self._pending_provider_failures.pop(attachment_token, None)
            async with self._lock:
                if self._closed:
                    raise RuntimeError("PeerCaptureSessionOwner is closed")
                if self._state is not PeerCaptureSessionState.STOPPED:
                    raise RuntimeError("Peer provider recovery requires suspended capture")
                self._config = config
                self._provider_signature = config.provider_signature
                self._commit_provider_attachment(
                    attachment_token,
                    recovery_handler=on_terminal_failure,
                )
                self._provider_status = PeerCaptureProviderStatus.READY
                self._notify_state_changed()
            if pending_failure is not None:
                await self._fault_current_generation_locked(
                    self._generation,
                    config=config,
                    reason=PeerCaptureFailureReason.PROVIDER_FAILED,
                )

    async def handle_terminal_provider_failure(self, exc: Exception) -> None:
        await self._on_terminal_stt_failure(
            exc,
            attachment_token=self._provider_attachment_token,
        )

    async def close(self) -> None:
        if self._closed:
            return
        await self._transition_coordinator.close()
        async with self._lock:
            self._closed = True
            self._generation += 1
            generation = self._generation
            self._desired_active = False
            self._state = PeerCaptureSessionState.STOPPING
            self._notify_state_changed()
        setup_task = self._provider_setup_task
        if setup_task is not None and setup_task is not asyncio.current_task():
            setup_task.cancel()
        async with self._activation_lock:
            await self._teardown_resources(
                target_state=PeerCaptureSessionState.STOPPED,
                generation=generation,
                release_mode="abort",
            )
        self._pending_provider_failures.clear()
        self._pending_provider_recoveries.clear()

    async def _transition_running_provider(
        self,
        config: PeerCaptureSessionConfig,
        *,
        generation: int,
    ) -> None:
        options = config.session_options or LocalASRSessionOptions(
            source_language=config.language.source_language,
            source_mode=config.language.source_mode,
        )
        if self._provider_signature == config.provider_signature:
            await self._provider.reconfigure(options)
            async with self._lock:
                if self._generation == generation and self._desired_active:
                    self._config = config
                    self._provider_signature = config.provider_signature
                    self._signature = config.runtime_signature
                    self._rebind_segment_ledger(generation, config)
            self._last_local_asr_transition_status = "applied"
            return
        transition_request = LocalASRTransitionRequest(
            channel="peer",
            requested_provider=config.provider_id,
            actual_provider=(
                self._config.provider_id if self._config is not None else config.provider_id
            ),
            model_id=config.model_id,
            session_options=options,
            trigger="settings",
        )

        async def prepare(
            prepared_request: LocalASRTransitionRequest,
            transition_generation: int,
        ) -> PreparedLocalASRTransition:
            return PreparedLocalASRTransition(
                request=prepared_request,
                provider=self._provider_request_factory(config, True),
                generation=transition_generation,
            )

        async def commit(prepared: PreparedLocalASRTransition) -> None:
            async with self._lock:
                if self._generation != generation or not self._desired_active:
                    raise RuntimeError("peer provider transition superseded")
            attachment_token = object()
            self._pending_provider_failures[attachment_token] = None
            try:
                result = await self._provider.handoff(
                    prepared.provider,
                    start=True,
                    on_terminal_failure=lambda exc: self._on_terminal_stt_failure(
                        exc,
                        attachment_token=attachment_token,
                    ),
                )
            except asyncio.CancelledError:
                self._pending_provider_failures.pop(attachment_token, None)
                await self._provider.cancel_handoff()
                raise
            except Exception:
                self._pending_provider_failures.pop(attachment_token, None)
                raise
            if result.status is not PeerCaptureProviderMutationStatus.APPLIED:
                self._pending_provider_failures.pop(attachment_token, None)
                raise RuntimeError("owned Peer STT handoff failed")
            pending_failure = self._pending_provider_failures.pop(attachment_token, None)
            async with self._lock:
                if self._generation == generation and self._desired_active:
                    self._config = config
                    self._provider_signature = config.provider_signature
                    self._signature = config.runtime_signature
                    self._commit_provider_attachment(attachment_token)
                    self._rebind_segment_ledger(generation, config)
            if pending_failure is not None:
                await self._fault_current_generation_locked(
                    generation,
                    config=config,
                    reason=PeerCaptureFailureReason.PROVIDER_FAILED,
                )

        outcome = await self._transition_coordinator.request_transition(
            transition_request,
            prepare=prepare,
            commit=commit,
        )
        self._last_local_asr_transition_status = outcome.status

    async def _start_generation(
        self,
        generation: int,
        config: PeerCaptureSessionConfig,
    ) -> None:
        source = None
        provider_ready = False
        load_started_at = self.clock.now()
        try:
            admission = await self._admission.admit(config)
            if self._is_superseded(generation):
                return
            self._emit_event(PeerCaptureDiagnosticEvent.ADMISSION_CHANGED)
            self._admission_reason = admission.reason
            if admission.status is PeerCaptureAdmissionStatus.PENDING:
                self._state = PeerCaptureSessionState.ADMISSION_PENDING
                self._provider_status = PeerCaptureProviderStatus.PENDING
                self._notify_state_changed()
                return
            if admission.status is PeerCaptureAdmissionStatus.REJECTED:
                self._desired_active = admission.retain_intent
                raise _PeerCaptureAdmissionRejected(
                    admission.reason or "peer capture admission rejected"
                )
            self._state = PeerCaptureSessionState.TARGET_RESOLVING
            self._target_status = PeerCaptureTargetStatus.PENDING
            self._notify_state_changed()
            resolution = await self._target_resolver.resolve(config.capture_target)
            if self._is_superseded(generation):
                return
            if (
                resolution.status is not PeerCaptureTargetStatus.RESOLVED
                or resolution.target is None
            ):
                self._last_failure_unavailable_reason = resolution.reason
                raise _PeerCaptureTargetUnavailable(resolution.reason or "target_unavailable")
            self._resolved_target = resolution.target
            self._target_status = PeerCaptureTargetStatus.RESOLVED
            self._emit_event(PeerCaptureDiagnosticEvent.TARGET_CHANGED)
            try:
                source = self._source_factory(config, resolution.target)
                if inspect.isawaitable(source):
                    source = await source
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise _PeerCaptureSourceOpenFailed from exc
            try:
                vad = self._vad_factory(config)
            except Exception as exc:
                raise _PeerCaptureVadFailed from exc
            if self._is_superseded(generation):
                await self._close_if_possible(source)
                return
            segment_ledger = PeerAudioSegmentLedger(
                activation_generation=generation,
                settings=self._segment_settings_snapshot(config),
            )
            provider_ingress_ready = asyncio.Event()
            async with self._lock:
                superseded = self._is_superseded(generation)
                if not superseded:
                    old_loop = self._loop_task
                    old_source = self._audio_source
                    old_segment_ledger = self._segment_ledger
                    self._loop_task = None
                    self._audio_source = None
                    self._vad = None
                    self._segment_ledger = None
                    self._capture_generation = None
                    self._provider_ingress_ready = None
                    self._signature = None
            if superseded:
                await self._close_if_possible(source)
                return
            await self._cancel_loop(old_loop)
            if old_segment_ledger is not None:
                old_segment_ledger.cancel_unfinished(
                    now_monotonic_s=self.clock.now()
                )
            await self._close_if_possible(old_source)
            async with self._lock:
                superseded = self._is_superseded(generation)
                if not superseded:
                    self._audio_source = source
                    self._vad = vad
                    self._signature = config.runtime_signature
                    self._segment_ledger = segment_ledger
                    self._segment_ledgers.append(segment_ledger)
                    capture_generation = _CaptureGeneration(generation)
                    self._capture_generation = capture_generation
                    self._provider_ingress_ready = provider_ingress_ready
                    self._state = PeerCaptureSessionState.STARTING
                    loop_task = self._create_task(
                        self._run_peer_loop_guarded(
                            source=source,
                            vad=vad,
                            target_sample_rate_hz=config.target_sample_rate_hz,
                            capture_generation=capture_generation,
                            segment_ledger=segment_ledger,
                            provider_ingress_ready=provider_ingress_ready,
                        ),
                        task_name="session-loop",
                    )
                    loop_task.add_done_callback(self._on_loop_task_done)
                    self._loop_task = loop_task
            if superseded:
                await self._close_if_possible(source)
                return
            self._notify_state_changed()
            reusable = (
                self._provider.is_ready(config) and self._provider_attachment_token is not None
            )
            attachment_token = self._provider_attachment_token
            if not reusable:
                attachment_token = object()
                self._pending_provider_failures[attachment_token] = None
                self._provider_status = PeerCaptureProviderStatus.PENDING
                self._notify_state_changed()
                request = self._provider_request_factory(
                    config,
                    config.local_provider,
                )
                try:
                    result = await self._provider.replace(
                        request,
                        start=False,
                        on_terminal_failure=lambda exc: self._on_terminal_stt_failure(
                            exc,
                            attachment_token=attachment_token,
                        ),
                    )
                except BaseException:
                    self._pending_provider_failures.pop(attachment_token, None)
                    raise
                if result.status is PeerCaptureProviderMutationStatus.PENDING:
                    self._pending_provider_failures.pop(attachment_token, None)
                    self._state = PeerCaptureSessionState.PROVIDER_PENDING
                    self._provider_status = PeerCaptureProviderStatus.PENDING
                    self._admission_reason = result.reason
                    self._notify_state_changed()
                    return
                if result.status is PeerCaptureProviderMutationStatus.SUPERSEDED:
                    self._pending_provider_failures.pop(attachment_token, None)
                    raise RuntimeError("owned Peer STT replacement was superseded")
                if result.status is not PeerCaptureProviderMutationStatus.APPLIED:
                    self._pending_provider_failures.pop(attachment_token, None)
                    raise RuntimeError("owned Peer STT replacement failed")
            provider_ready = True
            pending_failure = (
                self._pending_provider_failures.pop(attachment_token, None)
                if not reusable
                else None
            )
            if not reusable:
                self._commit_provider_attachment(attachment_token)
            if pending_failure is not None:
                await self._fault_current_generation_locked(
                    generation,
                    config=config,
                    reason=PeerCaptureFailureReason.PROVIDER_FAILED,
                )
                self._emit_local_asr_diagnostic(
                    config,
                    outcome="failed",
                    load_started_at=load_started_at,
                    failure_type=type(pending_failure).__name__,
                )
                return
            if self._is_superseded(generation):
                await self._provider.release(mode="abort")
                return
            await self._provider.start_ingress()
            if self._is_superseded(generation):
                await self._provider.release(mode="abort")
                return
            should_release = False
            async with self._lock:
                should_release = generation != self._generation or not self._desired_active
                if not should_release:
                    self._provider_signature = config.provider_signature
                    self._provider_status = PeerCaptureProviderStatus.READY
                    self._state = PeerCaptureSessionState.RUNNING
                    provider_ingress_ready.set()
            if should_release:
                await self._provider.release(mode="abort")
                return
            self._emit_event(PeerCaptureDiagnosticEvent.PROVIDER_CHANGED)
            self._notify_state_changed()
        except Exception as exc:
            if source is not None and self._audio_source is not source:
                await self._close_if_possible(source)
            if isinstance(
                exc,
                (
                    _PeerCaptureAdmissionRejected,
                    _PeerCaptureTargetUnavailable,
                    _PeerCaptureSourceOpenFailed,
                    _PeerCaptureVadFailed,
                ),
            ):
                reason = self._failure_reason_from_startup_exception(config, exc)
            elif provider_ready:
                reason = self._failure_reason_from_startup_exception(config, exc)
            else:
                reason = PeerCaptureFailureReason.PROVIDER_FAILED
            await self._fault_current_generation_locked(
                generation,
                config=config,
                reason=reason,
            )
            self._emit_local_asr_diagnostic(
                config,
                outcome="failed",
                load_started_at=load_started_at,
                failure_type=type(exc).__name__,
            )

    async def _run_peer_loop_guarded(
        self,
        *,
        source: object,
        vad: object,
        target_sample_rate_hz: int,
        capture_generation: _CaptureGeneration,
        segment_ledger: PeerAudioSegmentLedger,
        provider_ingress_ready: asyncio.Event,
    ) -> None:
        guarded_sink = _GenerationGuardedVadSink(
            sink=self._vad_sink,
            runtime=self,
            capture_generation=capture_generation,
            provider_ingress_ready=provider_ingress_ready,
        )
        try:
            await self._run_audio_loop(
                source=source,
                vad=vad,
                sink=guarded_sink,
                target_sample_rate_hz=target_sample_rate_hz,
                segment_ledger=segment_ledger,
                monotonic_clock=self.clock.now,
            )
            if self._terminal_reason_from_source(source) is None:
                await guarded_sink.finish()
            else:
                await guarded_sink.abort()
        except asyncio.CancelledError:
            await guarded_sink.abort()
            raise
        except Exception as exc:
            await guarded_sink.abort()
            await self._on_runtime_failure(
                exc,
                generation=capture_generation.value,
                config=self._config,
            )
            return
        terminal_reason = self._terminal_reason_from_source(source)
        if terminal_reason is not None:
            await self._fault_current_generation(
                capture_generation.value,
                config=self._config,
                reason=self._failure_reason_from_terminal_source(terminal_reason),
            )
        else:
            await self._complete_current_generation(capture_generation.value)

    async def _complete_current_generation(self, generation: int) -> None:
        async with self._activation_lock:
            async with self._lock:
                if (
                    generation != self._generation
                    or self._closed
                    or self._loop_task is not asyncio.current_task()
                ):
                    return
                self._generation += 1
                teardown_generation = self._generation
                self._desired_active = False
                self._state = PeerCaptureSessionState.STOPPING
                self._notify_state_changed()
            await self._teardown_resources(
                target_state=PeerCaptureSessionState.STOPPED,
                generation=teardown_generation,
                release_mode="drain",
                cancel_segments=False,
            )

    async def _on_runtime_failure(
        self,
        exc: Exception,
        *,
        generation: int,
        config: PeerCaptureSessionConfig | None,
    ) -> None:
        _ = exc
        await self._fault_current_generation(
            generation,
            config=config,
            reason=(
                PeerRuntimeFailureReason.PROCESS_SOURCE_FAILED
                if config is not None and config.capture_target.kind == "process"
                else PeerRuntimeFailureReason.PEER_RUNTIME_FAILED
            ),
        )

    async def _on_terminal_stt_failure(
        self,
        exc: Exception,
        *,
        attachment_token: object | None,
    ) -> None:
        if attachment_token is None:
            return
        if attachment_token in self._pending_provider_failures:
            self._pending_provider_failures[attachment_token] = exc
            return
        async with self._activation_lock:
            async with self._lock:
                if (
                    self._closed
                    or attachment_token is not self._provider_attachment_token
                    or self._provider_status
                    in {
                        PeerCaptureProviderStatus.DETACHED,
                        PeerCaptureProviderStatus.RELEASING,
                    }
                ):
                    return
                target_generation = self._generation
                config = self._config
            if config is None:
                return
            await self._fault_current_generation_locked(
                target_generation,
                config=config,
                reason=PeerCaptureFailureReason.PROVIDER_FAILED,
            )

    async def _fault_current_generation(
        self,
        generation: int,
        *,
        config: PeerCaptureSessionConfig | None,
        reason: PeerRuntimeFailureReason,
    ) -> None:
        async with self._activation_lock:
            await self._fault_current_generation_locked(
                generation,
                config=config,
                reason=reason,
            )

    async def _fault_current_generation_locked(
        self,
        generation: int,
        *,
        config: PeerCaptureSessionConfig | None,
        reason: PeerRuntimeFailureReason,
    ) -> None:
        current_task = asyncio.current_task()
        defer_diagnostic = current_task is not None and self._loop_task is current_task
        diagnostic = None
        if config is not None:
            unavailable_reason = None
            if reason is PeerRuntimeFailureReason.PROCESS_TARGET_UNAVAILABLE:
                unavailable_reason = self._last_failure_unavailable_reason
            diagnostic = PeerCaptureDiagnostic(
                event=PeerCaptureDiagnosticEvent.FAILURE,
                generation=generation,
                state=PeerCaptureSessionState.FAULTED,
                provider_id=config.provider_id,
                reason=reason,
                capture_kind=config.capture_target.kind,
                detail=unavailable_reason,
            )
            if config.capture_target.kind == "process":
                self._retry_required_capture_target = config.capture_target
            self._last_failure = diagnostic
        async with self._lock:
            if generation != self._generation or self._closed:
                return
            self._generation += 1
            teardown_generation = self._generation
            self._desired_active = False
            self._state = PeerCaptureSessionState.STOPPING
            self._notify_state_changed()
        try:
            await self._teardown_resources(
                target_state=PeerCaptureSessionState.FAULTED,
                generation=teardown_generation,
                release_mode="abort",
            )
        finally:
            if diagnostic is not None:
                if defer_diagnostic and current_task is not None:
                    self._deferred_loop_diagnostics[current_task] = diagnostic
                else:
                    self._emit_failure(diagnostic)

    async def _teardown_resources(
        self,
        *,
        target_state: PeerCaptureSessionState,
        generation: int,
        release_mode: Literal["drain", "dormant", "abort"],
        release_provider: bool = True,
        cancel_segments: bool = True,
    ) -> None:
        async with self._lock:
            if self._generation != generation:
                return
            loop_task = self._loop_task
            source = self._audio_source
            self._loop_task = None
            self._audio_source = None
            self._vad = None
            self._capture_generation = None
            self._provider_ingress_ready = None
            segment_ledger = self._segment_ledger
            self._segment_ledger = None
            self._resolved_target = None
            self._signature = None
            if release_mode == "abort" and release_provider:
                self._provider_signature = None
        failures: list[Exception] = []
        prior_cleanup_debt = tuple(self._retired_sources)
        await self._attempt_cleanup(failures, lambda: self._cancel_loop(loop_task))
        await self._attempt_cleanup(
            failures,
            lambda: self._close_if_possible(source),
            retain_on_failure=lambda: self._retain_retired_source(source),
        )
        if cancel_segments and segment_ledger is not None:
            segment_ledger.cancel_unfinished(now_monotonic_s=self.clock.now())
        await self._retry_retired_cleanup_debt(failures, prior_cleanup_debt)
        if release_provider:
            self._provider_status = PeerCaptureProviderStatus.RELEASING
            if release_mode == "abort":
                self._retire_provider_attachment()
            provider_failure_count = len(failures)
            await self._attempt_cleanup(
                failures,
                lambda: self._provider.release(
                    mode=release_mode,
                    release_backend_after=(
                        self._config.release_backend_after
                        if release_mode == "drain" and self._config is not None
                        else None
                    ),
                ),
            )
            if (
                release_mode == "drain"
                and not cancel_segments
                and segment_ledger is not None
                and len(failures) == provider_failure_count
            ):
                try:
                    segment_ledger.fail_unresolved_after_drain(
                        now_monotonic_s=self.clock.now()
                    )
                except Exception as exc:
                    failures.append(exc)
            if not failures:
                self._provider_status = PeerCaptureProviderStatus.DETACHED
        async with self._lock:
            if self._generation == generation:
                self._state = target_state
                self._target_status = None
                self._notify_state_changed()
        self._raise_cleanup_failures("peer owned runtime teardown failed", failures)

    def _commit_provider_attachment(
        self,
        attachment_token: object | None,
        *,
        recovery_handler: PeerCaptureTerminalFailureHandler | None = None,
    ) -> None:
        self._provider_attachment_token = attachment_token
        if recovery_handler is None:
            for pending in self._pending_provider_recoveries.values():
                self._pending_provider_failures.pop(pending[1], None)
            self._pending_provider_recoveries.clear()
        else:
            self._pending_provider_recoveries.pop(recovery_handler, None)

    def _retire_provider_attachment(self) -> None:
        self._commit_provider_attachment(None)

    def _create_task(self, coroutine: Awaitable[None], *, task_name: str) -> asyncio.Task[None]:
        return asyncio.create_task(coroutine, name=f"PeerCaptureSessionOwner:{task_name}")

    async def _attempt_cleanup(
        self,
        cleanup_failures: list[Exception],
        operation: Callable[[], Awaitable[None]],
        *,
        retain_on_failure: Callable[[], None] | None = None,
    ) -> None:
        try:
            await operation()
        except Exception as exc:
            if retain_on_failure is not None:
                retain_on_failure()
            cleanup_failures.append(exc)

    async def _retry_retired_cleanup_debt(
        self,
        cleanup_failures: list[Exception],
        sources: tuple[object, ...],
    ) -> None:
        for source in sources:
            try:
                await self._close_if_possible(source)
            except Exception as exc:
                cleanup_failures.append(exc)
            else:
                self._forget_retired_source(source)

    def _retain_retired_source(self, source: object | None) -> None:
        if source is None:
            return
        if any(retired_source is source for retired_source in self._retired_sources):
            return
        self._retired_sources.append(source)

    def _forget_retired_source(self, source: object) -> None:
        self._retired_sources = [
            retired_source
            for retired_source in self._retired_sources
            if retired_source is not source
        ]

    def _raise_cleanup_failures(
        self,
        message: str,
        cleanup_failures: list[Exception],
    ) -> None:
        if len(cleanup_failures) == 1:
            raise cleanup_failures[0]
        if cleanup_failures:
            raise ExceptionGroup(message, cleanup_failures)

    async def _cancel_loop(self, loop_task: asyncio.Task[None] | None) -> None:
        if loop_task is None or loop_task is asyncio.current_task():
            return
        loop_task.cancel()
        await asyncio.gather(loop_task, return_exceptions=True)

    def _on_loop_task_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            try:
                task.exception()
            except asyncio.CancelledError:
                pass
        diagnostic = self._deferred_loop_diagnostics.pop(task, None)
        if diagnostic is not None:
            self._emit_failure(diagnostic)

    async def _close_if_possible(self, resource: object | None) -> None:
        if resource is None or not hasattr(resource, "close"):
            return
        result = resource.close()
        if inspect.isawaitable(result):
            await result

    def _is_superseded(self, generation: int) -> bool:
        return generation != self._generation or not self._desired_active

    def is_current_generation(self, generation: int) -> bool:
        return (
            not self._is_superseded(generation)
            and self._state
            in {
                PeerCaptureSessionState.STARTING,
                PeerCaptureSessionState.PROVIDER_PENDING,
                PeerCaptureSessionState.RUNNING,
            }
            and self._loop_task is not None
        )

    def guard_vad_sink(self, generation: int | None = None) -> object:
        provider_ingress_ready = asyncio.Event()
        provider_ingress_ready.set()
        return _GenerationGuardedVadSink(
            sink=self._vad_sink,
            runtime=self,
            capture_generation=_CaptureGeneration(
                self._generation if generation is None else generation
            ),
            provider_ingress_ready=provider_ingress_ready,
        )

    def _rebind_capture_generation(self, generation: int) -> None:
        if self._capture_generation is not None:
            self._capture_generation.value = generation
        if self._config is not None:
            self._rebind_segment_ledger(generation, self._config)

    def _rebind_segment_ledger(
        self,
        generation: int,
        config: PeerCaptureSessionConfig,
    ) -> None:
        reconfigure_vad = getattr(self._vad, "reconfigure_next_segment", None)
        if callable(reconfigure_vad):
            reconfigure_vad(
                speech_threshold=config.vad_speech_threshold,
                hangover_ms=config.vad_hangover_ms,
                ring_buffer_ms=config.vad_pre_roll_ms,
            )
        if self._segment_ledger is not None:
            self._segment_ledger.rebind(
                activation_generation=generation,
                settings=self._segment_settings_snapshot(config),
            )

    @staticmethod
    def _segment_settings_snapshot(
        config: PeerCaptureSessionConfig,
    ) -> AudioSegmentSettingsSnapshot:
        return AudioSegmentSettingsSnapshot(
            provider_id=config.provider_id,
            provider_signature=config.provider_signature,
            runtime_signature=config.runtime_signature,
            source_mode=config.language.source_mode,
            source_language=config.language.source_language,
            expected_languages=config.language.expected_languages,
            target_sample_rate_hz=config.target_sample_rate_hz,
            vad_speech_threshold=config.vad_speech_threshold,
            vad_hangover_ms=config.vad_hangover_ms,
            vad_pre_roll_ms=config.vad_pre_roll_ms,
        )

    def _failure_reason_from_startup_exception(
        self,
        config: PeerCaptureSessionConfig,
        exc: Exception,
    ) -> PeerRuntimeFailureReason:
        if isinstance(exc, _PeerCaptureAdmissionRejected):
            return PeerCaptureFailureReason.ADMISSION_REJECTED
        if isinstance(exc, _PeerCaptureVadFailed):
            return PeerCaptureFailureReason.VAD_FAILED
        if isinstance(exc, _PeerCaptureSourceOpenFailed):
            return PeerCaptureFailureReason.SOURCE_OPEN_FAILED
        if config.capture_target.kind != "process":
            return PeerRuntimeFailureReason.PEER_RUNTIME_FAILED
        if isinstance(exc, _PeerCaptureTargetUnavailable):
            return PeerRuntimeFailureReason.PROCESS_TARGET_UNAVAILABLE
        if isinstance(exc, (ProcessAudioCaptureSetupError, ProcessAudioCaptureUnavailableError)):
            return PeerRuntimeFailureReason.PROCESS_SETUP_FAILED
        return PeerRuntimeFailureReason.PROCESS_SETUP_FAILED

    @staticmethod
    def _terminal_reason_from_source(source: object) -> str | None:
        current = source
        for _ in range(4):
            terminal_reason = getattr(current, "terminal_reason", None)
            if isinstance(terminal_reason, str):
                return terminal_reason
            current = getattr(current, "source", None)
            if current is None:
                return None
        return None

    @staticmethod
    def _failure_reason_from_terminal_source(reason: str) -> PeerRuntimeFailureReason:
        if reason == "target_exited":
            return PeerRuntimeFailureReason.PROCESS_TARGET_EXITED
        return PeerRuntimeFailureReason.PROCESS_SOURCE_FAILED

    def _emit_failure(self, diagnostic: PeerCaptureDiagnostic) -> None:
        self._last_failure = diagnostic
        self._notify_state_changed()
        if self._diagnostic_sink is not None:
            try:
                self._diagnostic_sink(diagnostic)
            except Exception:
                pass

    def _emit_local_asr_diagnostic(
        self,
        config: PeerCaptureSessionConfig,
        *,
        outcome: str,
        load_started_at: float,
        failure_type: str | None = None,
    ) -> None:
        sink = self._local_asr_diagnostic_sink
        if sink is None:
            return
        fields: dict[str, object] = {
            "channel": "peer",
            "requested_provider": config.provider_id,
            "actual_provider": config.provider_id,
            "model_id": config.model_id,
            "trigger": "activation",
            "load_ms": max(0, int(round((self.clock.now() - load_started_at) * 1000))),
            "outcome": outcome,
        }
        if failure_type is not None:
            fields["failure_type"] = failure_type
        try:
            sink(fields)
        except Exception:
            pass

    def _emit_event(self, event: PeerCaptureDiagnosticEvent) -> None:
        if self._diagnostic_sink is None:
            return
        config = self._config
        try:
            self._diagnostic_sink(
                PeerCaptureDiagnostic(
                    event=event,
                    generation=self._generation,
                    state=self._state,
                    provider_id=config.provider_id if config is not None else None,
                    capture_kind=(config.capture_target.kind if config is not None else None),
                )
            )
        except Exception:
            pass

    def _notify_state_changed(self) -> None:
        if self._state_changed is not None:
            self._state_changed(self.snapshot)


PeerChannelRuntime = PeerCaptureSessionOwner
