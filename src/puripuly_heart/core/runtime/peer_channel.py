from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Protocol, cast

from puripuly_heart.config.resolved import ResolvedDesktopAudioCaptureTarget, ResolvedSTTConfig
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
)
from puripuly_heart.core.runtime.local_asr_transition import (
    LocalASRSessionOptions,
    LocalASRTransitionCoordinator,
    LocalASRTransitionDiagnosticSink,
    LocalASRTransitionRequest,
    PreparedLocalASRTransition,
)

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

    async def adopt_recovered_provider(self, config: PeerRuntimeConfig) -> None: ...

    async def close(self) -> None: ...


class _VadSink(Protocol):
    async def handle_vad_event(self, event: object) -> None: ...


@dataclass(slots=True)
class _CaptureGeneration:
    value: int


@dataclass(slots=True)
class _GenerationGuardedVadSink:
    sink: object
    runtime: "PeerCaptureSessionOwner"
    capture_generation: _CaptureGeneration

    async def handle_vad_event(self, event: object) -> None:
        if not self.runtime.is_current_generation(self.capture_generation.value):
            return
        await cast(_VadSink, self.sink).handle_vad_event(event)


class PeerCaptureSessionOwner:
    resource_fields = (
        "_audio_source",
        "_vad",
        "_loop_task",
        "_generation",
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
        source_factory: Callable[
            [PeerCaptureSessionConfig, PeerCaptureResolvedTarget],
            Awaitable[object] | object,
        ],
        vad_factory: Callable[[PeerCaptureSessionConfig], object],
        run_audio_loop: Callable[..., Awaitable[None]],
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
        self._config: PeerCaptureSessionConfig | None = None
        self._resolved_target: PeerCaptureResolvedTarget | None = None
        self._audio_source: object | None = None
        self._vad: object | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._signature: tuple[object, ...] | None = None
        self._provider_signature: tuple[object, ...] | None = None
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
                self._state is PeerCaptureSessionState.RUNNING
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
    def last_failure(self) -> PeerCaptureDiagnostic | None:
        return self._last_failure

    @property
    def last_local_asr_transition_status(self) -> str:
        return self._last_local_asr_transition_status

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
        async with self._lock:
            if self._closed:
                raise RuntimeError("PeerCaptureSessionOwner is closed")
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
                self._config = config
                transition_only = True
                self._generation += 1
                generation = self._generation
                self._rebind_capture_generation(generation)
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
            self._notify_state_changed()
        async with self._activation_lock:
            if transition_only:
                await self._transition_running_provider(config, generation=generation)
                return self.snapshot
            if not enabled:
                release_mode = (
                    "drain"
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
        async with self._activation_lock:
            await self._teardown_resources(
                target_state=PeerCaptureSessionState.STOPPED,
                generation=generation,
                release_mode="drain",
                release_provider=False,
            )

    async def adopt_recovered_provider(self, config: PeerCaptureSessionConfig) -> None:
        if not self._provider.is_ready(config):
            raise RuntimeError("recovered Peer provider is not attached")
        async with self._lock:
            if self._closed:
                raise RuntimeError("PeerCaptureSessionOwner is closed")
            if self._state is not PeerCaptureSessionState.STOPPED:
                raise RuntimeError("Peer provider recovery requires suspended capture")
            self._config = config
            self._provider_signature = config.provider_signature
            self._provider_status = PeerCaptureProviderStatus.READY
            self._notify_state_changed()

    async def handle_terminal_provider_failure(self, exc: Exception) -> None:
        await self._on_terminal_stt_failure(exc)

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
        async with self._activation_lock:
            await self._teardown_resources(
                target_state=PeerCaptureSessionState.STOPPED,
                generation=generation,
                release_mode="abort",
            )

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
                if self._config is config and self._generation == generation:
                    self._provider_signature = config.provider_signature
                    self._signature = config.runtime_signature
            self._last_local_asr_transition_status = "applied"
            return
        transition_request = LocalASRTransitionRequest(
            channel="peer",
            requested_provider=config.provider_id,
            actual_provider=config.provider_id,
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
                if (
                    self._config is not config
                    or self._generation != generation
                    or not self._desired_active
                ):
                    raise RuntimeError("peer provider transition superseded")
            try:
                result = await self._provider.handoff(
                    prepared.provider,
                    start=True,
                    on_terminal_failure=lambda exc: self._on_terminal_stt_failure(
                        exc,
                        generation=generation,
                    ),
                )
            except asyncio.CancelledError:
                await self._provider.cancel_handoff()
                raise
            if result.status is not PeerCaptureProviderMutationStatus.APPLIED:
                raise RuntimeError("owned Peer STT handoff failed")
            async with self._lock:
                if self._config is config and self._generation == generation:
                    self._provider_signature = config.provider_signature
                    self._signature = config.runtime_signature

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
            reusable = self._provider.is_ready(config)
            if not reusable:
                self._provider_status = PeerCaptureProviderStatus.PENDING
                self._notify_state_changed()
                request = self._provider_request_factory(
                    config,
                    config.local_provider,
                )
                result = await self._provider.replace(
                    request,
                    start=False,
                    on_terminal_failure=lambda exc: self._on_terminal_stt_failure(
                        exc,
                        generation=generation,
                    ),
                )
                if result.status is PeerCaptureProviderMutationStatus.PENDING:
                    self._state = PeerCaptureSessionState.PROVIDER_PENDING
                    self._provider_status = PeerCaptureProviderStatus.PENDING
                    self._admission_reason = result.reason
                    self._notify_state_changed()
                    return
                if result.status is PeerCaptureProviderMutationStatus.SUPERSEDED:
                    self._state = PeerCaptureSessionState.STOPPED
                    self._provider_status = PeerCaptureProviderStatus.DETACHED
                    self._desired_active = False
                    self._notify_state_changed()
                    return
                if result.status is not PeerCaptureProviderMutationStatus.APPLIED:
                    raise RuntimeError("owned Peer STT replacement failed")
            provider_ready = True
            self._provider_status = PeerCaptureProviderStatus.READY
            self._emit_event(PeerCaptureDiagnosticEvent.PROVIDER_CHANGED)
            if self._is_superseded(generation):
                await self._provider.release(mode="abort")
                return
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
                await self._provider.release(mode="abort")
                return
            async with self._lock:
                superseded = self._is_superseded(generation)
                if not superseded:
                    old_loop = self._loop_task
                    old_source = self._audio_source
                    self._loop_task = None
                    self._audio_source = source
                    self._vad = vad
                    self._provider_signature = config.provider_signature
                    self._signature = config.runtime_signature
                    capture_generation = _CaptureGeneration(generation)
                    self._capture_generation = capture_generation
                    loop_task = self._create_task(
                        self._run_peer_loop_guarded(
                            source=source,
                            vad=vad,
                            target_sample_rate_hz=config.target_sample_rate_hz,
                            capture_generation=capture_generation,
                        ),
                        task_name="session-loop",
                    )
                    loop_task.add_done_callback(self._on_loop_task_done)
                    self._loop_task = loop_task
            if superseded:
                await self._close_if_possible(source)
                await self._provider.release(mode="abort")
                return
            await self._cancel_loop(old_loop)
            await self._close_if_possible(old_source)
            await self._provider.start_ingress()
            if self._is_superseded(generation):
                await self._fault_current_generation_locked(
                    generation,
                    config=config,
                    reason=PeerCaptureFailureReason.PROVIDER_FAILED,
                )
                return
            self._state = PeerCaptureSessionState.RUNNING
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
    ) -> None:
        try:
            await self._run_audio_loop(
                source=source,
                vad=vad,
                sink=_GenerationGuardedVadSink(
                    sink=self._vad_sink,
                    runtime=self,
                    capture_generation=capture_generation,
                ),
                target_sample_rate_hz=target_sample_rate_hz,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
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
        generation: int | None = None,
    ) -> None:
        _ = exc
        target_generation = self._generation if generation is None else generation
        config = self._config
        async with self._lock:
            if self._is_superseded(target_generation):
                return
            if (
                self._desired_active
                and self._state is PeerCaptureSessionState.RUNNING
                and config is not None
                and self._provider.is_ready(config)
                and (config is None or config.capture_target.kind != "process")
            ):
                return
        await self._fault_current_generation(
            target_generation,
            config=config,
            reason=(
                PeerRuntimeFailureReason.PROCESS_PROVIDER_FAILED
                if config is not None and config.capture_target.kind == "process"
                else PeerRuntimeFailureReason.PEER_RUNTIME_FAILED
            ),
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
        release_mode: Literal["drain", "abort"],
        release_provider: bool = True,
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
        await self._retry_retired_cleanup_debt(failures, prior_cleanup_debt)
        if release_provider:
            self._provider_status = PeerCaptureProviderStatus.RELEASING
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
            if not failures:
                self._provider_status = PeerCaptureProviderStatus.DETACHED
        async with self._lock:
            if self._generation == generation:
                self._state = target_state
                self._target_status = None
                self._notify_state_changed()
        self._raise_cleanup_failures("peer owned runtime teardown failed", failures)

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
            and self._state is PeerCaptureSessionState.RUNNING
            and self._loop_task is not None
        )

    def guard_vad_sink(self, generation: int | None = None) -> object:
        return _GenerationGuardedVadSink(
            sink=self._vad_sink,
            runtime=self,
            capture_generation=_CaptureGeneration(
                self._generation if generation is None else generation
            ),
        )

    def _rebind_capture_generation(self, generation: int) -> None:
        if self._capture_generation is not None:
            self._capture_generation.value = generation

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
