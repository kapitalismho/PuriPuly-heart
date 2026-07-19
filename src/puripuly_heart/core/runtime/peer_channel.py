from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Literal, Protocol

from puripuly_heart.config.process_capture_resolution import ProcessCaptureTargetUnavailableError
from puripuly_heart.config.resolved import ResolvedDesktopAudioCaptureTarget, ResolvedSTTConfig
from puripuly_heart.core.audio.process_source import (
    ProcessAudioCaptureSetupError,
    ProcessAudioCaptureUnavailableError,
)
from puripuly_heart.core.clock import Clock
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest
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

if TYPE_CHECKING:
    from puripuly_heart.core.orchestrator.hub import ClientHub


class PeerChannelRuntimeState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAULTED = "faulted"


class PeerRuntimeFailureReason(str, Enum):
    PROCESS_TARGET_UNAVAILABLE = "process_target_unavailable"
    PROCESS_SETUP_FAILED = "process_setup_failed"
    PROCESS_TARGET_EXITED = "process_target_exited"
    PROCESS_SOURCE_FAILED = "process_source_failed"
    PROCESS_PROVIDER_FAILED = "process_provider_failed"
    PEER_RUNTIME_FAILED = "peer_runtime_failed"


class PeerLocalASRTransitionSuperseded(RuntimeError):
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


@dataclass(slots=True)
class _PeerHubVadSink:
    hub: ClientHub
    runtime: "PeerChannelRuntime"
    generation: int

    async def handle_vad_event(self, event) -> None:  # noqa: ANN001
        if not self.runtime.is_current_generation(self.generation):
            return
        await self.hub.handle_peer_vad_event(event)


class PeerChannelRuntime:
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
        hub: ClientHub,
        clock: Clock,
        provider_request_factory: Callable[
            [PeerRuntimeConfig, bool],
            ProviderRuntimeBuildRequest,
        ],
        source_factory: Callable[[PeerRuntimeConfig], Awaitable[object] | object],
        vad_factory: Callable[[PeerRuntimeConfig, Path], object],
        vad_model_resolver: Callable[[], Path],
        run_audio_loop: Callable[..., Awaitable[None]],
        diagnostic_sink: Callable[[PeerRuntimeDiagnostic], object] | None = None,
        local_asr_diagnostic_sink: LocalASRTransitionDiagnosticSink | None = None,
        idle_release_seconds: float | None = None,
    ) -> None:
        self.hub = hub
        self.clock = clock
        self._provider_request_factory = provider_request_factory
        self._source_factory = source_factory
        self._vad_factory = vad_factory
        self._vad_model_resolver = vad_model_resolver
        self._run_audio_loop = run_audio_loop
        self._diagnostic_sink = diagnostic_sink
        self._local_asr_diagnostic_sink = local_asr_diagnostic_sink
        self._idle_release_seconds = idle_release_seconds
        self._config: PeerRuntimeConfig | None = None
        self._audio_source: object | None = None
        self._vad: object | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._signature: tuple[object, ...] | None = None
        self._provider_signature: tuple[object, ...] | None = None
        self._state = PeerChannelRuntimeState.STOPPED
        self._generation = 0
        self._desired_active = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._activation_lock = asyncio.Lock()
        self._retired_sources: list[object] = []
        self._last_failure: PeerRuntimeDiagnostic | None = None
        self._last_failure_unavailable_reason: str | None = None
        self._retry_required_capture_target: ResolvedDesktopAudioCaptureTarget | None = None
        self._deferred_loop_diagnostics: dict[asyncio.Task[None], PeerRuntimeDiagnostic] = {}
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
    def current_signature(self) -> object | None:
        return self._signature

    @property
    def loop_task(self) -> asyncio.Task[None] | None:
        return self._loop_task

    @property
    def last_failure(self) -> PeerRuntimeDiagnostic | None:
        return self._last_failure

    @property
    def last_local_asr_transition_status(self) -> str:
        return self._last_local_asr_transition_status

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": "PeerChannelRuntime",
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
            "local_asr_transition": self._transition_coordinator.lifecycle_snapshot(),
        }

    async def apply_policy(
        self,
        *,
        config: PeerRuntimeConfig,
        desired_active: bool,
        stop_mode: Literal["retain", "release"] = "retain",
    ) -> None:
        if stop_mode not in {"retain", "release"}:
            raise ValueError("stop_mode must be 'retain' or 'release'")
        transition_only = False
        async with self._lock:
            if self._closed:
                return
            if (
                desired_active
                and self._desired_active
                and self._state == PeerChannelRuntimeState.RUNNING
                and self._signature == config.runtime_signature
            ):
                self._config = config
                return
            current_config = self._config
            if (
                desired_active
                and self._desired_active
                and self._state == PeerChannelRuntimeState.RUNNING
                and current_config is not None
                and current_config.backend.provider in _LOCAL_ASR_PROVIDERS
                and config.backend.provider in _LOCAL_ASR_PROVIDERS
                and current_config.capture_vad_signature == config.capture_vad_signature
            ):
                self._config = config
                transition_only = True
                generation = self._generation
            else:
                self._generation += 1
                generation = self._generation
                self._config = config
                self._desired_active = desired_active
                self._state = (
                    PeerChannelRuntimeState.STARTING
                    if desired_active
                    else PeerChannelRuntimeState.STOPPING
                )
        async with self._activation_lock:
            if transition_only:
                await self._transition_running_provider(config, generation=generation)
                return
            if not desired_active:
                release_mode = (
                    "drain"
                    if stop_mode == "retain"
                    and config.backend.provider == "local_qwen"
                    and self._provider_signature == config.provider_signature
                    else "abort"
                )
                await self._teardown_resources(
                    target_state=PeerChannelRuntimeState.STOPPED,
                    generation=generation,
                    release_mode=release_mode,
                )
                return
            await self._start_generation(generation, config)

    async def retry_process_capture(self, *, config: PeerRuntimeConfig) -> bool:
        async with self._lock:
            if (
                config.capture_target.kind != "process"
                or self._retry_required_capture_target != config.capture_target
                or self._state != PeerChannelRuntimeState.FAULTED
            ):
                return False
            self._generation += 1
            generation = self._generation
            self._config = config
            self._desired_active = True
            self._retry_required_capture_target = None
            self._state = PeerChannelRuntimeState.STARTING
        async with self._activation_lock:
            await self._start_generation(generation, config)
        return self._state == PeerChannelRuntimeState.RUNNING

    async def warmup(self) -> None:
        if self._desired_active and self._state == PeerChannelRuntimeState.RUNNING:
            await self.hub.warmup_stt_channel("peer")

    async def suspend_provider_consumer(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._generation += 1
            generation = self._generation
            self._desired_active = False
            self._state = PeerChannelRuntimeState.STOPPING
        async with self._activation_lock:
            await self._teardown_resources(
                target_state=PeerChannelRuntimeState.STOPPED,
                generation=generation,
                release_mode="drain",
                release_provider=False,
            )

    async def adopt_recovered_provider(self, config: PeerRuntimeConfig) -> None:
        runtime = self.hub.local_asr_provider_runtime
        current = runtime.snapshot.channel_for("peer") if runtime is not None else None
        if (
            current is None
            or current.provider_id != config.backend.provider
            or not current.has_resources
        ):
            raise RuntimeError("recovered Peer provider is not attached")
        async with self._lock:
            if self._closed:
                raise RuntimeError("PeerChannelRuntime is closed")
            if self._state != PeerChannelRuntimeState.STOPPED:
                raise RuntimeError("Peer provider recovery requires suspended capture")
            self._config = config
            self._provider_signature = config.provider_signature

    async def handle_terminal_provider_failure(self, exc: Exception) -> None:
        await self._on_terminal_stt_failure(exc)

    async def close(self) -> None:
        await self._transition_coordinator.close()
        async with self._lock:
            self._closed = True
            self._generation += 1
            generation = self._generation
            self._desired_active = False
            self._state = PeerChannelRuntimeState.STOPPING
        async with self._activation_lock:
            await self._teardown_resources(
                target_state=PeerChannelRuntimeState.STOPPED,
                generation=generation,
                release_mode="abort",
            )

    async def _transition_running_provider(
        self,
        config: PeerRuntimeConfig,
        *,
        generation: int,
    ) -> None:
        runtime = self.hub.local_asr_provider_runtime
        if runtime is None:
            raise RuntimeError("local ASR provider runtime is unavailable")
        current = runtime.snapshot.channel_for("peer")
        options = config.session_options or LocalASRSessionOptions(
            source_language=config.backend.source_language,
            source_mode=config.backend.source_mode,
        )
        if current.model_id == config.model_id:
            await self.hub.reconfigure_stt_channel("peer", options)
            async with self._lock:
                if self._config is config and self._generation == generation:
                    self._provider_signature = config.provider_signature
                    self._signature = config.runtime_signature
            self._last_local_asr_transition_status = "applied"
            return
        transition_request = LocalASRTransitionRequest(
            channel="peer",
            requested_provider=config.backend.provider,
            actual_provider=config.backend.provider,
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
            if not isinstance(prepared.provider, ProviderRuntimeBuildRequest):
                raise TypeError("owned Peer STT transition requires a build request")
            try:
                result = await self.hub.handoff_peer_stt_provider_request(
                    prepared.provider,
                    start=True,
                    on_terminal_failure=lambda exc: self._on_terminal_stt_failure(
                        exc,
                        generation=generation,
                    ),
                )
            except asyncio.CancelledError:
                await self.hub.cancel_peer_stt_provider_request_handoff()
                raise
            if result.status != "applied":
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
        config: PeerRuntimeConfig,
    ) -> None:
        source = None
        provider_ready = False
        load_started_at = self.clock.now()
        try:
            owner = self.hub.local_asr_provider_runtime
            if owner is None:
                raise RuntimeError("local ASR provider runtime is unavailable")
            current = owner.snapshot.channel_for("peer")
            reusable = (
                self._provider_signature == config.provider_signature
                and current.provider_id == config.backend.provider
                and current.has_resources
            )
            if not reusable:
                request = self._provider_request_factory(
                    config,
                    config.backend.provider in _LOCAL_ASR_PROVIDERS,
                )
                result = await self.hub.replace_peer_stt_provider_request(
                    request,
                    start=False,
                    on_terminal_failure=lambda exc: self._on_terminal_stt_failure(
                        exc,
                        generation=generation,
                    ),
                )
                if result.status != "applied":
                    raise RuntimeError("owned Peer STT replacement failed")
            provider_ready = True
            if self._is_superseded(generation):
                await self.hub.abort_peer_stt_for_toggle_off()
                return
            source = self._source_factory(config)
            if inspect.isawaitable(source):
                source = await source
            vad = self._vad_factory(config, self._vad_model_resolver())
            if self._is_superseded(generation):
                await self._close_if_possible(source)
                await self.hub.abort_peer_stt_for_toggle_off()
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
                    loop_task = self._create_task(
                        self._run_peer_loop_guarded(
                            source=source,
                            vad=vad,
                            target_sample_rate_hz=config.backend.sample_rate_hz,
                            generation=generation,
                        ),
                        task_name="session-loop",
                    )
                    loop_task.add_done_callback(self._on_loop_task_done)
                    self._loop_task = loop_task
                    self._state = PeerChannelRuntimeState.RUNNING
            if superseded:
                await self._close_if_possible(source)
                await self.hub.abort_peer_stt_for_toggle_off()
                return
            await self._cancel_loop(old_loop)
            await self._close_if_possible(old_source)
            await self.hub.start_peer_stt_provider_ingress()
        except Exception as exc:
            if source is not None and self._audio_source is not source:
                await self._close_if_possible(source)
            reason = (
                self._failure_reason_from_startup_exception(config, exc)
                if provider_ready
                else (
                    PeerRuntimeFailureReason.PROCESS_PROVIDER_FAILED
                    if config.capture_target.kind == "process"
                    else PeerRuntimeFailureReason.PEER_RUNTIME_FAILED
                )
            )
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
        generation: int,
    ) -> None:
        try:
            await self._run_audio_loop(
                source=source,
                vad=vad,
                sink=_PeerHubVadSink(
                    hub=self.hub,
                    runtime=self,
                    generation=generation,
                ),
                target_sample_rate_hz=target_sample_rate_hz,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._on_runtime_failure(exc, generation=generation, config=self._config)
            return
        terminal_reason = self._terminal_reason_from_source(source)
        if terminal_reason is not None:
            await self._fault_current_generation(
                generation,
                config=self._config,
                reason=self._failure_reason_from_terminal_source(terminal_reason),
            )

    async def _on_runtime_failure(
        self,
        exc: Exception,
        *,
        generation: int,
        config: PeerRuntimeConfig | None,
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
            owner = self.hub.local_asr_provider_runtime
            current = owner.snapshot.channel_for("peer") if owner is not None else None
            if (
                self._desired_active
                and self._state == PeerChannelRuntimeState.RUNNING
                and current is not None
                and current.has_resources
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
        config: PeerRuntimeConfig | None,
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
        config: PeerRuntimeConfig | None,
        reason: PeerRuntimeFailureReason,
    ) -> None:
        current_task = asyncio.current_task()
        defer_diagnostic = current_task is not None and self._loop_task is current_task
        diagnostic = None
        if config is not None:
            unavailable_reason = None
            if reason is PeerRuntimeFailureReason.PROCESS_TARGET_UNAVAILABLE:
                unavailable_reason = self._last_failure_unavailable_reason
            diagnostic = PeerRuntimeDiagnostic(
                reason=reason,
                capture_kind=config.capture_target.kind,
                process_unavailable_reason=unavailable_reason,
            )
            if config.capture_target.kind == "process":
                self._retry_required_capture_target = config.capture_target
        async with self._lock:
            if self._is_superseded(generation):
                return
            self._generation += 1
            teardown_generation = self._generation
            self._desired_active = False
            self._state = PeerChannelRuntimeState.STOPPING
        try:
            await self._teardown_resources(
                target_state=PeerChannelRuntimeState.FAULTED,
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
        target_state: PeerChannelRuntimeState,
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
            self._signature = None
            if release_mode == "abort" and release_provider:
                self._provider_signature = None
        failures: list[Exception] = []
        await self._attempt_cleanup(failures, lambda: self._cancel_loop(loop_task))
        await self._attempt_cleanup(
            failures,
            lambda: self._close_if_possible(source),
            retain_on_failure=lambda: self._retain_retired_source(source),
        )
        await self._retry_retired_cleanup_debt(failures)
        if release_provider:
            runtime = self.hub.local_asr_provider_runtime
            if runtime is None:
                failures.append(RuntimeError("local ASR provider runtime is unavailable"))
            else:
                await self._attempt_cleanup(
                    failures,
                    lambda: runtime.release_channel(
                        "peer",
                        mode=release_mode,
                        release_backend_after=(
                            self._idle_release_seconds if release_mode == "drain" else None
                        ),
                    ),
                )
        async with self._lock:
            if self._generation == generation:
                self._state = target_state
        self._raise_cleanup_failures("peer owned runtime teardown failed", failures)

    def _create_task(self, coroutine: Awaitable[None], *, task_name: str) -> asyncio.Task[None]:
        return asyncio.create_task(coroutine, name=f"PeerChannelRuntime:{task_name}")

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
    ) -> None:
        for source in tuple(self._retired_sources):
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
        return not self._is_superseded(generation)

    def _failure_reason_from_startup_exception(
        self,
        config: PeerRuntimeConfig,
        exc: Exception,
    ) -> PeerRuntimeFailureReason:
        if config.capture_target.kind != "process":
            return PeerRuntimeFailureReason.PEER_RUNTIME_FAILED
        if isinstance(exc, ProcessCaptureTargetUnavailableError):
            self._last_failure_unavailable_reason = exc.reason
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

    def _emit_failure(self, diagnostic: PeerRuntimeDiagnostic) -> None:
        self._last_failure = diagnostic
        if self._diagnostic_sink is not None:
            try:
                self._diagnostic_sink(diagnostic)
            except Exception:
                pass

    def _emit_local_asr_diagnostic(
        self,
        config: PeerRuntimeConfig,
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
            "requested_provider": config.backend.provider,
            "actual_provider": config.backend.provider,
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
