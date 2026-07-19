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
        "_stt",
        "_retained_stt",
        "_audio_source",
        "_vad",
        "_loop_task",
        "_idle_release_task",
        "_generation",
        "_desired_active",
        "_lock",
    )
    stop_ingress = "invalidate generation and desired-active state"
    shutdown_policy = "cancel loop, close source, detach/close peer STT"
    late_callback_rule = "late peer callbacks cannot mutate current runtime or output to chatbox"

    def __init__(
        self,
        *,
        hub: ClientHub,
        clock: Clock,
        stt_factory: Callable[
            [PeerRuntimeConfig, Callable[[Exception], Awaitable[None]]],
            Awaitable[object] | object,
        ],
        provider_request_factory: (
            Callable[
                [PeerRuntimeConfig, bool],
                ProviderRuntimeBuildRequest,
            ]
            | None
        ) = None,
        source_factory: Callable[[PeerRuntimeConfig], Awaitable[object] | object],
        vad_factory: Callable[[PeerRuntimeConfig, Path], object],
        vad_model_resolver: Callable[[], Path],
        run_audio_loop: Callable[..., Awaitable[None]],
        diagnostic_sink: Callable[[PeerRuntimeDiagnostic], object] | None = None,
        local_asr_diagnostic_sink: LocalASRTransitionDiagnosticSink | None = None,
        idle_release_seconds: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.hub = hub
        self.clock = clock
        self._stt_factory = stt_factory
        self._provider_request_factory = provider_request_factory
        self._source_factory = source_factory
        self._vad_factory = vad_factory
        self._vad_model_resolver = vad_model_resolver
        self._run_audio_loop = run_audio_loop
        self._diagnostic_sink = diagnostic_sink
        self._local_asr_diagnostic_sink = local_asr_diagnostic_sink
        self._idle_release_seconds = idle_release_seconds
        self._sleep = sleep

        self._config: PeerRuntimeConfig | None = None
        self._stt: object | None = None
        self._retained_stt: object | None = None
        self._audio_source: object | None = None
        self._vad: object | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._idle_release_task: asyncio.Task[None] | None = None
        self._idle_release_started = False
        self._idle_release_deadline: float | None = None
        self._signature: tuple[object, ...] | None = None
        self._provider_signature: tuple[object, ...] | None = None
        self._state = PeerChannelRuntimeState.STOPPED
        self._generation = 0
        self._desired_active = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._activation_lock = asyncio.Lock()
        self._activation_events: dict[int, asyncio.Event] = {}
        self._retired_sources: list[object] = []
        self._retired_peer_providers: list[object] = []
        self._last_failure: PeerRuntimeDiagnostic | None = None
        self._last_failure_unavailable_reason: str | None = None
        self._last_idle_release_error_type: str | None = None
        self._retry_required_capture_target: ResolvedDesktopAudioCaptureTarget | None = None
        self._deferred_loop_diagnostics: dict[asyncio.Task[None], PeerRuntimeDiagnostic] = {}
        self._transition_coordinator = LocalASRTransitionCoordinator(
            channel="peer",
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
            "idle_release_error_type": self._last_idle_release_error_type,
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
        if self._provider_request_factory is not None:
            await self._apply_owned_policy(
                config=config,
                desired_active=desired_active,
                stop_mode=stop_mode,
            )
            return
        async with self._lock:
            if (
                not desired_active
                and not self._desired_active
                and self._state == PeerChannelRuntimeState.STOPPED
                and self._retained_stt is not None
                and self._provider_signature == config.provider_signature
                and self._idle_release_task is not None
            ):
                return
            if desired_active:
                self._idle_release_deadline = None
            elif config.backend.provider != "local_qwen" or self._provider_signature not in (
                None,
                config.provider_signature,
            ):
                self._idle_release_deadline = None
            elif self._idle_release_seconds is not None and self._idle_release_deadline is None:
                self._idle_release_deadline = self.clock.now() + self._idle_release_seconds
        await self._cancel_idle_release_task()
        provider_only_transition = False
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
                provider_only_transition = True
            if provider_only_transition:
                generation = self._generation
                activation_event = None
            else:
                self._generation += 1
                generation = self._generation
                self._config = config
                self._desired_active = desired_active
                activation_event = None
                if not desired_active:
                    self._retry_required_capture_target = None
                    self._state = PeerChannelRuntimeState.STOPPING
                elif (
                    self._signature == config.runtime_signature
                    and self._state == PeerChannelRuntimeState.RUNNING
                ):
                    return
                else:
                    self._state = PeerChannelRuntimeState.STARTING
                    activation_event = asyncio.Event()
                    self._activation_events[generation] = activation_event

            if (
                desired_active
                and self._state == PeerChannelRuntimeState.STARTING
                and self._retry_required_capture_target == config.capture_target
            ):
                self._desired_active = False
                self._state = PeerChannelRuntimeState.FAULTED
                if activation_event is not None:
                    activation_event.set()
                    self._activation_events.pop(generation, None)
                return

        if provider_only_transition:
            await self._transition_running_provider(config, generation=generation)
            return

        if not desired_active:
            await self._wait_for_activations_before(generation)
            async with self._lock:
                if self._generation != generation or self._desired_active:
                    return
            if stop_mode == "retain" and await self._stop_for_dormant_reuse(generation, config):
                return
            await self._teardown_resources(
                target_state=PeerChannelRuntimeState.STOPPED,
                generation=generation,
                abort_for_toggle_off=stop_mode == "release",
            )
            return

        try:
            if self._provider_request_factory is not None:
                await self._start_owned_generation(generation, config)
            else:
                await self._start_generation(generation, config)
        finally:
            assert activation_event is not None
            activation_event.set()
            async with self._lock:
                self._activation_events.pop(generation, None)

    async def retry_process_capture(self, *, config: PeerRuntimeConfig) -> bool:
        self._idle_release_deadline = None
        await self._cancel_idle_release_task()
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
            activation_event = asyncio.Event()
            self._activation_events[generation] = activation_event
        try:
            if self._provider_request_factory is not None:
                await self._start_owned_generation(generation, config)
            else:
                await self._start_generation(generation, config)
        finally:
            activation_event.set()
            async with self._lock:
                self._activation_events.pop(generation, None)
        return self._state == PeerChannelRuntimeState.RUNNING

    async def warmup(self) -> None:
        if self._provider_request_factory is not None:
            if self._desired_active and self._state == PeerChannelRuntimeState.RUNNING:
                await self.hub.warmup_stt_channel("peer")
            return
        async with self._lock:
            stt = self._stt
            if (
                self._desired_active
                and stt is not None
                and self._state == PeerChannelRuntimeState.RUNNING
                and hasattr(stt, "warmup")
            ):
                await stt.warmup()

    async def close(self) -> None:
        self._idle_release_deadline = None
        await self._cancel_idle_release_task()
        await self._transition_coordinator.close()
        if self._provider_request_factory is not None:
            async with self._lock:
                self._closed = True
                self._generation += 1
                generation = self._generation
                self._desired_active = False
                self._state = PeerChannelRuntimeState.STOPPING
            await self._teardown_owned_resources(
                target_state=PeerChannelRuntimeState.STOPPED,
                generation=generation,
                release_mode="abort",
            )
            return
        async with self._lock:
            self._closed = True
            self._generation += 1
            generation = self._generation
            self._desired_active = False
            self._state = PeerChannelRuntimeState.STOPPING
        await self._wait_for_activations_before(generation)
        await self._teardown_resources(
            target_state=PeerChannelRuntimeState.STOPPED,
            generation=generation,
        )

    async def _apply_owned_policy(
        self,
        *,
        config: PeerRuntimeConfig,
        desired_active: bool,
        stop_mode: Literal["retain", "release"],
    ) -> None:
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
        if transition_only:
            await self._transition_owned_running_provider(config, generation=generation)
            return
        if not desired_active:
            release_mode = (
                "drain"
                if stop_mode == "retain"
                and config.backend.provider == "local_qwen"
                and self._provider_signature == config.provider_signature
                else "abort"
            )
            await self._teardown_owned_resources(
                target_state=PeerChannelRuntimeState.STOPPED,
                generation=generation,
                release_mode=release_mode,
            )
            return
        await self._start_owned_generation(generation, config)

    async def _transition_owned_running_provider(
        self,
        config: PeerRuntimeConfig,
        *,
        generation: int,
    ) -> None:
        runtime = self.hub.local_asr_provider_runtime
        if runtime is None:
            self._last_local_asr_transition_status = "failed"
            return
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
            assert self._provider_request_factory is not None
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

    async def _start_owned_generation(
        self,
        generation: int,
        config: PeerRuntimeConfig,
    ) -> None:
        assert self._provider_request_factory is not None
        source = None
        attached = False
        try:
            owner = self.hub.local_asr_provider_runtime
            current = owner.snapshot.channel_for("peer")
            reusable = (
                self._provider_signature == config.provider_signature
                and current.provider_id == config.backend.provider
                and current.has_resources
            )
            if reusable:
                attached = True
            else:
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
                attached = True
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
                if self._is_superseded(generation):
                    return
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
            await self._cancel_loop(old_loop)
            await self._close_if_possible(old_source)
            await self.hub.start_peer_stt_provider_ingress()
        except Exception as exc:
            if source is not None:
                await self._close_if_possible(source)
            if attached:
                await self.hub.abort_peer_stt_for_toggle_off()
            await self._fault_current_generation(
                generation,
                config=config,
                reason=(
                    PeerRuntimeFailureReason.PROCESS_PROVIDER_FAILED
                    if config.capture_target.kind == "process"
                    else PeerRuntimeFailureReason.PEER_RUNTIME_FAILED
                ),
                detach_provider=False,
            )
            self._emit_local_asr_diagnostic(
                config,
                outcome="failed",
                load_started_at=self.clock.now(),
                failure_type=type(exc).__name__,
            )

    async def _teardown_owned_resources(
        self,
        *,
        target_state: PeerChannelRuntimeState,
        generation: int,
        release_mode: Literal["drain", "abort"],
    ) -> None:
        async with self._lock:
            loop_task = self._loop_task
            source = self._audio_source
            self._loop_task = None
            self._audio_source = None
            self._vad = None
            self._signature = None
            if release_mode == "abort":
                self._provider_signature = None
        failures: list[Exception] = []
        await self._attempt_cleanup(failures, lambda: self._cancel_loop(loop_task))
        await self._attempt_cleanup(
            failures,
            lambda: self._close_if_possible(source),
            retain_on_failure=lambda: self._retain_retired_source(source),
        )
        runtime = self.hub.local_asr_provider_runtime
        if runtime is not None:
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

    async def _transition_running_provider(
        self,
        config: PeerRuntimeConfig,
        *,
        generation: int,
    ) -> None:
        self._last_local_asr_transition_status = "preparing"
        current = self._stt
        if current is None:
            self._last_local_asr_transition_status = "failed"
            return
        options = config.session_options
        current_backend = getattr(current, "backend", None)
        current_model_id = getattr(current_backend, "resolved_model_id", None) or getattr(
            current_backend,
            "model_id",
            None,
        )
        if current_model_id == config.model_id and options is not None:
            reconfigure = getattr(current, "reconfigure_session_options", None)
            if callable(reconfigure):
                result = reconfigure(options)
                if inspect.isawaitable(result):
                    await result
            async with self._lock:
                if self._config is config and self._generation == generation:
                    if config.backend.provider == "local_qwen":
                        self._retained_stt = current
                    self._provider_signature = config.provider_signature
                    self._signature = config.runtime_signature
            self._last_local_asr_transition_status = "applied"
            return
        request = LocalASRTransitionRequest(
            channel="peer",
            requested_provider=config.backend.provider,
            actual_provider=config.backend.provider,
            model_id=config.model_id,
            session_options=options
            or LocalASRSessionOptions(source_language=config.backend.source_language),
            trigger="settings",
        )

        async def prepare(
            prepared_request: LocalASRTransitionRequest,
            transition_generation: int,
        ) -> PreparedLocalASRTransition:
            load_started_at = self.clock.now()
            candidate = self._stt_factory(
                config,
                lambda exc, *, _generation=generation: self._on_terminal_stt_failure(
                    exc,
                    generation=_generation,
                ),
            )
            if inspect.isawaitable(candidate):
                candidate = await candidate
            try:
                warmup = getattr(candidate, "warmup", None)
                if callable(warmup):
                    result = warmup()
                    if inspect.isawaitable(result):
                        await result
            except BaseException:
                await self._close_peer_provider_for_discard(candidate)
                raise
            return PreparedLocalASRTransition(
                request=prepared_request,
                provider=candidate,
                generation=transition_generation,
                load_ms=max(0, int(round((self.clock.now() - load_started_at) * 1000))),
            )

        async def commit(prepared: PreparedLocalASRTransition) -> None:
            async with self._lock:
                if (
                    self._config is not config
                    or self._generation != generation
                    or not self._desired_active
                ):
                    raise RuntimeError("peer provider transition superseded")
            handoff = getattr(self.hub, "handoff_peer_stt_provider", None)
            if callable(handoff):
                result = handoff(prepared.provider, start=True)
                try:
                    if inspect.isawaitable(result):
                        await result
                except asyncio.CancelledError:
                    cancel_handoff = getattr(self.hub, "cancel_peer_stt_provider_handoff", None)
                    if callable(cancel_handoff):
                        cancel_result = cancel_handoff(prepared.provider)
                        if inspect.isawaitable(cancel_result):
                            await cancel_result
                    raise
            else:
                await self._replace_peer_stt_provider(prepared.provider, start=True)
            async with self._lock:
                if self._config is config and self._generation == generation:
                    self._stt = prepared.provider
                    self._retained_stt = (
                        prepared.provider if config.backend.provider == "local_qwen" else None
                    )
                    self._provider_signature = config.provider_signature
                    self._signature = config.runtime_signature

        outcome = await self._transition_coordinator.request_transition(
            request,
            prepare=prepare,
            commit=commit,
        )
        self._last_local_asr_transition_status = outcome.status

    async def _wait_for_activations_before(self, generation: int) -> None:
        async with self._lock:
            pending = tuple(
                event
                for activation_generation, event in self._activation_events.items()
                if activation_generation < generation
            )
        if pending:
            await asyncio.gather(*(event.wait() for event in pending))

    async def _start_generation(self, generation: int, config: PeerRuntimeConfig) -> None:
        fresh_candidate = False
        created_candidate: object | None = None
        try:
            reusable = config.backend.provider == "local_qwen"

            async def build_and_warm() -> object:
                nonlocal created_candidate, fresh_candidate
                load_started_at = self.clock.now()
                try:
                    stt = (
                        self._retained_stt
                        if reusable and self._provider_signature == config.provider_signature
                        else None
                    )
                    if stt is None:
                        fresh_candidate = True
                        stt = self._stt_factory(
                            config,
                            lambda exc, *, _generation=generation: self._on_terminal_stt_failure(
                                exc, generation=_generation
                            ),
                        )
                        if inspect.isawaitable(stt):
                            stt = await stt
                        created_candidate = stt
                    warmup = (
                        getattr(stt, "warmup", None)
                        if config.backend.provider in _LOCAL_ASR_PROVIDERS
                        else None
                    )
                    if callable(warmup):
                        result = warmup()
                        if inspect.isawaitable(result):
                            await result
                except Exception as exc:
                    if fresh_candidate and config.backend.provider in _LOCAL_ASR_PROVIDERS:
                        self._emit_local_asr_diagnostic(
                            config,
                            outcome="failed",
                            load_started_at=load_started_at,
                            failure_type=type(exc).__name__,
                        )
                    raise
                if fresh_candidate and config.backend.provider in _LOCAL_ASR_PROVIDERS:
                    self._emit_local_asr_diagnostic(
                        config,
                        outcome="applied",
                        load_started_at=load_started_at,
                    )
                if reusable and fresh_candidate:
                    async with self._lock:
                        current_config = self._config
                        if (
                            current_config is not None
                            and current_config.provider_signature == config.provider_signature
                            and self._retained_stt is None
                        ):
                            self._retained_stt = stt
                            self._provider_signature = config.provider_signature
                return stt

            if reusable:
                async with self._activation_lock:
                    if self._is_superseded(generation):
                        return
                    stt = await build_and_warm()
            else:
                stt = await build_and_warm()
        except Exception:
            if created_candidate is not None and fresh_candidate:
                await self._close_peer_provider_for_discard(created_candidate)
            await self._fault_current_generation(
                generation,
                config=config,
                reason=(
                    PeerRuntimeFailureReason.PROCESS_PROVIDER_FAILED
                    if config.capture_target.kind == "process"
                    else PeerRuntimeFailureReason.PEER_RUNTIME_FAILED
                ),
                detach_provider=False,
            )
            return

        if self._is_superseded(generation):
            if not await self._retain_superseded_compatible_candidate(
                stt,
                config=config,
                fresh_candidate=fresh_candidate,
            ):
                await self._discard_unattached_peer_start(None, stt)
            return

        source = None
        try:
            source = self._source_factory(config)
            if inspect.isawaitable(source):
                source = await source
            model_path = self._vad_model_resolver()
            vad = self._vad_factory(config, model_path)
        except Exception as exc:
            await self._cleanup_failed_startup(generation, source, stt, config=config, exc=exc)
            return

        if self._is_superseded(generation):
            await self._discard_unattached_peer_start(source, stt)
            return

        cleanup_failures: list[Exception] = []
        loop_to_cancel = None
        source_to_close = None
        replacement_failure: Exception | None = None
        replacement_failed_before_attach = False
        superseded_before_replacement = False

        async with self._lock:
            if self._is_superseded(generation):
                superseded_before_replacement = True
            else:
                loop_to_cancel = self._loop_task
                source_to_close = self._audio_source
                self._loop_task = None
                self._audio_source = None
                self._vad = None

        if superseded_before_replacement:
            await self._discard_unattached_peer_start(source, stt)
            return

        try:
            await self._replace_peer_stt_provider(stt, start=False)
        except Exception as exc:
            replacement_failure = exc
            replacement_failed_before_attach = getattr(self.hub, "peer_stt", None) is not stt

        if replacement_failed_before_attach or (
            replacement_failure is not None and config.capture_target.kind == "process"
        ):
            if replacement_failure is not None:
                cleanup_failures.append(replacement_failure)
            await self._attempt_cleanup(
                cleanup_failures,
                lambda: self._cancel_loop(loop_to_cancel),
            )
            await self._attempt_cleanup(
                cleanup_failures,
                lambda: self._close_if_possible(source_to_close),
                retain_on_failure=lambda: self._retain_retired_source(source_to_close),
            )
            await self._attempt_cleanup(
                cleanup_failures,
                lambda: self._close_if_possible(source),
                retain_on_failure=lambda: self._retain_retired_source(source),
            )
            await self._attempt_cleanup(
                cleanup_failures,
                lambda: self._close_replacement_provider(stt),
                retain_on_failure=lambda: self._retain_retired_peer_provider(stt),
            )
            await self._attempt_cleanup(
                cleanup_failures,
                lambda: self._fault_current_generation(
                    generation,
                    config=config,
                    reason=(
                        PeerRuntimeFailureReason.PROCESS_PROVIDER_FAILED
                        if config.capture_target.kind == "process"
                        else PeerRuntimeFailureReason.PEER_RUNTIME_FAILED
                    ),
                    detach_provider=True,
                    retry_retired=False,
                ),
            )
            self._raise_cleanup_failures(
                "peer provider replacement failed before attach",
                cleanup_failures,
            )
            return

        if self._is_superseded(generation):
            if replacement_failure is not None:
                cleanup_failures.append(replacement_failure)
            await self._attempt_cleanup(
                cleanup_failures,
                lambda: self._cancel_loop(loop_to_cancel),
            )
            await self._attempt_cleanup(
                cleanup_failures,
                lambda: self._close_if_possible(source_to_close),
                retain_on_failure=lambda: self._retain_retired_source(source_to_close),
            )
            await self._attempt_cleanup(
                cleanup_failures,
                lambda: self._close_if_possible(source),
                retain_on_failure=lambda: self._retain_retired_source(source),
            )
            if stt is not self._retained_stt:
                await self._attempt_cleanup(
                    cleanup_failures,
                    lambda: self._close_peer_provider_if_current(stt),
                    retain_on_failure=lambda: self._retain_retired_peer_provider(stt),
                )
            self._raise_cleanup_failures(
                "peer channel superseded replacement cleanup failed",
                cleanup_failures,
            )
            return

        self._stt = stt
        if reusable:
            self._retained_stt = stt
            self._provider_signature = config.provider_signature
        self._audio_source = source
        self._vad = vad
        self._signature = config.runtime_signature

        await self._attempt_cleanup(
            cleanup_failures,
            lambda: self._cancel_loop(loop_to_cancel),
        )
        await self._attempt_cleanup(
            cleanup_failures,
            lambda: self._close_if_possible(source_to_close),
            retain_on_failure=lambda: self._retain_retired_source(source_to_close),
        )
        if self._is_superseded(generation):
            if replacement_failure is not None:
                cleanup_failures.append(replacement_failure)
            self._raise_cleanup_failures(
                "peer channel superseded running cleanup failed",
                cleanup_failures,
            )
            return

        loop_task = None
        async with self._lock:
            if (
                not self._is_superseded(generation)
                and self._stt is stt
                and self._audio_source is source
            ):
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

        if loop_task is None:
            if replacement_failure is not None:
                cleanup_failures.append(replacement_failure)
            self._raise_cleanup_failures(
                "peer channel superseded running cleanup failed",
                cleanup_failures,
            )
            return

        await self._attempt_cleanup(
            cleanup_failures,
            lambda: self._start_peer_stt_provider_ingress_if_current(stt, generation),
        )
        if replacement_failure is not None:
            cleanup_failures.append(replacement_failure)
        self._raise_cleanup_failures(
            "peer channel replacement cleanup failed",
            cleanup_failures,
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
                detach_provider=True,
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
            detach_provider=True,
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
        if self._provider_request_factory is not None:
            await self._fault_current_generation(
                target_generation,
                config=config,
                reason=(
                    PeerRuntimeFailureReason.PROCESS_PROVIDER_FAILED
                    if config is not None and config.capture_target.kind == "process"
                    else PeerRuntimeFailureReason.PEER_RUNTIME_FAILED
                ),
                detach_provider=True,
            )
            return
        async with self._lock:
            if self._is_superseded(target_generation):
                return
            if (
                self._desired_active
                and self._state == PeerChannelRuntimeState.RUNNING
                and self._stt is not None
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
            detach_provider=True,
        )

    async def _fault_current_generation(
        self,
        generation: int,
        *,
        config: PeerRuntimeConfig | None,
        reason: PeerRuntimeFailureReason,
        detach_provider: bool,
        retry_retired: bool = True,
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
        try:
            await self._mark_faulted_if_current(
                generation,
                detach_provider=detach_provider,
                retry_retired=retry_retired,
            )
        finally:
            if diagnostic is not None:
                if defer_diagnostic:
                    self._deferred_loop_diagnostics[current_task] = diagnostic
                else:
                    self._emit_failure(diagnostic)

    async def _mark_faulted_if_current(
        self,
        generation: int,
        *,
        detach_provider: bool,
        retry_retired: bool = True,
    ) -> None:
        async with self._lock:
            if self._is_superseded(generation):
                return
            self._generation += 1
            teardown_generation = self._generation
            self._desired_active = False
        await self._teardown_resources(
            target_state=PeerChannelRuntimeState.FAULTED,
            generation=teardown_generation,
            retry_retired=retry_retired,
        )
        if not detach_provider:
            return
        if getattr(self.hub, "peer_stt", None) is None:
            await self._replace_peer_stt_provider(None, start=False)

    async def _teardown_resources(
        self,
        *,
        target_state: PeerChannelRuntimeState,
        generation: int,
        retry_retired: bool = True,
        abort_for_toggle_off: bool = False,
    ) -> None:
        if self._provider_request_factory is not None:
            await self._teardown_owned_resources(
                target_state=target_state,
                generation=generation,
                release_mode="abort",
            )
            return
        async with self._lock:
            loop_task = self._loop_task
            source = self._audio_source
            stt = self._stt if self._stt is not None else self._retained_stt
            self._loop_task = None
            self._audio_source = None
            self._vad = None
            self._stt = None
            self._retained_stt = None
            self._provider_signature = None
            self._signature = None

        cleanup_failures: list[Exception] = []
        detached_peer_provider: object | None = None

        async def detach_current_peer_provider() -> None:
            nonlocal detached_peer_provider
            detached_peer_provider = await self._detach_current_peer_provider()

        abort_peer = getattr(self.hub, "abort_peer_stt_for_toggle_off", None)
        if (
            abort_for_toggle_off
            and stt is not None
            and getattr(self.hub, "peer_stt", None) is stt
            and callable(abort_peer)
        ):

            async def abort_current_peer_provider() -> None:
                nonlocal detached_peer_provider
                result = abort_peer(stt)
                if inspect.isawaitable(result):
                    await result
                detached_peer_provider = stt

            await self._attempt_cleanup(
                cleanup_failures,
                abort_current_peer_provider,
                retain_on_failure=lambda: self._retain_retired_peer_provider(stt),
            )
        else:
            await self._attempt_cleanup(
                cleanup_failures,
                detach_current_peer_provider,
                retain_on_failure=lambda: self._retain_retired_peer_provider(stt),
            )
        if stt is not None and detached_peer_provider is None:
            await self._attempt_cleanup(
                cleanup_failures,
                lambda: self._close_peer_provider_for_discard(stt),
                retain_on_failure=lambda: self._retain_retired_peer_provider(stt),
            )
        if retry_retired:
            await self._retry_retired_cleanup_debt(cleanup_failures)
        await self._attempt_cleanup(
            cleanup_failures,
            lambda: self._cancel_loop(loop_task),
        )
        await self._attempt_cleanup(
            cleanup_failures,
            lambda: self._close_if_possible(source),
            retain_on_failure=lambda: self._retain_retired_source(source),
        )

        async with self._lock:
            if self._generation == generation:
                self._state = target_state

        self._raise_cleanup_failures("peer channel teardown failed", cleanup_failures)

    async def _retain_superseded_compatible_candidate(
        self,
        stt: object,
        *,
        config: PeerRuntimeConfig,
        fresh_candidate: bool,
    ) -> bool:
        if not fresh_candidate or config.backend.provider != "local_qwen":
            return False
        async with self._lock:
            current_config = self._config
            if (
                current_config is None
                or current_config.provider_signature != config.provider_signature
                or self._retained_stt not in (None, stt)
            ):
                return False
            self._retained_stt = stt
            self._provider_signature = config.provider_signature
            return True

    async def _stop_for_dormant_reuse(
        self,
        generation: int,
        config: PeerRuntimeConfig,
    ) -> bool:
        drain = getattr(self.hub, "drain_peer_stt_for_toggle_off", None)
        async with self._lock:
            stt = self._stt if self._stt is not None else self._retained_stt
            if (
                stt is None
                or config.backend.provider != "local_qwen"
                or self._provider_signature != config.provider_signature
                or not callable(drain)
            ):
                return False
            loop_task = self._loop_task
            source = self._audio_source
            self._loop_task = None
            self._audio_source = None
            self._vad = None
            self._stt = None
            self._signature = None

        failures: list[Exception] = []
        if getattr(self.hub, "peer_stt", None) is stt:
            await self._attempt_cleanup(failures, lambda: drain(stt))
        else:
            await self._attempt_cleanup(
                failures,
                lambda: self._retire_unattached_provider_for_reuse(stt),
            )
        await self._attempt_cleanup(failures, lambda: self._cancel_loop(loop_task))
        await self._attempt_cleanup(
            failures,
            lambda: self._close_if_possible(source),
            retain_on_failure=lambda: self._retain_retired_source(source),
        )
        async with self._lock:
            if self._generation == generation:
                self._state = PeerChannelRuntimeState.STOPPED
        self._raise_cleanup_failures("peer channel dormant teardown failed", failures)
        await self._schedule_idle_release(
            stt=stt,
            provider_signature=config.provider_signature,
            generation=generation,
        )
        return True

    async def _retire_unattached_provider_for_reuse(self, stt: object) -> None:
        await self._close_if_possible(stt)
        discard_pending_events = getattr(stt, "discard_pending_events", None)
        if not callable(discard_pending_events):
            return
        result = discard_pending_events()
        if inspect.isawaitable(result):
            await result

    async def _schedule_idle_release(
        self,
        *,
        stt: object,
        provider_signature: tuple[object, ...],
        generation: int,
    ) -> None:
        if self._idle_release_seconds is None:
            return
        async with self._lock:
            if (
                self._closed
                or self._desired_active
                or self._generation != generation
                or self._retained_stt is not stt
                or self._provider_signature != provider_signature
            ):
                return
            self._last_idle_release_error_type = None
            self._idle_release_started = False
            delay_seconds = max(
                0.0,
                (self._idle_release_deadline or self.clock.now()) - self.clock.now(),
            )
            task = self._create_task(
                self._release_dormant_provider_after(
                    stt=stt,
                    provider_signature=provider_signature,
                    generation=generation,
                    delay_seconds=delay_seconds,
                ),
                task_name="idle-release",
            )
            self._idle_release_task = task
            task.add_done_callback(self._on_idle_release_task_done)

    async def _release_dormant_provider_after(
        self,
        *,
        stt: object,
        provider_signature: tuple[object, ...],
        generation: int,
        delay_seconds: float,
    ) -> None:
        await self._sleep(delay_seconds)
        self._idle_release_started = True
        async with self._activation_lock:
            async with self._lock:
                if (
                    self._closed
                    or self._desired_active
                    or self._generation != generation
                    or self._retained_stt is not stt
                    or self._provider_signature != provider_signature
                ):
                    return
                self._retained_stt = None
                self._provider_signature = None
            try:
                if getattr(self.hub, "peer_stt", None) is stt:
                    await self._replace_peer_stt_provider(None, start=False)
                else:
                    await self._close_peer_provider_for_discard(stt)
            except Exception:
                self._retain_retired_peer_provider(stt)
                raise

    async def _cancel_idle_release_task(self) -> None:
        task = self._idle_release_task
        if task is None:
            return
        self._idle_release_task = None
        if task is asyncio.current_task():
            return
        if not task.done() and not self._idle_release_started:
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _on_idle_release_task_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            exception = task.exception()
            if exception is not None:
                self._last_idle_release_error_type = type(exception).__name__
        if self._idle_release_task is task:
            self._idle_release_task = None
            self._idle_release_deadline = None
        self._idle_release_started = False

    def _create_task(self, coroutine: Awaitable[None], *, task_name: str) -> asyncio.Task[None]:
        return asyncio.create_task(coroutine, name=f"PeerChannelRuntime:{task_name}")

    async def _cleanup_failed_startup(
        self,
        generation: int,
        source: object | None,
        stt: object,
        *,
        config: PeerRuntimeConfig,
        exc: Exception,
    ) -> None:
        cleanup_failures: list[Exception] = []
        await self._attempt_cleanup(
            cleanup_failures,
            lambda: self._close_if_possible(source),
            retain_on_failure=lambda: self._retain_retired_source(source),
        )
        await self._attempt_cleanup(
            cleanup_failures,
            lambda: self._close_peer_provider_for_discard(stt),
            retain_on_failure=lambda: self._retain_retired_peer_provider(stt),
        )
        await self._attempt_cleanup(
            cleanup_failures,
            lambda: self._fault_current_generation(
                generation,
                config=config,
                reason=self._failure_reason_from_startup_exception(config, exc),
                detach_provider=True,
                retry_retired=False,
            ),
        )
        self._raise_cleanup_failures("peer channel startup cleanup failed", cleanup_failures)

    async def _discard_unattached_peer_start(
        self,
        source: object | None,
        stt: object,
    ) -> None:
        cleanup_failures: list[Exception] = []
        await self._attempt_cleanup(
            cleanup_failures,
            lambda: self._close_if_possible(source),
            retain_on_failure=lambda: self._retain_retired_source(source),
        )
        if stt is not self._retained_stt:
            await self._attempt_cleanup(
                cleanup_failures,
                lambda: self._close_peer_provider_for_discard(stt),
                retain_on_failure=lambda: self._retain_retired_peer_provider(stt),
            )
        self._raise_cleanup_failures(
            "peer channel discarded startup cleanup failed", cleanup_failures
        )

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
        retired_sources = tuple(self._retired_sources)
        retired_peer_providers = tuple(self._retired_peer_providers)

        for source in retired_sources:
            try:
                await self._close_if_possible(source)
            except Exception as exc:
                cleanup_failures.append(exc)
            else:
                self._forget_retired_source(source)

        for stt in retired_peer_providers:
            try:
                await self._close_peer_provider_for_discard(stt)
            except Exception as exc:
                cleanup_failures.append(exc)
            else:
                self._forget_retired_peer_provider(stt)

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

    def _retain_retired_peer_provider(self, stt: object | None) -> None:
        if stt is None:
            return
        if any(retired_provider is stt for retired_provider in self._retired_peer_providers):
            return
        self._retired_peer_providers.append(stt)

    def _forget_retired_peer_provider(self, stt: object) -> None:
        self._retired_peer_providers = [
            retired_provider
            for retired_provider in self._retired_peer_providers
            if retired_provider is not stt
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
        if loop_task is None:
            return
        if loop_task is asyncio.current_task():
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

    async def _close_peer_provider_if_current(self, stt: object | None) -> None:
        if stt is None:
            return
        if getattr(self.hub, "peer_stt", None) is stt:
            await self._replace_peer_stt_provider(None, start=False)
            return
        await self._close_peer_provider_for_discard(stt)

    async def _close_replacement_provider(self, stt: object) -> None:
        if getattr(self.hub, "peer_stt", None) is stt:
            await self._close_peer_provider_if_current(stt)
            return
        await self._close_peer_provider_for_discard(stt)

    async def _detach_current_peer_provider(self) -> object | None:
        current_peer_stt = getattr(self.hub, "peer_stt", None)
        if current_peer_stt is None:
            return None
        await self._replace_peer_stt_provider(None, start=False)
        return current_peer_stt

    async def _replace_peer_stt_provider(
        self,
        stt: object | None,
        *,
        start: bool,
    ) -> None:
        replace_provider = self.hub.replace_peer_stt_provider
        if self._call_supports_start_keyword(replace_provider):
            result = replace_provider(stt, start=start)
        else:
            result = replace_provider(stt)
        if inspect.isawaitable(result):
            await result

    async def _start_peer_stt_provider_ingress_if_current(
        self,
        stt: object,
        generation: int,
    ) -> None:
        async with self._lock:
            should_start = (
                not self._is_superseded(generation)
                and self._stt is stt
                and self._state == PeerChannelRuntimeState.RUNNING
            )
        if not should_start:
            return
        start_ingress = getattr(self.hub, "start_peer_stt_provider_ingress", None)
        if not callable(start_ingress):
            return
        result = start_ingress(stt)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _call_supports_start_keyword(callable_object: object) -> bool:
        try:
            signature = inspect.signature(callable_object)
        except (TypeError, ValueError):
            return False
        return any(
            name == "start" or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for name, parameter in signature.parameters.items()
        )

    async def _close_peer_provider_for_discard(self, stt: object | None) -> None:
        if stt is None:
            return
        close_backend = getattr(stt, "close_backend", None)
        if callable(close_backend):
            result = close_backend()
            if inspect.isawaitable(result):
                await result
            return
        await self._close_if_possible(stt)

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
