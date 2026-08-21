from __future__ import annotations

import asyncio
import inspect
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace

from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeSnapshot,
    ProviderGpuRuntimePort,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeChannel,
    ProviderRuntimeChannelPhase,
    ProviderRuntimeChannelSnapshot,
    ProviderRuntimeDiagnostic,
    ProviderRuntimeEventHandler,
    ProviderRuntimeExceptionHandler,
    ProviderRuntimeGpuPhase,
    ProviderRuntimeGpuRecoveryRequest,
    ProviderRuntimeGpuSnapshot,
    ProviderRuntimeMutationResult,
    ProviderRuntimeProviderFactoryPort,
    ProviderRuntimeRecoveryQuiesce,
    ProviderRuntimeReleaseMode,
    ProviderRuntimeTerminalFailureSink,
)
from puripuly_heart.core.local_asr_provisioning import LocalASRProvisioningPort
from puripuly_heart.core.runtime.gpu_asr import GpuASRDiagnostic
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle

ProviderRuntimeStateChanged = Callable[
    [LocalASRProviderRuntimeSnapshot],
    Awaitable[None] | None,
]
ProviderRuntimeDiagnosticSink = Callable[
    [ProviderRuntimeDiagnostic],
    Awaitable[None] | None,
]
ProviderGpuRuntimeFactory = Callable[
    [Callable[[GpuASRDiagnostic], Awaitable[None]]],
    ProviderGpuRuntimePort,
]

_CHANNELS: tuple[ProviderRuntimeChannel, ...] = ("self", "peer")
_GPU_PROVIDER_ID = "local_qwen_gpu"
_COMPLETED_NO_GPU_FAILURE_CODES = frozenset({"unsupported_capability"})


class LocalASRProviderRuntimeOwner:
    resource_fields = (
        "_self_provider_handle",
        "_peer_provider_handle",
        "_gpu_runtime",
        "_gpu_recovery_lock",
        "_gpu_discovery_task",
        "_operation_tasks",
        "_callback_tasks",
        "_scope",
        "pending provider handoffs",
        "retired providers and event drains",
        "authenticated GPU worker and model residency",
    )
    stop_ingress = "reject new provider, discovery, activation, handoff, and retry commands"
    shutdown_policy = (
        "cancel and await owner operations, close Self and Peer providers, then close the shared "
        "GPU runtime with bounded worker termination"
    )
    late_callback_rule = "provider generations reject stale events and GPU-runtime generations reject late diagnostics"

    def __init__(
        self,
        *,
        provider_factory: ProviderRuntimeProviderFactoryPort,
        gpu_runtime_factory: ProviderGpuRuntimeFactory,
        provisioning: LocalASRProvisioningPort,
        self_event_handler: ProviderRuntimeEventHandler | None = None,
        peer_event_handler: ProviderRuntimeEventHandler | None = None,
        retired_event_handler: ProviderRuntimeEventHandler | None = None,
        self_exception_handler: ProviderRuntimeExceptionHandler | None = None,
        peer_exception_handler: ProviderRuntimeExceptionHandler | None = None,
        state_changed: ProviderRuntimeStateChanged | None = None,
        diagnostic_sink: ProviderRuntimeDiagnosticSink | None = None,
        diagnostics_capacity: int = 256,
        prebuilt_providers: Mapping[ProviderRuntimeChannel, object | None] | None = None,
    ) -> None:
        if diagnostics_capacity < 1:
            raise ValueError("diagnostics_capacity must be positive")
        self._provider_factory = provider_factory
        self._gpu_runtime_factory = gpu_runtime_factory
        self._provisioning = provisioning
        self._state_changed = state_changed
        self._diagnostic_sink = diagnostic_sink
        self._diagnostics: deque[ProviderRuntimeDiagnostic] = deque(maxlen=diagnostics_capacity)
        self._revision = 0
        self._closing = False
        self._closed = False
        self._close_complete = False
        self._close_lock = asyncio.Lock()
        self._gpu_recovery_lock = asyncio.Lock()
        self._started = False
        initial_providers = dict(prebuilt_providers or {})
        self._channel_phases: dict[ProviderRuntimeChannel, ProviderRuntimeChannelPhase] = {
            channel: "ready" if initial_providers.get(channel) is not None else "inactive"
            for channel in _CHANNELS
        }
        self._provider_ids: dict[ProviderRuntimeChannel, str | None] = {
            channel: _prebuilt_provider_id(initial_providers.get(channel)) for channel in _CHANNELS
        }
        self._model_ids: dict[ProviderRuntimeChannel, str | None] = {
            channel: _prebuilt_model_id(initial_providers.get(channel)) for channel in _CHANNELS
        }
        self._last_requests: dict[ProviderRuntimeChannel, ProviderRuntimeBuildRequest] = {}
        self._last_terminal_failure_sinks: dict[
            ProviderRuntimeChannel,
            ProviderRuntimeTerminalFailureSink | None,
        ] = {}
        self._pending_candidates: dict[ProviderRuntimeChannel, object] = {}
        self._pending_requests: dict[ProviderRuntimeChannel, ProviderRuntimeBuildRequest] = {}
        self._operation_tasks: dict[asyncio.Task[object], int] = {}
        self._callback_tasks: set[asyncio.Task[object]] = set()
        self._scope = LifecycleScope("local-asr-provider-runtime")
        self._gpu_generation = 0
        self._gpu_phase: ProviderRuntimeGpuPhase = "inactive"
        self._gpu_failure_code: str | None = None
        self._gpu_devices = ()
        self._gpu_discovery_attempted = False
        self._gpu_discovery_task: asyncio.Task[LocalASRProviderRuntimeSnapshot] | None = None
        self._gpu_runtime = self._create_gpu_runtime()
        self._handles: dict[ProviderRuntimeChannel, ProviderRuntimeHandle] = {
            "self": ProviderRuntimeHandle(
                name="self_stt",
                provider=initial_providers.get("self"),
                event_handler=self_event_handler,
                retired_event_handler=retired_event_handler,
                exception_handler=self_exception_handler,
                state_changed=self._on_handle_state_changed,
            ),
            "peer": ProviderRuntimeHandle(
                name="peer_stt",
                provider=initial_providers.get("peer"),
                event_handler=peer_event_handler,
                retired_event_handler=retired_event_handler,
                exception_handler=peer_exception_handler,
                state_changed=self._on_handle_state_changed,
            ),
        }

    @property
    def owner_name(self) -> str:
        return "LocalASRProviderRuntimeOwner"

    @property
    def snapshot(self) -> LocalASRProviderRuntimeSnapshot:
        channel_states = tuple(self._channel_snapshot(channel) for channel in _CHANNELS)
        runtime_state = _enum_value(getattr(self._gpu_runtime, "state", "idle"))
        worker_pid = getattr(self._gpu_runtime, "worker_pid", None)
        active_channels = frozenset(getattr(self._gpu_runtime, "active_channels", ()))
        failure_code = self._gpu_failure_code or getattr(
            self._gpu_runtime,
            "last_failure_code",
            None,
        )
        return LocalASRProviderRuntimeSnapshot(
            channels=channel_states,
            gpu=ProviderRuntimeGpuSnapshot(
                phase="closed" if self._closed else self._gpu_phase,
                devices=tuple(self._gpu_devices),
                active_channels=active_channels,
                pending_count=int(getattr(self._gpu_runtime, "pending_count", 0)),
                worker_pid=worker_pid,
                configured_device_id=getattr(
                    self._gpu_runtime,
                    "configured_device_id",
                    None,
                ),
                model_resident=bool(worker_pid is not None and runtime_state == "ready"),
                retry_required=bool(runtime_state == "failed" or self._gpu_phase == "failed"),
                failure_code=failure_code,
            ),
            revision=self._revision,
            closed=self._closed,
        )

    @property
    def diagnostics(self) -> tuple[ProviderRuntimeDiagnostic, ...]:
        return tuple(self._diagnostics)

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
            "provider_handles": {
                channel: self._handles[channel].lifecycle_owner_snapshot() for channel in _CHANNELS
            },
        }

    async def start(self) -> None:
        self._require_open("start provider runtime")
        if self._started:
            return
        self._started = True
        await asyncio.gather(*(handle.start() for handle in self._handles.values()))
        for channel in _CHANNELS:
            if self._handles[channel].provider is not None:
                self._channel_phases[channel] = "running"
        await self._publish_state()

    async def discover_gpu(
        self,
        *,
        force: bool = False,
    ) -> LocalASRProviderRuntimeSnapshot:
        self._require_open("discover GPU devices")
        task = self._gpu_discovery_task
        if task is not None and not task.done():
            return await asyncio.shield(task)
        if not force and self._gpu_discovery_attempted:
            return self.snapshot
        task = start_lifecycle_task(
            self._scope,
            self._run_gpu_discovery(),
            name=f"{self.owner_name}:gpu-discovery:{self._gpu_generation}",
        )
        self._gpu_discovery_task = task
        try:
            return await asyncio.shield(task)
        finally:
            if self._gpu_discovery_task is task and task.done():
                self._gpu_discovery_task = None

    async def inspect_gpu_readiness(
        self,
        *,
        explicit_intent: bool,
        device_id: str,
    ) -> LocalASRProviderRuntimeSnapshot:
        return await self._inspect_gpu_readiness(
            explicit_intent=explicit_intent,
            device_id=device_id,
            allow_device_change=False,
        )

    async def _inspect_gpu_readiness(
        self,
        *,
        explicit_intent: bool,
        device_id: str,
        allow_device_change: bool,
    ) -> LocalASRProviderRuntimeSnapshot:
        self._require_open("inspect GPU readiness")
        if not device_id.strip():
            raise ValueError("device_id must be non-empty")
        async with self._operation():
            if not explicit_intent:
                self._gpu_phase = "inactive"
                self._gpu_failure_code = None
                await self._publish_state()
                return self.snapshot
            await self.discover_gpu()
            known_devices = {device.device_id for device in self._gpu_devices}
            if not self._gpu_devices:
                if self._gpu_phase == "failed":
                    return self.snapshot
                self._gpu_phase = "unsupported"
                self._gpu_failure_code = "no_supported_gpu"
                await self._publish_state()
                return self.snapshot
            if device_id != "auto" and device_id not in known_devices:
                self._gpu_phase = "failed"
                self._gpu_failure_code = "saved_device_missing"
                await self._emit_diagnostic(
                    ProviderRuntimeDiagnostic(
                        event="gpu_readiness",
                        outcome="failed",
                        phase="device_validation",
                        device_id=device_id,
                        failure_code="saved_device_missing",
                    )
                )
                await self._publish_state()
                return self.snapshot
            active_channels = frozenset(getattr(self._gpu_runtime, "active_channels", ()))
            configured_device_id = getattr(
                self._gpu_runtime,
                "configured_device_id",
                None,
            )
            if (
                active_channels
                and configured_device_id is not None
                and device_id != configured_device_id
                and not allow_device_change
            ):
                self._gpu_phase = "failed"
                self._gpu_failure_code = "device_change_requires_quiesce"
                await self._emit_diagnostic(
                    ProviderRuntimeDiagnostic(
                        event="gpu_readiness",
                        outcome="failed",
                        phase="device_validation",
                        device_id=device_id,
                        failure_code="device_change_requires_quiesce",
                    )
                )
                await self._publish_state()
                return self.snapshot
            self._gpu_phase = "validating"
            self._gpu_failure_code = None
            await self._publish_state()
            provisioning = await self._provisioning.inspect_gpu(
                explicit_intent=True,
                verify_checksums=False,
            )
            status = provisioning.state_for(provisioning.gpu_model_id).status
            if status == "ready":
                runtime_state = _enum_value(getattr(self._gpu_runtime, "state", "idle"))
                self._gpu_phase = "ready" if runtime_state == "ready" else "available"
                self._gpu_failure_code = None
            elif status in {"missing", "not_requested"}:
                self._gpu_phase = "not_installed"
                self._gpu_failure_code = status
            elif status == "downloading":
                self._gpu_phase = "installing"
                self._gpu_failure_code = status
            else:
                self._gpu_phase = "invalid"
                self._gpu_failure_code = status
            await self._emit_diagnostic(
                ProviderRuntimeDiagnostic(
                    event="gpu_readiness",
                    outcome="ready" if status == "ready" else "unavailable",
                    phase=self._gpu_phase,
                    device_id=device_id,
                    model_id=provisioning.gpu_model_id,
                    failure_code=None if status == "ready" else status,
                )
            )
            await self._publish_state()
            return self.snapshot

    async def replace_provider(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        start: bool,
        on_terminal_failure: ProviderRuntimeTerminalFailureSink | None = None,
    ) -> ProviderRuntimeMutationResult:
        self._require_open("replace provider")
        async with self._operation():
            ready = await self._ensure_request_ready(request)
            if not ready:
                return self._failed_result(request, failure_type="ProviderReadinessError")
            channel = request.channel
            previous_provider_id = self._provider_ids[channel]
            provider = await self._build_provider(
                request,
                on_terminal_failure=on_terminal_failure,
            )
            if provider is None:
                return self._failed_result(
                    request,
                    previous_provider_id=previous_provider_id,
                )
            handle = self._handles[channel]
            previous_phase = self._channel_phases[channel]
            next_phase: ProviderRuntimeChannelPhase = (
                "running" if start else ("ready" if request.warmup else "dormant")
            )
            try:
                await handle.replace_provider(provider, start=start)
            except Exception as exc:
                if handle.provider is provider:
                    self._provider_ids[channel] = request.provider_id
                    self._model_ids[channel] = request.model_id
                    self._last_requests[channel] = request
                    self._last_terminal_failure_sinks[channel] = on_terminal_failure
                    self._channel_phases[channel] = next_phase
                else:
                    self._channel_phases[channel] = previous_phase
                    await _close_provider_for_discard(provider)
                await self._emit_provider_failure(
                    event="provider_replace",
                    request=request,
                    exc=exc,
                )
                await self._publish_state()
                return self._failed_result(
                    request,
                    previous_provider_id=previous_provider_id,
                    failure_type=type(exc).__name__,
                )
            self._provider_ids[channel] = request.provider_id
            self._model_ids[channel] = request.model_id
            self._last_requests[channel] = request
            self._last_terminal_failure_sinks[channel] = on_terminal_failure
            self._channel_phases[channel] = next_phase
            await self._emit_diagnostic(
                ProviderRuntimeDiagnostic(
                    event="provider_replace",
                    outcome="applied",
                    channel=channel,
                    provider_id=request.provider_id,
                    phase=self._channel_phases[channel],
                )
            )
            await self._publish_state()
            return ProviderRuntimeMutationResult(
                status="applied",
                request=request,
                previous_provider_id=previous_provider_id,
                snapshot=self.snapshot,
            )

    def current_provider(self, channel: ProviderRuntimeChannel) -> object | None:
        self._validate_channel(channel)
        return self._handles[channel].provider

    async def replace_prebuilt_provider(
        self,
        channel: ProviderRuntimeChannel,
        provider: object | None,
        *,
        start: bool,
    ) -> object | None:
        self._require_open("replace prebuilt provider")
        self._validate_channel(channel)
        async with self._operation():
            try:
                return await self._handles[channel].replace_provider(provider, start=start)
            finally:
                current = self._handles[channel].provider
                self._provider_ids[channel] = _prebuilt_provider_id(current)
                self._model_ids[channel] = _prebuilt_model_id(current)
                self._channel_phases[channel] = (
                    "inactive" if current is None else "running" if start else "ready"
                )
                await self._publish_state()

    async def handoff_prebuilt_provider(
        self,
        channel: ProviderRuntimeChannel,
        provider: object,
        *,
        start: bool,
    ) -> object | None:
        self._require_open("handoff prebuilt provider")
        self._validate_channel(channel)
        async with self._operation():
            self._pending_candidates[channel] = provider
            try:
                previous = await self._handles[channel].handoff_provider_at_boundary(
                    provider,
                    start=start,
                )
            except asyncio.CancelledError:
                await self._handles[channel].cancel_pending_handoff(provider)
                await self._discard_pending_candidate(channel, provider)
                raise
            except Exception:
                await self._discard_pending_candidate(channel, provider)
                raise
            self._pending_candidates.pop(channel, None)
            self._provider_ids[channel] = _prebuilt_provider_id(provider)
            self._model_ids[channel] = _prebuilt_model_id(provider)
            self._channel_phases[channel] = "running" if start else "ready"
            await self._publish_state()
            return previous

    async def handoff_provider(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        start: bool,
        on_terminal_failure: ProviderRuntimeTerminalFailureSink | None = None,
    ) -> ProviderRuntimeMutationResult:
        self._require_open("handoff provider")
        async with self._operation():
            ready = await self._ensure_request_ready(request)
            if not ready:
                return self._failed_result(request, failure_type="ProviderReadinessError")
            channel = request.channel
            previous_provider_id = self._provider_ids[channel]
            provider = await self._build_provider(
                request,
                on_terminal_failure=on_terminal_failure,
            )
            if provider is None:
                return self._failed_result(
                    request,
                    previous_provider_id=previous_provider_id,
                )
            self._pending_candidates[channel] = provider
            self._pending_requests[channel] = request
            try:
                await self._handles[channel].handoff_provider_at_boundary(
                    provider,
                    start=start,
                )
            except asyncio.CancelledError:
                await self._handles[channel].cancel_pending_handoff(provider)
                await self._discard_pending_candidate(channel, provider)
                raise
            except Exception as exc:
                await self._discard_pending_candidate(channel, provider)
                await self._emit_provider_failure(
                    event="provider_handoff",
                    request=request,
                    exc=exc,
                )
                await self._publish_state()
                return self._failed_result(
                    request,
                    previous_provider_id=previous_provider_id,
                    failure_type=type(exc).__name__,
                )
            self._pending_candidates.pop(channel, None)
            self._pending_requests.pop(channel, None)
            self._provider_ids[channel] = request.provider_id
            self._model_ids[channel] = request.model_id
            self._last_requests[channel] = request
            self._last_terminal_failure_sinks[channel] = on_terminal_failure
            self._channel_phases[channel] = (
                "running" if start else ("ready" if request.warmup else "dormant")
            )
            await self._emit_diagnostic(
                ProviderRuntimeDiagnostic(
                    event="provider_handoff",
                    outcome="applied",
                    channel=channel,
                    provider_id=request.provider_id,
                    phase=self._channel_phases[channel],
                )
            )
            await self._publish_state()
            return ProviderRuntimeMutationResult(
                status="applied",
                request=request,
                previous_provider_id=previous_provider_id,
                snapshot=self.snapshot,
            )

    async def commit_handoff(self, channel: ProviderRuntimeChannel) -> None:
        self._require_open("commit provider handoff")
        self._validate_channel(channel)
        request = self._pending_requests.get(channel)
        await self._handles[channel].commit_pending_handoff()
        if request is not None:
            self._provider_ids[channel] = request.provider_id
            self._model_ids[channel] = request.model_id
        await self._publish_state()

    async def cancel_handoff(self, channel: ProviderRuntimeChannel) -> bool:
        self._require_open("cancel provider handoff")
        self._validate_channel(channel)
        provider = self._pending_candidates.get(channel)
        if provider is None:
            return False
        cancelled = await self._handles[channel].cancel_pending_handoff(provider)
        if cancelled:
            await self._discard_pending_candidate(channel, provider)
            await self._publish_state()
        return cancelled

    async def release_channel(
        self,
        channel: ProviderRuntimeChannel,
        *,
        mode: ProviderRuntimeReleaseMode,
        release_backend_after: float | None = None,
    ) -> None:
        self._require_open("release provider channel")
        self._validate_channel(channel)
        async with self._operation():
            handle = self._handles[channel]
            if mode == "drain":
                await handle.drain_for_toggle_off(
                    release_backend_after=release_backend_after,
                )
                self._channel_phases[channel] = "dormant"
            elif mode == "dormant":
                provider = handle.provider
                if provider is not None:
                    await handle.retire_for_dormant_reuse(provider)
                self._channel_phases[channel] = "dormant"
            elif mode == "abort":
                try:
                    await handle.abort_and_release()
                finally:
                    if handle.provider is None:
                        self._provider_ids[channel] = None
                        self._model_ids[channel] = None
                        self._channel_phases[channel] = "inactive"
            else:
                raise ValueError("unsupported provider release mode")
            await self._emit_diagnostic(
                ProviderRuntimeDiagnostic(
                    event="provider_release",
                    outcome="applied",
                    channel=channel,
                    provider_id=self._provider_ids[channel],
                    phase=mode,
                )
            )
            await self._publish_state()

    async def start_channel(self, channel: ProviderRuntimeChannel) -> None:
        self._require_open("start provider channel")
        self._validate_channel(channel)
        await self._handles[channel].start()
        if self._handles[channel].provider is not None:
            self._channel_phases[channel] = "running"
        await self._publish_state()

    async def warmup_channel(self, channel: ProviderRuntimeChannel) -> None:
        self._require_open("warm provider channel")
        self._validate_channel(channel)
        async with self._operation():
            provider, generation = self._handles[channel].current_provider_generation()
            if provider is None:
                raise RuntimeError(f"no provider is attached for {channel}")
            previous_phase = self._channel_phases[channel]
            self._channel_phases[channel] = "building"
            await self._publish_state()
            try:
                await _call_async_method(provider, "warmup")
            except Exception as exc:
                self._channel_phases[channel] = "failed"
                await self._emit_diagnostic(
                    ProviderRuntimeDiagnostic(
                        event="provider_warmup",
                        outcome="failed",
                        channel=channel,
                        provider_id=self._provider_ids[channel],
                        model_id=self._model_ids[channel],
                        phase="failed",
                        failure_code=_optional_string(getattr(exc, "code", None)),
                        failure_type=type(exc).__name__,
                    )
                )
                await self._publish_state()
                raise
            if self._handles[channel].is_current_provider_generation(
                provider=provider,
                generation=generation,
            ):
                self._channel_phases[channel] = (
                    "running" if previous_phase == "running" else "ready"
                )
            await self._publish_state()

    async def reconfigure_channel(
        self,
        channel: ProviderRuntimeChannel,
        options: LocalASRSessionOptions,
    ) -> None:
        self._require_open("reconfigure provider channel")
        self._validate_channel(channel)
        async with self._operation():
            provider, generation = self._handles[channel].current_provider_generation()
            if provider is None:
                raise RuntimeError(f"no provider is attached for {channel}")
            await _call_async_method_with_argument(
                provider,
                "reconfigure_session_options",
                options,
            )
            if self._handles[channel].is_current_provider_generation(
                provider=provider,
                generation=generation,
            ):
                request = self._last_requests.get(channel)
                if request is not None:
                    self._last_requests[channel] = replace(
                        request,
                        session_options=options,
                    )
            await self._publish_state()

    async def handle_vad_event(
        self,
        channel: ProviderRuntimeChannel,
        event: object,
    ) -> None:
        self._require_open("dispatch provider VAD event")
        self._validate_channel(channel)
        async with self._operation():
            provider, generation = self._handles[channel].current_provider_generation()
            if provider is None:
                return
            await _call_async_method_with_argument(provider, "handle_vad_event", event)
            if not self._handles[channel].is_current_provider_generation(
                provider=provider,
                generation=generation,
            ):
                return

    async def recover_gpu(
        self,
        request: ProviderRuntimeGpuRecoveryRequest,
        *,
        quiesce: ProviderRuntimeRecoveryQuiesce | None = None,
    ) -> LocalASRProviderRuntimeSnapshot:
        self._require_open("recover GPU runtime")
        targets = {item.request.channel: item for item in request.channels}
        async with self._gpu_recovery_lock:
            async with self._operation():
                unavailable_phase: ProviderRuntimeGpuPhase | None = None
                unavailable_failure_code: str | None = None
                if targets:
                    readiness = await self._inspect_gpu_readiness(
                        explicit_intent=True,
                        device_id=request.device_id,
                        allow_device_change=True,
                    )
                    if readiness.gpu.phase not in {"available", "ready"}:
                        unavailable_phase = readiness.gpu.phase
                        unavailable_failure_code = readiness.gpu.failure_code
                current_gpu_channels = {
                    channel
                    for channel in _CHANNELS
                    if self._provider_ids[channel] == _GPU_PROVIDER_ID
                    or channel in getattr(self._gpu_runtime, "active_channels", frozenset())
                }
                affected_channels = tuple(
                    channel
                    for channel in _CHANNELS
                    if channel in current_gpu_channels | targets.keys()
                )
                for channel, target in targets.items():
                    self._last_requests[channel] = target.request
                    self._last_terminal_failure_sinks[channel] = target.on_terminal_failure
                if unavailable_phase is None:
                    self._gpu_phase = "validating"
                    self._gpu_failure_code = None
                await self._publish_state()
                try:
                    if quiesce is not None and affected_channels:
                        await quiesce(affected_channels)
                    for channel in _CHANNELS:
                        if self._provider_ids[channel] == _GPU_PROVIDER_ID:
                            await self.release_channel(channel, mode="abort")
                    previous_runtime = self._gpu_runtime
                    await previous_runtime.close()
                    self._gpu_runtime = self._create_gpu_runtime()
                    self._gpu_phase = "idle"
                    if unavailable_phase is not None:
                        self._gpu_phase = unavailable_phase
                        self._gpu_failure_code = unavailable_failure_code
                        await self._emit_diagnostic(
                            ProviderRuntimeDiagnostic(
                                event="gpu_recovery",
                                outcome="unavailable",
                                phase=unavailable_phase,
                                device_id=request.device_id,
                                failure_code=unavailable_failure_code,
                            )
                        )
                        await self._publish_state()
                        return self.snapshot
                    for channel in _CHANNELS:
                        target = targets.get(channel)
                        if target is None:
                            continue
                        result = await self.replace_provider(
                            target.request,
                            start=target.start,
                            on_terminal_failure=target.on_terminal_failure,
                        )
                        if result.status != "applied":
                            raise RuntimeError(f"GPU provider recovery failed for {channel}")
                except asyncio.CancelledError as exc:
                    if self._closing:
                        raise
                    cleanup_failures = await asyncio.shield(self._cleanup_failed_gpu_recovery())
                    if cleanup_failures:
                        raise BaseExceptionGroup(
                            "GPU provider recovery cancellation cleanup failed",
                            [exc, *cleanup_failures],
                        ) from exc
                    raise
                except Exception as exc:
                    cleanup_failures = await self._cleanup_failed_gpu_recovery()
                    self._gpu_phase = "failed"
                    self._gpu_failure_code = (
                        _optional_string(getattr(exc, "code", None)) or type(exc).__name__
                    )
                    await self._emit_diagnostic(
                        ProviderRuntimeDiagnostic(
                            event="gpu_recovery",
                            outcome="failed",
                            phase="failed",
                            device_id=request.device_id,
                            failure_code=self._gpu_failure_code,
                            failure_type=type(exc).__name__,
                        )
                    )
                    await self._publish_state()
                    if cleanup_failures:
                        raise ExceptionGroup(
                            "GPU provider recovery and cleanup failed",
                            [exc, *cleanup_failures],
                        ) from exc
                    raise
                self._gpu_phase = "ready" if targets else "inactive"
                self._gpu_failure_code = None
                await self._emit_diagnostic(
                    ProviderRuntimeDiagnostic(
                        event="gpu_recovery",
                        outcome="applied",
                        phase=self._gpu_phase,
                        device_id=request.device_id,
                    )
                )
                await self._publish_state()
                return self.snapshot

    async def _cleanup_failed_gpu_recovery(self) -> list[Exception]:
        failures: list[Exception] = []
        for channel in _CHANNELS:
            if self._provider_ids[channel] != _GPU_PROVIDER_ID:
                continue
            try:
                await self.release_channel(channel, mode="abort")
            except Exception as exc:
                failures.append(exc)
        if failures:
            return failures
        runtime = self._gpu_runtime
        try:
            await runtime.close()
        except Exception as exc:
            failures.append(exc)
        else:
            if self._gpu_runtime is runtime:
                self._gpu_runtime = self._create_gpu_runtime()
        return failures

    async def close(self) -> None:
        if self._close_complete:
            return
        async with self._close_lock:
            if self._close_complete:
                return
            self._closing = True
            self._closed = True
            self._gpu_phase = "closed"
            for channel in _CHANNELS:
                self._channel_phases[channel] = "closed"
            await self._publish_state()
            current = asyncio.current_task()
            operation_tasks = tuple(
                task for task in self._operation_tasks if task is not current and not task.done()
            )
            for task in operation_tasks:
                task.cancel()
            discovery_task = self._gpu_discovery_task
            if (
                discovery_task is not None
                and discovery_task is not current
                and not discovery_task.done()
            ):
                discovery_task.cancel()
            failures: list[Exception] = []
            if operation_tasks:
                operation_results = await asyncio.gather(
                    *operation_tasks,
                    return_exceptions=True,
                )
                failures.extend(
                    failure
                    for result in operation_results
                    if (failure := _close_failure_from_task_result(result)) is not None
                )
            if discovery_task is not None and discovery_task is not current:
                discovery_results = await asyncio.gather(
                    discovery_task,
                    return_exceptions=True,
                )
                failures.extend(
                    failure
                    for result in discovery_results
                    if (failure := _close_failure_from_task_result(result)) is not None
                )
            for channel in _CHANNELS:
                try:
                    await self._handles[channel].close()
                except Exception as exc:
                    failures.append(exc)
            for channel, provider in tuple(self._pending_candidates.items()):
                try:
                    await self._discard_pending_candidate(channel, provider)
                except Exception as exc:
                    failures.append(exc)
            try:
                await self._gpu_runtime.close()
            except Exception as exc:
                failures.append(exc)
            callback_tasks = tuple(
                task for task in self._callback_tasks if task is not current and not task.done()
            )
            if callback_tasks:
                callback_results = await asyncio.gather(
                    *callback_tasks,
                    return_exceptions=True,
                )
                failures.extend(
                    result for result in callback_results if isinstance(result, Exception)
                )
            try:
                await self._scope.close()
            except Exception as exc:
                failures.append(exc)
            self._closing = False
            if failures:
                raise ExceptionGroup("Local ASR provider runtime close failed", failures)
            self._close_complete = True

    async def _run_gpu_discovery(self) -> LocalASRProviderRuntimeSnapshot:
        async with self._operation():
            self._gpu_phase = "discovering"
            self._gpu_failure_code = None
            await self._emit_diagnostic(
                ProviderRuntimeDiagnostic(
                    event="gpu_discovery",
                    outcome="started",
                    phase="discovering",
                )
            )
            await self._publish_state()
            try:
                devices = await self._gpu_runtime.discover_devices()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failure_code = _gpu_discovery_failure_code(exc)
                if failure_code in _COMPLETED_NO_GPU_FAILURE_CODES:
                    return await self._finish_gpu_discovery(
                        devices=(),
                        phase="unsupported",
                        outcome="unsupported",
                        failure_code="no_supported_gpu",
                    )
                self._gpu_discovery_attempted = False
                self._gpu_devices = ()
                self._gpu_phase = "failed"
                self._gpu_failure_code = failure_code
                await self._emit_diagnostic(
                    ProviderRuntimeDiagnostic(
                        event="gpu_discovery",
                        outcome="failed",
                        phase="failed",
                        failure_code=self._gpu_failure_code,
                        failure_type=type(exc).__name__,
                    )
                )
                await self._publish_state()
                return self.snapshot
            return await self._finish_gpu_discovery(
                devices=devices,
                phase="idle" if devices else "unsupported",
                outcome="ready" if devices else "unsupported",
                failure_code=None if devices else "no_supported_gpu",
            )

    async def _finish_gpu_discovery(
        self,
        *,
        devices: tuple[object, ...],
        phase: ProviderRuntimeGpuPhase,
        outcome: str,
        failure_code: str | None,
    ) -> LocalASRProviderRuntimeSnapshot:
        self._gpu_discovery_attempted = True
        self._gpu_devices = devices
        self._gpu_phase = phase
        self._gpu_failure_code = failure_code
        await self._emit_diagnostic(
            ProviderRuntimeDiagnostic(
                event="gpu_discovery",
                outcome=outcome,
                phase=self._gpu_phase,
                failure_code=failure_code,
            )
        )
        await self._publish_state()
        return self.snapshot

    async def _ensure_request_ready(self, request: ProviderRuntimeBuildRequest) -> bool:
        if request.provider_id != _GPU_PROVIDER_ID:
            return True
        snapshot = await self.inspect_gpu_readiness(
            explicit_intent=True,
            device_id=request.gpu_device_id,
        )
        return snapshot.gpu.phase == "available" or snapshot.gpu.phase == "ready"

    async def _build_provider(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        on_terminal_failure: ProviderRuntimeTerminalFailureSink | None,
    ) -> object | None:
        channel = request.channel
        previous_phase = self._channel_phases[channel]
        self._channel_phases[channel] = "building"
        await self._publish_state()
        provider: object | None = None
        try:
            result = self._provider_factory.create(
                request,
                gpu_runtime=self._gpu_runtime,
                on_terminal_failure=on_terminal_failure,
            )
            provider = await result if inspect.isawaitable(result) else result
            if provider is None:
                raise RuntimeError("provider factory returned no provider")
            if request.warmup:
                await _call_async_method(provider, "warmup")
            return provider
        except asyncio.CancelledError:
            if provider is not None:
                await _close_provider_for_discard(provider)
            raise
        except Exception as exc:
            if provider is not None:
                try:
                    await _close_provider_for_discard(provider)
                except Exception:
                    pass
            self._channel_phases[channel] = (
                previous_phase if self._handles[channel].provider is not None else "failed"
            )
            await self._emit_provider_failure(
                event="provider_build",
                request=request,
                exc=exc,
            )
            await self._publish_state()
            return None

    async def _discard_pending_candidate(
        self,
        channel: ProviderRuntimeChannel,
        provider: object,
    ) -> None:
        if self._pending_candidates.get(channel) is not provider:
            return
        await _close_provider_for_discard(provider)
        if self._pending_candidates.get(channel) is provider:
            self._pending_candidates.pop(channel, None)
            self._pending_requests.pop(channel, None)

    async def _on_gpu_diagnostic(
        self,
        diagnostic: GpuASRDiagnostic,
        *,
        generation: int,
    ) -> None:
        if generation != self._gpu_generation or self._closing or self._closed:
            return
        fields = diagnostic.fields
        if diagnostic.kind == "worker_lifecycle":
            phase = str(fields.get("phase") or "")
            if phase in {"validating", "loading", "warming", "ready"}:
                self._gpu_phase = phase
        elif diagnostic.kind == "discovery_pending":
            self._gpu_phase = "discovery_pending"
        elif diagnostic.kind == "activation_ready":
            self._gpu_phase = "ready"
            self._gpu_failure_code = None
        elif diagnostic.kind in {"activation_failed", "worker_failed"}:
            self._gpu_phase = "failed"
            self._gpu_failure_code = _optional_string(fields.get("failure"))
        channel = _safe_channel(fields.get("channel"))
        await self._emit_diagnostic(
            ProviderRuntimeDiagnostic(
                event=diagnostic.kind,
                outcome=_optional_string(fields.get("result") or fields.get("outcome")),
                channel=channel,
                phase=_optional_string(fields.get("phase")) or self._gpu_phase,
                model_id=_optional_string(fields.get("model")),
                device_id=_optional_string(fields.get("device")),
                failure_code=_optional_string(fields.get("failure")),
                progress_percent=_optional_int(fields.get("progress_percent")),
                model_load_seconds=_optional_float(fields.get("model_load_seconds")),
                warmup_seconds=_optional_float(fields.get("warmup_seconds")),
                audio_seconds=_optional_float(fields.get("audio_seconds")),
                decode_seconds=_optional_float(fields.get("decode_seconds")),
                rtf=_optional_float(fields.get("rtf")),
                queue_wait_seconds=_optional_float(fields.get("queue_wait_seconds")),
                worker_exit_code=_optional_int(fields.get("exit_code")),
            )
        )
        await self._publish_state()

    def _create_gpu_runtime(self) -> ProviderGpuRuntimePort:
        self._gpu_generation += 1
        generation = self._gpu_generation

        async def diagnostic_sink(diagnostic: GpuASRDiagnostic) -> None:
            await self._on_gpu_diagnostic(diagnostic, generation=generation)

        return self._gpu_runtime_factory(diagnostic_sink)

    def _channel_snapshot(
        self,
        channel: ProviderRuntimeChannel,
    ) -> ProviderRuntimeChannelSnapshot:
        handle = self._handles[channel]
        lifecycle = handle.lifecycle_owner_snapshot()
        return ProviderRuntimeChannelSnapshot(
            channel=channel,
            provider_id=self._provider_ids[channel],
            model_id=self._model_ids[channel],
            phase="closed" if self._closed else self._channel_phases[channel],
            generation=handle.generation,
            pending_handoff=bool(lifecycle["pending_handoff"]),
            has_resources=handle.has_resources,
        )

    def _failed_result(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        previous_provider_id: str | None = None,
        failure_type: str | None = None,
    ) -> ProviderRuntimeMutationResult:
        if previous_provider_id is None:
            previous_provider_id = self._provider_ids[request.channel]
        if failure_type is None and self._diagnostics:
            failure_type = self._diagnostics[-1].failure_type
        return ProviderRuntimeMutationResult(
            status="failed",
            request=request,
            previous_provider_id=previous_provider_id,
            snapshot=self.snapshot,
            failure_type=failure_type,
        )

    async def _emit_provider_failure(
        self,
        *,
        event: str,
        request: ProviderRuntimeBuildRequest,
        exc: Exception,
    ) -> None:
        await self._emit_diagnostic(
            ProviderRuntimeDiagnostic(
                event=event,
                outcome="failed",
                channel=request.channel,
                provider_id=request.provider_id,
                phase=self._channel_phases[request.channel],
                failure_code=_optional_string(getattr(exc, "code", None)),
                failure_type=type(exc).__name__,
            )
        )

    async def _emit_diagnostic(self, diagnostic: ProviderRuntimeDiagnostic) -> None:
        self._diagnostics.append(diagnostic)
        if self._diagnostic_sink is None or self._closing or self._closed:
            return
        try:
            result = self._diagnostic_sink(diagnostic)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            self._diagnostics.append(
                ProviderRuntimeDiagnostic(
                    event="diagnostic_sink",
                    outcome="failed",
                    failure_type=type(exc).__name__,
                )
            )

    async def _publish_state(self) -> None:
        self._revision += 1
        if self._state_changed is None:
            return
        try:
            result = self._state_changed(self.snapshot)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            self._diagnostics.append(
                ProviderRuntimeDiagnostic(
                    event="state_changed_sink",
                    outcome="failed",
                    failure_type=type(exc).__name__,
                )
            )

    def _on_handle_state_changed(self, _handle: ProviderRuntimeHandle) -> None:
        if self._state_changed is None or self._closing or self._closed:
            return
        self._revision += 1
        try:
            result = self._state_changed(self.snapshot)
        except Exception as exc:
            self._diagnostics.append(
                ProviderRuntimeDiagnostic(
                    event="state_changed_sink",
                    outcome="failed",
                    failure_type=type(exc).__name__,
                )
            )
            return
        if not inspect.isawaitable(result):
            return
        try:
            task = start_lifecycle_task(
                self._scope,
                result,
                name=f"{self.owner_name}:state-changed",
            )
        except RuntimeError:
            if inspect.iscoroutine(result):
                result.close()
            return
        self._callback_tasks.add(task)
        task.add_done_callback(self._on_callback_task_done)

    def _on_callback_task_done(self, task: asyncio.Task[object]) -> None:
        self._callback_tasks.discard(task)
        if task.cancelled():
            return
        try:
            exception = task.exception()
        except asyncio.CancelledError:
            return
        if exception is not None:
            self._diagnostics.append(
                ProviderRuntimeDiagnostic(
                    event="state_changed_sink",
                    outcome="failed",
                    failure_type=type(exception).__name__,
                )
            )

    @asynccontextmanager
    async def _operation(self):
        task = asyncio.current_task()
        if task is not None:
            self._operation_tasks[task] = self._operation_tasks.get(task, 0) + 1
        try:
            self._require_open("continue provider runtime operation")
            yield
        finally:
            if task is not None:
                depth = self._operation_tasks.get(task, 0)
                if depth <= 1:
                    self._operation_tasks.pop(task, None)
                else:
                    self._operation_tasks[task] = depth - 1

    def _require_open(self, operation: str) -> None:
        if self._closing or self._closed:
            state = "closing" if self._closing else "closed"
            raise RuntimeError(f"{self.owner_name} is {state}; cannot {operation}")

    @staticmethod
    def _validate_channel(channel: ProviderRuntimeChannel) -> None:
        if channel not in _CHANNELS:
            raise ValueError("provider runtime channel must be self or peer")


async def _call_async_method(resource: object, method_name: str) -> None:
    method = getattr(resource, method_name, None)
    if not callable(method):
        return
    result = method()
    if inspect.isawaitable(result):
        await result


async def _call_async_method_with_argument(
    resource: object,
    method_name: str,
    argument: object,
) -> None:
    method = getattr(resource, method_name, None)
    if not callable(method):
        return
    result = method(argument)
    if inspect.isawaitable(result):
        await result


async def _close_provider_for_discard(provider: object) -> None:
    failures: list[Exception] = []
    for method_name in ("close", "close_backend"):
        try:
            await _call_async_method(provider, method_name)
        except Exception as exc:
            failures.append(exc)
    if failures:
        raise ExceptionGroup("discarded provider close failed", failures)


def _enum_value(value: object) -> str:
    enum_value = getattr(value, "value", value)
    return str(enum_value)


def _safe_channel(value: object) -> ProviderRuntimeChannel | None:
    return value if value in _CHANNELS else None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _gpu_discovery_failure_code(exc: BaseException) -> str:
    return _optional_string(getattr(exc, "code", None)) or type(exc).__name__


def _prebuilt_provider_id(provider: object | None) -> str | None:
    if provider is None:
        return None
    provider_name = getattr(provider, "stt_provider_name", None)
    provider_id = getattr(provider_name, "value", provider_name)
    return provider_id if isinstance(provider_id, str) else "prebuilt"


def _prebuilt_model_id(provider: object | None) -> str | None:
    model_id = getattr(getattr(provider, "backend", None), "model_id", None)
    return model_id if isinstance(model_id, str) else None


def _close_failure_from_task_result(result: object) -> Exception | None:
    if isinstance(result, Exception):
        return result
    if isinstance(result, BaseExceptionGroup):
        _cancelled, failure = result.split(asyncio.CancelledError)
        if isinstance(failure, Exception):
            return failure
    return None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


__all__ = [
    "LocalASRProviderRuntimeOwner",
    "ProviderGpuRuntimeFactory",
    "ProviderRuntimeDiagnosticSink",
    "ProviderRuntimeStateChanged",
]
