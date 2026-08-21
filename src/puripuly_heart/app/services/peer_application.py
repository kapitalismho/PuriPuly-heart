from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from puripuly_heart.app.services.local_asr_selection import LOCAL_CPU_PROVIDERS
from puripuly_heart.core.peer_capture import (
    PeerCaptureDiagnostic,
    PeerCaptureFailureReason,
    PeerCaptureProviderStatus,
    PeerCaptureSessionConfig,
    PeerCaptureSessionSnapshot,
)
from puripuly_heart.core.runtime.peer_channel import (
    PeerCaptureSessionOwner,
    PeerLocalASRTransitionSuperseded,
)
from puripuly_heart.core.runtime.provider_rebuild import ProviderRuntimeRebuildService


@dataclass(frozen=True, slots=True)
class PeerApplicationState:
    settings_available: bool
    peer_intent_enabled: bool
    eula_accepted: bool
    overlay_intent_enabled: bool
    peer_provider_id: str | None
    runtime_available: bool
    peer_provider_available: bool
    overlay_state: str
    overlay_command_available: bool
    ingress_frozen: bool = False


@dataclass(frozen=True, slots=True)
class PeerApplicationSnapshot:
    intent_enabled: bool
    activation_requested: bool
    effective_enabled: bool
    desired_active: bool
    activation_generation: int
    activation_starting: bool
    model_loading: bool
    process_warning_reason: str | None
    runtime_signature: tuple[object, ...] | None
    provider_signature: tuple[object, ...] | None


PeerApplicationStateProvider = Callable[[], PeerApplicationState]
PeerApplicationConfigFactory = Callable[[], PeerCaptureSessionConfig]
PeerApplicationManualFallback = Callable[[], bool]
PeerApplicationEnsureReady = Callable[[int], Awaitable[bool]]
PeerApplicationEffect = Callable[[], None]
PeerApplicationIntentSink = Callable[[bool], None]
PeerApplicationOverlayStart = Callable[[], Awaitable[None]]
PeerApplicationEffectiveSink = Callable[[bool, bool], None]
PeerApplicationLogSink = Callable[[str], object]
PeerApplicationSupersededSink = Callable[[], None]
PeerApplicationLifecycleTraceSink = Callable[[str, dict[str, object]], None]
PeerApplicationTranslationDemandSink = Callable[[], Awaitable[None]]


@dataclass(slots=True)
class PeerApplicationOwner:
    state_provider: PeerApplicationStateProvider = field(repr=False)
    config_factory: PeerApplicationConfigFactory = field(repr=False)
    peer_intent_sink: PeerApplicationIntentSink = field(repr=False)
    overlay_intent_sink: PeerApplicationIntentSink = field(repr=False)
    persist_manual_fallback: PeerApplicationManualFallback = field(repr=False)
    ensure_local_ready: PeerApplicationEnsureReady = field(repr=False)
    clear_cpu_pending: PeerApplicationEffect = field(repr=False)
    clear_gpu_pending: PeerApplicationEffect = field(repr=False)
    clear_switched_pending: PeerApplicationEffect = field(repr=False)
    sync_local_notice: PeerApplicationEffect = field(repr=False)
    presentation_changed: PeerApplicationEffect = field(repr=False)
    begin_overlay_start: PeerApplicationOverlayStart = field(repr=False)
    effective_sink: PeerApplicationEffectiveSink = field(repr=False)
    disclosure_sink: PeerApplicationEffect = field(repr=False)
    superseded_sink: PeerApplicationSupersededSink = field(repr=False)
    log_basic: PeerApplicationLogSink = field(repr=False)
    log_detailed: PeerApplicationLogSink = field(repr=False)
    log_failure: PeerApplicationLogSink = field(repr=False)
    lifecycle_trace_sink: PeerApplicationLifecycleTraceSink | None = field(
        default=None,
        repr=False,
    )
    translation_demand_sink: PeerApplicationTranslationDemandSink | None = field(
        default=None,
        repr=False,
    )
    _runtime: PeerCaptureSessionOwner | None = field(init=False, default=None, repr=False)
    _rebuild: ProviderRuntimeRebuildService = field(
        init=False,
        default_factory=ProviderRuntimeRebuildService,
        repr=False,
    )
    _last_runtime_signature: tuple[object, ...] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _last_provider_signature: tuple[object, ...] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _last_intent_enabled: bool | None = field(init=False, default=None, repr=False)
    _last_activation_requested: bool | None = field(init=False, default=None, repr=False)
    _activation_generation: int = field(init=False, default=0, repr=False)
    _activation_starting: bool = field(init=False, default=False, repr=False)
    _model_loading: bool = field(init=False, default=False, repr=False)
    _model_loading_generation: int | None = field(init=False, default=None, repr=False)
    _runtime_refresh_hold_generation: int | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _process_warning_reason: str | None = field(init=False, default=None, repr=False)
    _effective_trace_generation: int | None = field(init=False, default=None, repr=False)
    _ingress_stopped: bool = field(init=False, default=False, repr=False)
    _runtime_lock: asyncio.Lock = field(
        init=False,
        default_factory=asyncio.Lock,
        repr=False,
    )

    @property
    def runtime(self) -> PeerCaptureSessionOwner | None:
        return self._runtime

    @property
    def last_runtime_signature(self) -> tuple[object, ...] | None:
        return self._last_runtime_signature

    @last_runtime_signature.setter
    def last_runtime_signature(self, value: tuple[object, ...] | None) -> None:
        self._last_runtime_signature = value

    @property
    def last_provider_signature(self) -> tuple[object, ...] | None:
        return self._last_provider_signature

    @last_provider_signature.setter
    def last_provider_signature(self, value: tuple[object, ...] | None) -> None:
        self._last_provider_signature = value

    @property
    def last_intent_enabled(self) -> bool | None:
        return self._last_intent_enabled

    @last_intent_enabled.setter
    def last_intent_enabled(self, value: bool | None) -> None:
        self._last_intent_enabled = value

    @property
    def last_activation_requested(self) -> bool | None:
        return self._last_activation_requested

    @last_activation_requested.setter
    def last_activation_requested(self, value: bool | None) -> None:
        self._last_activation_requested = value

    @property
    def activation_generation(self) -> int:
        return self._activation_generation

    @activation_generation.setter
    def activation_generation(self, value: int) -> None:
        self._activation_generation = int(value)

    @property
    def activation_starting(self) -> bool:
        return self._activation_starting

    @activation_starting.setter
    def activation_starting(self, value: bool) -> None:
        self._activation_starting = bool(value)

    @property
    def model_loading(self) -> bool:
        return self._model_loading

    @model_loading.setter
    def model_loading(self, value: bool) -> None:
        self._model_loading = bool(value)
        self._model_loading_generation = self._activation_generation if value else None

    @property
    def process_warning_reason(self) -> str | None:
        return self._process_warning_reason

    @process_warning_reason.setter
    def process_warning_reason(self, value: str | None) -> None:
        self._process_warning_reason = value

    @staticmethod
    def activation_requested(*, intent_enabled: bool, eula_accepted: bool) -> bool:
        return bool(intent_enabled and eula_accepted)

    def effective_enabled(self, state: PeerApplicationState | None = None) -> bool:
        current = state or self.state_provider()
        runtime = self._runtime
        return bool(
            current.settings_available
            and self.activation_requested(
                intent_enabled=current.peer_intent_enabled,
                eula_accepted=current.eula_accepted,
            )
            and current.overlay_state == "connected"
            and current.runtime_available
            and current.peer_provider_available
            and runtime is not None
            and runtime.snapshot.effective_active
        )

    def desired_active(self, state: PeerApplicationState | None = None) -> bool:
        current = state or self.state_provider()
        return bool(
            current.settings_available
            and self.activation_requested(
                intent_enabled=current.peer_intent_enabled,
                eula_accepted=current.eula_accepted,
            )
            and current.overlay_state == "connected"
            and current.runtime_available
            and current.overlay_command_available
        )

    def snapshot(self, state: PeerApplicationState | None = None) -> PeerApplicationSnapshot:
        current = state or self.state_provider()
        intent_enabled = bool(current.settings_available and current.peer_intent_enabled)
        activation_requested = bool(
            current.settings_available
            and self.activation_requested(
                intent_enabled=current.peer_intent_enabled,
                eula_accepted=current.eula_accepted,
            )
        )
        effective_enabled = self.effective_enabled(current)
        if effective_enabled or not intent_enabled:
            self._process_warning_reason = None
        return PeerApplicationSnapshot(
            intent_enabled=intent_enabled,
            activation_requested=activation_requested,
            effective_enabled=effective_enabled,
            desired_active=self.desired_active(current),
            activation_generation=self._activation_generation,
            activation_starting=self._activation_starting,
            model_loading=self._model_loading,
            process_warning_reason=self._process_warning_reason,
            runtime_signature=self._last_runtime_signature,
            provider_signature=self._last_provider_signature,
        )

    def sync_effective_flags(self, state: PeerApplicationState | None = None) -> None:
        current = state or self.state_provider()
        enabled = self.effective_enabled(current)
        if enabled:
            self._activation_starting = False
            if self._effective_trace_generation != self._activation_generation:
                self._effective_trace_generation = self._activation_generation
                sink = self.lifecycle_trace_sink
                if sink is not None:
                    sink(
                        "peer_capture_effective",
                        {
                            "activation_generation": self._activation_generation,
                            "accepted": True,
                        },
                    )
        self.effective_sink(enabled, enabled)

    def record_settings(
        self,
        *,
        intent_enabled: bool,
        eula_accepted: bool,
        config: PeerCaptureSessionConfig,
    ) -> None:
        self._last_runtime_signature = config.runtime_signature
        self._last_provider_signature = config.provider_signature
        self._last_intent_enabled = intent_enabled
        self._last_activation_requested = self.activation_requested(
            intent_enabled=intent_enabled,
            eula_accepted=eula_accepted,
        )

    def bind_runtime(self, runtime: PeerCaptureSessionOwner | None) -> None:
        self._runtime = runtime

    async def replace_runtime(self, runtime: PeerCaptureSessionOwner) -> None:
        async with self._runtime_lock:
            previous = self._runtime
            if previous is runtime:
                if self._ingress_stopped:
                    await runtime.close()
                    if self._runtime is runtime:
                        self._runtime = None
                return
            if self._ingress_stopped:
                await self._close_rejected_runtime(runtime)
                return
            if previous is not None:
                await previous.close()
            if self._ingress_stopped:
                if self._runtime is previous:
                    self._runtime = None
                await self._close_rejected_runtime(runtime)
                return
            self._runtime = runtime

    async def _close_rejected_runtime(self, runtime: PeerCaptureSessionOwner) -> None:
        try:
            await runtime.close()
        except BaseException:
            if self._runtime is None:
                self._runtime = runtime
            raise

    async def set_enabled(self, enabled: bool) -> None:
        state = self.state_provider()
        if not state.settings_available or self._ingress_stopped or state.ingress_frozen:
            return
        enabled = bool(enabled)
        if (
            enabled
            and state.peer_provider_id != "local_cpu_auto"
            and not self.persist_manual_fallback()
        ):
            return
        self._activation_generation += 1
        generation = self._activation_generation
        self.log_basic(f"[Peer] Toggle request: enabled={enabled}")
        self.log_detailed(
            "[Peer] Toggle detail: "
            f"overlay_enabled={state.overlay_intent_enabled} "
            f"overlay_state={state.overlay_state} "
            f"peer_stt_available={state.peer_provider_available} "
            f"eula_accepted={state.eula_accepted}"
        )
        if enabled and not state.eula_accepted:
            self.disable_intent()
            self.sync_effective_flags()
            self.presentation_changed()
            self.log_basic("[Peer] Toggle ignored: eula_accepted=False")
            await self._notify_translation_demand()
            return
        if enabled and not state.overlay_intent_enabled:
            self.overlay_intent_sink(True)
        self.peer_intent_sink(enabled)
        self._last_intent_enabled = enabled
        self._last_activation_requested = self.activation_requested(
            intent_enabled=enabled,
            eula_accepted=state.eula_accepted,
        )
        self._activation_starting = enabled
        if enabled:
            current = self.state_provider()
            if current.overlay_state not in {"starting", "connected"}:
                await self.begin_overlay_start()
                if generation != self._activation_generation:
                    return
        self.presentation_changed()
        await self._notify_translation_demand()
        ready = False
        if enabled:
            if self.local_stt_requested():
                self._runtime_refresh_hold_generation = generation
            try:
                ready = await self.ensure_local_ready(generation)
            except BaseException:
                if self._runtime_refresh_hold_generation == generation:
                    self._runtime_refresh_hold_generation = None
                raise
            if generation != self._activation_generation:
                return
            if not ready:
                self._activation_starting = False
                if self._runtime_refresh_hold_generation == generation:
                    self._runtime_refresh_hold_generation = None
        else:
            self.clear_cpu_pending()
            self.clear_gpu_pending()
        self.clear_switched_pending()
        self.sync_local_notice()
        self.presentation_changed()
        current = self.state_provider()
        prepared = False
        runtime = self._runtime
        prepare_local = bool(
            enabled
            and ready
            and current.peer_provider_id in LOCAL_CPU_PROVIDERS
            and runtime is not None
        )
        config = self.config_factory() if prepare_local else None
        if prepare_local:
            self._model_loading = True
            self._model_loading_generation = generation
            self.sync_local_notice()
            self.presentation_changed()
        try:
            if prepare_local and runtime is not None and config is not None:
                prepared_snapshot = await runtime.prepare_provider(config)
        finally:
            if self._model_loading_generation == generation:
                self._model_loading = False
                self._model_loading_generation = None
                self.sync_local_notice()
                self.presentation_changed()
            if self._runtime_refresh_hold_generation == generation:
                self._runtime_refresh_hold_generation = None
        if prepare_local and config is not None:
            if generation != self._activation_generation or self._runtime is not runtime:
                return
            prepared = prepared_snapshot.provider_status is PeerCaptureProviderStatus.READY
            if prepared:
                self._last_provider_signature = config.provider_signature
                self._last_runtime_signature = config.runtime_signature
            else:
                self._activation_starting = False
        current = self.state_provider()
        if not enabled:
            await self.refresh_dependencies(stop_mode="release")
        elif current.overlay_state == "connected" and (not prepare_local or prepared):
            await self.refresh_dependencies()
        elif current.overlay_state == "starting" and not prepare_local:
            await self.refresh_dependencies()
        if generation != self._activation_generation:
            return
        self.sync_effective_flags()
        if enabled:
            self.disclosure_sink()
        self.presentation_changed()

    async def _notify_translation_demand(self) -> None:
        sink = self.translation_demand_sink
        if sink is None:
            return
        await sink()

    def disable_intent(self) -> None:
        if self.state_provider().settings_available:
            self.peer_intent_sink(False)
        self._last_intent_enabled = False
        self._last_activation_requested = False
        self._activation_starting = False

    def cancel_activation_starting(self) -> None:
        self._activation_starting = False

    def invalidate_activation(self) -> None:
        self._activation_generation += 1

    def disable_for_overlay(self) -> None:
        self.invalidate_activation()
        self.disable_intent()

    async def refresh_dependencies(
        self,
        *,
        stop_mode: Literal["retain", "release"] = "retain",
    ) -> None:
        await self.refresh_runtime(stop_mode=stop_mode)
        self.sync_effective_flags()
        self.presentation_changed()

    async def refresh_runtime(
        self,
        *,
        stop_mode: Literal["retain", "release"] = "retain",
    ) -> None:
        state = self.state_provider()
        runtime = self._runtime
        if not state.settings_available or not state.runtime_available or runtime is None:
            return
        if self._should_hold_runtime_refresh(state):
            return
        config = self.config_factory()
        desired_active = self.desired_active(state)
        previous_signature = getattr(runtime, "current_signature", None)
        peer_local_provider = bool(desired_active and config.local_provider)
        peer_local_transition = bool(
            peer_local_provider
            and previous_signature is not None
            and previous_signature != config.runtime_signature
        )
        peer_local_loading = bool(
            peer_local_provider
            and (
                not state.peer_provider_available or previous_signature != config.runtime_signature
            )
        )
        if peer_local_loading:
            self._model_loading = True
            self.sync_local_notice()
            self.presentation_changed()
        try:
            await self._rebuild.apply_peer_policy(
                peer_runtime=runtime,
                config=config,
                desired_active=desired_active,
                stop_mode=stop_mode,
            )
            transition_status = getattr(
                runtime,
                "last_local_asr_transition_status",
                "idle",
            )
            if peer_local_transition and transition_status == "superseded":
                raise PeerLocalASRTransitionSuperseded
            if peer_local_transition and transition_status == "failed":
                raise RuntimeError("peer local ASR transition failed")
        except PeerLocalASRTransitionSuperseded:
            self.superseded_sink()
            raise
        finally:
            if peer_local_loading:
                self._model_loading = False
                self.sync_local_notice()
                self.presentation_changed()
        self._last_runtime_signature = config.runtime_signature
        self.sync_effective_flags()

    async def retry_process_capture(self) -> bool:
        generation = self._activation_generation
        state = self.state_provider()
        runtime = self._runtime
        if runtime is None or not self._retry_context_is_current(
            state=state,
            generation=generation,
            runtime=runtime,
        ):
            return False
        if not await self.ensure_local_ready(generation):
            return False
        current = self.state_provider()
        if not self._retry_context_is_current(
            state=current,
            generation=generation,
            runtime=runtime,
        ):
            return False
        retried = await runtime.retry_process_capture(config=self.config_factory())
        current = self.state_provider()
        if not self._retry_context_is_current(
            state=current,
            generation=generation,
            runtime=runtime,
        ):
            return False
        if retried:
            self._process_warning_reason = None
        self.sync_effective_flags()
        self.presentation_changed()
        return retried

    def _retry_context_is_current(
        self,
        *,
        state: PeerApplicationState,
        generation: int,
        runtime: PeerCaptureSessionOwner | None,
    ) -> bool:
        return bool(
            not self._ingress_stopped
            and not state.ingress_frozen
            and generation == self._activation_generation
            and runtime is not None
            and self._runtime is runtime
            and state.settings_available
            and self.desired_active(state)
        )

    def _should_hold_runtime_refresh(self, state: PeerApplicationState) -> bool:
        if not self.local_stt_requested(state):
            return False
        if self._model_loading and self._model_loading_generation == self._activation_generation:
            return True
        return self._runtime_refresh_hold_generation == self._activation_generation

    def local_stt_requested(self, state: PeerApplicationState | None = None) -> bool:
        resolved = state or self.state_provider()
        return bool(
            resolved.settings_available
            and resolved.peer_provider_id in LOCAL_CPU_PROVIDERS
            and self.activation_requested(
                intent_enabled=resolved.peer_intent_enabled,
                eula_accepted=resolved.eula_accepted,
            )
        )

    def on_runtime_state_changed(self, snapshot: PeerCaptureSessionSnapshot) -> None:
        if snapshot.state.value == "running":
            self._process_warning_reason = None

    def on_runtime_diagnostic(self, diagnostic: PeerCaptureDiagnostic) -> None:
        unavailable_reason = getattr(
            diagnostic,
            "detail",
            getattr(diagnostic, "process_unavailable_reason", None),
        )
        self.log_detailed(
            "[PeerRuntime] "
            f"reason={diagnostic.reason.value} "
            f"capture_kind={diagnostic.capture_kind} "
            f"unavailable_reason={unavailable_reason}"
        )
        if diagnostic.reason is not PeerCaptureFailureReason.PROCESS_PROVIDER_FAILED:
            suffix = (
                f" unavailable_reason={unavailable_reason}"
                if unavailable_reason is not None
                else ""
            )
            self.log_failure(
                "[PeerRuntime] outcome=failed "
                f"reason={diagnostic.reason.value} capture_kind={diagnostic.capture_kind}{suffix}"
            )
        if diagnostic.capture_kind == "process":
            self._process_warning_reason = self.warning_reason_for_diagnostic(diagnostic)
            self._activation_starting = False
            self.presentation_changed()

    @staticmethod
    def warning_reason_for_diagnostic(diagnostic: PeerCaptureDiagnostic) -> str:
        if diagnostic.reason is PeerCaptureFailureReason.PROCESS_TARGET_UNAVAILABLE:
            unavailable = (
                getattr(
                    diagnostic,
                    "detail",
                    getattr(diagnostic, "process_unavailable_reason", None),
                )
                or "no_process"
            )
            return f"process_unavailable_{unavailable}"
        return diagnostic.reason.value

    def stop_ingress(self) -> None:
        self._ingress_stopped = True
        self.invalidate_activation()
        self._activation_starting = False

    async def close(self) -> None:
        self.stop_ingress()
        async with self._runtime_lock:
            runtime = self._runtime
            if runtime is not None:
                await runtime.close()
            if self._runtime is runtime:
                self._runtime = None


__all__ = [
    "PeerApplicationOwner",
    "PeerApplicationSnapshot",
    "PeerApplicationState",
]
