from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from puripuly_heart.app.services.local_asr_cpu_repair import (
    LocalASRCpuRepairOwner,
    LocalASRCpuRepairRequest,
)
from puripuly_heart.app.services.local_asr_selection import (
    LOCAL_CPU_AUTO_PROVIDER,
    LOCAL_CPU_PROVIDERS,
    local_asr_status_for_provider,
    required_local_asr_model_ids,
    resolve_local_asr_selection,
)
from puripuly_heart.core.local_asr_provider_runtime import (
    ProviderRuntimeChannelSnapshot,
)
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRProvisioningPort,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_stt_assets import (
    LocalParakeetSherpaLoadError,
    LocalQwenSherpaLoadError,
    LocalSTTManifestInvalidError,
    LocalSTTModelMissingError,
)
from puripuly_heart.core.local_stt_catalog import LocalCPUAutoUnavailableError


class LocalASRReadinessEffectType(str, Enum):
    DISABLE_SELF_UNSUPPORTED = "disable_self_unsupported"
    DISABLE_SELF_INVALID = "disable_self_invalid"
    SELF_DOWNLOAD_IN_PROGRESS = "self_download_in_progress"
    DISABLE_PEER_UNSUPPORTED = "disable_peer_unsupported"
    SYNC_NOTICE = "sync_notice"


@dataclass(frozen=True, slots=True)
class LocalASRReadinessEffect:
    type: LocalASRReadinessEffectType


@dataclass(frozen=True, slots=True)
class LocalASRReadinessState:
    settings_available: bool
    runtime_available: bool
    self_provider: str | None
    peer_provider: str | None
    self_source_language: str
    peer_source_language: str
    self_desired: bool
    peer_requested: bool
    self_activation_generation: int
    peer_activation_generation: int


LocalASRReadinessStateProvider = Callable[[], LocalASRReadinessState]
LocalASRReadinessEffectSink = Callable[[LocalASRReadinessEffect], None]
LocalASRReadinessAsyncEffect = Callable[[], Awaitable[None]]
LocalASRReadinessFallback = Callable[[str], bool]
LocalASRReadinessProviderAvailable = Callable[[], bool]
LocalASRReadinessChannelProvider = Callable[
    [],
    ProviderRuntimeChannelSnapshot | None,
]
LocalASRReadinessGpuValidator = Callable[[], Awaitable[bool]]
LocalASRReadinessGpuStateProvider = Callable[[], str | None]
LocalASRReadinessChannel = Literal["self", "peer"]
LocalASRReadinessGpuPendingSink = Callable[[LocalASRReadinessChannel], None]
LocalASRReadinessLoadLogSink = Callable[..., None]
LocalASRReadinessProvisioningProvider = Callable[[], LocalASRProvisioningPort]


@dataclass(slots=True)
class LocalASRReadinessOwner:
    provisioning_provider: LocalASRReadinessProvisioningProvider = field(repr=False)
    cpu_repair_owner: LocalASRCpuRepairOwner = field(repr=False)
    state_provider: LocalASRReadinessStateProvider = field(repr=False)
    effect_sink: LocalASRReadinessEffectSink = field(repr=False)
    self_provider_available: LocalASRReadinessProviderAvailable = field(repr=False)
    self_channel_provider: LocalASRReadinessChannelProvider = field(repr=False)
    rebuild_self_provider: LocalASRReadinessAsyncEffect = field(repr=False)
    probe_self_provider: LocalASRReadinessAsyncEffect = field(repr=False)
    persist_manual_fallback: LocalASRReadinessFallback = field(repr=False)
    validate_gpu_activation: LocalASRReadinessGpuValidator = field(repr=False)
    gpu_state_provider: LocalASRReadinessGpuStateProvider = field(repr=False)
    retain_gpu_pending: LocalASRReadinessGpuPendingSink = field(repr=False)
    load_log_sink: LocalASRReadinessLoadLogSink = field(repr=False)

    @property
    def owner_name(self) -> str:
        return "LocalASRReadinessOwner"

    def current_self_status(self) -> str:
        provider = self.state_provider().self_provider
        return (
            local_asr_status_for_provider(
                self.provisioning_provider().snapshot,
                provider,
            )
            if provider is not None
            else "ready"
        )

    async def ensure_self_ready(
        self,
        *,
        activation_generation: int | None = None,
    ) -> bool:
        if not self._self_activation_is_current(activation_generation):
            return False
        state = self.state_provider()
        if not state.settings_available or state.self_provider not in LOCAL_CPU_PROVIDERS:
            return True
        decision = resolve_local_asr_selection(
            state.self_provider,
            state.self_source_language,
        )
        if not decision.supported:
            self._emit(LocalASRReadinessEffectType.DISABLE_SELF_UNSUPPORTED)
            return False
        current_status = self.current_self_status()
        if current_status == "downloading":
            self.cpu_repair_owner.retain_pending(
                "self",
                activation_generation=(
                    activation_generation
                    if activation_generation is not None
                    else state.self_activation_generation
                ),
            )
            self._emit(LocalASRReadinessEffectType.SELF_DOWNLOAD_IN_PROGRESS)
            return False
        if current_status in {"missing", "invalid", "download_failed"}:
            return self._request_repair(
                current_status,
                channel="self",
                activation_generation=activation_generation,
            )
        if state.runtime_available and not self.self_provider_available():
            await self.rebuild_self_provider()
            if not self._self_activation_is_current(activation_generation):
                return False
        if not self.self_provider_available():
            self._emit(LocalASRReadinessEffectType.DISABLE_SELF_INVALID)
            return False
        channel_snapshot = self.self_channel_provider()
        if channel_snapshot is None:
            raise RuntimeError("local ASR provider runtime is unavailable")
        was_loaded = channel_snapshot.phase in {"ready", "running"}
        load_started_at = time.monotonic()
        try:
            await self.probe_self_provider()
            if not self._self_activation_is_current(activation_generation):
                return False
            loaded_model_id = channel_snapshot.model_id or decision.model_id
            if not was_loaded:
                self.load_log_sink(
                    channel="self",
                    model_id=str(loaded_model_id or "unknown"),
                    backend="CPU",
                    outcome="ready",
                    load_seconds=time.monotonic() - load_started_at,
                )
            current_state = self.state_provider()
            if current_state.self_provider is None:
                return False
            await self.provisioning_provider().inspect_cpu(
                required_local_asr_model_ids(current_state.self_provider)
            )
            self._emit(LocalASRReadinessEffectType.SYNC_NOTICE)
            return True
        except LocalCPUAutoUnavailableError:
            current_state = self.state_provider()
            if current_state.self_provider is None:
                return False
            required_model_ids = required_local_asr_model_ids(current_state.self_provider)
            snapshot = await self.provisioning_provider().inspect_cpu(required_model_ids)
            current_state = self.state_provider()
            if current_state.self_provider == LOCAL_CPU_AUTO_PROVIDER:
                if not self.persist_manual_fallback("self"):
                    return False
                await self.rebuild_self_provider()
                return await self.ensure_self_ready(activation_generation=activation_generation)
            return self._request_repair(
                self._snapshot_unavailable_status(snapshot, required_model_ids),
                channel="self",
                model_ids=snapshot.unavailable_model_ids(required_model_ids),
                activation_generation=activation_generation,
            )
        except LocalSTTModelMissingError as exc:
            self._log_load_failure(
                decision.model_id,
                load_started_at,
                exc,
            )
            return self._request_repair(
                "missing",
                channel="self",
                model_ids=((decision.model_id,) if decision.model_id is not None else None),
                activation_generation=activation_generation,
            )
        except (
            LocalSTTManifestInvalidError,
            LocalQwenSherpaLoadError,
            LocalParakeetSherpaLoadError,
        ) as exc:
            self._log_load_failure(
                decision.model_id,
                load_started_at,
                exc,
            )
            if decision.model_id is not None:
                await self.provisioning_provider().report_model_validation_failure(
                    decision.model_id,
                    failure_type=type(exc).__name__,
                )
            return self._request_repair(
                "invalid",
                channel="self",
                model_ids=((decision.model_id,) if decision.model_id is not None else None),
                activation_generation=activation_generation,
            )

    async def ensure_peer_ready(
        self,
        *,
        activation_generation: int | None = None,
        gpu_provider_id: str,
    ) -> bool:
        if not self._peer_activation_is_current(activation_generation):
            return False
        state = self.state_provider()
        if state.peer_provider == gpu_provider_id:
            ready = await self.validate_gpu_activation()
            if not ready and self.gpu_state_provider() in {
                "not_installed",
                "invalid",
                "install_failed",
                "installing",
            }:
                self.retain_gpu_pending("peer")
            return ready
        if not state.settings_available or not state.peer_requested:
            return True
        if state.peer_provider is None:
            return False
        decision = resolve_local_asr_selection(
            state.peer_provider,
            state.peer_source_language,
        )
        if not decision.supported:
            self._emit(LocalASRReadinessEffectType.DISABLE_PEER_UNSUPPORTED)
            return False
        current_status = local_asr_status_for_provider(
            self.provisioning_provider().snapshot,
            state.peer_provider,
        )
        if current_status == "downloading":
            self.cpu_repair_owner.retain_pending("peer")
            self._emit(LocalASRReadinessEffectType.SYNC_NOTICE)
            return False
        if current_status in {"missing", "invalid", "download_failed"}:
            return self._request_repair(
                current_status,
                channel="peer",
                activation_generation=activation_generation,
            )
        required_model_ids = required_local_asr_model_ids(state.peer_provider)
        strict_snapshot = await self.provisioning_provider().inspect_cpu(required_model_ids)
        if not self._peer_activation_is_current(activation_generation):
            return False
        unavailable_model_ids = strict_snapshot.unavailable_model_ids(required_model_ids)
        if unavailable_model_ids:
            return self._request_repair(
                self._snapshot_unavailable_status(strict_snapshot, required_model_ids),
                channel="peer",
                model_ids=unavailable_model_ids,
                activation_generation=activation_generation,
            )
        self._emit(LocalASRReadinessEffectType.SYNC_NOTICE)
        return True

    def _request_repair(
        self,
        status: str,
        *,
        channel: LocalASRReadinessChannel,
        model_ids: tuple[str, ...] | None = None,
        activation_generation: int | None = None,
    ) -> bool:
        return self.cpu_repair_owner.request_repair(
            LocalASRCpuRepairRequest(
                status=status,
                channel=channel,
                model_ids=model_ids,
                activation_generation=activation_generation,
            )
        )

    def _self_activation_is_current(self, generation: int | None) -> bool:
        state = self.state_provider()
        return generation is None or (
            generation == state.self_activation_generation and state.self_desired
        )

    def _peer_activation_is_current(self, generation: int | None) -> bool:
        state = self.state_provider()
        return generation is None or (
            generation == state.peer_activation_generation and state.peer_requested
        )

    @staticmethod
    def _snapshot_unavailable_status(
        snapshot: LocalASRProvisioningSnapshot,
        model_ids: tuple[str, ...],
    ) -> str:
        if any(snapshot.state_for(model_id).integrity == "invalid" for model_id in model_ids):
            return "invalid"
        return "missing"

    def _log_load_failure(
        self,
        model_id: str | None,
        load_started_at: float,
        exception: BaseException,
    ) -> None:
        self.load_log_sink(
            channel="self",
            model_id=str(model_id or "unknown"),
            backend="CPU",
            outcome="failed",
            load_seconds=time.monotonic() - load_started_at,
            failure_type=type(exception).__name__,
        )

    def _emit(self, effect_type: LocalASRReadinessEffectType) -> None:
        self.effect_sink(LocalASRReadinessEffect(type=effect_type))


__all__ = [
    "LocalASRReadinessAsyncEffect",
    "LocalASRReadinessChannel",
    "LocalASRReadinessChannelProvider",
    "LocalASRReadinessEffect",
    "LocalASRReadinessEffectSink",
    "LocalASRReadinessEffectType",
    "LocalASRReadinessFallback",
    "LocalASRReadinessGpuPendingSink",
    "LocalASRReadinessGpuStateProvider",
    "LocalASRReadinessGpuValidator",
    "LocalASRReadinessLoadLogSink",
    "LocalASRReadinessOwner",
    "LocalASRReadinessProviderAvailable",
    "LocalASRReadinessProvisioningProvider",
    "LocalASRReadinessState",
    "LocalASRReadinessStateProvider",
]
