from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from puripuly_heart.app.ports.self_capture_admission import (
    SelfCaptureAdmissionEffect,
    SelfCaptureAdmissionEffectType,
    SelfCaptureAdmissionState,
)
from puripuly_heart.app.services.local_asr_cpu_repair import (
    LocalASRCpuRepairEffect,
    LocalASRCpuRepairEffectType,
    LocalASRCpuRepairOwner,
    LocalASRCpuRepairRuntimeState,
)
from puripuly_heart.app.services.local_asr_readiness import (
    LocalASRReadinessEffect,
    LocalASRReadinessEffectType,
    LocalASRReadinessState,
)
from puripuly_heart.app.services.local_asr_selection import (
    LOCAL_CPU_PROVIDERS,
    local_asr_status_for_provider,
    required_local_asr_model_ids,
    resolve_local_asr_selection,
)
from puripuly_heart.app.services.peer_application import PeerApplicationOwner
from puripuly_heart.core.local_asr_provisioning import LocalASRProvisioningSnapshot
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import (
    SelfCaptureSessionConfig,
    SelfCaptureSessionSnapshot,
    SelfCaptureSessionState,
)


class LocalASRNoticeSink(Protocol):
    def __call__(
        self,
        *,
        status: str | None,
        model_id: str | None,
        percent: int | None,
        starting: bool,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class LocalASRApplicationSettings:
    locale: str
    self_provider: str
    peer_provider: str
    self_source_language: str
    peer_source_language: str
    self_gpu_provider: bool
    peer_gpu_provider: bool
    peer_requested: bool
    peer_activation_requested: bool


@dataclass(frozen=True, slots=True)
class LocalASRApplicationStateAdapter:
    settings_provider: Callable[[], LocalASRApplicationSettings | None]
    runtime_available: Callable[[], bool]
    self_capture_provider: Callable[[], SelfCaptureSessionOwner | None]
    peer: Callable[[], PeerApplicationOwner]
    gpu_state_provider: Callable[[], str]
    provisioning_snapshot: Callable[[], LocalASRProvisioningSnapshot]

    def cpu_repair(self) -> LocalASRCpuRepairRuntimeState:
        settings = self.settings_provider()
        snapshot = self._self_snapshot()
        self_provider = settings.self_provider if settings is not None else None
        peer_provider = settings.peer_provider if settings is not None else None
        return LocalASRCpuRepairRuntimeState(
            settings_available=settings is not None,
            locale=settings.locale if settings is not None else None,
            self_provider=self_provider,
            peer_provider=peer_provider,
            self_provider_local=bool(self_provider in LOCAL_CPU_PROVIDERS),
            peer_requested=bool(settings is not None and settings.peer_requested),
            self_activation_generation=snapshot.generation if snapshot is not None else 0,
            peer_activation_generation=self.peer().activation_generation,
            self_desired=bool(snapshot is not None and snapshot.desired_active),
        )

    def readiness(self) -> LocalASRReadinessState:
        settings = self.settings_provider()
        snapshot = self._self_snapshot()
        return LocalASRReadinessState(
            settings_available=settings is not None,
            runtime_available=self.runtime_available(),
            self_provider=settings.self_provider if settings is not None else None,
            peer_provider=settings.peer_provider if settings is not None else None,
            self_source_language=settings.self_source_language if settings is not None else "",
            peer_source_language=settings.peer_source_language if settings is not None else "",
            self_desired=bool(snapshot is not None and snapshot.desired_active),
            peer_requested=bool(settings is not None and settings.peer_requested),
            self_activation_generation=snapshot.generation if snapshot is not None else 0,
            peer_activation_generation=self.peer().activation_generation,
        )

    def self_admission(
        self,
        config: SelfCaptureSessionConfig,
    ) -> SelfCaptureAdmissionState:
        settings = self.settings_provider()
        snapshot = self._self_snapshot()
        decision = (
            resolve_local_asr_selection(
                settings.self_provider,
                settings.self_source_language,
            )
            if settings is not None and config.local_cpu
            else None
        )
        return SelfCaptureAdmissionState(
            settings_available=settings is not None,
            runtime_available=self.runtime_available(),
            gpu_status=self.gpu_state_provider(),
            local_cpu_supported=bool(decision is None or decision.supported),
            local_runtime_status=(
                local_asr_status_for_provider(
                    self.provisioning_snapshot(),
                    settings.self_provider,
                )
                if decision is not None and decision.supported
                else "ready"
            ),
            activation_generation=snapshot.generation if snapshot is not None else 0,
        )

    def _self_snapshot(self) -> SelfCaptureSessionSnapshot | None:
        owner = self.self_capture_provider()
        return owner.snapshot if owner is not None else None


@dataclass(frozen=True, slots=True)
class LocalASRApplicationEffectsAdapter:
    self_capture_provider: Callable[[], SelfCaptureSessionOwner | None]
    peer: Callable[[], PeerApplicationOwner]
    cpu_repair: Callable[[], LocalASRCpuRepairOwner]
    retain_gpu_pending: Callable[[str], None]
    repair_request: Callable[[str, str, int | None], bool]
    dashboard_enabled_sink: Callable[[bool], None]
    dashboard_needs_key_sink: Callable[[bool], None]
    message_sink: Callable[[str], None]
    sync_notice: Callable[[], None]

    def apply_cpu_repair(self, effect: LocalASRCpuRepairEffect) -> None:
        if effect.type is LocalASRCpuRepairEffectType.DISABLE_SELF_INTENT:
            self._invalidate_self()
            return
        if effect.type is LocalASRCpuRepairEffectType.DISABLE_SELF_DASHBOARD:
            self.dashboard_enabled_sink(False)
            self.dashboard_needs_key_sink(False)
            return
        if effect.type is LocalASRCpuRepairEffectType.SYNC_NOTICE:
            self.sync_notice()
            return
        if effect.type is LocalASRCpuRepairEffectType.SHOW_DOWNLOAD_FAILED:
            self.message_sink("local_stt.download_failed")
            return
        raise ValueError(f"Unsupported Local ASR CPU repair effect: {effect.type}")

    def apply_readiness(self, effect: LocalASRReadinessEffect) -> None:
        if effect.type is LocalASRReadinessEffectType.DISABLE_SELF_UNSUPPORTED:
            self._disable_self("local_stt.language_unsupported", needs_key=False)
            return
        if effect.type is LocalASRReadinessEffectType.DISABLE_SELF_INVALID:
            self._disable_self("error.local_stt_model_invalid", needs_key=False)
            return
        if effect.type is LocalASRReadinessEffectType.SELF_DOWNLOAD_IN_PROGRESS:
            self._disable_self("local_stt.download_in_progress")
            return
        if effect.type is LocalASRReadinessEffectType.DISABLE_PEER_UNSUPPORTED:
            peer = self.peer()
            peer.disable_intent()
            peer.sync_effective_flags()
            self.message_sink("local_stt.language_unsupported")
            return
        if effect.type is LocalASRReadinessEffectType.SYNC_NOTICE:
            self.sync_notice()
            return
        raise ValueError(f"Unsupported Local ASR readiness effect: {effect.type}")

    def apply_self_admission(self, effect: SelfCaptureAdmissionEffect) -> None:
        if effect.type is SelfCaptureAdmissionEffectType.RETAIN_GPU_PENDING_INTENT:
            self.retain_gpu_pending("self")
            return
        if effect.type is SelfCaptureAdmissionEffectType.REJECT_UNSUPPORTED_LANGUAGE:
            self.dashboard_enabled_sink(False)
            self.dashboard_needs_key_sink(False)
            self.message_sink("local_stt.language_unsupported")
            return
        if effect.type is SelfCaptureAdmissionEffectType.RETAIN_DOWNLOAD_PENDING_INTENT:
            self.cpu_repair().retain_pending(
                "self",
                activation_generation=effect.activation_generation,
            )
            self.dashboard_enabled_sink(False)
            self.message_sink("local_stt.download_in_progress")
            return
        if effect.type is SelfCaptureAdmissionEffectType.REQUEST_LOCAL_REPAIR:
            if effect.status is None:
                raise ValueError("Local ASR repair effect requires a status")
            self.repair_request(
                effect.status,
                "self",
                effect.activation_generation,
            )
            return
        raise ValueError(f"Unsupported Self capture admission effect: {effect.type}")

    def _invalidate_self(self) -> None:
        owner = self.self_capture_provider()
        if owner is not None:
            owner.invalidate_intent()

    def _disable_self(self, message: str, *, needs_key: bool | None = None) -> None:
        self._invalidate_self()
        self.dashboard_enabled_sink(False)
        if needs_key is not None:
            self.dashboard_needs_key_sink(needs_key)
        self.message_sink(message)


@dataclass(frozen=True, slots=True)
class LocalASRNoticeProjector:
    settings_provider: Callable[[], LocalASRApplicationSettings | None]
    self_capture_provider: Callable[[], SelfCaptureSessionOwner | None]
    peer: Callable[[], PeerApplicationOwner]
    provisioning_snapshot: Callable[[], LocalASRProvisioningSnapshot]
    sink: LocalASRNoticeSink

    def sync(self) -> None:
        settings = self.settings_provider()
        if settings is None:
            return
        owner = self.self_capture_provider()
        self_snapshot = owner.snapshot if owner is not None else None
        self_provider = settings.self_provider
        peer_provider = settings.peer_provider
        self_local = self_provider in LOCAL_CPU_PROVIDERS
        peer_local = settings.peer_requested
        self_local_asr = self_local or settings.self_gpu_provider
        peer_local_asr = peer_local or (
            settings.peer_gpu_provider and settings.peer_activation_requested
        )
        provisioning = self.provisioning_snapshot()
        status = local_asr_status_for_provider(
            provisioning,
            self_provider if self_local else peer_provider,
        )
        visible_model_ids = required_local_asr_model_ids(
            self_provider if self_local else peer_provider
        )
        activity = provisioning.activity_for("cpu")
        notice_model_id = (
            activity.model_id
            if activity is not None and activity.model_id in visible_model_ids
            else next(
                (
                    model_id
                    for model_id in visible_model_ids
                    if provisioning.state_for(model_id).status != "ready"
                ),
                None,
            )
        )
        self_starting = bool(
            self_snapshot is not None
            and self_snapshot.state
            in {
                SelfCaptureSessionState.STARTING,
                SelfCaptureSessionState.ADMISSION_PENDING,
            }
        )
        if self_starting and self_local_asr:
            status = "self_loading"
        elif (self.peer().activation_starting or self.peer().model_loading) and peer_local_asr:
            status = "peer_loading"
        elif (
            self_snapshot is not None
            and self_snapshot.state is SelfCaptureSessionState.FAULTED
            and self_local_asr
        ):
            status = "start_failed"
        should_show = status in {"self_loading", "peer_loading", "downloading"} or (
            (self_local_asr or peer_local_asr) and status != "ready"
        )
        with contextlib.suppress(Exception):
            self.sink(
                status=status if should_show else None,
                model_id=notice_model_id if should_show else None,
                percent=(
                    activity.progress_percent
                    if status == "downloading" and activity is not None
                    else None
                ),
                starting=self_starting,
            )


@dataclass(frozen=True, slots=True)
class LocalASRApplicationAdapters:
    state: LocalASRApplicationStateAdapter
    effects: LocalASRApplicationEffectsAdapter
    notice: LocalASRNoticeProjector


__all__ = [
    "LocalASRApplicationAdapters",
    "LocalASRApplicationEffectsAdapter",
    "LocalASRApplicationSettings",
    "LocalASRApplicationStateAdapter",
    "LocalASRNoticeProjector",
]
