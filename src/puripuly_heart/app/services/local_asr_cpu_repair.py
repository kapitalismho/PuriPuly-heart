from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from puripuly_heart.core.local_asr_provisioning import (
    LocalASRInstallRequest,
    LocalASRInstallResult,
    LocalASRProvisioningPort,
)
from puripuly_heart.core.local_stt_assets import LOCAL_STT_MODEL_ID

LocalASRCpuRepairChannel = Literal["self", "peer"]


class LocalASRCpuRepairEffectType(str, Enum):
    DISABLE_SELF_INTENT = "disable_self_intent"
    DISABLE_SELF_DASHBOARD = "disable_self_dashboard"
    SYNC_NOTICE = "sync_notice"
    SHOW_DOWNLOAD_FAILED = "show_download_failed"


@dataclass(frozen=True, slots=True)
class LocalASRCpuRepairEffect:
    type: LocalASRCpuRepairEffectType


@dataclass(frozen=True, slots=True)
class LocalASRCpuRepairRuntimeState:
    settings_available: bool
    locale: str | None
    self_provider: str | None
    peer_provider: str | None
    self_provider_local: bool
    peer_requested: bool
    self_activation_generation: int
    peer_activation_generation: int
    self_desired: bool


@dataclass(frozen=True, slots=True)
class LocalASRCpuRepairRequest:
    status: str
    channel: LocalASRCpuRepairChannel
    model_ids: tuple[str, ...] | None = None
    activation_generation: int | None = None


@dataclass(frozen=True, slots=True)
class LocalASRCpuRepairSnapshot:
    self_pending: bool
    self_activation_generation: int | None
    peer_pending: bool


LocalASRCpuRepairRuntimeStateProvider = Callable[[], LocalASRCpuRepairRuntimeState]
LocalASRCpuRepairEffectSink = Callable[[LocalASRCpuRepairEffect], None]
LocalASRCpuModelIdsForProvider = Callable[[str], tuple[str, ...]]
LocalASRCpuStatusForProvider = Callable[[str], str]
LocalASRCpuSelfProviderRebuild = Callable[[], Awaitable[None]]
LocalASRCpuSelfResume = Callable[[], Awaitable[bool]]
LocalASRCpuPeerResume = Callable[[], Awaitable[None]]
LocalASRCpuProvisioningProvider = Callable[[], LocalASRProvisioningPort]


@dataclass(slots=True)
class LocalASRCpuRepairOwner:
    provisioning_provider: LocalASRCpuProvisioningProvider = field(repr=False)
    state_provider: LocalASRCpuRepairRuntimeStateProvider = field(repr=False)
    model_ids_for_provider: LocalASRCpuModelIdsForProvider = field(repr=False)
    status_for_provider: LocalASRCpuStatusForProvider = field(repr=False)
    effect_sink: LocalASRCpuRepairEffectSink = field(repr=False)
    rebuild_self_provider: LocalASRCpuSelfProviderRebuild = field(repr=False)
    resume_self: LocalASRCpuSelfResume = field(repr=False)
    resume_peer: LocalASRCpuPeerResume = field(repr=False)
    _self_pending: bool = field(init=False, default=False, repr=False)
    _self_activation_generation: int | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _peer_pending: bool = field(init=False, default=False, repr=False)

    @property
    def owner_name(self) -> str:
        return "LocalASRCpuRepairOwner"

    @property
    def snapshot(self) -> LocalASRCpuRepairSnapshot:
        return LocalASRCpuRepairSnapshot(
            self_pending=self._self_pending,
            self_activation_generation=self._self_activation_generation,
            peer_pending=self._peer_pending,
        )

    def retain_pending(
        self,
        channel: LocalASRCpuRepairChannel,
        *,
        activation_generation: int | None = None,
    ) -> None:
        if channel == "self":
            self._self_pending = True
            self._self_activation_generation = activation_generation
            return
        self._peer_pending = True

    def set_self_pending(self, pending: bool) -> None:
        self._self_pending = pending

    def set_self_activation_generation(self, generation: int | None) -> None:
        self._self_activation_generation = generation

    def set_peer_pending(self, pending: bool) -> None:
        self._peer_pending = pending

    def reset_self(self) -> None:
        self._self_pending = False
        self._self_activation_generation = None

    def reset_peer(self) -> None:
        self._peer_pending = False

    def reset_all(self) -> None:
        self.reset_self()
        self.reset_peer()

    def clear_if_provider_switched_away(self) -> None:
        state = self.state_provider()
        if not state.settings_available:
            return
        if not state.self_provider_local:
            self.reset_self()
        if not state.peer_requested:
            self.reset_peer()

    def request_repair(self, request: LocalASRCpuRepairRequest) -> bool:
        state = self.state_provider()
        if not state.settings_available:
            return False
        if request.activation_generation is not None:
            if request.channel == "self" and (
                request.activation_generation != state.self_activation_generation
                or not state.self_desired
            ):
                return False
            if request.channel == "peer" and (
                request.activation_generation != state.peer_activation_generation
                or not state.peer_requested
            ):
                return False
        provider = state.self_provider if request.channel == "self" else state.peer_provider
        if provider is None:
            return False
        if request.channel == "self":
            self.retain_pending(
                "self",
                activation_generation=(
                    request.activation_generation
                    if request.activation_generation is not None
                    else state.self_activation_generation
                ),
            )
            self._emit(LocalASRCpuRepairEffectType.DISABLE_SELF_INTENT)
        else:
            self.retain_pending("peer")
        affected_model_ids = request.model_ids
        if not affected_model_ids:
            required_model_ids = self.model_ids_for_provider(provider)
            affected_model_ids = self.provisioning_provider().snapshot.unavailable_model_ids(
                required_model_ids
            )
            if not affected_model_ids:
                affected_model_ids = required_model_ids
        if request.channel == "self":
            self._emit(LocalASRCpuRepairEffectType.DISABLE_SELF_DASHBOARD)
        self._emit(LocalASRCpuRepairEffectType.SYNC_NOTICE)
        self.request_install(
            origin="manual",
            model_ids=affected_model_ids,
        )
        return False

    def request_install(
        self,
        *,
        origin: str,
        model_ids: tuple[str, ...] | None = None,
    ) -> bool:
        state = self.state_provider()
        if not state.settings_available:
            return False
        provisioning = self.provisioning_provider()
        if provisioning.snapshot.activity_for("cpu") is not None:
            return False
        requested_model_ids = model_ids or (LOCAL_STT_MODEL_ID,)
        try:
            provisioning.start_install(
                LocalASRInstallRequest(
                    backend="cpu",
                    model_ids=requested_model_ids,
                    locale=self.state_provider().locale,
                    origin=origin,
                ),
                result_handler=lambda result: self.handle_install_result(
                    result,
                    origin=origin,
                ),
            )
        except RuntimeError:
            return False
        return True

    async def handle_install_result(
        self,
        result: LocalASRInstallResult,
        *,
        origin: str,
    ) -> None:
        if result.cancelled:
            return
        if result.failed_model_ids:
            if origin == "manual":
                self._emit(LocalASRCpuRepairEffectType.SHOW_DOWNLOAD_FAILED)
            return

        self.clear_if_provider_switched_away()
        state = self.state_provider()
        should_resume_self = (
            origin == "manual"
            and state.settings_available
            and state.self_provider_local
            and state.self_provider is not None
            and self.status_for_provider(state.self_provider) == "ready"
            and self._self_pending
            and (
                self._self_activation_generation is None
                or self._self_activation_generation == state.self_activation_generation
            )
        )
        should_resume_peer = (
            origin == "manual"
            and state.settings_available
            and state.peer_requested
            and state.peer_provider is not None
            and self.status_for_provider(state.peer_provider) == "ready"
            and self._peer_pending
        )

        if should_resume_self:
            resume_generation = (
                self._self_activation_generation
                if self._self_activation_generation is not None
                else state.self_activation_generation
            )
            await self.rebuild_self_provider()
            self.clear_if_provider_switched_away()
            if not self._self_pending or self._self_activation_generation != resume_generation:
                return
            self.reset_self()
            if not await self.resume_self():
                return

        if should_resume_peer:
            self.reset_peer()
            await self.resume_peer()

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "state_fields": (
                "_self_pending",
                "_self_activation_generation",
                "_peer_pending",
            ),
            "operation_policy": (
                "retain repair intent, request one CPU install, then revalidate and "
                "resume Self before Peer"
            ),
            "shutdown_policy": "no task or external resource is retained",
        }

    def _emit(self, effect_type: LocalASRCpuRepairEffectType) -> None:
        self.effect_sink(LocalASRCpuRepairEffect(type=effect_type))


__all__ = [
    "LocalASRCpuModelIdsForProvider",
    "LocalASRCpuPeerResume",
    "LocalASRCpuProvisioningProvider",
    "LocalASRCpuRepairChannel",
    "LocalASRCpuRepairEffect",
    "LocalASRCpuRepairEffectSink",
    "LocalASRCpuRepairEffectType",
    "LocalASRCpuRepairOwner",
    "LocalASRCpuRepairRequest",
    "LocalASRCpuRepairRuntimeState",
    "LocalASRCpuRepairRuntimeStateProvider",
    "LocalASRCpuRepairSnapshot",
    "LocalASRCpuSelfProviderRebuild",
    "LocalASRCpuSelfResume",
    "LocalASRCpuStatusForProvider",
]
