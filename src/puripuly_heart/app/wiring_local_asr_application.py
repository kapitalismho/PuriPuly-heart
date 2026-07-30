from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from puripuly_heart.app.adapters.local_asr_application import (
    LocalASRApplicationAdapters,
    LocalASRApplicationEffectsAdapter,
    LocalASRApplicationSettings,
    LocalASRApplicationStateAdapter,
    LocalASRNoticeProjector,
    LocalASRNoticeSink,
)
from puripuly_heart.app.services.local_asr_cpu_repair import (
    LocalASRCpuRepairOwner,
    LocalASRCpuRepairRequest,
)
from puripuly_heart.app.services.local_asr_readiness import LocalASRReadinessOwner
from puripuly_heart.app.services.local_asr_selection import (
    local_asr_status_for_provider,
    required_local_asr_model_ids,
)
from puripuly_heart.app.services.peer_application import PeerApplicationOwner
from puripuly_heart.app.wiring_composition import (
    create_local_asr_cpu_repair_owner,
    create_local_asr_readiness_owner,
)
from puripuly_heart.config.settings import AppSettings, STTProviderName
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimePort,
    ProviderRuntimeChannelSnapshot,
)
from puripuly_heart.core.local_asr_provisioning import LocalASRProvisioningPort
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner

LocalASRChannel = Literal["self", "peer"]


@dataclass(frozen=True, slots=True)
class LocalASRApplicationRuntime:
    adapters: LocalASRApplicationAdapters
    cpu_repair: LocalASRCpuRepairOwner
    readiness: LocalASRReadinessOwner
    provisioning_provider: Callable[[], LocalASRProvisioningPort]

    @property
    def self_pending(self) -> bool:
        return self.cpu_repair.snapshot.self_pending

    def request_repair(
        self,
        status: str,
        *,
        channel: LocalASRChannel,
        model_ids: tuple[str, ...] | None = None,
        activation_generation: int | None = None,
    ) -> bool:
        return self.cpu_repair.request_repair(
            LocalASRCpuRepairRequest(
                status=status,
                channel=channel,
                model_ids=model_ids,
                activation_generation=activation_generation,
            )
        )

    async def ensure_self_ready(
        self,
        *,
        activation_generation: int | None = None,
    ) -> bool:
        return await self.readiness.ensure_self_ready(
            activation_generation=activation_generation,
        )

    async def ensure_peer_ready(
        self,
        *,
        activation_generation: int | None = None,
    ) -> bool:
        return await self.readiness.ensure_peer_ready(
            activation_generation=activation_generation,
            gpu_provider_id=STTProviderName.LOCAL_QWEN_GPU.value,
        )

    async def close(self) -> None:
        self.cpu_repair.reset_all()
        await self.provisioning_provider().close()


def compose_local_asr_application(
    *,
    settings_provider: Callable[[], AppSettings | None],
    runtime_provider: Callable[[], LocalASRProviderRuntimePort | None],
    self_capture_provider: Callable[[], SelfCaptureSessionOwner | None],
    peer_provider: Callable[[], PeerApplicationOwner],
    peer_requested: Callable[[AppSettings | None], bool],
    peer_activation_requested: Callable[[AppSettings], bool],
    provisioning_provider: Callable[[], LocalASRProvisioningPort],
    gpu_state_provider: Callable[[], str],
    retain_gpu_pending: Callable[[LocalASRChannel], None],
    validate_gpu_activation: Callable[[], Awaitable[bool]],
    dashboard_enabled_sink: Callable[[bool], None],
    dashboard_needs_key_sink: Callable[[bool], None],
    message_sink: Callable[[str], None],
    notice_sink: LocalASRNoticeSink,
    rebuild_self_provider: Callable[[], Awaitable[None]],
    resume_self: Callable[[], Awaitable[bool]],
    resume_peer: Callable[[], Awaitable[None]],
    persist_manual_fallback: Callable[[str], bool],
    load_log_sink: Callable[..., None],
) -> LocalASRApplicationRuntime:
    def application_settings() -> LocalASRApplicationSettings | None:
        settings = settings_provider()
        if settings is None:
            return None
        return LocalASRApplicationSettings(
            locale=settings.ui.locale,
            self_provider=settings.provider.stt.value,
            peer_provider=settings.provider.peer_stt.value,
            self_source_language=settings.languages.source_language,
            peer_source_language=settings.languages.effective_peer_source,
            self_gpu_provider=settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU,
            peer_gpu_provider=settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU,
            peer_requested=peer_requested(settings),
            peer_activation_requested=peer_activation_requested(settings),
        )

    def self_provider_available() -> bool:
        runtime = runtime_provider()
        return runtime is not None and runtime.snapshot.channel_for("self").provider_id is not None

    def self_channel_provider() -> ProviderRuntimeChannelSnapshot | None:
        runtime = runtime_provider()
        return runtime.snapshot.channel_for("self") if runtime is not None else None

    async def probe_self_provider() -> None:
        runtime = runtime_provider()
        if runtime is None or runtime.snapshot.channel_for("self").provider_id is None:
            raise RuntimeError("self STT provider is unavailable")
        await runtime.warmup_channel("self")

    state = LocalASRApplicationStateAdapter(
        settings_provider=application_settings,
        runtime_available=lambda: runtime_provider() is not None,
        self_capture_provider=self_capture_provider,
        peer=peer_provider,
        gpu_state_provider=gpu_state_provider,
        provisioning_snapshot=lambda: provisioning_provider().snapshot,
    )
    notice = LocalASRNoticeProjector(
        settings_provider=application_settings,
        self_capture_provider=self_capture_provider,
        peer=peer_provider,
        provisioning_snapshot=lambda: provisioning_provider().snapshot,
        sink=notice_sink,
    )
    repair_owner: LocalASRCpuRepairOwner | None = None

    def repair() -> LocalASRCpuRepairOwner:
        if repair_owner is None:
            raise RuntimeError("Local ASR CPU repair owner is not composed")
        return repair_owner

    effects = LocalASRApplicationEffectsAdapter(
        self_capture_provider=self_capture_provider,
        peer=peer_provider,
        cpu_repair=repair,
        retain_gpu_pending=retain_gpu_pending,
        repair_request=lambda status, channel, generation: repair().request_repair(
            LocalASRCpuRepairRequest(
                status=status,
                channel=channel,
                activation_generation=generation,
            )
        ),
        dashboard_enabled_sink=dashboard_enabled_sink,
        dashboard_needs_key_sink=dashboard_needs_key_sink,
        message_sink=message_sink,
        sync_notice=notice.sync,
    )
    repair_owner = create_local_asr_cpu_repair_owner(
        provisioning_provider=provisioning_provider,
        state_provider=state.cpu_repair,
        model_ids_for_provider=required_local_asr_model_ids,
        status_for_provider=lambda provider: local_asr_status_for_provider(
            provisioning_provider().snapshot,
            provider,
        ),
        effect_sink=effects.apply_cpu_repair,
        rebuild_self_provider=rebuild_self_provider,
        resume_self=resume_self,
        resume_peer=resume_peer,
    )
    readiness = create_local_asr_readiness_owner(
        provisioning_provider=provisioning_provider,
        cpu_repair_owner=repair_owner,
        state_provider=state.readiness,
        effect_sink=effects.apply_readiness,
        self_provider_available=self_provider_available,
        self_channel_provider=self_channel_provider,
        rebuild_self_provider=rebuild_self_provider,
        probe_self_provider=probe_self_provider,
        persist_manual_fallback=persist_manual_fallback,
        validate_gpu_activation=validate_gpu_activation,
        gpu_state_provider=gpu_state_provider,
        retain_gpu_pending=retain_gpu_pending,
        load_log_sink=load_log_sink,
    )
    return LocalASRApplicationRuntime(
        adapters=LocalASRApplicationAdapters(
            state=state,
            effects=effects,
            notice=notice,
        ),
        cpu_repair=repair_owner,
        readiness=readiness,
        provisioning_provider=provisioning_provider,
    )


__all__ = [
    "LocalASRApplicationRuntime",
    "compose_local_asr_application",
]
