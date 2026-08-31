from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from puripuly_heart.app.ports.capture_vad_runtime import (
    PeerCaptureVadEventRuntime,
    SelfCaptureVadEventRuntime,
)
from puripuly_heart.app.ports.local_asr_production_evidence import (
    LocalASRProductionCompositionAccessPort,
    LocalASRProductionEvidencePort,
)
from puripuly_heart.app.ports.provider_channel_runtime import ProviderChannelResetPort
from puripuly_heart.app.ports.ui_application import UiApplicationPort
from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.wiring_local_asr_provider_runtime import (
    LocalASRProviderRuntimeFactory,
)
from puripuly_heart.app.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_peer_stt_provider_request,
    build_self_stt_provider_request_from_vnext,
)
from puripuly_heart.composition.application_runtime import (
    compose_application_runtime,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


@dataclass(slots=True)
class _ApplicationLocalASRProductionEvidence:
    access: LocalASRProductionCompositionAccessPort
    application: UiApplicationPort

    @property
    def config_path(self) -> Path:
        return self.access.config_path

    def load_compatibility_settings(self) -> AppSettingsVNext:
        return self.access.load_compatibility_settings()

    async def initialize(self, settings: AppSettingsVNext) -> None:
        await self.access.initialize(settings)
        _ = self.owner

    @property
    def owner(self) -> LocalASRProviderRuntimeOwner:
        owner = self.access.owner
        if not isinstance(owner, LocalASRProviderRuntimeOwner):
            raise RuntimeError("production application did not compose the canonical owner")
        return owner

    @property
    def llm_runtime(self) -> ProviderRuntimeHandle:
        return self.access.llm_runtime

    @property
    def translation_runtime_configuration(self) -> TranslationRuntimeConfigurationOwner:
        return self.access.translation_runtime_configuration

    @property
    def self_vad(self) -> SelfCaptureVadEventRuntime:
        return self.access.self_vad

    @property
    def peer_vad(self) -> PeerCaptureVadEventRuntime:
        return self.access.peer_vad

    @property
    def channel_reset(self) -> ProviderChannelResetPort:
        return self.access.channel_reset

    async def start_runtime(self) -> None:
        callbacks = self.access.start_callbacks
        await callbacks.start_output(False)
        await callbacks.open_self_ingress()
        await callbacks.open_peer_ingress()
        await callbacks.start_translation_turns()
        await callbacks.start_local_asr()

    def composition_facts(self) -> dict[str, object]:
        return {
            "application": type(self.application).__name__,
            "factory": LocalASRProviderRuntimeFactory.__name__,
            "owner": type(self.owner).__name__,
            "llm_owner": type(self.llm_runtime).__name__,
            "self_vad": type(self.self_vad).__name__,
            "peer_vad": type(self.peer_vad).__name__,
        }

    def build_self_provider_request(
        self,
        settings: AppSettingsVNext,
        *,
        warmup: bool,
    ) -> ProviderRuntimeBuildRequest:
        return build_self_stt_provider_request_from_vnext(settings, warmup=warmup)

    def build_peer_provider_request(
        self,
        settings: AppSettingsVNext,
        *,
        warmup: bool,
    ) -> ProviderRuntimeBuildRequest:
        config = build_peer_capture_session_config_from_vnext(settings)
        return build_peer_stt_provider_request(
            config,
            gpu_device_id=settings.intent.stt.gpu_device_id,
            warmup=warmup,
        )

    async def retry_gpu_activation(self) -> None:
        await self.access.retry_gpu_activation()

    async def close(self) -> None:
        await self.application.stop()


def compose_local_asr_production_evidence(
    *,
    config_path: Path,
) -> LocalASRProductionEvidencePort:
    presentation: UiPresentationPort = FletUiPresentationAdapter(
        SimpleNamespace(debug_ui_preview=False),
    )
    captured: list[LocalASRProductionCompositionAccessPort] = []
    application = compose_application_runtime(
        presentation=presentation,
        config_path=config_path,
        local_asr_evidence_sink=captured.append,
    )
    if len(captured) != 1:
        raise RuntimeError("production application did not expose Local ASR evidence")
    return _ApplicationLocalASRProductionEvidence(
        access=captured[0],
        application=application,
    )
