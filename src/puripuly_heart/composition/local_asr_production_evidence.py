from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from puripuly_heart.app.ports.local_asr_production_evidence import (
    LocalASRProductionCompositionAccessPort,
    LocalASRProductionEvidencePort,
)
from puripuly_heart.app.ports.ui_application import UiApplicationPort
from puripuly_heart.app.ports.ui_presentation import UiPresentationPort
from puripuly_heart.app.wiring_local_asr_provider_runtime import (
    LocalASRProviderRuntimeFactory,
)
from puripuly_heart.app.wiring_stt_factory import (
    build_peer_capture_session_config,
    build_peer_stt_provider_request,
    build_self_stt_provider_request,
)
from puripuly_heart.composition.application_runtime import (
    compose_application_runtime,
)
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


@dataclass(slots=True)
class _ApplicationLocalASRProductionEvidence:
    access: LocalASRProductionCompositionAccessPort
    application: UiApplicationPort

    @property
    def config_path(self) -> Path:
        return self.access.config_path

    def load_compatibility_settings(self) -> object:
        return self.access.load_compatibility_settings()

    async def initialize(self, settings: object) -> None:
        await self.access.initialize(settings)
        self._validated_hub_and_owner()

    def _validated_hub_and_owner(
        self,
    ) -> tuple[ClientHub, LocalASRProviderRuntimeOwner]:
        hub = self.access.hub
        if not isinstance(
            hub.local_asr_provider_runtime,
            LocalASRProviderRuntimeOwner,
        ):
            raise RuntimeError("production application did not compose the canonical owner")
        return hub, hub.local_asr_provider_runtime

    @property
    def hub(self) -> ClientHub:
        return self._validated_hub_and_owner()[0]

    @property
    def owner(self) -> LocalASRProviderRuntimeOwner:
        return self._validated_hub_and_owner()[1]

    def composition_facts(self) -> dict[str, object]:
        return {
            "application": type(self.application).__name__,
            "hub": type(self.hub).__name__,
            "factory": LocalASRProviderRuntimeFactory.__name__,
            "owner": type(self.owner).__name__,
        }

    def build_self_provider_request(
        self,
        settings: object,
        *,
        warmup: bool,
    ) -> ProviderRuntimeBuildRequest:
        return build_self_stt_provider_request(settings, warmup=warmup)

    def build_peer_provider_request(
        self,
        settings: object,
        *,
        warmup: bool,
    ) -> ProviderRuntimeBuildRequest:
        config = build_peer_capture_session_config(settings)
        return build_peer_stt_provider_request(
            config,
            gpu_device_id=settings.stt.gpu_device_id,
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
