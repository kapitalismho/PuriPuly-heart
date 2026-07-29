from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from puripuly_heart.app.ports.local_asr_production_evidence import (
    LocalASRProductionEvidencePort,
)
from puripuly_heart.app.wiring_local_asr_provider_runtime import (
    LocalASRProviderRuntimeFactory,
)
from puripuly_heart.app.wiring_stt_factory import (
    build_peer_capture_session_config,
    build_peer_stt_provider_request,
    build_self_stt_provider_request,
)
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)
from puripuly_heart.ui.controller import GuiController


@dataclass(slots=True)
class _ControllerBackedLocalASRProductionEvidence:
    backend: GuiController

    @property
    def config_path(self) -> Path:
        return self.backend.config_path

    def load_compatibility_settings(self) -> object:
        return self.backend._load_or_init_settings(self.backend.config_path)

    async def initialize(self, settings: object) -> None:
        self.backend.settings = settings
        await self.backend._init_pipeline()
        self._validated_hub_and_owner()

    def _validated_hub_and_owner(
        self,
    ) -> tuple[ClientHub, LocalASRProviderRuntimeOwner]:
        hub = self.backend.hub
        if hub is None or not isinstance(
            hub.local_asr_provider_runtime,
            LocalASRProviderRuntimeOwner,
        ):
            raise RuntimeError("production controller did not compose the canonical owner")
        return hub, hub.local_asr_provider_runtime

    @property
    def hub(self) -> ClientHub:
        return self._validated_hub_and_owner()[0]

    @property
    def owner(self) -> LocalASRProviderRuntimeOwner:
        return self._validated_hub_and_owner()[1]

    def composition_facts(self) -> dict[str, object]:
        return {
            "controller": type(self.backend).__name__,
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
        await self.backend.retry_gpu_activation()

    async def close(self) -> None:
        await self.backend.stop()


def compose_local_asr_production_evidence(
    *,
    config_path: Path,
) -> LocalASRProductionEvidencePort:
    backend = GuiController(
        page=None,
        app=SimpleNamespace(debug_ui_preview=False),
        config_path=config_path,
    )
    return _ControllerBackedLocalASRProductionEvidence(backend)
