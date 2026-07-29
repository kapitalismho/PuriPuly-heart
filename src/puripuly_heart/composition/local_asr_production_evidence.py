from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from puripuly_heart.app.ports.local_asr_production_evidence import (
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
from puripuly_heart.composition.ui_application import (
    compose_gui_application_boundary,
    compose_gui_controller,
)
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)
from puripuly_heart.ui.controller import GuiController
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


@dataclass(slots=True)
class _ControllerBackedLocalASRProductionEvidence:
    backend: GuiController
    application: UiApplicationPort

    @property
    def config_path(self) -> Path:
        return self.backend.config_path

    def load_compatibility_settings(self) -> object:
        return self.backend._load_or_init_settings(self.backend.config_path)

    async def initialize(self, settings: object) -> None:
        self.backend.settings = settings
        self.backend._get_local_asr_provisioning_owner()
        self.backend._sync_signature_caches(settings)
        await self.backend.runtime_composition.pipeline_launcher.launch(
            settings,
            vrc_mic_state=self.backend.vrc_mic_state,
            vrc_mic_audio_gate=self.backend.vrc_mic_audio_gate,
            receiver_active=self.backend.receiver is not None,
        )
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
        await self.application.stop()


def compose_local_asr_production_evidence(
    *,
    config_path: Path,
) -> LocalASRProductionEvidencePort:
    presentation: UiPresentationPort = FletUiPresentationAdapter(
        SimpleNamespace(debug_ui_preview=False),
    )
    backend = compose_gui_controller(
        presentation=presentation,
        config_path=config_path,
    )
    return _ControllerBackedLocalASRProductionEvidence(
        backend,
        compose_gui_application_boundary(backend),
    )
