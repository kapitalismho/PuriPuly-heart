from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from puripuly_heart.app.ports.capture_vad_runtime import (
    PeerCaptureVadEventRuntime,
    SelfCaptureVadEventRuntime,
)
from puripuly_heart.app.ports.provider_channel_runtime import ProviderChannelResetPort
from puripuly_heart.app.ports.runtime_pipeline_lifecycle import (
    RuntimePipelineStartCallbacks,
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


class LocalASRProductionEvidencePort(Protocol):
    @property
    def config_path(self) -> Path: ...

    def load_compatibility_settings(self) -> AppSettingsVNext: ...

    async def initialize(self, settings: AppSettingsVNext) -> None: ...

    @property
    def owner(self) -> LocalASRProviderRuntimeOwner: ...

    @property
    def llm_runtime(self) -> ProviderRuntimeHandle: ...

    @property
    def translation_runtime_configuration(self) -> TranslationRuntimeConfigurationOwner: ...

    @property
    def self_vad(self) -> SelfCaptureVadEventRuntime: ...

    @property
    def peer_vad(self) -> PeerCaptureVadEventRuntime: ...

    @property
    def channel_reset(self) -> ProviderChannelResetPort: ...

    async def start_runtime(self) -> None: ...

    def composition_facts(self) -> Mapping[str, object]: ...

    def build_self_provider_request(
        self,
        settings: AppSettingsVNext,
        *,
        warmup: bool,
    ) -> ProviderRuntimeBuildRequest: ...

    def build_peer_provider_request(
        self,
        settings: AppSettingsVNext,
        *,
        warmup: bool,
    ) -> ProviderRuntimeBuildRequest: ...

    async def retry_gpu_activation(self) -> None: ...

    async def close(self) -> None: ...


class LocalASRProductionCompositionAccessPort(Protocol):
    @property
    def config_path(self) -> Path: ...

    def load_compatibility_settings(self) -> AppSettingsVNext: ...

    async def initialize(self, settings: AppSettingsVNext) -> None: ...

    @property
    def owner(self) -> LocalASRProviderRuntimeOwner: ...

    @property
    def llm_runtime(self) -> ProviderRuntimeHandle: ...

    @property
    def translation_runtime_configuration(self) -> TranslationRuntimeConfigurationOwner: ...

    @property
    def self_vad(self) -> SelfCaptureVadEventRuntime: ...

    @property
    def peer_vad(self) -> PeerCaptureVadEventRuntime: ...

    @property
    def channel_reset(self) -> ProviderChannelResetPort: ...

    @property
    def start_callbacks(self) -> RuntimePipelineStartCallbacks: ...

    async def retry_gpu_activation(self) -> None: ...


class LocalASRProductionEvidenceFactoryPort(Protocol):
    def __call__(
        self,
        *,
        config_path: Path,
    ) -> LocalASRProductionEvidencePort: ...
