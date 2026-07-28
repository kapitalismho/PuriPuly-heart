from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)


class LocalASRProductionEvidencePort(Protocol):
    @property
    def config_path(self) -> Path: ...

    def load_compatibility_settings(self) -> object: ...

    async def initialize(self, settings: object) -> None: ...

    @property
    def hub(self) -> ClientHub: ...

    @property
    def owner(self) -> LocalASRProviderRuntimeOwner: ...

    def composition_facts(self) -> Mapping[str, object]: ...

    def build_self_provider_request(
        self,
        settings: object,
        *,
        warmup: bool,
    ) -> ProviderRuntimeBuildRequest: ...

    def build_peer_provider_request(
        self,
        settings: object,
        *,
        warmup: bool,
    ) -> ProviderRuntimeBuildRequest: ...

    async def retry_gpu_activation(self) -> None: ...

    async def close(self) -> None: ...


class LocalASRProductionEvidenceFactoryPort(Protocol):
    def __call__(
        self,
        *,
        config_path: Path,
    ) -> LocalASRProductionEvidencePort: ...
