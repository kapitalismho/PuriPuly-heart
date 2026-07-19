from __future__ import annotations

from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeCallbacks,
    LocalASRProviderRuntimePort,
    ProviderRuntimeBuildRequest,
)
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)


class _PrebuiltProviderFactory:
    async def create(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        gpu_runtime,
        on_terminal_failure=None,
    ) -> object:
        _ = request, gpu_runtime, on_terminal_failure
        raise RuntimeError("prebuilt compatibility cannot construct providers")


class _UnavailableGpuRuntime:
    state = "idle"
    discovery_state = "idle"
    active_channels = frozenset()
    pending_count = 0
    worker_pid = None
    last_failure_code = None
    configured_device_id = None

    async def discover_devices(self):
        return ()

    async def activate_channel(self, channel, *, model_path, model_id, device_id):
        _ = channel, model_path, model_id, device_id
        raise RuntimeError("prebuilt compatibility has no GPU worker")

    async def retry(self):
        raise RuntimeError("prebuilt compatibility has no GPU worker")

    async def submit(
        self,
        channel,
        samples_f32,
        *,
        speech_end_at,
        language_hint=None,
    ):
        _ = channel, samples_f32, speech_end_at, language_hint
        raise RuntimeError("prebuilt compatibility has no GPU worker")

    async def deactivate_channel(self, channel) -> None:
        _ = channel

    async def close(self) -> None:
        return


class _UnavailableProvisioning:
    async def inspect_gpu(self, *, explicit_intent: bool, verify_checksums: bool = False):
        _ = explicit_intent, verify_checksums
        raise RuntimeError("prebuilt compatibility has no GPU provisioning")


class PrebuiltLocalASRProviderRuntimeFactory:
    def __init__(self, *, self_provider: object | None, peer_provider: object | None) -> None:
        self._self_provider = self_provider
        self._peer_provider = peer_provider

    def create(
        self,
        callbacks: LocalASRProviderRuntimeCallbacks,
    ) -> LocalASRProviderRuntimePort:
        return LocalASRProviderRuntimeOwner(
            provider_factory=_PrebuiltProviderFactory(),
            gpu_runtime_factory=lambda _diagnostic_sink: _UnavailableGpuRuntime(),
            provisioning=_UnavailableProvisioning(),
            self_event_handler=callbacks.self_event_handler,
            peer_event_handler=callbacks.peer_event_handler,
            retired_event_handler=callbacks.retired_event_handler,
            self_exception_handler=callbacks.self_exception_handler,
            peer_exception_handler=callbacks.peer_exception_handler,
            prebuilt_providers={
                "self": self._self_provider,
                "peer": self._peer_provider,
            },
        )


__all__ = ["PrebuiltLocalASRProviderRuntimeFactory"]
