from __future__ import annotations

from typing import Literal

from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.self_capture import (
    SelfCaptureProviderMutation,
    SelfCaptureProviderMutationStatus,
    SelfCaptureSessionConfig,
    SelfCaptureTerminalFailureHandler,
)


class SelfCaptureProviderAdapter:
    def __init__(self, hub: ClientHub | None) -> None:
        self._hub = hub
        self._config: SelfCaptureSessionConfig | None = None

    def is_ready(self, config: SelfCaptureSessionConfig) -> bool:
        self._config = config
        runtime = getattr(self._hub, "local_asr_provider_runtime", None)
        if runtime is None:
            return False
        channel = runtime.snapshot.channel_for("self")
        return channel.provider_id == config.provider_id and channel.has_resources

    async def replace(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: SelfCaptureTerminalFailureHandler,
    ) -> SelfCaptureProviderMutation:
        build_request = self._require_request(request)
        result = await self._require_hub().replace_stt_provider_request(
            build_request,
            start=start,
            on_terminal_failure=on_terminal_failure,
        )
        return self._mutation(result.status, result.failure_type)

    async def handoff(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: SelfCaptureTerminalFailureHandler,
    ) -> SelfCaptureProviderMutation:
        build_request = self._require_request(request)
        result = await self._require_hub().handoff_stt_provider_request(
            build_request,
            start=start,
            on_terminal_failure=on_terminal_failure,
        )
        return self._mutation(result.status, result.failure_type)

    async def cancel_handoff(self) -> bool:
        return await self._require_hub().cancel_stt_provider_request_handoff()

    async def start_ingress(self) -> None:
        hub = self._require_hub()
        await hub.resume_self_stt_after_toggle_on()
        config = self._config
        runtime = getattr(hub, "local_asr_provider_runtime", None)
        if config is None or runtime is None:
            raise RuntimeError("Self provider runtime is unavailable")
        channel = runtime.snapshot.channel_for("self")
        if channel.provider_id != config.provider_id or not channel.has_resources:
            raise RuntimeError("Self provider ingress did not become ready")
        if config.local_gpu:
            gpu = runtime.snapshot.gpu
            if gpu.phase != "ready" or "self" not in gpu.active_channels:
                raise RuntimeError("Self GPU provider ingress did not become ready")

    async def warmup(self) -> None:
        await self._require_hub().warmup_stt_channel("self")

    async def reconfigure(self, session_options: object) -> None:
        await self._require_hub().reconfigure_stt_channel("self", session_options)

    async def release(
        self,
        *,
        mode: Literal["drain", "abort"],
        release_backend_after: float | None = None,
    ) -> None:
        if self._hub is None:
            return
        if mode == "abort":
            await self._hub.abort_self_stt_for_toggle_off()
            return
        if mode != "drain":
            raise ValueError("unsupported Self provider release mode")
        await self._hub.drain_self_stt_for_toggle_off(release_backend_after=release_backend_after)

    def _require_hub(self) -> ClientHub:
        if self._hub is None:
            raise RuntimeError("Self capture provider adapter requires the production hub")
        return self._hub

    @staticmethod
    def _require_request(request: object) -> ProviderRuntimeBuildRequest:
        if not isinstance(request, ProviderRuntimeBuildRequest):
            raise TypeError("Self capture owner requires a provider build request")
        return request

    @staticmethod
    def _mutation(status: str, failure_type: str | None) -> SelfCaptureProviderMutation:
        try:
            mapped_status = SelfCaptureProviderMutationStatus(status)
        except ValueError:
            mapped_status = SelfCaptureProviderMutationStatus.FAILED
        return SelfCaptureProviderMutation(mapped_status, reason=failure_type)


__all__ = ["SelfCaptureProviderAdapter"]
