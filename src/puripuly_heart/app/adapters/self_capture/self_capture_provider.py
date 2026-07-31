from __future__ import annotations

from typing import Literal

from puripuly_heart.app.ports.provider_channel_runtime import ProviderChannelResetPort
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimePort,
    ProviderRuntimeBuildRequest,
)
from puripuly_heart.core.self_capture import (
    SelfCaptureProviderMutation,
    SelfCaptureProviderMutationStatus,
    SelfCaptureSessionConfig,
    SelfCaptureTerminalFailureHandler,
)


class SelfCaptureProviderAdapter:
    def __init__(
        self,
        runtime: LocalASRProviderRuntimePort | None,
        channel_reset: ProviderChannelResetPort | None,
    ) -> None:
        self._runtime = runtime
        self._channel_reset = channel_reset
        self._config: SelfCaptureSessionConfig | None = None

    def is_ready(self, config: SelfCaptureSessionConfig) -> bool:
        self._config = config
        runtime = self._runtime
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
        await self._require_channel_reset().reset_provider_channel("self")
        result = await self._require_runtime().replace_provider(
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
        result = await self._require_runtime().handoff_provider(
            build_request,
            start=start,
            on_terminal_failure=on_terminal_failure,
        )
        return self._mutation(result.status, result.failure_type)

    async def cancel_handoff(self) -> bool:
        return await self._require_runtime().cancel_handoff("self")

    async def start_ingress(self) -> None:
        runtime = self._require_runtime()
        await runtime.start_channel("self")
        config = self._config
        if config is None:
            raise RuntimeError("Self provider runtime is unavailable")
        channel = runtime.snapshot.channel_for("self")
        if channel.provider_id != config.provider_id or not channel.has_resources:
            raise RuntimeError("Self provider ingress did not become ready")
        if config.local_gpu:
            gpu = runtime.snapshot.gpu
            if gpu.phase != "ready" or "self" not in gpu.active_channels:
                raise RuntimeError("Self GPU provider ingress did not become ready")

    async def warmup(self) -> None:
        await self._require_runtime().warmup_channel("self")

    async def reconfigure(self, session_options: object) -> None:
        await self._require_runtime().reconfigure_channel("self", session_options)

    async def release(
        self,
        *,
        mode: Literal["drain", "abort"],
        release_backend_after: float | None = None,
    ) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        if mode == "abort":
            await self._require_channel_reset().reset_provider_channel("self")
        elif mode != "drain":
            raise ValueError("unsupported Self provider release mode")
        await runtime.release_channel(
            "self",
            mode=mode,
            release_backend_after=release_backend_after,
        )

    def _require_runtime(self) -> LocalASRProviderRuntimePort:
        if self._runtime is None:
            raise RuntimeError("Self capture provider adapter requires the production runtime")
        return self._runtime

    def _require_channel_reset(self) -> ProviderChannelResetPort:
        if self._channel_reset is None:
            raise RuntimeError("Self capture provider adapter requires the channel reset port")
        return self._channel_reset

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
