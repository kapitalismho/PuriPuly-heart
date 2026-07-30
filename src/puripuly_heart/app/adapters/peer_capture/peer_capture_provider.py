from __future__ import annotations

from typing import Literal

from puripuly_heart.app.ports.provider_channel_runtime import ProviderChannelResetPort
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimePort,
    ProviderRuntimeBuildRequest,
)
from puripuly_heart.core.peer_capture import (
    PeerCaptureProviderMutation,
    PeerCaptureProviderMutationStatus,
    PeerCaptureSessionConfig,
    PeerCaptureTerminalFailureHandler,
)


class PeerCaptureProviderAdapter:
    def __init__(
        self,
        runtime: LocalASRProviderRuntimePort | None,
        channel_reset: ProviderChannelResetPort | None,
    ) -> None:
        self._runtime = runtime
        self._channel_reset = channel_reset
        self._config: PeerCaptureSessionConfig | None = None

    def is_ready(self, config: PeerCaptureSessionConfig) -> bool:
        self._config = config
        runtime = self._runtime
        if runtime is None:
            return False
        channel = runtime.snapshot.channel_for("peer")
        return channel.provider_id == config.provider_id and channel.has_resources

    async def replace(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: PeerCaptureTerminalFailureHandler,
    ) -> PeerCaptureProviderMutation:
        await self._require_channel_reset().reset_provider_channel("peer")
        result = await self._require_runtime().replace_provider(
            self._require_request(request),
            start=start,
            on_terminal_failure=on_terminal_failure,
        )
        return self._mutation(result.status, result.failure_type)

    async def handoff(
        self,
        request: object,
        *,
        start: bool,
        on_terminal_failure: PeerCaptureTerminalFailureHandler,
    ) -> PeerCaptureProviderMutation:
        result = await self._require_runtime().handoff_provider(
            self._require_request(request),
            start=start,
            on_terminal_failure=on_terminal_failure,
        )
        return self._mutation(result.status, result.failure_type)

    async def cancel_handoff(self) -> bool:
        return await self._require_runtime().cancel_handoff("peer")

    async def start_ingress(self) -> None:
        await self._require_runtime().start_channel("peer")
        config = self._config
        if config is None or not self.is_ready(config):
            raise RuntimeError("Peer provider ingress did not become ready")

    async def warmup(self) -> None:
        await self._require_runtime().warmup_channel("peer")

    async def reconfigure(self, session_options: object) -> None:
        await self._require_runtime().reconfigure_channel("peer", session_options)

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
            await self._require_channel_reset().reset_provider_channel("peer")
        elif mode != "drain":
            raise ValueError("unsupported Peer provider release mode")
        await runtime.release_channel(
            "peer",
            mode=mode,
            release_backend_after=release_backend_after,
        )

    def _require_runtime(self) -> LocalASRProviderRuntimePort:
        if self._runtime is None:
            raise RuntimeError("Peer capture provider adapter requires the production runtime")
        return self._runtime

    def _require_channel_reset(self) -> ProviderChannelResetPort:
        if self._channel_reset is None:
            raise RuntimeError("Peer capture provider adapter requires the channel reset port")
        return self._channel_reset

    @staticmethod
    def _require_request(request: object) -> ProviderRuntimeBuildRequest:
        if not isinstance(request, ProviderRuntimeBuildRequest):
            raise TypeError("Peer capture owner requires a provider build request")
        return request

    @staticmethod
    def _mutation(status: str, failure_type: str | None) -> PeerCaptureProviderMutation:
        try:
            mapped_status = PeerCaptureProviderMutationStatus(status)
        except ValueError:
            mapped_status = PeerCaptureProviderMutationStatus.FAILED
        return PeerCaptureProviderMutation(mapped_status, reason=failure_type)


__all__ = ["PeerCaptureProviderAdapter"]
