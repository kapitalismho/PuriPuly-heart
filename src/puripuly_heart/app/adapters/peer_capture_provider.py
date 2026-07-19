from __future__ import annotations

from typing import Literal

from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.peer_capture import (
    PeerCaptureProviderMutation,
    PeerCaptureProviderMutationStatus,
    PeerCaptureSessionConfig,
    PeerCaptureTerminalFailureHandler,
)


class PeerCaptureProviderAdapter:
    def __init__(self, hub: ClientHub | None) -> None:
        self._hub = hub
        self._config: PeerCaptureSessionConfig | None = None

    def is_ready(self, config: PeerCaptureSessionConfig) -> bool:
        self._config = config
        runtime = getattr(self._hub, "local_asr_provider_runtime", None)
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
        result = await self._require_hub().replace_peer_stt_provider_request(
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
        result = await self._require_hub().handoff_peer_stt_provider_request(
            self._require_request(request),
            start=start,
            on_terminal_failure=on_terminal_failure,
        )
        return self._mutation(result.status, result.failure_type)

    async def cancel_handoff(self) -> bool:
        return await self._require_hub().cancel_peer_stt_provider_request_handoff()

    async def start_ingress(self) -> None:
        await self._require_hub().start_peer_stt_provider_ingress()
        config = self._config
        if config is None or not self.is_ready(config):
            raise RuntimeError("Peer provider ingress did not become ready")

    async def warmup(self) -> None:
        await self._require_hub().warmup_stt_channel("peer")

    async def reconfigure(self, session_options: object) -> None:
        await self._require_hub().reconfigure_stt_channel("peer", session_options)

    async def release(
        self,
        *,
        mode: Literal["drain", "abort"],
        release_backend_after: float | None = None,
    ) -> None:
        hub = self._hub
        if hub is None:
            return
        runtime = hub.local_asr_provider_runtime
        if runtime is None:
            raise RuntimeError("Peer provider runtime is unavailable")
        await runtime.release_channel(
            "peer",
            mode=mode,
            release_backend_after=release_backend_after,
        )

    def _require_hub(self) -> ClientHub:
        if self._hub is None:
            raise RuntimeError("Peer capture provider adapter requires the production hub")
        return self._hub

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
