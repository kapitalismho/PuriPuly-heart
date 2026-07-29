from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from puripuly_heart.app.services.gpu_provider_recovery import (
    GpuProviderRecoveryChannelPlan,
    GpuProviderRecoveryExecution,
    GpuProviderRecoveryOwner,
    GpuProviderRecoveryReason,
    GpuProviderRecoveryResult,
)
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimePort,
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeChannel,
    ProviderRuntimeRecoveryQuiesce,
    ProviderRuntimeTerminalFailureSink,
)
from puripuly_heart.core.peer_capture import PeerCaptureSessionConfig
from puripuly_heart.core.runtime.peer_channel import PeerCaptureSessionOwner
from puripuly_heart.core.runtime.self_capture import SelfCaptureSessionOwner
from puripuly_heart.core.self_capture import (
    SelfCaptureSessionConfig,
    SelfCaptureSessionSnapshot,
)

GpuProviderRecoveryRuntimeProvider = Callable[[], LocalASRProviderRuntimePort | None]
GpuProviderRecoveryPendingProvider = Callable[[], frozenset[ProviderRuntimeChannel]]
GpuProviderRecoveryPendingClear = Callable[[frozenset[ProviderRuntimeChannel]], None]
GpuProviderRecoveryFailureSink = Callable[[GpuProviderRecoveryReason], None]
GpuProviderRecoveryStateSink = Callable[[LocalASRProviderRuntimeSnapshot], None]
GpuProviderRecoverySelfOwnerFactory = Callable[[], SelfCaptureSessionOwner]
GpuProviderRecoveryPeerOwnerProvider = Callable[[], PeerCaptureSessionOwner | None]
GpuProviderRecoverySelfStateSink = Callable[[SelfCaptureSessionSnapshot], None]
GpuProviderRecoveryAsyncEffect = Callable[[], Awaitable[None]]
GpuProviderRecoverySelfConfigFactory = Callable[[], SelfCaptureSessionConfig]
GpuProviderRecoveryPeerConfigFactory = Callable[[], PeerCaptureSessionConfig]
GpuProviderRecoveryProviderRequestFactory = Callable[[], ProviderRuntimeBuildRequest]
GpuProviderRecoveryPeerRequestFactory = Callable[
    [PeerCaptureSessionConfig],
    ProviderRuntimeBuildRequest,
]


@dataclass(frozen=True, slots=True)
class GpuProviderRecoveryApplicationRequest:
    device_id: str
    reason: GpuProviderRecoveryReason
    self_gpu_selected: bool
    peer_gpu_selected: bool
    self_desired: bool
    peer_enabled: bool
    self_config_factory: GpuProviderRecoverySelfConfigFactory = field(repr=False)
    peer_config_factory: GpuProviderRecoveryPeerConfigFactory = field(repr=False)
    self_request_factory: GpuProviderRecoveryProviderRequestFactory = field(repr=False)
    peer_request_factory: GpuProviderRecoveryPeerRequestFactory = field(repr=False)
    should_refresh_self: bool = False
    should_refresh_peer: bool = False


GpuProviderRecoveryApplicationRequestFactory = Callable[
    [],
    GpuProviderRecoveryApplicationRequest,
]


@dataclass(slots=True)
class GpuProviderRecoveryApplicationOwner:
    recovery_owner: GpuProviderRecoveryOwner = field(repr=False)
    runtime_provider: GpuProviderRecoveryRuntimeProvider = field(repr=False)
    pending_provider: GpuProviderRecoveryPendingProvider = field(repr=False)
    pending_clear: GpuProviderRecoveryPendingClear = field(repr=False)
    failure_sink: GpuProviderRecoveryFailureSink = field(repr=False)
    runtime_state_sink: GpuProviderRecoveryStateSink = field(repr=False)
    quiesce: ProviderRuntimeRecoveryQuiesce = field(repr=False)
    self_owner_factory: GpuProviderRecoverySelfOwnerFactory = field(repr=False)
    peer_owner_provider: GpuProviderRecoveryPeerOwnerProvider = field(repr=False)
    self_state_sink: GpuProviderRecoverySelfStateSink = field(repr=False)
    ensure_self_switch: GpuProviderRecoveryAsyncEffect = field(repr=False)
    refresh_self: GpuProviderRecoveryAsyncEffect = field(repr=False)
    refresh_peer: GpuProviderRecoveryAsyncEffect = field(repr=False)

    @property
    def owner_name(self) -> str:
        return "GpuProviderRecoveryApplicationOwner"

    async def recover(
        self,
        request_factory: GpuProviderRecoveryApplicationRequestFactory,
    ) -> GpuProviderRecoveryResult:
        return await self.recovery_owner.recover(
            lambda: self._execution(request_factory()),
        )

    def _execution(
        self,
        request: GpuProviderRecoveryApplicationRequest,
    ) -> GpuProviderRecoveryExecution:
        runtime = self.runtime_provider()
        if runtime is None:
            raise RuntimeError("local ASR provider runtime is unavailable")
        desired_channels = self._desired_channels(request, runtime.snapshot)
        if request.reason == "manual_retry":
            desired_channels = frozenset(
                {
                    *desired_channels,
                    *self.pending_provider(),
                }
            )
        return GpuProviderRecoveryExecution(
            runtime=runtime,
            device_id=request.device_id,
            reason=request.reason,
            channels=self._channel_plans(request, desired_channels),
            quiesce=self.quiesce,
            on_incomplete=self.runtime_state_sink,
            on_applied=lambda recovered_channels: self._complete(
                request,
                recovered_channels,
            ),
            on_failure=lambda: self.failure_sink(request.reason),
            skip_if_no_channels=request.reason == "manual_retry",
        )

    @staticmethod
    def _desired_channels(
        request: GpuProviderRecoveryApplicationRequest,
        snapshot: LocalASRProviderRuntimeSnapshot,
    ) -> frozenset[ProviderRuntimeChannel]:
        desired: set[ProviderRuntimeChannel] = set()
        if request.self_gpu_selected and (
            request.self_desired or "self" in snapshot.gpu.active_channels
        ):
            desired.add("self")
        if request.peer_gpu_selected and (
            request.peer_enabled or "peer" in snapshot.gpu.active_channels
        ):
            desired.add("peer")
        return frozenset(desired)

    def _channel_plans(
        self,
        request: GpuProviderRecoveryApplicationRequest,
        channels: frozenset[ProviderRuntimeChannel],
    ) -> tuple[GpuProviderRecoveryChannelPlan, ...]:
        plans: list[GpuProviderRecoveryChannelPlan] = []
        if "self" in channels and request.self_gpu_selected:
            self_config = request.self_config_factory()
            self_owner = self.self_owner_factory()

            async def adopt_self(
                failure_handler: ProviderRuntimeTerminalFailureSink,
            ) -> None:
                snapshot = await self_owner.adopt_recovered_provider(
                    self_config,
                    on_terminal_failure=failure_handler,
                )
                self.self_state_sink(snapshot)
                if request.self_desired:
                    await self.ensure_self_switch()

            plans.append(
                GpuProviderRecoveryChannelPlan(
                    request=request.self_request_factory(),
                    start=request.self_desired,
                    prepare=lambda: self_owner.prepare_provider_recovery(self_config),
                    abort=self_owner.abort_provider_recovery,
                    adopt=adopt_self,
                )
            )
        if "peer" in channels and request.peer_gpu_selected:
            peer_config = request.peer_config_factory()
            peer_owner = self.peer_owner_provider()
            if peer_owner is None:
                raise RuntimeError("Peer recovery requires a capture owner")

            async def adopt_peer(
                failure_handler: ProviderRuntimeTerminalFailureSink,
            ) -> None:
                await peer_owner.adopt_recovered_provider(
                    peer_config,
                    on_terminal_failure=failure_handler,
                )
                await self.refresh_peer()

            plans.append(
                GpuProviderRecoveryChannelPlan(
                    request=request.peer_request_factory(peer_config),
                    start=False,
                    prepare=lambda: peer_owner.prepare_provider_recovery(peer_config),
                    abort=peer_owner.abort_provider_recovery,
                    adopt=adopt_peer,
                )
            )
        return tuple(plans)

    async def _complete(
        self,
        request: GpuProviderRecoveryApplicationRequest,
        recovered_channels: frozenset[ProviderRuntimeChannel],
    ) -> None:
        if request.reason == "manual_retry":
            self.pending_clear(recovered_channels)
            return
        if request.should_refresh_self and "self" not in recovered_channels:
            await self.refresh_self()
        if request.should_refresh_peer and "peer" not in recovered_channels:
            await self.refresh_peer()


__all__ = [
    "GpuProviderRecoveryApplicationOwner",
    "GpuProviderRecoveryApplicationRequest",
    "GpuProviderRecoveryApplicationRequestFactory",
    "GpuProviderRecoveryAsyncEffect",
    "GpuProviderRecoveryFailureSink",
    "GpuProviderRecoveryPeerConfigFactory",
    "GpuProviderRecoveryPeerOwnerProvider",
    "GpuProviderRecoveryPeerRequestFactory",
    "GpuProviderRecoveryPendingClear",
    "GpuProviderRecoveryPendingProvider",
    "GpuProviderRecoveryProviderRequestFactory",
    "GpuProviderRecoveryRuntimeProvider",
    "GpuProviderRecoverySelfConfigFactory",
    "GpuProviderRecoverySelfOwnerFactory",
    "GpuProviderRecoverySelfStateSink",
    "GpuProviderRecoveryStateSink",
]
