from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimePort,
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeChannel,
    ProviderRuntimeGpuRecoveryRequest,
    ProviderRuntimeRecoveryChannel,
    ProviderRuntimeRecoveryQuiesce,
    ProviderRuntimeTerminalFailureSink,
)

GpuProviderRecoveryReason = Literal["manual_retry", "settings_restart"]
GpuProviderRecoveryStatus = Literal["skipped", "applied", "incomplete", "failed"]
GpuProviderRecoveryPrepare = Callable[[], ProviderRuntimeTerminalFailureSink]
GpuProviderRecoveryAbort = Callable[[ProviderRuntimeTerminalFailureSink], object]
GpuProviderRecoveryAdopt = Callable[[ProviderRuntimeTerminalFailureSink], Awaitable[None]]
GpuProviderRecoveryIncompleteHandler = Callable[[LocalASRProviderRuntimeSnapshot], None]
GpuProviderRecoveryAppliedHandler = Callable[
    [frozenset[ProviderRuntimeChannel]],
    Awaitable[None],
]
GpuProviderRecoveryFailureHandler = Callable[[], None]


@dataclass(frozen=True, slots=True)
class GpuProviderRecoveryChannelPlan:
    request: ProviderRuntimeBuildRequest
    start: bool
    prepare: GpuProviderRecoveryPrepare
    abort: GpuProviderRecoveryAbort
    adopt: GpuProviderRecoveryAdopt


@dataclass(frozen=True, slots=True)
class GpuProviderRecoveryExecution:
    runtime: LocalASRProviderRuntimePort
    device_id: str
    reason: GpuProviderRecoveryReason
    channels: tuple[GpuProviderRecoveryChannelPlan, ...]
    quiesce: ProviderRuntimeRecoveryQuiesce
    on_incomplete: GpuProviderRecoveryIncompleteHandler
    on_applied: GpuProviderRecoveryAppliedHandler
    on_failure: GpuProviderRecoveryFailureHandler
    skip_if_no_channels: bool = False

    def __post_init__(self) -> None:
        channel_ids = tuple(item.request.channel for item in self.channels)
        if len(frozenset(channel_ids)) != len(channel_ids):
            raise ValueError("GPU provider recovery channel plans must be unique")


GpuProviderRecoveryExecutionFactory = Callable[[], GpuProviderRecoveryExecution]


@dataclass(frozen=True, slots=True)
class GpuProviderRecoveryResult:
    status: GpuProviderRecoveryStatus
    reason: GpuProviderRecoveryReason
    channels: frozenset[ProviderRuntimeChannel]
    snapshot: LocalASRProviderRuntimeSnapshot | None


@dataclass(frozen=True, slots=True)
class GpuProviderRecoveryDiagnostic:
    outcome: Literal["applied", "cancelled", "failed", "incomplete", "prepare_failed", "skipped"]
    reason: GpuProviderRecoveryReason
    channels: tuple[ProviderRuntimeChannel, ...]
    failure_type: str | None = None


GpuProviderRecoveryDiagnosticSink = Callable[[GpuProviderRecoveryDiagnostic], None]


@dataclass(frozen=True, slots=True)
class _PreparedRecoveryChannel:
    plan: GpuProviderRecoveryChannelPlan
    failure_handler: ProviderRuntimeTerminalFailureSink


@dataclass(slots=True)
class GpuProviderRecoveryOwner:
    diagnostic_sink: GpuProviderRecoveryDiagnosticSink | None = field(
        default=None,
        repr=False,
    )
    _lock: asyncio.Lock | None = field(init=False, default=None, repr=False)

    @property
    def owner_name(self) -> str:
        return "GpuProviderRecoveryOwner"

    async def recover(
        self,
        execution_factory: GpuProviderRecoveryExecutionFactory,
    ) -> GpuProviderRecoveryResult:
        async with self._serialization_lock():
            execution = execution_factory()
            channels = frozenset(item.request.channel for item in execution.channels)
            if execution.skip_if_no_channels and not channels:
                self._emit(
                    GpuProviderRecoveryDiagnostic(
                        outcome="skipped",
                        reason=execution.reason,
                        channels=(),
                    )
                )
                return GpuProviderRecoveryResult(
                    status="skipped",
                    reason=execution.reason,
                    channels=channels,
                    snapshot=None,
                )
            prepared = self._prepare_channels(execution)
            try:
                request = ProviderRuntimeGpuRecoveryRequest(
                    device_id=execution.device_id,
                    channels=tuple(
                        ProviderRuntimeRecoveryChannel(
                            request=item.plan.request,
                            start=item.plan.start,
                            on_terminal_failure=item.failure_handler,
                        )
                        for item in prepared
                    ),
                    reason=execution.reason,
                )
            except BaseException as exc:
                self._abort_channels(prepared)
                self._emit(
                    GpuProviderRecoveryDiagnostic(
                        outcome="prepare_failed",
                        reason=execution.reason,
                        channels=tuple(sorted(channels)),
                        failure_type=type(exc).__name__,
                    )
                )
                raise
            try:
                snapshot = await execution.runtime.recover_gpu(
                    request,
                    quiesce=execution.quiesce,
                )
                if channels and (
                    snapshot.gpu.retry_required
                    or not channels.issubset(snapshot.gpu.active_channels)
                ):
                    execution.on_incomplete(snapshot)
                    self._emit(
                        GpuProviderRecoveryDiagnostic(
                            outcome="incomplete",
                            reason=execution.reason,
                            channels=tuple(sorted(channels)),
                        )
                    )
                    return GpuProviderRecoveryResult(
                        status="incomplete",
                        reason=execution.reason,
                        channels=channels,
                        snapshot=snapshot,
                    )
                for item in sorted(
                    prepared,
                    key=lambda candidate: (0 if candidate.plan.request.channel == "peer" else 1),
                ):
                    await item.plan.adopt(item.failure_handler)
                await execution.on_applied(channels)
                self._emit(
                    GpuProviderRecoveryDiagnostic(
                        outcome="applied",
                        reason=execution.reason,
                        channels=tuple(sorted(channels)),
                    )
                )
                return GpuProviderRecoveryResult(
                    status="applied",
                    reason=execution.reason,
                    channels=channels,
                    snapshot=snapshot,
                )
            except asyncio.CancelledError:
                self._emit(
                    GpuProviderRecoveryDiagnostic(
                        outcome="cancelled",
                        reason=execution.reason,
                        channels=tuple(sorted(channels)),
                    )
                )
                raise
            except Exception as exc:
                execution.on_failure()
                self._emit(
                    GpuProviderRecoveryDiagnostic(
                        outcome="failed",
                        reason=execution.reason,
                        channels=tuple(sorted(channels)),
                        failure_type=type(exc).__name__,
                    )
                )
                return GpuProviderRecoveryResult(
                    status="failed",
                    reason=execution.reason,
                    channels=channels,
                    snapshot=None,
                )
            finally:
                self._abort_channels(prepared)

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": ("_lock",),
            "operation_policy": (
                "serialize preparation, runtime recovery, consumer adoption and cleanup"
            ),
            "cancellation_policy": "propagate cancellation after prepared callback cleanup",
            "shutdown_policy": "no background task or external resource is retained",
        }

    def _prepare_channels(
        self,
        execution: GpuProviderRecoveryExecution,
    ) -> tuple[_PreparedRecoveryChannel, ...]:
        prepared: list[_PreparedRecoveryChannel] = []
        try:
            for plan in execution.channels:
                prepared.append(
                    _PreparedRecoveryChannel(
                        plan=plan,
                        failure_handler=plan.prepare(),
                    )
                )
        except BaseException as exc:
            self._abort_channels(tuple(prepared))
            self._emit(
                GpuProviderRecoveryDiagnostic(
                    outcome="prepare_failed",
                    reason=execution.reason,
                    channels=tuple(item.plan.request.channel for item in prepared),
                    failure_type=type(exc).__name__,
                )
            )
            raise
        return tuple(prepared)

    @staticmethod
    def _abort_channels(channels: tuple[_PreparedRecoveryChannel, ...]) -> None:
        for item in channels:
            with contextlib.suppress(Exception):
                item.plan.abort(item.failure_handler)

    def _emit(self, diagnostic: GpuProviderRecoveryDiagnostic) -> None:
        if self.diagnostic_sink is None:
            return
        with contextlib.suppress(Exception):
            self.diagnostic_sink(diagnostic)

    def _serialization_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock


__all__ = [
    "GpuProviderRecoveryAbort",
    "GpuProviderRecoveryAdopt",
    "GpuProviderRecoveryAppliedHandler",
    "GpuProviderRecoveryChannelPlan",
    "GpuProviderRecoveryDiagnostic",
    "GpuProviderRecoveryDiagnosticSink",
    "GpuProviderRecoveryExecution",
    "GpuProviderRecoveryExecutionFactory",
    "GpuProviderRecoveryFailureHandler",
    "GpuProviderRecoveryIncompleteHandler",
    "GpuProviderRecoveryOwner",
    "GpuProviderRecoveryPrepare",
    "GpuProviderRecoveryReason",
    "GpuProviderRecoveryResult",
    "GpuProviderRecoveryStatus",
]
