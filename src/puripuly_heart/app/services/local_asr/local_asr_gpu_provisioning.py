from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from puripuly_heart.core.local_asr_provisioning import (
    LocalASRInstallRequest,
    LocalASRInstallResult,
    LocalASRProvisioningPort,
)
from puripuly_heart.core.local_stt_assets import LOCAL_QWEN_GPU_MODEL_ID
from puripuly_heart.core.runtime.gpu_asr import GpuASRChannel

LocalASRGpuProvisioningUiState = Literal[
    "installing",
    "install_failed",
    "installed",
]


@dataclass(frozen=True, slots=True)
class LocalASRGpuProvisioningState:
    selected_provider_requires_model: bool
    locale: str | None
    pending_channels: frozenset[GpuASRChannel]


@dataclass(frozen=True, slots=True)
class LocalASRGpuProvisioningEffect:
    state: LocalASRGpuProvisioningUiState
    origin: str
    progress_percent: int | None = None
    publish_notice: bool = False


@dataclass(frozen=True, slots=True)
class LocalASRGpuProvisioningDiagnostic:
    event: Literal["model_install"]
    outcome: Literal["failed"]
    origin: str
    failure_type: str
    exception: BaseException = field(repr=False, compare=False)


LocalASRGpuProvisioningStateProvider = Callable[[], LocalASRGpuProvisioningState]
LocalASRGpuProvisioningEffectSink = Callable[[LocalASRGpuProvisioningEffect], None]
LocalASRGpuProvisioningDiagnosticSink = Callable[
    [LocalASRGpuProvisioningDiagnostic],
    None,
]
LocalASRGpuActivationRetry = Callable[[], Awaitable[None]]
LocalASRProvisioningProvider = Callable[[], LocalASRProvisioningPort]


@dataclass(slots=True)
class LocalASRGpuProvisioningOwner:
    provisioning_provider: LocalASRProvisioningProvider = field(repr=False)
    state_provider: LocalASRGpuProvisioningStateProvider = field(repr=False)
    effect_sink: LocalASRGpuProvisioningEffectSink = field(repr=False)
    retry_activation: LocalASRGpuActivationRetry = field(repr=False)
    diagnostic_sink: LocalASRGpuProvisioningDiagnosticSink | None = field(
        default=None,
        repr=False,
    )
    _install_task: asyncio.Task[LocalASRInstallResult] | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def owner_name(self) -> str:
        return "LocalASRGpuProvisioningOwner"

    def request_install(self, *, origin: str = "activation") -> bool:
        if self._install_task is not None and not self._install_task.done():
            return False
        task = self._start_install(origin=origin, deliver_result=True)
        if task is None:
            return False
        self._install_task = task
        return True

    async def install_selected_model_if_needed(self) -> bool:
        if not self.state_provider().selected_provider_requires_model:
            return False
        provisioning = self.provisioning_provider()
        if provisioning.snapshot.activity_for("gpu") is not None:
            return False
        snapshot = await provisioning.inspect_gpu(
            explicit_intent=True,
            verify_checksums=False,
        )
        if not self.state_provider().selected_provider_requires_model:
            return False
        if snapshot.state_for(LOCAL_QWEN_GPU_MODEL_ID).status == "ready":
            return False
        await self.install_or_repair(origin="settings_exit")
        return True

    async def install_or_repair(self, *, origin: str = "manual") -> None:
        task = self._start_install(origin=origin, deliver_result=False)
        if task is None:
            return
        try:
            result = await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit_diagnostic(
                LocalASRGpuProvisioningDiagnostic(
                    event="model_install",
                    outcome="failed",
                    origin=origin,
                    failure_type=type(exc).__name__,
                    exception=exc,
                )
            )
            self.effect_sink(
                LocalASRGpuProvisioningEffect(
                    state="install_failed",
                    origin=origin,
                    publish_notice=True,
                )
            )
            return
        await self._handle_install_result(result, origin=origin)

    def _start_install(
        self,
        *,
        origin: str,
        deliver_result: bool,
    ) -> asyncio.Task[LocalASRInstallResult] | None:
        provisioning = self.provisioning_provider()
        if provisioning.snapshot.activity_for("gpu") is not None:
            return None
        self.effect_sink(
            LocalASRGpuProvisioningEffect(
                state="installing",
                origin=origin,
                progress_percent=0,
                publish_notice=True,
            )
        )
        return provisioning.start_install(
            LocalASRInstallRequest(
                backend="gpu",
                model_ids=(LOCAL_QWEN_GPU_MODEL_ID,),
                locale=self.state_provider().locale,
                origin=origin,
                explicit_gpu_intent=True,
            ),
            result_handler=(
                (lambda result: self._handle_install_result(result, origin=origin))
                if deliver_result
                else None
            ),
        )

    async def _handle_install_result(
        self,
        result: LocalASRInstallResult,
        *,
        origin: str,
    ) -> None:
        if result.cancelled:
            return
        if result.failed_model_ids:
            self.effect_sink(
                LocalASRGpuProvisioningEffect(
                    state="install_failed",
                    origin=origin,
                    publish_notice=True,
                )
            )
            return
        pending_channels = self.state_provider().pending_channels
        self.effect_sink(
            LocalASRGpuProvisioningEffect(
                state="installed",
                origin=origin,
            )
        )
        if pending_channels:
            await self.retry_activation()

    def _emit_diagnostic(
        self,
        diagnostic: LocalASRGpuProvisioningDiagnostic,
    ) -> None:
        if self.diagnostic_sink is None:
            return
        with contextlib.suppress(Exception):
            self.diagnostic_sink(diagnostic)


__all__ = [
    "LocalASRGpuActivationRetry",
    "LocalASRGpuProvisioningDiagnostic",
    "LocalASRGpuProvisioningDiagnosticSink",
    "LocalASRGpuProvisioningEffect",
    "LocalASRGpuProvisioningEffectSink",
    "LocalASRGpuProvisioningOwner",
    "LocalASRGpuProvisioningState",
    "LocalASRGpuProvisioningStateProvider",
    "LocalASRGpuProvisioningUiState",
    "LocalASRProvisioningProvider",
]
