from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from puripuly_heart.app.ports.ui_models import (
    GpuDashboardNotice,
    GpuDeviceOption,
    GpuNoticeAction,
)
from puripuly_heart.app.services.local_asr_diagnostics import (
    LocalASRDiagnosticsGpuEffect,
)
from puripuly_heart.app.services.local_asr_gpu_provisioning import (
    LocalASRGpuProvisioningDiagnostic,
    LocalASRGpuProvisioningEffect,
    LocalASRGpuProvisioningOwner,
    LocalASRGpuProvisioningState,
)
from puripuly_heart.core.local_asr_provider_runtime import (
    GpuWorkerDevice,
    LocalASRProviderRuntimePort,
    LocalASRProviderRuntimeSnapshot,
)
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRProvisioningPort,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_stt_assets import LOCAL_QWEN_GPU_MODEL_ID
from puripuly_heart.core.runtime.gpu_asr import GpuASRChannel


@dataclass(frozen=True, slots=True)
class GpuRuntimeInteractionState:
    settings_available: bool
    selected_provider_requires_model: bool
    locale: str | None
    device_id: str


@dataclass(frozen=True, slots=True)
class GpuRuntimeInteractionSnapshot:
    ui_state: str | None
    devices: tuple[GpuWorkerDevice, ...]
    discovery_attempted: bool
    discovery_failed: bool
    discovery_failure_state: str | None
    discovery_origin: str
    pending_channels: frozenset[GpuASRChannel]


@dataclass(frozen=True, slots=True)
class GpuRuntimePresentation:
    devices: tuple[GpuDeviceOption, ...]
    state: str
    progress_percent: int | None
    notice: GpuDashboardNotice | None
    publish_notice: bool


GpuRuntimeProvider = Callable[[], LocalASRProviderRuntimePort]
GpuRuntimeProvisioningProvider = Callable[[], LocalASRProvisioningPort]
GpuRuntimeInteractionStateProvider = Callable[[], GpuRuntimeInteractionState]
GpuRuntimePresentationSink = Callable[[GpuRuntimePresentation], None]
GpuRuntimeDetailedLogSink = Callable[[str], object]
GpuRuntimeActivationRetry = Callable[[], Awaitable[None]]
GpuRuntimeInstallDiagnosticSink = Callable[[LocalASRGpuProvisioningDiagnostic], None]


@dataclass(slots=True)
class GpuRuntimeInteractionOwner:
    runtime_provider: GpuRuntimeProvider = field(repr=False)
    provisioning_provider: GpuRuntimeProvisioningProvider = field(repr=False)
    state_provider: GpuRuntimeInteractionStateProvider = field(repr=False)
    presentation_sink: GpuRuntimePresentationSink = field(repr=False)
    detailed_log_sink: GpuRuntimeDetailedLogSink = field(repr=False)
    retry_activation: GpuRuntimeActivationRetry = field(repr=False)
    install_diagnostic_sink: GpuRuntimeInstallDiagnosticSink | None = field(
        default=None,
        repr=False,
    )
    _ui_state: str | None = field(init=False, default=None, repr=False)
    _devices: tuple[GpuWorkerDevice, ...] = field(init=False, default=(), repr=False)
    _discovery_attempted: bool = field(init=False, default=False, repr=False)
    _discovery_failed: bool = field(init=False, default=False, repr=False)
    _discovery_failure_state: str | None = field(init=False, default=None, repr=False)
    _discovery_origin: str = field(init=False, default="settings", repr=False)
    _pending_channels: frozenset[GpuASRChannel] = field(
        init=False,
        default=frozenset(),
        repr=False,
    )
    _provisioning_owner: LocalASRGpuProvisioningOwner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._provisioning_owner = LocalASRGpuProvisioningOwner(
            provisioning_provider=self.provisioning_provider,
            state_provider=self._provisioning_state,
            effect_sink=self.apply_provisioning_effect,
            retry_activation=self.retry_activation,
            diagnostic_sink=self.install_diagnostic_sink,
        )

    @property
    def owner_name(self) -> str:
        return "GpuRuntimeInteractionOwner"

    @property
    def snapshot(self) -> GpuRuntimeInteractionSnapshot:
        return GpuRuntimeInteractionSnapshot(
            ui_state=self._ui_state,
            devices=self._devices,
            discovery_attempted=self._discovery_attempted,
            discovery_failed=self._discovery_failed,
            discovery_failure_state=self._discovery_failure_state,
            discovery_origin=self._discovery_origin,
            pending_channels=self._pending_channels,
        )

    async def preload_saved_device_discovery(self) -> tuple[GpuWorkerDevice, ...]:
        if not self.state_provider().selected_provider_requires_model:
            return ()
        return await self.ensure_device_discovery(origin="startup")

    async def handle_notice_action(self, action: GpuNoticeAction) -> None:
        if action in {"install", "repair", "reinstall"}:
            await self.install_or_repair()
            return
        if action == "rediscover":
            await self.ensure_device_discovery(
                force=True,
                origin="manual_rediscovery",
            )
            if self._discovery_failed:
                self.set_ui_state(
                    self._discovery_failure_state or "discovery_failed",
                    publish_notice=True,
                    origin="manual_rediscovery",
                )
            elif not self._devices:
                self.set_ui_state(
                    "unsupported",
                    publish_notice=True,
                    origin="manual_rediscovery",
                )
            return
        if action == "restart":
            await self.retry_activation()

    async def ensure_device_discovery(
        self,
        *,
        force: bool = False,
        origin: str = "settings",
    ) -> tuple[GpuWorkerDevice, ...]:
        self._discovery_origin = origin
        self.set_ui_state("discovering", origin=origin)
        snapshot = await self.runtime_provider().discover_gpu(force=force)
        self.observe_runtime(snapshot)
        if snapshot.gpu.phase == "unsupported":
            self.set_ui_state("unsupported", origin=origin)
        elif snapshot.gpu.phase == "failed":
            self.set_ui_state(
                "discovery_failed",
                publish_notice=True,
                origin=origin,
            )
        else:
            self.set_ui_state(self.idle_ui_state(), origin=origin)
        return snapshot.gpu.devices

    async def validate_activation(self) -> bool:
        state = self.state_provider()
        if not state.settings_available:
            return False
        self.set_ui_state("validating", origin="activation")
        snapshot = await self.runtime_provider().inspect_gpu_readiness(
            explicit_intent=True,
            device_id=state.device_id,
        )
        self.observe_runtime(snapshot)
        phase = snapshot.gpu.phase
        if phase in {"available", "ready"}:
            self.set_ui_state(
                "ready" if phase == "ready" else "loading",
                origin="activation",
            )
            return True
        if phase in {"not_installed", "invalid"}:
            if self._provisioning_owner.request_install(origin="activation"):
                self.set_ui_state(
                    "installing",
                    progress_percent=0,
                    publish_notice=True,
                    origin="activation",
                )
                return False
        state_by_phase = {
            "unsupported": "unsupported",
            "not_installed": "not_installed",
            "installing": "installing",
            "invalid": "invalid",
            "failed": (
                "unavailable_device"
                if snapshot.gpu.failure_code == "saved_device_missing"
                else "activation_failed"
            ),
        }
        self.set_ui_state(
            state_by_phase.get(phase, "activation_failed"),
            publish_notice=True,
            origin="activation",
        )
        return False

    async def install_selected_model_if_needed(self) -> bool:
        return await self._provisioning_owner.install_selected_model_if_needed()

    async def install_or_repair(self, *, origin: str = "manual") -> None:
        await self._provisioning_owner.install_or_repair(origin=origin)

    def observe_runtime(self, snapshot: LocalASRProviderRuntimeSnapshot) -> None:
        self._devices = snapshot.gpu.devices
        self._discovery_attempted = bool(snapshot.gpu.devices) or snapshot.gpu.phase not in {
            "inactive",
            "idle",
        }
        self._discovery_failed = snapshot.gpu.phase in {"failed", "unsupported"}
        self._discovery_failure_state = (
            "unsupported"
            if snapshot.gpu.phase == "unsupported"
            else "discovery_failed" if snapshot.gpu.phase == "failed" else None
        )

    def observe_provisioning(self, snapshot: LocalASRProvisioningSnapshot) -> None:
        activity = snapshot.activity_for("gpu")
        if activity is not None:
            self.set_ui_state(
                "installing",
                progress_percent=activity.progress_percent,
                publish_notice=True,
                origin=activity.origin,
            )

    def apply_provisioning_effect(self, effect: LocalASRGpuProvisioningEffect) -> None:
        self.set_ui_state(
            effect.state,
            progress_percent=effect.progress_percent,
            publish_notice=effect.publish_notice,
            origin=effect.origin,
        )

    def apply_diagnostics_effect(self, effect: LocalASRDiagnosticsGpuEffect) -> None:
        self.set_ui_state(
            effect.state,
            publish_notice=effect.publish_notice,
            origin=effect.origin,
        )

    def set_ui_state(
        self,
        state: str,
        *,
        progress_percent: int | None = None,
        publish_notice: bool = False,
        origin: str = "runtime",
    ) -> None:
        self._ui_state = state
        fields = [f"state={state}", f"origin={origin}"]
        if progress_percent is not None:
            fields.append(f"progress_percent={progress_percent}")
        self.detailed_log_sink(f"[GPU ASR] {' '.join(fields)}")
        devices = tuple(
            GpuDeviceOption(
                device_id=device.device_id,
                display_name=device.description.strip() or device.name,
                backend_name=device.name,
            )
            for device in self._devices
        )
        action_by_state: dict[str, GpuNoticeAction] = {
            "discovery_failed": "rediscover",
            "activation_failed": "restart",
        }
        notice = (
            GpuDashboardNotice(
                status=state,
                progress_percent=progress_percent,
                action=action_by_state.get(state),
            )
            if publish_notice
            else None
        )
        self.presentation_sink(
            GpuRuntimePresentation(
                devices=devices,
                state=state,
                progress_percent=progress_percent,
                notice=notice,
                publish_notice=publish_notice,
            )
        )

    def idle_ui_state(self) -> str:
        status = self.provisioning_provider().snapshot.state_for(LOCAL_QWEN_GPU_MODEL_ID).status
        if status in {"not_requested", "missing"}:
            return "not_installed"
        if status in {"invalid", "download_failed", "cancelled"}:
            return "invalid"
        if status == "downloading":
            return "installing"
        return "installed"

    def retain_pending(self, channel: GpuASRChannel) -> None:
        self._pending_channels = frozenset({*self._pending_channels, channel})

    def clear_pending(self, *channels: GpuASRChannel) -> None:
        removed = frozenset(channels)
        self._pending_channels = frozenset(
            channel for channel in self._pending_channels if channel not in removed
        )

    def complete_manual_recovery(
        self,
        channels: frozenset[GpuASRChannel],
    ) -> None:
        self.clear_pending(*channels)
        self.set_ui_state("ready", origin="manual_retry")

    def _provisioning_state(self) -> LocalASRGpuProvisioningState:
        state = self.state_provider()
        return LocalASRGpuProvisioningState(
            selected_provider_requires_model=state.selected_provider_requires_model,
            locale=state.locale,
            pending_channels=self._pending_channels,
        )


__all__ = [
    "GpuRuntimeActivationRetry",
    "GpuRuntimeDetailedLogSink",
    "GpuRuntimeInstallDiagnosticSink",
    "GpuRuntimeInteractionOwner",
    "GpuRuntimeInteractionSnapshot",
    "GpuRuntimeInteractionState",
    "GpuRuntimeInteractionStateProvider",
    "GpuRuntimePresentation",
    "GpuRuntimePresentationSink",
    "GpuRuntimeProvider",
    "GpuRuntimeProvisioningProvider",
]
