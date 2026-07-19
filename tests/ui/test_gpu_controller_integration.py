from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from puripuly_heart.app.ports.gpu_worker import GpuWorkerDevice
from puripuly_heart.app.services.provider_runtime_apply import _ProviderRuntimeApplyPlan
from puripuly_heart.config.settings import AppSettings, STTProviderName
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeChannelSnapshot,
    ProviderRuntimeDiagnostic,
    ProviderRuntimeGpuSnapshot,
)
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRModelProvisioningState,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
)
from puripuly_heart.ui.controller import GuiController
from puripuly_heart.ui.gpu_device import GpuDeviceOption

pytestmark = pytest.mark.asyncio


def _owned_gpu_snapshot(
    device: GpuWorkerDevice | None = None,
    *,
    phase: str,
    active_channels: frozenset[str] = frozenset(),
    retry_required: bool = False,
    failure_code: str | None = None,
    attached_channels: frozenset[str] = frozenset(),
) -> LocalASRProviderRuntimeSnapshot:
    return LocalASRProviderRuntimeSnapshot(
        channels=tuple(
            ProviderRuntimeChannelSnapshot(
                channel=channel,
                provider_id=(
                    STTProviderName.LOCAL_QWEN_GPU.value if channel in attached_channels else None
                ),
                model_id=LOCAL_QWEN_GPU_MODEL_ID if channel in attached_channels else None,
                phase="ready" if channel in active_channels else "inactive",
                generation=0,
                pending_handoff=False,
                has_resources=channel in attached_channels,
            )
            for channel in ("self", "peer")
        ),
        gpu=ProviderRuntimeGpuSnapshot(
            phase=phase,
            devices=(device,) if device is not None else (),
            active_channels=active_channels,
            pending_count=0,
            worker_pid=None,
            configured_device_id=None,
            model_resident=False,
            retry_required=retry_required,
            failure_code=failure_code,
        ),
    )


class CapturingGpuSettingsView:
    def __init__(self) -> None:
        self.states: list[tuple[str, tuple[tuple[str, str], ...], int | None]] = []

    def set_gpu_runtime_state(
        self,
        state: str,
        *,
        devices: tuple[tuple[str, str], ...],
        progress_percent: int | None = None,
    ) -> None:
        self.states.append((state, devices, progress_percent))


class ReadyProvisioningPort:
    def __init__(self) -> None:
        self.snapshot = LocalASRProvisioningSnapshot(
            models=(
                LocalASRModelProvisioningState(PARAKEET_V3_MODEL_ID, "cpu", "ready"),
                LocalASRModelProvisioningState(PARAKEET_JAPANESE_MODEL_ID, "cpu", "ready"),
                LocalASRModelProvisioningState(LOCAL_STT_MODEL_ID, "cpu", "ready"),
                LocalASRModelProvisioningState(LOCAL_QWEN_GPU_MODEL_ID, "gpu", "ready"),
            ),
            required_cpu_model_ids=REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
            gpu_model_id=LOCAL_QWEN_GPU_MODEL_ID,
        )

    @property
    def diagnostics(self):
        return ()

    async def inspect_cpu(self, model_ids=None, *, verify_checksums=False):
        _ = (model_ids, verify_checksums)
        return self.snapshot

    async def inspect_gpu(self, *, explicit_intent, verify_checksums=False):
        _ = (explicit_intent, verify_checksums)
        return self.snapshot

    def start_install(self, request):
        raise AssertionError(f"unexpected provisioning install: {request}")

    async def report_model_validation_failure(self, model_id, *, failure_type):
        _ = (model_id, failure_type)
        return self.snapshot

    async def cancel_install(self, backend):
        _ = backend

    async def close(self):
        return


def _controller() -> tuple[GuiController, CapturingGpuSettingsView]:
    view = CapturingGpuSettingsView()
    controller = GuiController(
        page=SimpleNamespace(),
        app=SimpleNamespace(view_settings=view),
        config_path=Path("settings.json"),
        local_asr_provisioning=ReadyProvisioningPort(),
    )
    controller.settings = AppSettings()
    return controller, view


@pytest.mark.parametrize(
    "state",
    (
        "discovering",
        "discovery_pending",
        "installed",
        "validating",
        "loading",
        "warming",
        "ready",
    ),
)
async def test_internal_gpu_states_are_logged_without_dashboard_notice(
    state: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, view = _controller()
    messages: list[str] = []

    def log(_self: GuiController, message: str, **_kwargs) -> bool:
        messages.append(message)
        return True

    monkeypatch.setattr(GuiController, "log_detailed", log)

    controller._set_gpu_ui_state(state, origin="startup")

    assert controller._gpu_ui_state == state
    assert view.states == []
    assert messages[-1] == f"[GPU ASR] state={state} origin=startup"


async def test_gpu_settings_receive_hardware_name_separately_from_vulkan_slot() -> None:
    controller, view = _controller()
    captured: list[tuple[GpuDeviceOption, ...]] = []
    view.set_gpu_devices = lambda *, devices: captured.append(devices)
    controller._gpu_devices = (
        GpuWorkerDevice(
            device_id="0000:01:00.0",
            registry_index=0,
            name="Vulkan0",
            description="NVIDIA GeForce RTX 4070",
            device_type="gpu",
            memory_total_bytes=12_000_000_000,
            memory_free_bytes=8_000_000_000,
        ),
    )

    controller._set_gpu_ui_state("installed", origin="settings")

    assert captured[-1] == (
        GpuDeviceOption(
            device_id="0000:01:00.0",
            display_name="NVIDIA GeForce RTX 4070",
            backend_name="Vulkan0",
        ),
    )


async def test_public_self_gpu_toggle_surfaces_provider_teardown_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    controller._stt_desired = True
    failure = RuntimeError("provider teardown failed")
    abort_calls = 0

    async def abort_self_stt_for_toggle_off() -> None:
        nonlocal abort_calls
        abort_calls += 1
        raise failure

    async def stop_mic_loop(_self: GuiController) -> None:
        return None

    controller.hub = SimpleNamespace(abort_self_stt_for_toggle_off=abort_self_stt_for_toggle_off)
    monkeypatch.setattr(GuiController, "_stop_mic_loop", stop_mic_loop)

    with pytest.raises(RuntimeError, match="provider teardown failed") as exc_info:
        await controller.set_stt_enabled(False)

    assert exc_info.value is failure
    assert abort_calls == 1
    assert controller._stt_desired is False


async def test_unavailable_saved_gpu_device_retains_gpu_without_runtime_start() -> None:
    controller, view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.stt.gpu_device_id = "vk:missing"
    device = GpuWorkerDevice(
        device_id="vk:0",
        registry_index=0,
        name="GPU",
        description="GPU",
        device_type="discrete",
        memory_total_bytes=1,
        memory_free_bytes=1,
    )

    class Owner:
        snapshot = _owned_gpu_snapshot(device, phase="idle")

        async def inspect_gpu_readiness(self, *, explicit_intent: bool, device_id: str):
            assert explicit_intent is True
            assert device_id == "vk:missing"
            self.snapshot = _owned_gpu_snapshot(
                device,
                phase="failed",
                failure_code="saved_device_missing",
            )
            return self.snapshot

    owner = Owner()
    controller.hub = SimpleNamespace(local_asr_provider_runtime=owner)

    ready = await controller._validate_gpu_activation()

    assert ready is False
    assert controller.settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
    assert controller.settings.stt.gpu_device_id == "vk:missing"
    assert owner.snapshot.gpu.active_channels == frozenset()
    assert view.states[-1][0] == "unavailable_device"


async def test_gpu_install_notices_have_no_install_action() -> None:
    controller, _view = _controller()
    notices = []
    controller.app.view_dashboard = SimpleNamespace(set_gpu_notice=notices.append)

    controller._set_gpu_ui_state("not_installed", publish_notice=True)
    controller._set_gpu_ui_state("invalid", publish_notice=True)
    controller._set_gpu_ui_state("install_failed", publish_notice=True)

    assert [notice.action for notice in notices] == [None, None, None]


async def test_missing_gpu_model_preserves_self_enable_intent_without_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    dashboard_enabled: list[bool] = []
    controller.app.view_dashboard = SimpleNamespace(
        set_stt_enabled=dashboard_enabled.append,
        set_local_stt_notice=lambda *_args, **_kwargs: None,
        set_local_stt_notice_model=lambda *_args, **_kwargs: None,
        set_gpu_notice=lambda _notice: None,
    )
    install = AsyncMock()

    async def unavailable(_self: GuiController) -> bool:
        _self._set_gpu_ui_state("not_installed")
        return False

    monkeypatch.setattr(GuiController, "_validate_gpu_activation", unavailable)
    monkeypatch.setattr(GuiController, "install_or_repair_gpu_model", install)

    await controller.set_stt_enabled(True)

    assert controller._stt_desired is True
    assert controller._gpu_pending_enable_channels == frozenset({"self"})
    assert dashboard_enabled == []
    install.assert_not_awaited()


async def test_gpu_discovery_keeps_startup_progress_off_dashboard() -> None:
    controller, view = _controller()
    gate = asyncio.Event()
    device = GpuWorkerDevice(
        device_id="vk:0",
        registry_index=0,
        name="GPU",
        description="GPU",
        device_type="discrete",
        memory_total_bytes=1,
        memory_free_bytes=1,
    )

    class Owner:
        snapshot = _owned_gpu_snapshot(phase="idle")

        async def discover_gpu(self, *, force: bool):
            assert force is False
            await gate.wait()
            self.snapshot = _owned_gpu_snapshot(device, phase="available")
            return self.snapshot

    controller.hub = SimpleNamespace(local_asr_provider_runtime=Owner())
    controller.local_asr_provisioning = ReadyProvisioningPort()

    task = asyncio.create_task(controller.ensure_gpu_device_discovery(origin="startup"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    controller._on_local_asr_provider_runtime_diagnostic(
        ProviderRuntimeDiagnostic(event="discovery_pending")
    )
    assert controller._gpu_ui_state == "discovery_pending"
    assert view.states == []

    gate.set()
    await task
    assert controller._gpu_ui_state == "installed"
    assert view.states == []


async def test_gpu_worker_failure_keeps_recovery_membership_out_of_controller_state() -> None:
    controller, view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    controller._stt_desired = True

    controller._on_local_asr_provider_runtime_diagnostic(
        ProviderRuntimeDiagnostic(
            event="worker_failed",
            outcome="failed",
            failure_code="out_of_memory",
        )
    )

    assert not hasattr(controller, "_gpu_manual_retry_channels")
    assert view.states[-1][0] == "activation_failed"


async def test_saved_gpu_preload_skips_non_gpu_and_runs_once_for_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _view = _controller()
    calls: list[str] = []

    async def discover(_self, *, force: bool = False, origin: str = "settings"):
        _ = force
        calls.append(origin)
        return ()

    monkeypatch.setattr(GuiController, "ensure_gpu_device_discovery", discover)

    assert await controller.preload_saved_gpu_device_discovery() == ()
    assert calls == []

    controller.settings.provider.peer_stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.ui.peer_translation_enabled = True
    assert await controller.preload_saved_gpu_device_discovery() == ()
    assert calls == ["startup"]


def _gpu_restart_plan() -> _ProviderRuntimeApplyPlan:
    return _ProviderRuntimeApplyPlan(
        should_rebuild_llm=False,
        should_refresh_peer=True,
        should_refresh_self_stt=True,
        coordinated_gpu_restart=True,
    )


async def test_valid_device_change_quiesces_both_before_either_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.provider.peer_stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.stt.gpu_device_id = "vk:1"
    controller._stt_desired = True
    events: list[str] = []

    class Owner:
        def __init__(self) -> None:
            self.snapshot = _owned_gpu_snapshot(
                phase="ready",
                active_channels=frozenset({"self", "peer"}),
                attached_channels=frozenset({"self", "peer"}),
            )

        async def recover_gpu(self, request, *, quiesce):
            events.append("owner_started")
            assert tuple(item.request.channel for item in request.channels) == ("self", "peer")
            await quiesce(("self", "peer"))
            events.append("owner_rebuilt")
            self.snapshot = _owned_gpu_snapshot(
                phase="ready",
                active_channels=frozenset({"self", "peer"}),
                attached_channels=frozenset({"self", "peer"}),
            )
            return self.snapshot

    controller.hub = SimpleNamespace(local_asr_provider_runtime=Owner())

    async def suspend(_self: GuiController, _channels) -> None:
        events.extend(("self_closed", "peer_closed"))

    async def resume(_self: GuiController, **_kwargs) -> None:
        assert events[-1] == "owner_rebuilt"
        events.append("consumers_resumed")

    monkeypatch.setattr(GuiController, "_suspend_gpu_provider_consumers", suspend)
    monkeypatch.setattr(GuiController, "_resume_gpu_provider_consumers", resume)

    await controller._apply_gpu_runtime_owner_recovery(controller.settings, _gpu_restart_plan())

    assert events == [
        "owner_started",
        "self_closed",
        "peer_closed",
        "owner_rebuilt",
        "consumers_resumed",
    ]


async def test_unavailable_device_change_retains_selection_and_stops_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.provider.peer_stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.stt.gpu_device_id = "vk:missing"
    controller._stt_desired = True
    suspend = AsyncMock()
    resume = AsyncMock()

    class Owner:
        def __init__(self) -> None:
            self.snapshot = _owned_gpu_snapshot(
                phase="failed",
                active_channels=frozenset({"self", "peer"}),
                retry_required=True,
                failure_code="saved_device_missing",
                attached_channels=frozenset({"self", "peer"}),
            )

        async def recover_gpu(self, _request, *, quiesce):
            await quiesce(("self", "peer"))
            self.snapshot = _owned_gpu_snapshot(
                phase="failed",
                retry_required=True,
                failure_code="saved_device_missing",
            )
            return self.snapshot

    controller.hub = SimpleNamespace(local_asr_provider_runtime=Owner())
    monkeypatch.setattr(
        GuiController,
        "_suspend_gpu_provider_consumers",
        lambda _self, channels: suspend(channels),
    )
    monkeypatch.setattr(
        GuiController,
        "_resume_gpu_provider_consumers",
        lambda _self, **kwargs: resume(**kwargs),
    )

    await controller._apply_gpu_runtime_owner_recovery(controller.settings, _gpu_restart_plan())

    suspend.assert_awaited_once_with(("self", "peer"))
    resume.assert_not_awaited()
    assert controller.settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
    assert controller.settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
    assert controller.settings.stt.gpu_device_id == "vk:missing"


async def test_device_activation_failure_releases_both_and_requires_manual_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.provider.peer_stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.stt.gpu_device_id = "vk:1"
    controller._stt_desired = True
    suspend = AsyncMock()

    class Owner:
        snapshot = _owned_gpu_snapshot(
            phase="ready",
            active_channels=frozenset({"self", "peer"}),
            attached_channels=frozenset({"self", "peer"}),
        )

        async def recover_gpu(self, _request, *, quiesce):
            await quiesce(("self", "peer"))
            raise RuntimeError("activation failed")

    controller.hub = SimpleNamespace(local_asr_provider_runtime=Owner())
    monkeypatch.setattr(
        GuiController,
        "_suspend_gpu_provider_consumers",
        lambda _self, channels: suspend(channels),
    )

    await controller._apply_gpu_runtime_owner_recovery(controller.settings, _gpu_restart_plan())

    suspend.assert_awaited_once_with(("self", "peer"))
    assert view.states[-1][0] == "activation_failed"
    assert controller.settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
    assert controller.settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
    assert controller.settings.stt.gpu_device_id == "vk:1"


async def test_gpu_consumer_suspension_does_not_detach_unrelated_non_gpu_self_channel() -> None:
    controller, _view = _controller()
    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    controller.settings.provider.peer_stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.ui.peer_translation_enabled = True

    class PeerRuntime:
        def __init__(self) -> None:
            self.suspend_calls = 0

        async def suspend_provider_consumer(self) -> None:
            self.suspend_calls += 1

    peer_runtime = PeerRuntime()
    controller._peer_runtime = peer_runtime

    await controller._suspend_gpu_provider_consumers(("peer",))

    assert peer_runtime.suspend_calls == 1


@pytest.mark.parametrize(
    "channels",
    [frozenset({"self"}), frozenset({"peer"}), frozenset({"self", "peer"})],
)
async def test_manual_retry_delegates_attached_gpu_channels_to_owner(
    channels: frozenset[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _view = _controller()
    device = GpuWorkerDevice(
        device_id="vk:0",
        registry_index=0,
        name="GPU",
        description="GPU",
        device_type="discrete",
        memory_total_bytes=8,
        memory_free_bytes=4,
    )
    controller.settings.stt.gpu_device_id = device.device_id
    controller.settings.provider.stt = (
        STTProviderName.LOCAL_QWEN_GPU if "self" in channels else STTProviderName.DEEPGRAM
    )
    controller.settings.provider.peer_stt = (
        STTProviderName.LOCAL_QWEN_GPU if "peer" in channels else STTProviderName.DEEPGRAM
    )
    controller.settings.ui.peer_translation_enabled = "peer" in channels
    controller._stt_desired = "self" in channels
    controller._gpu_devices = (device,)
    resumed = AsyncMock()
    monkeypatch.setattr(
        GuiController,
        "_resume_gpu_provider_consumers",
        lambda _self, **kwargs: resumed(**kwargs),
    )

    class Owner:
        def __init__(self) -> None:
            self.snapshot = _owned_gpu_snapshot(
                device,
                phase="available",
                retry_required=True,
                failure_code="worker_failed",
                attached_channels=channels,
            )
            self.recovery_calls = []

        async def recover_gpu(self, request, *, quiesce):
            self.recovery_calls.append(request)
            await quiesce(tuple(item.request.channel for item in request.channels))
            self.snapshot = _owned_gpu_snapshot(
                device,
                phase="ready",
                active_channels=channels,
                attached_channels=channels,
            )
            return self.snapshot

    owner = Owner()
    controller.hub = SimpleNamespace(local_asr_provider_runtime=owner)

    await controller.retry_gpu_activation()

    assert len(owner.recovery_calls) == 1
    recovery = owner.recovery_calls[0]
    assert recovery.reason == "manual_retry"
    assert tuple(item.request.channel for item in recovery.channels) == tuple(
        channel for channel in ("self", "peer") if channel in channels
    )
    assert all(item.request.warmup for item in recovery.channels)
    assert owner.snapshot.gpu.active_channels == channels
    assert controller.settings.stt.gpu_device_id == device.device_id
    assert controller._gpu_ui_state == "ready"
    resumed.assert_awaited_once()


async def test_controller_gpu_discovery_and_readiness_delegate_to_owned_runtime() -> None:
    controller, _view = _controller()
    device = GpuWorkerDevice(
        device_id="vk:0",
        registry_index=0,
        name="GPU",
        description="GPU",
        device_type="discrete",
        memory_total_bytes=8,
        memory_free_bytes=4,
    )

    class Owner:
        def __init__(self) -> None:
            self.snapshot = _owned_gpu_snapshot(device, phase="idle")
            self.discovery_calls: list[bool] = []
            self.readiness_calls: list[tuple[bool, str]] = []

        async def discover_gpu(self, *, force: bool):
            self.discovery_calls.append(force)
            return self.snapshot

        async def inspect_gpu_readiness(self, *, explicit_intent: bool, device_id: str):
            self.readiness_calls.append((explicit_intent, device_id))
            self.snapshot = _owned_gpu_snapshot(device, phase="available")
            return self.snapshot

    owner = Owner()
    controller.hub = SimpleNamespace(local_asr_provider_runtime=owner)

    devices = await controller.ensure_gpu_device_discovery(force=True)
    ready = await controller._validate_gpu_activation()

    assert devices == (device,)
    assert ready is True
    assert owner.discovery_calls == [True]
    assert owner.readiness_calls == [(True, controller.settings.stt.gpu_device_id)]
