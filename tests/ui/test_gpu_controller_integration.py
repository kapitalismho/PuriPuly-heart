from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import puripuly_heart.ui.controller as controller_module
from puripuly_heart.app.ports.gpu_worker import (
    GpuWorkerActivation,
    GpuWorkerClosedError,
    GpuWorkerDevice,
    GpuWorkerEvent,
    GpuWorkerRequestError,
)
from puripuly_heart.app.services.provider_runtime_apply import _ProviderRuntimeApplyPlan
from puripuly_heart.config.settings import AppSettings, STTProviderName
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeChannelSnapshot,
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
from puripuly_heart.core.runtime.gpu_asr import GpuASRRuntimeState, SharedGpuASRRuntime
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend
from puripuly_heart.ui.controller import GuiController
from puripuly_heart.ui.gpu_device import GpuDeviceOption

pytestmark = pytest.mark.asyncio


def _owned_gpu_snapshot(
    device: GpuWorkerDevice,
    *,
    phase: str,
) -> LocalASRProviderRuntimeSnapshot:
    return LocalASRProviderRuntimeSnapshot(
        channels=tuple(
            ProviderRuntimeChannelSnapshot(
                channel=channel,
                provider_id=None,
                model_id=None,
                phase="inactive",
                generation=0,
                pending_handoff=False,
                has_resources=False,
            )
            for channel in ("self", "peer")
        ),
        gpu=ProviderRuntimeGpuSnapshot(
            phase=phase,
            devices=(device,),
            active_channels=frozenset(),
            pending_count=0,
            worker_pid=None,
            configured_device_id=None,
            model_resident=False,
            retry_required=False,
            failure_code=None,
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


class RecoveryGpuWorkerClient:
    def __init__(
        self,
        device: GpuWorkerDevice,
        *,
        activation_error: bool = False,
    ) -> None:
        self.device = device
        self.activation_error = activation_error
        self.activate_calls: list[str] = []
        self.close_calls = 0
        self._closed = False
        self._events: asyncio.Queue[GpuWorkerEvent | None] = asyncio.Queue()

    @property
    def pid(self) -> int | None:
        return None if self._closed else 4321

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def discover(self) -> tuple[GpuWorkerDevice, ...]:
        return (self.device,)

    async def activate(self, *, model_path: Path, device_id: str) -> GpuWorkerActivation:
        assert model_path == Path("gpu-model.gguf").resolve()
        self.activate_calls.append(device_id)
        if self.activation_error:
            raise GpuWorkerRequestError("activation_failed")
        return GpuWorkerActivation(
            device=self.device,
            model_load_seconds=0.01,
            warmup_seconds=0.01,
        )

    async def transcribe(self, **_kwargs):
        raise AssertionError("transcribe is not used by recovery tests")

    async def cancel(self, _target_request_id: str) -> None:
        return

    async def next_event(self) -> GpuWorkerEvent:
        event = await self._events.get()
        if event is None:
            raise GpuWorkerClosedError("closed")
        return event

    async def close(self) -> None:
        self.close_calls += 1
        if self._closed:
            return
        self._closed = True
        await self._events.put(None)

    async def force_close(self) -> None:
        await self.close()


class RecoveryGpuWorkerFactory:
    def __init__(self, clients: list[RecoveryGpuWorkerClient]) -> None:
        self.clients = clients
        self.modes: list[str] = []

    async def start(self, *, mode: str) -> RecoveryGpuWorkerClient:
        self.modes.append(mode)
        return self.clients.pop(0)


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


async def test_self_gpu_toggle_off_releases_worker_and_toggle_on_rebuilds_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    device = GpuWorkerDevice(
        device_id="vk:0",
        registry_index=0,
        name="GPU",
        description="GPU",
        device_type="discrete",
        memory_total_bytes=8,
        memory_free_bytes=4,
    )
    first_client = RecoveryGpuWorkerClient(device)
    second_client = RecoveryGpuWorkerClient(device)
    factory = RecoveryGpuWorkerFactory([first_client, second_client])
    runtime = SharedGpuASRRuntime(process_factory=factory)

    def create_provider() -> ManagedSTTProvider:
        return ManagedSTTProvider(
            backend=LocalGpuSTTBackend(
                runtime=runtime,
                channel="self",
                model_path=Path("gpu-model.gguf"),
                model_id="gpu-model",
                device_id="vk:0",
            ),
            sample_rate_hz=16_000,
            stt_provider_name=STTProviderName.LOCAL_QWEN_GPU,
        )

    handle = ProviderRuntimeHandle(name="self-stt", provider=create_provider())

    class Hub:
        @property
        def stt(self):
            return handle.provider

        async def replace_stt_provider(self, provider) -> None:
            await handle.stop_ingress()
            await handle.replace_provider(provider, start=True)

        async def resume_self_stt_after_toggle_on(self) -> None:
            await handle.start()

    hub = Hub()
    controller.hub = hub
    assert hub.stt is not None
    await hub.stt.warmup()
    first_pid = runtime.worker_pid
    assert first_pid is not None
    assert runtime.active_channels == frozenset({"self"})

    await controller._drain_self_stt_for_toggle_off()

    assert hub.stt is None
    assert runtime.state == GpuASRRuntimeState.STOPPED
    assert runtime.active_channels == frozenset()
    assert runtime.worker_pid is None
    assert first_client.close_calls == 1

    async def rebuild(_self: GuiController) -> None:
        await hub.replace_stt_provider(create_provider())

    async def start_mic(_self: GuiController) -> bool:
        assert hub.stt is not None
        await hub.stt.warmup()
        return True

    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", rebuild)
    monkeypatch.setattr(GuiController, "_start_mic_loop", start_mic)
    controller._stt_desired = True
    controller._stt_activation_generation = 1

    await controller._run_stt_switch()

    assert hub.stt is not None
    assert runtime.state == GpuASRRuntimeState.READY
    assert runtime.active_channels == frozenset({"self"})
    assert runtime.worker_pid is not None
    assert second_client.activate_calls == ["vk:0"]
    assert factory.modes == ["persistent", "persistent"]
    await handle.close()
    await runtime.close()


async def test_public_self_gpu_toggle_surfaces_provider_teardown_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    controller._stt_desired = True
    failure = RuntimeError("provider teardown failed")
    replace_calls: list[object | None] = []

    async def replace_stt_provider(provider: object | None) -> None:
        replace_calls.append(provider)
        raise failure

    async def stop_mic_loop(_self: GuiController) -> None:
        return None

    controller.hub = SimpleNamespace(replace_stt_provider=replace_stt_provider)
    monkeypatch.setattr(GuiController, "_stop_mic_loop", stop_mic_loop)

    with pytest.raises(RuntimeError, match="provider teardown failed") as exc_info:
        await controller.set_stt_enabled(False)

    assert exc_info.value is failure
    assert replace_calls == [None]
    assert controller._stt_desired is False


async def test_unavailable_saved_gpu_device_retains_gpu_without_runtime_start() -> None:
    controller, view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.stt.gpu_device_id = "vk:missing"
    controller._gpu_devices = (
        GpuWorkerDevice(
            device_id="vk:0",
            registry_index=0,
            name="GPU",
            description="GPU",
            device_type="discrete",
            memory_total_bytes=1,
            memory_free_bytes=1,
        ),
    )

    ready = await controller._validate_gpu_activation()

    assert ready is False
    assert controller.settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
    assert controller.settings.stt.gpu_device_id == "vk:missing"
    assert controller._gpu_asr_runtime is None
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


async def test_gpu_discovery_keeps_startup_progress_off_dashboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    class Runtime:
        async def discover_devices(self):
            await gate.wait()
            return (device,)

    monkeypatch.setattr(GuiController, "_get_gpu_asr_runtime", lambda _self: Runtime())
    monkeypatch.setattr(GuiController, "_gpu_idle_ui_state", lambda _self: "not_installed")

    task = asyncio.create_task(controller.ensure_gpu_device_discovery(origin="startup"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await controller._on_gpu_asr_diagnostic(SimpleNamespace(kind="discovery_pending", fields={}))
    assert controller._gpu_ui_state == "discovery_pending"
    assert view.states == []

    gate.set()
    await task
    assert controller._gpu_ui_state == "not_installed"
    assert view.states == []


async def test_gpu_worker_failure_preserves_channels_for_restart_action() -> None:
    controller, view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    controller._stt_desired = True

    await controller._on_gpu_asr_diagnostic(
        SimpleNamespace(kind="worker_failed", fields={"code": "out_of_memory"})
    )

    assert controller._gpu_manual_retry_channels == frozenset({"self"})
    assert view.states[-1][0] == "activation_failed"


async def test_gpu_discovery_success_and_empty_results_are_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for devices in (
        (
            GpuWorkerDevice(
                device_id="vk:0",
                registry_index=0,
                name="GPU",
                description="GPU",
                device_type="discrete",
                memory_total_bytes=1,
                memory_free_bytes=1,
            ),
        ),
        (),
    ):
        controller, _view = _controller()
        calls = 0

        class Runtime:
            async def discover_devices(self):
                nonlocal calls
                calls += 1
                return devices

        monkeypatch.setattr(GuiController, "_get_gpu_asr_runtime", lambda _self: Runtime())
        monkeypatch.setattr(GuiController, "_gpu_idle_ui_state", lambda _self: "not_installed")

        assert await controller.ensure_gpu_device_discovery() == devices
        assert await controller.ensure_gpu_device_discovery() == devices
        assert calls == 1


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
    assert await controller.preload_saved_gpu_device_discovery() == ()
    assert calls == ["startup"]


async def test_gpu_discovery_concurrent_requests_and_activation_share_one_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    gate = asyncio.Event()
    calls = 0
    device = GpuWorkerDevice(
        device_id="vk:0",
        registry_index=0,
        name="GPU",
        description="GPU",
        device_type="discrete",
        memory_total_bytes=1,
        memory_free_bytes=1,
    )

    class Runtime:
        async def discover_devices(self):
            nonlocal calls
            calls += 1
            await gate.wait()
            return (device,)

    monkeypatch.setattr(GuiController, "_get_gpu_asr_runtime", lambda _self: Runtime())
    monkeypatch.setattr(GuiController, "_gpu_idle_ui_state", lambda _self: "installed")

    first = asyncio.create_task(controller.ensure_gpu_device_discovery())
    second = asyncio.create_task(controller.ensure_gpu_device_discovery())
    activation = asyncio.create_task(controller._validate_gpu_activation())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert calls == 1

    gate.set()
    assert await first == (device,)
    assert await second == (device,)
    assert await activation is True
    assert calls == 1


async def test_gpu_discovery_failure_is_cached_until_forced_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, view = _controller()
    calls = 0

    class Runtime:
        async def discover_devices(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise GpuWorkerRequestError("discovery_failed")
            return ()

    monkeypatch.setattr(GuiController, "_get_gpu_asr_runtime", lambda _self: Runtime())

    assert await controller.ensure_gpu_device_discovery() == ()
    assert await controller.ensure_gpu_device_discovery() == ()
    assert calls == 1
    assert controller._gpu_ui_state == "discovery_failed"
    assert view.states == []

    assert await controller.ensure_gpu_device_discovery(force=True) == ()
    assert calls == 2
    assert controller._gpu_ui_state == "unsupported"
    assert view.states == []


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

    async def validate(_self: GuiController) -> bool:
        events.append("validated")
        return True

    async def quiesce(_self: GuiController, _settings: AppSettings) -> None:
        events.extend(("self_closed", "peer_closed"))

    async def restore_self(_self: GuiController) -> None:
        assert events[:3] == ["validated", "self_closed", "peer_closed"]
        events.append("self_restored")

    async def restore_peer(_self: GuiController) -> None:
        assert "self_closed" in events and "peer_closed" in events
        events.append("peer_restored")

    monkeypatch.setattr(GuiController, "_validate_gpu_activation", validate)
    monkeypatch.setattr(GuiController, "_quiesce_shared_gpu_consumers", quiesce)
    monkeypatch.setattr(GuiController, "_replace_runtime_stt_provider", restore_self)
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", restore_peer)
    monkeypatch.setattr(GuiController, "_refresh_overlay_peer_consumers", lambda _self: None)

    await controller._apply_coordinated_gpu_restart(controller.settings, _gpu_restart_plan())

    assert events == [
        "validated",
        "self_closed",
        "peer_closed",
        "self_restored",
        "peer_restored",
    ]


async def test_unavailable_device_change_retains_selection_and_stops_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _view = _controller()
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.provider.peer_stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.stt.gpu_device_id = "vk:missing"
    controller._stt_desired = True
    quiesce = AsyncMock()
    restore_self = AsyncMock()
    restore_peer = AsyncMock()

    async def validate(_self: GuiController) -> bool:
        return False

    async def quiesce_call(_self: GuiController, settings: AppSettings) -> None:
        await quiesce(settings)

    async def restore_self_call(_self: GuiController) -> None:
        await restore_self()

    async def restore_peer_call(_self: GuiController) -> None:
        await restore_peer()

    monkeypatch.setattr(GuiController, "_validate_gpu_activation", validate)
    monkeypatch.setattr(GuiController, "_quiesce_shared_gpu_consumers", quiesce_call)
    monkeypatch.setattr(GuiController, "_replace_runtime_stt_provider", restore_self_call)
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", restore_peer_call)

    await controller._apply_coordinated_gpu_restart(controller.settings, _gpu_restart_plan())

    quiesce.assert_awaited_once_with(controller.settings)
    restore_self.assert_not_awaited()
    restore_peer.assert_not_awaited()
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
    quiesce = AsyncMock()

    async def validate(_self: GuiController) -> bool:
        return True

    async def quiesce_call(_self: GuiController, settings: AppSettings) -> None:
        await quiesce(settings)

    async def fail_restore(_self: GuiController) -> None:
        raise RuntimeError("activation failed")

    monkeypatch.setattr(GuiController, "_validate_gpu_activation", validate)
    monkeypatch.setattr(GuiController, "_quiesce_shared_gpu_consumers", quiesce_call)
    monkeypatch.setattr(GuiController, "_replace_runtime_stt_provider", fail_restore)

    await controller._apply_coordinated_gpu_restart(controller.settings, _gpu_restart_plan())

    assert quiesce.await_count == 2
    assert view.states[-1][0] == "activation_failed"
    assert controller.settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
    assert controller.settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
    assert controller.settings.stt.gpu_device_id == "vk:1"


async def test_gpu_quiescence_does_not_detach_unrelated_non_gpu_self_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, _view = _controller()
    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    controller.settings.provider.peer_stt = STTProviderName.LOCAL_QWEN_GPU
    controller.settings.ui.peer_translation_enabled = True
    controller._gpu_asr_runtime = SimpleNamespace(active_channels=frozenset({"peer"}))
    replace_self = AsyncMock()
    controller.hub = SimpleNamespace(replace_stt_provider=replace_self)

    class PeerRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[object, bool]] = []

        async def apply_policy(self, *, config: object, desired_active: bool) -> None:
            self.calls.append((config, desired_active))

    peer_runtime = PeerRuntime()
    controller._peer_runtime = peer_runtime
    config = object()
    monkeypatch.setattr(
        GuiController,
        "_build_peer_runtime_config",
        lambda _self, _settings: config,
    )

    await controller._quiesce_shared_gpu_consumers(controller.settings)

    replace_self.assert_not_awaited()
    assert peer_runtime.calls == [(config, False)]


@pytest.mark.parametrize(
    "channels",
    [frozenset({"self"}), frozenset({"peer"}), frozenset({"self", "peer"})],
)
@pytest.mark.parametrize("initial_failure", ["unavailable", "activation"])
async def test_manual_retry_restores_detached_gpu_providers_on_one_fresh_runtime(
    channels: frozenset[str],
    initial_failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, view = _controller()
    new_device = GpuWorkerDevice(
        device_id="vk:new",
        registry_index=1,
        name="New GPU",
        description="New GPU",
        device_type="discrete",
        memory_total_bytes=8,
        memory_free_bytes=4,
    )
    stale_device = GpuWorkerDevice(
        device_id="vk:stale",
        registry_index=0,
        name="Stale GPU",
        description="Stale GPU",
        device_type="discrete",
        memory_total_bytes=8,
        memory_free_bytes=4,
    )
    discovery_client = RecoveryGpuWorkerClient(new_device)
    failed_client = RecoveryGpuWorkerClient(new_device, activation_error=True)
    successful_client = RecoveryGpuWorkerClient(new_device)
    clients = (
        [discovery_client, successful_client]
        if initial_failure == "unavailable"
        else [failed_client, successful_client]
    )
    factory = RecoveryGpuWorkerFactory(clients)
    runtime = SharedGpuASRRuntime(process_factory=factory)
    controller._gpu_asr_runtime = runtime
    controller.settings.stt.gpu_device_id = "vk:new"
    controller.settings.provider.stt = (
        STTProviderName.LOCAL_QWEN_GPU if "self" in channels else STTProviderName.DEEPGRAM
    )
    controller.settings.provider.peer_stt = (
        STTProviderName.LOCAL_QWEN_GPU if "peer" in channels else STTProviderName.DEEPGRAM
    )
    controller.settings.ui.peer_translation_enabled = "peer" in channels
    controller._stt_desired = "self" in channels
    controller._gpu_manual_retry_channels = channels
    controller._gpu_devices = (stale_device,) if initial_failure == "unavailable" else (new_device,)
    monkeypatch.setattr(controller_module, "local_gpu_model_path", lambda: Path("gpu-model.gguf"))
    backends: dict[str, LocalGpuSTTBackend] = {}
    sessions: dict[str, object] = {}
    restores: list[str] = []

    def backend(channel: str) -> LocalGpuSTTBackend:
        return LocalGpuSTTBackend(
            runtime=runtime,
            channel=channel,
            model_path=Path("gpu-model.gguf"),
            model_id="gpu-model",
            device_id=controller.settings.stt.gpu_device_id,
        )

    async def rebuild_self(_self: GuiController) -> None:
        restores.append("self_rebuilt")
        backends["self"] = backend("self")

    async def enable_self(
        _self: GuiController,
        enabled: bool,
        *,
        force_immediate: bool = False,
    ) -> None:
        _ = force_immediate
        assert enabled
        assert "self" in backends
        sessions["self"] = await backends["self"].open_session()
        restores.append("self_enabled")

    async def refresh_peer(_self: GuiController) -> None:
        restores.append("peer_rebuilt")
        backends["peer"] = backend("peer")
        sessions["peer"] = await backends["peer"].open_session()
        restores.append("peer_enabled")

    async def quiesce(_self: GuiController, _settings: AppSettings) -> None:
        current = tuple(backends.values())
        backends.clear()
        sessions.clear()
        await asyncio.gather(*(item.close() for item in current))

    monkeypatch.setattr(GuiController, "_rebuild_stt_provider", rebuild_self)
    monkeypatch.setattr(GuiController, "set_stt_enabled", enable_self)
    monkeypatch.setattr(GuiController, "_refresh_peer_stt_runtime", refresh_peer)
    monkeypatch.setattr(GuiController, "_quiesce_shared_gpu_consumers", quiesce)

    if initial_failure == "activation":
        await controller.retry_gpu_activation()
        assert factory.modes == ["persistent"]
        assert runtime.state == GpuASRRuntimeState.STOPPED
        assert runtime.active_channels == frozenset()
        assert backends == {}
        assert controller._gpu_manual_retry_channels == channels
        assert view.states[-1][0] == "activation_failed"

    await controller.retry_gpu_activation()

    expected_modes = (
        ["discovery", "persistent"]
        if initial_failure == "unavailable"
        else ["persistent", "persistent"]
    )
    assert factory.modes == expected_modes
    assert controller._gpu_devices == (new_device,)
    assert runtime.state == GpuASRRuntimeState.READY
    assert runtime.active_channels == channels
    assert runtime.configured_device_id == "vk:new"
    assert successful_client.activate_calls == ["vk:new"]
    assert set(backends) == set(channels)
    assert set(sessions) == set(channels)
    assert controller._gpu_manual_retry_channels == frozenset()
    assert controller.settings.stt.gpu_device_id == "vk:new"
    if "self" in channels:
        assert controller.settings.provider.stt == STTProviderName.LOCAL_QWEN_GPU
        assert restores[-2:] == (
            ["self_rebuilt", "self_enabled"]
            if channels == frozenset({"self"})
            else ["peer_rebuilt", "peer_enabled"]
        )
    if "peer" in channels:
        assert controller.settings.provider.peer_stt == STTProviderName.LOCAL_QWEN_GPU
    assert successful_client.close_calls == 0

    await asyncio.gather(*(item.close() for item in tuple(backends.values())))
    await runtime.close()
    assert successful_client.close_calls == 1


async def test_controller_gpu_discovery_and_readiness_delegate_to_owned_runtime(
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
    monkeypatch.setattr(
        GuiController,
        "_get_gpu_asr_runtime",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("legacy Controller GPU runtime was constructed")
        ),
    )

    devices = await controller.ensure_gpu_device_discovery(force=True)
    ready = await controller._validate_gpu_activation()

    assert devices == (device,)
    assert ready is True
    assert owner.discovery_calls == [True]
    assert owner.readiness_calls == [(True, controller.settings.stt.gpu_device_id)]
