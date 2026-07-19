from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from puripuly_heart.config.settings import AppSettings, STTProviderName
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRInstallRequest,
    LocalASRInstallResult,
    LocalASRModelProvisioningState,
    LocalASRProvisioningActivity,
    LocalASRProvisioningDiagnostic,
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


def _snapshot(
    *,
    parakeet_v3: str = "ready",
    parakeet_ja: str = "ready",
    qwen: str = "ready",
    gpu: str = "not_requested",
    activities: tuple[LocalASRProvisioningActivity, ...] = (),
    closed: bool = False,
) -> LocalASRProvisioningSnapshot:
    states = (
        LocalASRModelProvisioningState(PARAKEET_V3_MODEL_ID, "cpu", parakeet_v3),
        LocalASRModelProvisioningState(PARAKEET_JAPANESE_MODEL_ID, "cpu", parakeet_ja),
        LocalASRModelProvisioningState(LOCAL_STT_MODEL_ID, "cpu", qwen),
        LocalASRModelProvisioningState(LOCAL_QWEN_GPU_MODEL_ID, "gpu", gpu),
    )
    return LocalASRProvisioningSnapshot(
        models=states,
        required_cpu_model_ids=REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
        gpu_model_id=LOCAL_QWEN_GPU_MODEL_ID,
        activities=activities,
        closed=closed,
    )


class RecordingProvisioningPort:
    def __init__(
        self,
        snapshot: LocalASRProvisioningSnapshot,
        *,
        release: asyncio.Event | None = None,
        fail: bool = False,
    ) -> None:
        self._snapshot = snapshot
        self.release = release
        self.fail = fail
        self.requests: list[LocalASRInstallRequest] = []
        self.cpu_inspections: list[tuple[tuple[str, ...] | None, bool]] = []
        self.gpu_inspections: list[tuple[bool, bool]] = []
        self.validation_failures: list[tuple[str, str]] = []
        self.close_calls = 0
        self.tasks: list[asyncio.Task[LocalASRInstallResult]] = []

    @property
    def snapshot(self) -> LocalASRProvisioningSnapshot:
        return self._snapshot

    @property
    def diagnostics(self) -> tuple[LocalASRProvisioningDiagnostic, ...]:
        return ()

    async def inspect_cpu(
        self,
        model_ids: tuple[str, ...] | None = None,
        *,
        verify_checksums: bool = False,
    ) -> LocalASRProvisioningSnapshot:
        self.cpu_inspections.append((model_ids, verify_checksums))
        return self._snapshot

    async def inspect_gpu(
        self,
        *,
        explicit_intent: bool,
        verify_checksums: bool = False,
    ) -> LocalASRProvisioningSnapshot:
        self.gpu_inspections.append((explicit_intent, verify_checksums))
        return self._snapshot

    def start_install(
        self,
        request: LocalASRInstallRequest,
    ) -> asyncio.Task[LocalASRInstallResult]:
        if self._snapshot.activity_for(request.backend) is not None:
            raise RuntimeError("install already active")
        self.requests.append(request)
        activity = LocalASRProvisioningActivity(
            backend=request.backend,
            model_id=request.model_ids[0],
            origin=request.origin,
            progress_percent=0,
            generation=len(self.requests),
        )
        models = tuple(
            (
                replace(model, operation="downloading")
                if model.model_id in request.model_ids
                else model
            )
            for model in self._snapshot.models
        )
        self._snapshot = replace(
            self._snapshot,
            models=models,
            activities=(*self._snapshot.activities, activity),
        )
        task = asyncio.create_task(self._finish(request))
        self.tasks.append(task)
        return task

    async def _finish(self, request: LocalASRInstallRequest) -> LocalASRInstallResult:
        try:
            if self.release is not None:
                await self.release.wait()
        except asyncio.CancelledError:
            self._snapshot = replace(
                self._snapshot,
                activities=tuple(
                    activity
                    for activity in self._snapshot.activities
                    if activity.backend != request.backend
                ),
            )
            return LocalASRInstallResult(
                request=request,
                installed_model_ids=(),
                failed_model_ids=(),
                cancelled=True,
                snapshot=self._snapshot,
            )
        failed = request.model_ids if self.fail else ()
        installed = () if self.fail else request.model_ids
        models = tuple(
            (
                replace(
                    model,
                    integrity=(model.integrity if self.fail else "ready"),
                    operation=("download_failed" if self.fail else "idle"),
                )
                if model.model_id in request.model_ids
                else model
            )
            for model in self._snapshot.models
        )
        self._snapshot = replace(
            self._snapshot,
            models=models,
            activities=tuple(
                activity
                for activity in self._snapshot.activities
                if activity.backend != request.backend
            ),
        )
        return LocalASRInstallResult(
            request=request,
            installed_model_ids=installed,
            failed_model_ids=failed,
            cancelled=False,
            snapshot=self._snapshot,
        )

    async def report_model_validation_failure(
        self,
        model_id: str,
        *,
        failure_type: str,
    ) -> LocalASRProvisioningSnapshot:
        self.validation_failures.append((model_id, failure_type))
        self._snapshot = replace(
            self._snapshot,
            models=tuple(
                replace(model, integrity="invalid") if model.model_id == model_id else model
                for model in self._snapshot.models
            ),
        )
        return self._snapshot

    async def cancel_install(self, backend: str) -> None:
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self._snapshot = replace(
            self._snapshot,
            activities=tuple(
                activity for activity in self._snapshot.activities if activity.backend != backend
            ),
        )

    async def close(self) -> None:
        if self._snapshot.closed:
            return
        self.close_calls += 1
        await self.cancel_install("cpu")
        await self.cancel_install("gpu")
        self._snapshot = replace(self._snapshot, closed=True)


class Dashboard:
    def __init__(self) -> None:
        self.enabled: list[bool] = []
        self.notice_models: list[str | None] = []
        self.notices: list[tuple[str | None, int | None]] = []

    def set_stt_enabled(self, enabled: bool) -> None:
        self.enabled.append(enabled)

    def set_stt_needs_key(self, needs_key: bool) -> None:
        _ = needs_key

    def set_stt_starting(self, starting: bool) -> None:
        _ = starting

    def set_local_stt_notice_model(self, model_id: str | None) -> None:
        self.notice_models.append(model_id)

    def set_local_stt_notice(self, status: str | None, *, percent: int | None = None) -> None:
        self.notices.append((status, percent))


def _controller(
    port: RecordingProvisioningPort,
    *,
    dashboard: Dashboard | None = None,
) -> GuiController:
    controller = GuiController(
        page=SimpleNamespace(),
        app=(
            SimpleNamespace(view_dashboard=dashboard)
            if dashboard is not None
            else SimpleNamespace()
        ),
        config_path=Path("settings.json"),
        local_asr_provisioning=port,
    )
    controller.settings = AppSettings()
    return controller


@pytest.mark.asyncio
async def test_missing_direct_cpu_enable_requests_exact_model_through_owner() -> None:
    release = asyncio.Event()
    port = RecordingProvisioningPort(_snapshot(qwen="missing"), release=release)
    dashboard = Dashboard()
    controller = _controller(port, dashboard=dashboard)
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN

    await controller.set_stt_enabled(True)

    assert port.requests == [
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(LOCAL_STT_MODEL_ID,),
            locale=controller.settings.ui.locale,
            origin="manual",
        )
    ]
    assert controller._local_stt_pending_enable_after_install is True
    assert dashboard.enabled[-1] is False

    await port.close()


@pytest.mark.asyncio
async def test_repeated_enable_during_install_is_single_flight() -> None:
    release = asyncio.Event()
    port = RecordingProvisioningPort(_snapshot(qwen="missing"), release=release)
    controller = _controller(port, dashboard=Dashboard())
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN

    await controller.set_stt_enabled(True)
    await controller.set_stt_enabled(True)

    assert len(port.requests) == 1

    await port.close()


@pytest.mark.asyncio
async def test_successful_manual_repair_resumes_only_current_self_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()
    port = RecordingProvisioningPort(_snapshot(qwen="missing"), release=release)
    controller = _controller(port, dashboard=Dashboard())
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN
    rebuilds: list[str] = []
    switches: list[str] = []
    monkeypatch.setattr(
        GuiController,
        "_rebuild_stt_provider",
        lambda self: asyncio.sleep(0, result=rebuilds.append("rebuild")),
    )
    monkeypatch.setattr(
        GuiController,
        "_ensure_stt_switch",
        lambda self: asyncio.sleep(0, result=switches.append("switch")),
    )

    await controller.set_stt_enabled(True)
    release.set()
    await asyncio.gather(*port.tasks)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert rebuilds == ["rebuild"]
    assert switches == ["switch"]
    assert controller._local_stt_pending_enable_after_install is False


@pytest.mark.asyncio
async def test_provider_switch_before_repair_completion_suppresses_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()
    port = RecordingProvisioningPort(_snapshot(qwen="missing"), release=release)
    controller = _controller(port, dashboard=Dashboard())
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN
    rebuilds: list[str] = []
    monkeypatch.setattr(
        GuiController,
        "_rebuild_stt_provider",
        lambda self: asyncio.sleep(0, result=rebuilds.append("rebuild")),
    )

    await controller.set_stt_enabled(True)
    controller.settings.provider.stt = STTProviderName.DEEPGRAM
    release.set()
    await asyncio.gather(*port.tasks)
    await asyncio.sleep(0)

    assert rebuilds == []
    assert controller._local_stt_pending_enable_after_install is False


@pytest.mark.asyncio
async def test_peer_disable_before_repair_completion_suppresses_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()
    port = RecordingProvisioningPort(_snapshot(qwen="missing"), release=release)
    controller = _controller(port, dashboard=Dashboard())
    controller.settings.provider.peer_stt = STTProviderName.LOCAL_QWEN
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    refreshes: list[str] = []
    monkeypatch.setattr(
        GuiController,
        "_refresh_overlay_runtime_dependencies",
        lambda self: asyncio.sleep(0, result=refreshes.append("refresh")),
    )

    assert await controller._ensure_peer_local_stt_ready() is False
    assert port.requests[0].model_ids == (LOCAL_STT_MODEL_ID,)
    controller.settings.ui.peer_translation_enabled = False
    release.set()
    await asyncio.gather(*port.tasks)
    await asyncio.sleep(0)

    assert refreshes == []
    assert controller._local_stt_pending_peer_enable_after_install is False


@pytest.mark.asyncio
async def test_selected_gpu_install_uses_explicit_owner_intent() -> None:
    port = RecordingProvisioningPort(_snapshot(gpu="missing"))
    controller = _controller(port)
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU

    assert await controller.install_selected_gpu_model_if_needed() is True

    assert port.gpu_inspections == [(True, False)]
    assert port.requests == [
        LocalASRInstallRequest(
            backend="gpu",
            model_ids=(LOCAL_QWEN_GPU_MODEL_ID,),
            locale=controller.settings.ui.locale,
            origin="settings_exit",
            explicit_gpu_intent=True,
        )
    ]
    assert controller._gpu_ui_state == "installed"


@pytest.mark.asyncio
async def test_gpu_install_retries_only_channels_still_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()
    port = RecordingProvisioningPort(_snapshot(gpu="missing"), release=release)
    controller = _controller(port)
    controller._gpu_pending_enable_channels = frozenset({"self"})
    retries: list[str] = []
    monkeypatch.setattr(
        GuiController,
        "retry_gpu_activation",
        lambda self: asyncio.sleep(0, result=retries.append("retry")),
    )

    install = asyncio.create_task(controller.install_or_repair_gpu_model())
    await asyncio.sleep(0)
    release.set()
    await install

    assert retries == ["retry"]

    release = asyncio.Event()
    port = RecordingProvisioningPort(_snapshot(gpu="missing"), release=release)
    controller = _controller(port)
    controller._gpu_pending_enable_channels = frozenset({"self"})
    monkeypatch.setattr(
        GuiController,
        "retry_gpu_activation",
        lambda self: asyncio.sleep(0, result=retries.append("late-retry")),
    )
    install = asyncio.create_task(controller.install_or_repair_gpu_model())
    await asyncio.sleep(0)
    controller._gpu_pending_enable_channels = frozenset()
    release.set()
    await install

    assert retries == ["retry"]


def test_provisioning_snapshot_is_localized_only_at_ui_projection() -> None:
    activity = LocalASRProvisioningActivity(
        backend="cpu",
        model_id=PARAKEET_JAPANESE_MODEL_ID,
        origin="manual",
        progress_percent=42,
        generation=1,
    )
    port = RecordingProvisioningPort(
        _snapshot(
            parakeet_ja="missing",
            activities=(activity,),
        )
    )
    port._snapshot = replace(
        port.snapshot,
        models=tuple(
            (
                replace(model, operation="downloading")
                if model.model_id == PARAKEET_JAPANESE_MODEL_ID
                else model
            )
            for model in port.snapshot.models
        ),
    )
    dashboard = Dashboard()
    controller = _controller(port, dashboard=dashboard)
    controller.settings.provider.stt = STTProviderName.LOCAL_CPU_AUTO

    controller._on_local_asr_provisioning_state_changed(port.snapshot)

    assert dashboard.notice_models[-1] == PARAKEET_JAPANESE_MODEL_ID
    assert dashboard.notices[-1] == ("downloading", 42)


@pytest.mark.asyncio
async def test_controller_shutdown_awaits_idempotent_provisioning_owner_close() -> None:
    port = RecordingProvisioningPort(_snapshot())
    controller = _controller(port)

    await controller._close_local_asr_provisioning()
    await controller._close_local_asr_provisioning()

    assert port.close_calls == 1
    assert port.snapshot.closed is True
