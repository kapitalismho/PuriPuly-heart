from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
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
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("timed out waiting for condition")


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
        self.result_tasks: list[asyncio.Task[None]] = []

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
        *,
        result_handler=None,
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
        if result_handler is not None:

            def schedule_result(completed: asyncio.Task[LocalASRInstallResult]) -> None:
                async def deliver() -> None:
                    result = await completed
                    outcome = result_handler(result)
                    if inspect.isawaitable(outcome):
                        await outcome

                self.result_tasks.append(asyncio.create_task(deliver()))

            task.add_done_callback(schedule_result)
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
        await asyncio.sleep(0)
        for task in self.result_tasks:
            if not task.done():
                task.cancel()
        if self.result_tasks:
            await asyncio.gather(*self.result_tasks, return_exceptions=True)
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
        app=FletUiPresentationAdapter(
            (
                SimpleNamespace(view_dashboard=dashboard)
                if dashboard is not None
                else SimpleNamespace()
            )
        ),
        config_path=Path("settings.json"),
        local_asr_provisioning=port,
    )
    controller.settings = AppSettings()
    return controller


def test_cpu_repair_pending_compatibility_properties_preserve_independent_generation() -> None:
    controller = _controller(RecordingProvisioningPort(_snapshot()))

    controller._local_stt_pending_enable_generation = 17
    controller._local_stt_pending_enable_after_install = True
    controller._local_stt_pending_enable_after_install = False

    assert controller._local_stt_pending_enable_after_install is False
    assert controller._local_stt_pending_enable_generation == 17

    controller._local_stt_pending_enable_after_install = True

    assert controller._local_stt_pending_enable_after_install is True
    assert controller._local_stt_pending_enable_generation == 17

    controller._reset_local_stt_pending_enable_after_install()

    assert controller._local_stt_pending_enable_after_install is False
    assert controller._local_stt_pending_enable_generation is None


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
    controller._get_local_asr_cpu_repair_owner()

    async def rebuild_with_owner_generation(self) -> None:
        rebuilds.append("rebuild")
        self._stt_activation_generation += 1

    monkeypatch.setattr(
        GuiController,
        "_rebuild_stt_provider",
        rebuild_with_owner_generation,
    )
    monkeypatch.setattr(
        GuiController,
        "_ensure_stt_switch",
        lambda self: asyncio.sleep(0, result=switches.append("switch")),
    )

    controller._stt_desired = True
    controller._stt_activation_generation = 7
    assert (
        controller._request_unavailable_local_asr_repair(
            "missing",
            channel="self",
            activation_generation=7,
        )
        is False
    )
    release.set()
    await asyncio.gather(*port.tasks)
    await _wait_until(lambda: bool(rebuilds))

    assert rebuilds == ["rebuild"]
    assert switches == ["switch"]
    assert controller._local_stt_pending_enable_after_install is False


@pytest.mark.asyncio
async def test_cpu_repair_composition_routes_peer_resume_to_runtime_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()
    port = RecordingProvisioningPort(_snapshot(qwen="missing"), release=release)
    controller = _controller(port, dashboard=Dashboard())
    controller.settings.provider.peer_stt = STTProviderName.LOCAL_QWEN
    controller.settings.ui.peer_translation_enabled = True
    controller.settings.ui.peer_translation_eula_accepted = True
    refreshes: list[str] = []
    controller._get_local_asr_cpu_repair_owner()
    monkeypatch.setattr(
        GuiController,
        "_refresh_overlay_runtime_dependencies",
        lambda self: asyncio.sleep(0, result=refreshes.append("refresh")),
    )

    assert await controller._ensure_peer_local_stt_ready() is False
    assert port.requests[0].model_ids == (LOCAL_STT_MODEL_ID,)
    release.set()
    await asyncio.gather(*port.tasks)
    await _wait_until(lambda: bool(refreshes))

    assert refreshes == ["refresh"]
    assert controller._local_stt_pending_peer_enable_after_install is False


@pytest.mark.asyncio
async def test_gpu_provisioning_composition_re_resolves_port_after_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspected = RecordingProvisioningPort(_snapshot(gpu="missing"))
    installed = RecordingProvisioningPort(_snapshot(gpu="missing"))
    controller = _controller(inspected)
    controller.settings.provider.stt = STTProviderName.LOCAL_QWEN_GPU
    inspect_gpu = inspected.inspect_gpu

    async def inspect_and_replace(
        *,
        explicit_intent: bool,
        verify_checksums: bool = False,
    ) -> LocalASRProvisioningSnapshot:
        snapshot = await inspect_gpu(
            explicit_intent=explicit_intent,
            verify_checksums=verify_checksums,
        )
        controller.local_asr_provisioning = installed
        return snapshot

    monkeypatch.setattr(inspected, "inspect_gpu", inspect_and_replace)

    assert await controller.install_selected_gpu_model_if_needed() is True

    assert inspected.gpu_inspections == [(True, False)]
    assert inspected.requests == []
    assert installed.requests == [
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
async def test_gpu_provisioning_composition_preserves_sync_and_awaited_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronous = RecordingProvisioningPort(_snapshot(gpu="missing"))
    synchronous_controller = _controller(synchronous)
    synchronous_logs: list[tuple[str, int, BaseException | None]] = []

    def fail_start_install(*_args, **_kwargs):
        raise RuntimeError("synchronous")

    def record_synchronous_log(
        _controller: GuiController,
        message: str,
        *,
        level: int = logging.DEBUG,
        exception: BaseException | None = None,
    ) -> None:
        synchronous_logs.append((message, level, exception))

    monkeypatch.setattr(synchronous, "start_install", fail_start_install)
    monkeypatch.setattr(
        GuiController,
        "log_detailed",
        record_synchronous_log,
    )

    with pytest.raises(RuntimeError, match="synchronous"):
        await synchronous_controller.install_or_repair_gpu_model()

    assert synchronous_controller._gpu_ui_state == "installing"
    assert synchronous_logs == [
        (
            "[GPU ASR] state=installing origin=manual progress_percent=0",
            logging.DEBUG,
            None,
        )
    ]

    awaited = RecordingProvisioningPort(_snapshot(gpu="missing"))
    awaited_controller = _controller(awaited)
    awaited_logs: list[tuple[str, int, BaseException | None]] = []
    failure = RuntimeError("awaited")

    def start_failed_install(
        request: LocalASRInstallRequest,
        *,
        result_handler=None,
    ) -> asyncio.Task[LocalASRInstallResult]:
        assert result_handler is None
        awaited.requests.append(request)

        async def fail() -> LocalASRInstallResult:
            raise failure

        return asyncio.create_task(fail())

    def record_awaited_log(
        _controller: GuiController,
        message: str,
        *,
        level: int = logging.DEBUG,
        exception: BaseException | None = None,
    ) -> None:
        awaited_logs.append((message, level, exception))

    monkeypatch.setattr(awaited, "start_install", start_failed_install)
    monkeypatch.setattr(
        GuiController,
        "log_detailed",
        record_awaited_log,
    )

    await awaited_controller.install_or_repair_gpu_model(origin="repair")

    failure_logs = [
        item for item in awaited_logs if item[0] == "[GPU ASR] model_install failure=unexpected"
    ]
    assert len(failure_logs) == 1
    message, level, exception = failure_logs[0]
    assert message == "[GPU ASR] model_install failure=unexpected"
    assert level == logging.WARNING
    assert exception is failure
    assert awaited_controller._gpu_ui_state == "install_failed"


@pytest.mark.asyncio
async def test_gpu_provisioning_composition_contains_retry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = RecordingProvisioningPort(_snapshot(gpu="missing"))
    controller = _controller(port)
    controller._gpu_pending_enable_channels = frozenset({"self"})
    failure = RuntimeError("retry")
    diagnostics: list[BaseException | None] = []

    async def fail_retry(_controller: GuiController) -> None:
        raise failure

    def record_diagnostic(
        _controller: GuiController,
        _message: str,
        *,
        level: int = logging.DEBUG,
        exception: BaseException | None = None,
    ) -> None:
        _ = level
        if _message == "[GPU ASR] model_install failure=unexpected":
            diagnostics.append(exception)

    monkeypatch.setattr(GuiController, "retry_gpu_activation", fail_retry)
    monkeypatch.setattr(
        GuiController,
        "log_detailed",
        record_diagnostic,
    )

    await controller.install_or_repair_gpu_model()

    assert diagnostics == [failure]
    assert diagnostics[0] is failure
    assert controller._gpu_ui_state == "install_failed"


@pytest.mark.asyncio
async def test_gpu_provisioning_composition_propagates_retry_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = RecordingProvisioningPort(_snapshot(gpu="missing"))
    controller = _controller(port)
    controller._gpu_pending_enable_channels = frozenset({"peer"})

    async def cancel_retry(_controller: GuiController) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(GuiController, "retry_gpu_activation", cancel_retry)

    with pytest.raises(asyncio.CancelledError):
        await controller.install_or_repair_gpu_model()

    assert controller._gpu_ui_state == "installed"


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
