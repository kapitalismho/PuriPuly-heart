from __future__ import annotations

import asyncio
import inspect
from dataclasses import replace

import pytest
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeChannelSnapshot,
    ProviderRuntimeGpuSnapshot,
)
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRInstallRequest,
    LocalASRInstallResult,
    LocalASRModelProvisioningState,
    LocalASRProvisioningActivity,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_stt_assets import LOCAL_QWEN_GPU_MODEL_ID

from puripuly_heart.app.services.gpu_runtime_interaction import (
    GpuRuntimeInteractionOwner,
    GpuRuntimeInteractionState,
)
from puripuly_heart.core.gpu_worker import GpuWorkerDevice


def _gpu_snapshot(
    *, phase: str, failure_code: str | None = None
) -> LocalASRProviderRuntimeSnapshot:
    channels = tuple(
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
    )
    device = GpuWorkerDevice(
        device_id="vulkan-index-0",
        registry_index=0,
        name="GPU 0",
        description="GPU 0",
        device_type="discrete",
        memory_total_bytes=8_000_000_000,
        memory_free_bytes=4_000_000_000,
    )
    return LocalASRProviderRuntimeSnapshot(
        channels=channels,
        gpu=ProviderRuntimeGpuSnapshot(
            phase=phase,
            devices=(device,),
            active_channels=frozenset(),
            pending_count=0,
            worker_pid=None,
            configured_device_id=None,
            model_resident=False,
            retry_required=False,
            failure_code=failure_code,
        ),
    )


class PhaseRuntime:
    def __init__(self, phase: str, *, failure_code: str | None = None) -> None:
        self.phase = phase
        self.failure_code = failure_code

    async def inspect_gpu_readiness(
        self,
        *,
        explicit_intent: bool,
        device_id: str,
    ) -> LocalASRProviderRuntimeSnapshot:
        assert explicit_intent is True
        assert device_id == "auto"
        return _gpu_snapshot(phase=self.phase, failure_code=self.failure_code)


class RecordingProvisioning:
    def __init__(self) -> None:
        self.requests: list[LocalASRInstallRequest] = []
        self._snapshot = LocalASRProvisioningSnapshot(
            models=(),
            required_cpu_model_ids=(),
            gpu_model_id=LOCAL_QWEN_GPU_MODEL_ID,
            activities=(),
        )

    @property
    def snapshot(self) -> LocalASRProvisioningSnapshot:
        return self._snapshot

    def start_install(
        self,
        request: LocalASRInstallRequest,
        *,
        result_handler=None,
    ) -> asyncio.Task[LocalASRInstallResult]:
        self.requests.append(request)

        async def finish() -> LocalASRInstallResult:
            result = LocalASRInstallResult(
                request=request,
                installed_model_ids=(LOCAL_QWEN_GPU_MODEL_ID,),
                failed_model_ids=(),
                cancelled=False,
                snapshot=self._snapshot,
            )
            if result_handler is not None:
                outcome = result_handler(result)
                if inspect.isawaitable(outcome):
                    await outcome
            return result

        return asyncio.create_task(finish())


def _owner(
    runtime: PhaseRuntime,
    provisioning: RecordingProvisioning | None = None,
) -> tuple[GpuRuntimeInteractionOwner, list[object]]:
    presentations: list[object] = []

    async def retry_activation() -> None:
        return None

    owner = GpuRuntimeInteractionOwner(
        runtime_provider=lambda: runtime,
        provisioning_provider=(
            (lambda: provisioning)
            if provisioning is not None
            else (lambda: pytest.fail("provisioning should not be queried directly"))
        ),
        state_provider=lambda: GpuRuntimeInteractionState(
            settings_available=True,
            selected_provider_requires_model=True,
            locale="ko",
            device_id="auto",
        ),
        presentation_sink=presentations.append,
        diagnostic_log_sink=lambda _message: None,
        retry_activation=retry_activation,
    )
    return owner, presentations


@pytest.mark.asyncio
async def test_validate_activation_keeps_active_download_in_installing_state() -> None:
    owner, presentations = _owner(PhaseRuntime("installing", failure_code="downloading"))

    assert await owner.validate_activation() is False
    assert owner.snapshot.ui_state == "installing"
    assert presentations[-1].state == "installing"
    assert presentations[-1].notice is not None
    assert presentations[-1].notice.status == "installing"
    assert presentations[-1].publish_notice is True


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["not_installed", "invalid"])
async def test_validate_activation_starts_install_when_model_is_missing(
    phase: str,
) -> None:
    provisioning = RecordingProvisioning()
    owner, presentations = _owner(PhaseRuntime(phase), provisioning)

    assert await owner.validate_activation() is False
    assert owner.snapshot.ui_state == "installing"
    assert presentations[-1].state == "installing"
    await asyncio.sleep(0)
    assert provisioning.requests == [
        LocalASRInstallRequest(
            backend="gpu",
            model_ids=(LOCAL_QWEN_GPU_MODEL_ID,),
            locale="ko",
            origin="activation",
            explicit_gpu_intent=True,
        )
    ]
    await asyncio.sleep(0)


@pytest.mark.parametrize(
    ("integrity", "operation", "expected_state", "expected_notice"),
    [
        ("ready", "idle", "installed", None),
        ("missing", "download_failed", "install_failed", "install_failed"),
        ("missing", "cancelled", "not_installed", None),
        ("ready", "cancelled", "installed", None),
        ("invalid", "cancelled", "invalid", None),
    ],
)
def test_model_manager_gpu_download_settles_dashboard_notice(
    integrity: str,
    operation: str,
    expected_state: str,
    expected_notice: str | None,
) -> None:
    owner, presentations = _owner(PhaseRuntime("inactive"))
    model = LocalASRModelProvisioningState(
        model_id=LOCAL_QWEN_GPU_MODEL_ID,
        backend="gpu",
        integrity="missing",
        operation="downloading",
    )
    snapshot = LocalASRProvisioningSnapshot(
        models=(model,),
        required_cpu_model_ids=(),
        gpu_model_id=LOCAL_QWEN_GPU_MODEL_ID,
    )
    for percent in (0, 37, 100):
        owner.observe_provisioning(
            replace(
                snapshot,
                activities=(
                    LocalASRProvisioningActivity(
                        backend="gpu",
                        model_id=LOCAL_QWEN_GPU_MODEL_ID,
                        origin="model_assets",
                        progress_percent=percent,
                        generation=1,
                    ),
                ),
            )
        )
        assert presentations[-1].notice.progress_percent == percent

    owner.observe_provisioning(
        replace(snapshot, models=(replace(model, integrity=integrity, operation=operation),))
    )

    assert presentations[-1].state == expected_state
    notice = presentations[-1].notice
    assert (notice.status if notice is not None else None) == expected_notice
    assert presentations[-1].publish_notice is (expected_notice is not None)
