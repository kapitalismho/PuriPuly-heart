from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from puripuly_heart.core.local_asr_provisioning import (
    LocalASRInstallRequest,
    LocalASRInstallResult,
)
from puripuly_heart.core.local_gpu_assets import LocalGPUInstallSnapshot
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LOCAL_STT_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    PARAKEET_V3_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
    InstalledLocalSTTManifest,
    LocalSTTInstallState,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_stt_catalog import (
    LocalCPUInstallSnapshot,
    LocalCPUModelInstall,
)
from puripuly_heart.core.local_stt_runtime_installer import RuntimeLocalSTTStatusUpdate
from puripuly_heart.core.runtime.local_asr_provisioning import LocalASRProvisioningOwner
from puripuly_heart.core.runtime.local_stt_download import LocalSTTDownloadRuntime


def _installed(model_id: str) -> InstalledLocalSTTManifest:
    manifest = load_local_stt_asset_manifest(model_id)
    source = next(iter(manifest.sources.values()))
    return InstalledLocalSTTManifest(
        manifest_version=manifest.installed_manifest_version,
        model_id=model_id,
        engine=manifest.engine,
        install_dirname=manifest.install_dirname,
        selected_source=source.name,
        selected_revision=source.revision,
    )


def _state(status: str, model_id: str):
    return LocalSTTInstallState(
        status=status,
        installed_manifest=_installed(model_id) if status == "ready" else None,
    )


@dataclass(slots=True)
class MutableProvisioningBackend:
    states: dict[str, LocalSTTInstallState]
    install_calls: list[str] = field(default_factory=list)
    status_callbacks: list[Callable] = field(default_factory=list)
    install_started: asyncio.Event = field(default_factory=asyncio.Event)
    release_install: asyncio.Event | None = None
    failure: Exception | None = None

    def inspect_cpu(
        self,
        model_ids: tuple[str, ...],
        *_args,
        **_kwargs,
    ) -> LocalCPUInstallSnapshot:
        return LocalCPUInstallSnapshot(
            models=tuple(
                LocalCPUModelInstall(model_id=model_id, state=self.states[model_id])
                for model_id in model_ids
            )
        )

    def inspect_gpu(self, **_kwargs) -> LocalGPUInstallSnapshot:
        state = self.states[LOCAL_QWEN_GPU_MODEL_ID]
        return LocalGPUInstallSnapshot(
            explicit_opt_in=True,
            status=state.status,
            state=state,
        )

    async def install(self, **kwargs) -> InstalledLocalSTTManifest:
        model_id = kwargs["model_id"]
        on_status = kwargs["on_status"]
        self.install_calls.append(model_id)
        self.status_callbacks.append(on_status)
        self.install_started.set()
        await on_status(RuntimeLocalSTTStatusUpdate(status="downloading", percent=37))
        if self.release_install is not None:
            await self.release_install.wait()
        if self.failure is not None:
            raise self.failure
        installed = _installed(model_id)
        self.states[model_id] = LocalSTTInstallState(
            status="ready",
            installed_manifest=installed,
        )
        await on_status(RuntimeLocalSTTStatusUpdate(status="ready", percent=None))
        return installed


def _owner(backend: MutableProvisioningBackend, **kwargs) -> LocalASRProvisioningOwner:
    return LocalASRProvisioningOwner(
        cpu_model_inspector=backend.inspect_cpu,
        gpu_model_inspector=backend.inspect_gpu,
        installer=backend.install,
        **kwargs,
    )


def _all_states(status: str = "ready") -> dict[str, LocalSTTInstallState]:
    return {
        model_id: _state(status, model_id)
        for model_id in (*REQUIRED_CPU_LOCAL_STT_MODEL_IDS, LOCAL_QWEN_GPU_MODEL_ID)
    }


@pytest.mark.asyncio
async def test_owner_snapshot_keeps_independent_cpu_availability_and_cpu_auto_completeness() -> (
    None
):
    states = _all_states()
    states[PARAKEET_JAPANESE_MODEL_ID] = _state("invalid", PARAKEET_JAPANESE_MODEL_ID)
    owner = _owner(MutableProvisioningBackend(states))

    snapshot = await owner.inspect_cpu()

    assert snapshot.cpu_auto_available is False
    assert snapshot.state_for(PARAKEET_V3_MODEL_ID).status == "ready"
    assert snapshot.state_for(LOCAL_STT_MODEL_ID).status == "ready"
    assert snapshot.state_for(PARAKEET_JAPANESE_MODEL_ID).status == "invalid"
    assert snapshot.status_for((LOCAL_STT_MODEL_ID,)) == "ready"
    assert snapshot.unavailable_model_ids(REQUIRED_CPU_LOCAL_STT_MODEL_IDS) == (
        PARAKEET_JAPANESE_MODEL_ID,
    )

    await owner.close()


@pytest.mark.asyncio
async def test_owner_marks_only_the_model_with_invalid_manifest_authority_invalid() -> None:
    backend = MutableProvisioningBackend(_all_states())

    def manifest_loader(model_id: str):
        if model_id == PARAKEET_JAPANESE_MODEL_ID:
            return load_local_stt_asset_manifest(PARAKEET_V3_MODEL_ID)
        return load_local_stt_asset_manifest(model_id)

    owner = _owner(backend, manifest_loader=manifest_loader)

    snapshot = await owner.inspect_cpu()

    assert snapshot.state_for(PARAKEET_JAPANESE_MODEL_ID).status == "invalid"
    assert snapshot.state_for(PARAKEET_V3_MODEL_ID).status == "ready"
    assert snapshot.state_for(LOCAL_STT_MODEL_ID).status == "ready"

    await owner.close()


@pytest.mark.asyncio
async def test_owner_cleans_only_known_stale_install_residue_before_inspection(
    tmp_path,
) -> None:
    manifest = load_local_stt_asset_manifest(LOCAL_STT_MODEL_ID)
    stale_staging = tmp_path / f"{manifest.install_dirname}.staging-abcd"
    stale_backup = tmp_path / f"{manifest.install_dirname}.backup"
    restored_install = tmp_path / manifest.install_dirname
    unrelated = tmp_path / f"{manifest.install_dirname}.evidence"
    stale_staging.mkdir()
    stale_backup.mkdir()
    unrelated.mkdir()
    owner = _owner(MutableProvisioningBackend(_all_states()), model_root=tmp_path)

    await owner.inspect_cpu()

    assert not stale_staging.exists()
    assert not stale_backup.exists()
    assert restored_install.exists()
    assert unrelated.exists()
    assert owner.diagnostics[0].event == "cleanup"
    assert owner.diagnostics[0].outcome == "ready"

    await owner.close()


@pytest.mark.asyncio
async def test_cleanup_failure_is_safely_diagnosed_without_blocking_availability() -> None:
    def fail_cleanup(**_kwargs):
        raise PermissionError("sensitive-path")

    owner = _owner(
        MutableProvisioningBackend(_all_states()),
        residue_cleaner=fail_cleanup,
    )

    snapshot = await owner.inspect_cpu()

    assert snapshot.cpu_auto_available is True
    assert owner.diagnostics[0].event == "cleanup"
    assert owner.diagnostics[0].outcome == "failed"
    assert owner.diagnostics[0].failure_type == "PermissionError"
    assert "sensitive-path" not in repr(owner.diagnostics)

    await owner.close()


@pytest.mark.asyncio
async def test_owner_retries_startup_cleanup_after_another_process_holds_lease() -> None:
    cleanup_calls = 0

    def cleanup(**_kwargs):
        nonlocal cleanup_calls
        cleanup_calls += 1
        return None if cleanup_calls == 1 else ()

    owner = _owner(
        MutableProvisioningBackend(_all_states()),
        residue_cleaner=cleanup,
    )

    await owner.inspect_cpu()
    await owner.inspect_cpu()

    assert cleanup_calls == 2
    assert [item.event for item in owner.diagnostics] == ["cleanup"]

    await owner.close()


@pytest.mark.asyncio
async def test_targeted_cpu_repair_preserves_valid_models_and_reports_progress() -> None:
    states = _all_states()
    states[PARAKEET_JAPANESE_MODEL_ID] = _state("invalid", PARAKEET_JAPANESE_MODEL_ID)
    backend = MutableProvisioningBackend(states)
    snapshots = []
    owner = _owner(backend, state_changed=snapshots.append)
    await owner.inspect_cpu()

    task = owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(PARAKEET_JAPANESE_MODEL_ID,),
            locale="ja",
            origin="manual",
        )
    )
    result = await task

    assert backend.install_calls == [PARAKEET_JAPANESE_MODEL_ID]
    assert result.installed_model_ids == (PARAKEET_JAPANESE_MODEL_ID,)
    assert result.failed_model_ids == ()
    assert result.snapshot.cpu_auto_available is True
    assert result.snapshot.state_for(PARAKEET_V3_MODEL_ID).status == "ready"
    assert result.snapshot.state_for(LOCAL_STT_MODEL_ID).status == "ready"
    assert any(
        snapshot.activity_for("cpu") is not None
        and snapshot.activity_for("cpu").progress_percent == 37
        for snapshot in snapshots
    )
    assert [
        diagnostic.outcome for diagnostic in owner.diagnostics if diagnostic.event == "install"
    ] == ["started", "ready"]

    await owner.close()


@pytest.mark.asyncio
async def test_owner_delivers_install_result_through_owned_task() -> None:
    states = _all_states()
    states[LOCAL_STT_MODEL_ID] = _state("missing", LOCAL_STT_MODEL_ID)
    owner = _owner(MutableProvisioningBackend(states))
    delivered: list[LocalASRInstallResult] = []

    result = await owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(LOCAL_STT_MODEL_ID,),
            locale="en",
            origin="manual",
        ),
        result_handler=lambda current: delivered.append(current),
    )
    for _ in range(20):
        if delivered and not owner.active_result_delivery_task_names:
            break
        await asyncio.sleep(0)

    assert delivered == [result]
    assert owner.active_result_delivery_task_names == ()

    await owner.close()


@pytest.mark.asyncio
async def test_owner_close_cancels_active_install_result_delivery() -> None:
    states = _all_states()
    states[LOCAL_STT_MODEL_ID] = _state("missing", LOCAL_STT_MODEL_ID)
    owner = _owner(MutableProvisioningBackend(states))
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def handle_result(_result: LocalASRInstallResult) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    await owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(LOCAL_STT_MODEL_ID,),
            locale="en",
            origin="manual",
        ),
        result_handler=handle_result,
    )
    await entered.wait()

    assert len(owner.active_result_delivery_task_names) == 1

    await owner.close()

    assert cancelled.is_set()
    assert owner.active_result_delivery_task_names == ()


@pytest.mark.asyncio
async def test_owner_contains_install_result_handler_failure_as_safe_diagnostic() -> None:
    states = _all_states()
    states[LOCAL_STT_MODEL_ID] = _state("missing", LOCAL_STT_MODEL_ID)
    owner = _owner(MutableProvisioningBackend(states))

    def fail(_result: LocalASRInstallResult) -> None:
        raise RuntimeError("secret result payload")

    await owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(LOCAL_STT_MODEL_ID,),
            locale="en",
            origin="manual",
        ),
        result_handler=fail,
    )
    for _ in range(20):
        if any(item.event == "result_delivery" for item in owner.diagnostics):
            break
        await asyncio.sleep(0)

    diagnostic = owner.diagnostics[-1]
    assert diagnostic.event == "result_delivery"
    assert diagnostic.backend == "cpu"
    assert diagnostic.origin == "manual"
    assert diagnostic.failure_type == "RuntimeError"
    assert "secret result payload" not in repr(owner.diagnostics)

    await owner.close()


@pytest.mark.asyncio
async def test_failed_repair_keeps_integrity_and_exposes_only_safe_failure_type() -> None:
    states = _all_states()
    states[LOCAL_STT_MODEL_ID] = _state("missing", LOCAL_STT_MODEL_ID)
    backend = MutableProvisioningBackend(
        states,
        failure=RuntimeError("credential=do-not-publish"),
    )
    owner = _owner(backend)
    await owner.inspect_cpu()

    result = await owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(LOCAL_STT_MODEL_ID,),
            locale="en",
            origin="manual",
        )
    )

    assert result.failed_model_ids == (LOCAL_STT_MODEL_ID,)
    assert result.snapshot.state_for(LOCAL_STT_MODEL_ID).integrity == "missing"
    assert result.snapshot.state_for(LOCAL_STT_MODEL_ID).status == "download_failed"
    assert owner.diagnostics[-1].failure_type == "RuntimeError"
    assert "do-not-publish" not in repr(owner.diagnostics)

    await owner.close()


@pytest.mark.asyncio
async def test_runtime_validation_failure_is_owned_and_safely_diagnosed() -> None:
    owner = _owner(MutableProvisioningBackend(_all_states()))
    await owner.inspect_cpu()

    snapshot = await owner.report_model_validation_failure(
        LOCAL_STT_MODEL_ID,
        failure_type="LocalQwenSherpaLoadError",
    )

    assert snapshot.state_for(LOCAL_STT_MODEL_ID).status == "invalid"
    assert snapshot.state_for(PARAKEET_V3_MODEL_ID).status == "ready"
    assert owner.diagnostics[-1].event == "validation"
    assert owner.diagnostics[-1].failure_type == "LocalQwenSherpaLoadError"

    await owner.close()


@pytest.mark.asyncio
async def test_gpu_install_requires_explicit_intent_and_uses_only_gpu_model() -> None:
    states = _all_states()
    states[LOCAL_QWEN_GPU_MODEL_ID] = _state("missing", LOCAL_QWEN_GPU_MODEL_ID)
    backend = MutableProvisioningBackend(states)
    owner = _owner(backend)

    snapshot = await owner.inspect_gpu(explicit_intent=False)
    assert snapshot.state_for(LOCAL_QWEN_GPU_MODEL_ID).status == "not_requested"

    with pytest.raises(ValueError, match="explicit application intent"):
        owner.start_install(
            LocalASRInstallRequest(
                backend="gpu",
                model_ids=(LOCAL_QWEN_GPU_MODEL_ID,),
                locale="en",
                origin="settings_exit",
            )
        )

    await owner.inspect_gpu(explicit_intent=True)
    result = await owner.start_install(
        LocalASRInstallRequest(
            backend="gpu",
            model_ids=(LOCAL_QWEN_GPU_MODEL_ID,),
            locale="en",
            origin="settings_exit",
            explicit_gpu_intent=True,
        )
    )

    assert backend.install_calls == [LOCAL_QWEN_GPU_MODEL_ID]
    assert result.installed_model_ids == (LOCAL_QWEN_GPU_MODEL_ID,)
    assert result.snapshot.state_for(LOCAL_QWEN_GPU_MODEL_ID).status == "ready"

    await owner.close()


@pytest.mark.asyncio
async def test_cancel_restores_integrity_and_ignores_late_status_from_old_generation() -> None:
    states = _all_states()
    states[LOCAL_STT_MODEL_ID] = _state("missing", LOCAL_STT_MODEL_ID)
    backend = MutableProvisioningBackend(states, release_install=asyncio.Event())
    owner = _owner(backend)
    await owner.inspect_cpu()
    task = owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(LOCAL_STT_MODEL_ID,),
            locale="en",
            origin="manual",
        )
    )
    await backend.install_started.wait()
    old_status_callback = backend.status_callbacks[0]

    await owner.cancel_install("cpu")
    result = await task
    revision_after_cancel = owner.snapshot.revision
    await old_status_callback(RuntimeLocalSTTStatusUpdate(status="downloading", percent=99))

    assert result.cancelled is True
    assert owner.snapshot.activity_for("cpu") is None
    assert owner.snapshot.state_for(LOCAL_STT_MODEL_ID).integrity == "missing"
    assert owner.snapshot.state_for(LOCAL_STT_MODEL_ID).status == "cancelled"
    assert owner.snapshot.revision == revision_after_cancel
    assert owner.diagnostics[-1].outcome == "cancelled"

    await owner.close()


@pytest.mark.asyncio
async def test_owner_can_hold_independent_cpu_and_gpu_install_activities() -> None:
    states = _all_states()
    states[LOCAL_STT_MODEL_ID] = _state("missing", LOCAL_STT_MODEL_ID)
    states[LOCAL_QWEN_GPU_MODEL_ID] = _state("missing", LOCAL_QWEN_GPU_MODEL_ID)
    backend = MutableProvisioningBackend(states, release_install=asyncio.Event())
    owner = _owner(backend)
    cpu = owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(LOCAL_STT_MODEL_ID,),
            locale="en",
            origin="manual",
        )
    )
    gpu = owner.start_install(
        LocalASRInstallRequest(
            backend="gpu",
            model_ids=(LOCAL_QWEN_GPU_MODEL_ID,),
            locale="en",
            origin="settings_exit",
            explicit_gpu_intent=True,
        )
    )
    while len(owner.snapshot.activities) < 2:
        await asyncio.sleep(0)

    assert {activity.backend for activity in owner.snapshot.activities} == {"cpu", "gpu"}

    await owner.close()
    await asyncio.gather(cpu, gpu)

    assert owner.snapshot.closed is True
    assert owner.snapshot.activities == ()
    with pytest.raises(RuntimeError, match="closed"):
        owner.start_install(
            LocalASRInstallRequest(
                backend="cpu",
                model_ids=(LOCAL_STT_MODEL_ID,),
                locale="en",
                origin="manual",
            )
        )


@pytest.mark.asyncio
async def test_owner_close_retains_timed_out_install_cleanup_for_retry() -> None:
    states = _all_states()
    states[LOCAL_STT_MODEL_ID] = _state("missing", LOCAL_STT_MODEL_ID)
    backend = MutableProvisioningBackend(states)
    started = asyncio.Event()
    release = asyncio.Event()

    async def cancellation_resistant_install(**kwargs) -> InstalledLocalSTTManifest:
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        return _installed(kwargs["model_id"])

    owner = LocalASRProvisioningOwner(
        cpu_model_inspector=backend.inspect_cpu,
        gpu_model_inspector=backend.inspect_gpu,
        installer=cancellation_resistant_install,
        download_runtime_factory=lambda: LocalSTTDownloadRuntime(cancel_timeout_s=0.01),
    )
    task = owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(LOCAL_STT_MODEL_ID,),
            locale="en",
            origin="manual",
        )
    )
    await started.wait()

    with pytest.raises(TimeoutError, match="cancellation timed out"):
        await owner.close()

    assert task.done() is False
    assert task in asyncio.all_tasks()
    assert owner.snapshot.closed is True
    assert owner.snapshot.activity_for("cpu") is not None
    assert owner._cpu_install_runtime.download_task is task
    with pytest.raises(RuntimeError, match="closed"):
        owner.start_install(
            LocalASRInstallRequest(
                backend="cpu",
                model_ids=(LOCAL_STT_MODEL_ID,),
                locale="en",
                origin="manual",
            )
        )

    release.set()
    await owner.close()
    result = await task

    assert result.cancelled is True
    assert task.done() is True
    assert owner.snapshot.activities == ()
    assert owner._cpu_install_runtime.download_task is None


def test_owner_lifecycle_inventory_names_all_provisioning_resources() -> None:
    owner = _owner(MutableProvisioningBackend(_all_states()))

    snapshot = owner.lifecycle_owner_snapshot()

    assert snapshot["owner"] == "LocalASRProvisioningOwner"
    assert snapshot["resource_fields"] == (
        "_cpu_install_runtime",
        "_gpu_install_runtime",
        "_result_delivery_tasks",
        "installer cancel events",
        "Xet helper processes",
        "staging and backup directories",
        "model-root cross-process provisioning lease",
    )
    assert "cancel CPU and GPU install and result-delivery tasks" in snapshot["shutdown_policy"]
    assert snapshot["late_callback_rule"] == (
        "ignore stale status generations and drop install-result delivery after close"
    )
