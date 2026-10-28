from __future__ import annotations

import asyncio
import errno
import hashlib
import io
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.core import feature_diagnostics
from puripuly_heart.core.diagnostics import diagnostic_operation
from puripuly_heart.core.local_asr import local_stt_runtime_installer as installer_module
from puripuly_heart.core.local_asr.local_asr_provisioning import (
    LocalASRInstallRequest,
    LocalASRInstallResult,
)
from puripuly_heart.core.local_asr.local_gpu_assets import LocalGPUInstallSnapshot
from puripuly_heart.core.local_asr.local_stt_assets import (
    FUN_ASR_NANO_MODEL_ID,
    LOCAL_QWEN_GPU_MODEL_ID,
    ORUKEET_MODEL_ID,
    PARAKEET_JAPANESE_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
    InstalledLocalSTTManifest,
    LocalSTTAssetFile,
    LocalSTTInstallState,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_asr.local_stt_catalog import (
    LocalCPUInstallSnapshot,
    LocalCPUModelInstall,
)
from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadProgress,
    LocalSTTDownloadPortCancelled,
)
from puripuly_heart.core.local_asr.local_stt_runtime_installer import (
    LocalSTTProvisioningLease,
    LocalSTTRuntimeInstallCancelled,
    LocalSTTRuntimeInstallError,
    RuntimeLocalSTTStatusUpdate,
)
from puripuly_heart.core.runtime import local_asr_provisioning as provisioning_module
from puripuly_heart.core.runtime.local_asr_provisioning import LocalASRProvisioningOwner
from puripuly_heart.core.runtime.local_stt_download import LocalSTTDownloadRuntime
from puripuly_heart.core.runtime_logging import SessionRuntimeLoggingService, configure_main_logging


@pytest.fixture(autouse=True)
def isolated_provisioning_root(tmp_path, monkeypatch):
    root = tmp_path / "isolated-models"
    monkeypatch.setattr(provisioning_module, "default_local_stt_model_root", lambda: root)
    monkeypatch.setattr(installer_module, "default_local_stt_model_root", lambda: root)


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
    assert snapshot.state_for(ORUKEET_MODEL_ID).status == "ready"
    assert snapshot.state_for(FUN_ASR_NANO_MODEL_ID).status == "ready"
    assert snapshot.state_for(PARAKEET_JAPANESE_MODEL_ID).status == "invalid"
    assert snapshot.status_for((FUN_ASR_NANO_MODEL_ID,)) == "ready"
    assert snapshot.unavailable_model_ids(REQUIRED_CPU_LOCAL_STT_MODEL_IDS) == (
        PARAKEET_JAPANESE_MODEL_ID,
    )

    await owner.close()


@pytest.mark.asyncio
async def test_owner_marks_only_the_model_with_invalid_manifest_authority_invalid() -> None:
    backend = MutableProvisioningBackend(_all_states())

    def manifest_loader(model_id: str):
        if model_id == PARAKEET_JAPANESE_MODEL_ID:
            return load_local_stt_asset_manifest(ORUKEET_MODEL_ID)
        return load_local_stt_asset_manifest(model_id)

    owner = _owner(backend, manifest_loader=manifest_loader)

    snapshot = await owner.inspect_cpu()

    assert snapshot.state_for(PARAKEET_JAPANESE_MODEL_ID).status == "invalid"
    assert snapshot.state_for(ORUKEET_MODEL_ID).status == "ready"
    assert snapshot.state_for(FUN_ASR_NANO_MODEL_ID).status == "ready"

    await owner.close()


@pytest.mark.asyncio
async def test_owner_cleans_only_known_stale_install_residue_before_inspection(
    tmp_path,
) -> None:
    manifest = load_local_stt_asset_manifest(FUN_ASR_NANO_MODEL_ID)
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
    assert result.snapshot.state_for(ORUKEET_MODEL_ID).status == "ready"
    assert result.snapshot.state_for(FUN_ASR_NANO_MODEL_ID).status == "ready"
    progress = [
        snapshot.activity_for("cpu").progress_percent
        for snapshot in snapshots
        if snapshot.activity_for("cpu") is not None
    ]
    assert progress[-1] == 37
    assert all(percent is not None for percent in progress)
    assert result.snapshot.activities == ()
    assert [
        diagnostic.outcome for diagnostic in owner.diagnostics if diagnostic.event == "install"
    ] == ["started", "ready"]

    await owner.close()


@pytest.mark.asyncio
async def test_owner_delivers_install_result_through_owned_task() -> None:
    states = _all_states()
    states[FUN_ASR_NANO_MODEL_ID] = _state("missing", FUN_ASR_NANO_MODEL_ID)
    owner = _owner(MutableProvisioningBackend(states))
    delivered: list[LocalASRInstallResult] = []

    result = await owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(FUN_ASR_NANO_MODEL_ID,),
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
    states[FUN_ASR_NANO_MODEL_ID] = _state("missing", FUN_ASR_NANO_MODEL_ID)
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
            model_ids=(FUN_ASR_NANO_MODEL_ID,),
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
    states[FUN_ASR_NANO_MODEL_ID] = _state("missing", FUN_ASR_NANO_MODEL_ID)
    owner = _owner(MutableProvisioningBackend(states))

    def fail(_result: LocalASRInstallResult) -> None:
        raise RuntimeError("secret result payload")

    await owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(FUN_ASR_NANO_MODEL_ID,),
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
    states[FUN_ASR_NANO_MODEL_ID] = _state("missing", FUN_ASR_NANO_MODEL_ID)
    backend = MutableProvisioningBackend(
        states,
        failure=RuntimeError("credential=do-not-publish"),
    )
    owner = _owner(backend)
    await owner.inspect_cpu()

    result = await owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(FUN_ASR_NANO_MODEL_ID,),
            locale="en",
            origin="manual",
        )
    )

    assert result.failed_model_ids == (FUN_ASR_NANO_MODEL_ID,)
    assert result.snapshot.state_for(FUN_ASR_NANO_MODEL_ID).integrity == "missing"
    assert result.snapshot.state_for(FUN_ASR_NANO_MODEL_ID).status == "download_failed"
    assert owner.diagnostics[-1].failure_type == "RuntimeError"
    assert "do-not-publish" not in repr(owner.diagnostics)

    await owner.close()


@pytest.mark.parametrize("failure_kind", ["http", "unknown", "cancelled"])
@pytest.mark.asyncio
async def test_install_failure_retains_admission_identity_and_safe_native_cause(
    tmp_path, monkeypatch, failure_kind
) -> None:
    private = "private-model-body-key-path-237"
    if failure_kind == "http":
        response = httpx.Response(
            503, request=httpx.Request("GET", f"https://example.invalid/{private}")
        )
        native = httpx.HTTPStatusError(private, request=response.request, response=response)
        failure = LocalSTTRuntimeInstallError(private)
        failure.__cause__ = native
    elif failure_kind == "cancelled":
        failure = LocalSTTRuntimeInstallCancelled(private)
    else:
        failure = RuntimeError(f"status=503 auth timeout {private}")
    states = _all_states()
    states[FUN_ASR_NANO_MODEL_ID] = _state("missing", FUN_ASR_NANO_MODEL_ID)
    release = asyncio.Event()
    backend = MutableProvisioningBackend(states, release_install=release, failure=failure)
    owner = _owner(backend, model_root=tmp_path / "models")
    root = logging.getLogger(f"test.model-diagnostic.{uuid4().hex}")
    root.propagate = False
    sinks = configure_main_logging(root_logger=root, log_dir=tmp_path / "logs")
    basic = io.StringIO()
    sinks.stream_handler.setStream(basic)
    service = SessionRuntimeLoggingService(root_logger=root, sinks=sinks)
    instance, operation, request = str(uuid4()), uuid4().hex, uuid4().hex
    events = []
    service.configure_diagnostics(instance_id=instance, event_sink=events.append)
    monkeypatch.setattr(provisioning_module, "logger", root)
    try:
        with diagnostic_operation(instance, operation, request):
            task = owner.start_install(
                LocalASRInstallRequest(
                    backend="cpu",
                    model_ids=(FUN_ASR_NANO_MODEL_ID,),
                    locale="en",
                    origin="application_control",
                )
            )
        await backend.install_started.wait()
        generation = owner.snapshot.activity_for("cpu").generation
        unrelated_operation = uuid4().hex
        with diagnostic_operation(instance, unrelated_operation, uuid4().hex):
            release.set()
            result = await task
        records = service.diagnostic_history({"operation_id": operation})["records"]
        assert service.diagnostic_history({"operation_id": unrelated_operation})["records"] == []
        assert result.snapshot.state_for(FUN_ASR_NANO_MODEL_ID).integrity == "missing"
        if failure_kind == "cancelled":
            assert result.cancelled is True
            assert result.failed_model_ids == ()
            assert records == []
            assert events == []
        else:
            assert result.cancelled is False
            assert result.failed_model_ids == (FUN_ASR_NANO_MODEL_ID,)
            assert result.snapshot.state_for(FUN_ASR_NANO_MODEL_ID).status == "download_failed"
            assert len(records) == 1
            record = records[0]
            assert record["component"] == "local_asr_provisioning"
            assert record["operation"] == "models.install"
            assert record["stage"] == "installation"
            assert record["code"] == "local_asr_install_failed"
            assert record["disposition"] == "failed"
            assert record["request_id"] == request
            assert record["generation"] == generation
            assert record["fields"]["model_id"] == FUN_ASR_NANO_MODEL_ID
            assert record["category"] == (
                "service_unavailable" if failure_kind == "http" else "unknown"
            )
            assert record["status_code"] == (503 if failure_kind == "http" else None)
            assert events[0]["diagnostic"] == record
        assert basic.getvalue() == ""
        assert private not in json.dumps(records)
        assert private not in json.dumps(events)
    finally:
        await owner.close()
        service.close()
        sinks.close(force=True)
        root.handlers.clear()
    if failure_kind == "cancelled":
        return
    file_text = sinks.log_file.read_text(encoding="utf-8")
    persisted = [
        json.loads(line.partition("[Failure] ")[2])
        for line in file_text.splitlines()
        if "[Failure] {" in line
    ]
    assert persisted == records
    assert private not in file_text
    assert str(tmp_path) not in file_text


@pytest.mark.asyncio
async def test_runtime_validation_failure_is_owned_and_safely_diagnosed() -> None:
    owner = _owner(MutableProvisioningBackend(_all_states()))
    await owner.inspect_cpu()

    snapshot = await owner.report_model_validation_failure(
        FUN_ASR_NANO_MODEL_ID,
        failure_type="LocalSTTManifestInvalidError",
    )

    assert snapshot.state_for(FUN_ASR_NANO_MODEL_ID).status == "invalid"
    assert snapshot.state_for(ORUKEET_MODEL_ID).status == "ready"
    assert owner.diagnostics[-1].event == "validation"
    assert owner.diagnostics[-1].failure_type == "LocalSTTManifestInvalidError"

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
    states[FUN_ASR_NANO_MODEL_ID] = _state("missing", FUN_ASR_NANO_MODEL_ID)
    backend = MutableProvisioningBackend(states, release_install=asyncio.Event())
    owner = _owner(backend)
    await owner.inspect_cpu()
    task = owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(FUN_ASR_NANO_MODEL_ID,),
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
    assert owner.snapshot.state_for(FUN_ASR_NANO_MODEL_ID).integrity == "missing"
    assert owner.snapshot.state_for(FUN_ASR_NANO_MODEL_ID).status == "cancelled"
    assert owner.snapshot.revision == revision_after_cancel
    assert owner.diagnostics[-1].outcome == "cancelled"

    await owner.close()


@pytest.mark.asyncio
async def test_owner_can_hold_independent_cpu_and_gpu_install_activities() -> None:
    states = _all_states()
    states[FUN_ASR_NANO_MODEL_ID] = _state("missing", FUN_ASR_NANO_MODEL_ID)
    states[LOCAL_QWEN_GPU_MODEL_ID] = _state("missing", LOCAL_QWEN_GPU_MODEL_ID)
    backend = MutableProvisioningBackend(states, release_install=asyncio.Event())
    owner = _owner(backend)
    cpu = owner.start_install(
        LocalASRInstallRequest(
            backend="cpu",
            model_ids=(FUN_ASR_NANO_MODEL_ID,),
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
                model_ids=(FUN_ASR_NANO_MODEL_ID,),
                locale="en",
                origin="manual",
            )
        )


@pytest.mark.asyncio
async def test_owner_close_retains_timed_out_install_cleanup_for_retry() -> None:
    states = _all_states()
    states[FUN_ASR_NANO_MODEL_ID] = _state("missing", FUN_ASR_NANO_MODEL_ID)
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
            model_ids=(FUN_ASR_NANO_MODEL_ID,),
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
                model_ids=(FUN_ASR_NANO_MODEL_ID,),
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


class _HeldProvisioningDownload:
    def __init__(self, payload: bytes, cleanup_failure: Exception | None) -> None:
        self.payload = payload
        self.cleanup_failure = cleanup_failure
        self.release = asyncio.Event()
        self.cancellation_requested = asyncio.Event()
        self.active = 0
        self.requests = []

    async def download(self, request, *, cancel_event, on_progress):
        self.requests.append(request)
        self.active += 1
        path = request.local_dir / request.remote_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload[:1])
        on_progress(HuggingFaceDownloadProgress(1, len(self.payload)))
        try:
            try:
                while not self.release.is_set():
                    if cancel_event.is_set():
                        self.cancellation_requested.set()
                        await self.release.wait()
                        raise LocalSTTDownloadPortCancelled("controlled cancellation")
                    await asyncio.sleep(0.005)
            except asyncio.CancelledError:
                self.cancellation_requested.set()
                await self.release.wait()
                raise
            path.write_bytes(self.payload)
            return path
        finally:
            self.active -= 1
            if self.cleanup_failure is not None:
                raise self.cleanup_failure


async def _wait_for_condition(predicate) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["observer", "grouped_observer", "cancel"])
@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["settles", "cleanup-fails"])
async def test_composed_installer_failure_reconciles_results_safe_diagnostics_and_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, cleanup_fails: bool
) -> None:
    private = "private-observer-cleanup-path-and-secret-312"
    payload = b"synthetic-model"
    base = load_local_stt_asset_manifest(LOCAL_QWEN_GPU_MODEL_ID)
    manifest = replace(
        base,
        sources={"huggingface": base.sources["huggingface"]},
        files=(
            LocalSTTAssetFile(
                relative_path="synthetic.gguf",
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            ),
        ),
    )
    downloader = _HeldProvisioningDownload(
        payload, PermissionError(errno.EACCES, private, private) if cleanup_fails else None
    )
    snapshots = []
    owner = LocalASRProvisioningOwner(
        model_root=tmp_path / "models",
        gpu_manifest_loader=lambda: manifest,
        huggingface_downloader=downloader,
        state_changed=snapshots.append,
    )
    root = logging.getLogger(f"test.composed-model-diagnostic.{uuid4().hex}")
    root.propagate = False
    sinks = configure_main_logging(root_logger=root, log_dir=tmp_path / "logs")
    basic = io.StringIO()
    sinks.stream_handler.setStream(basic)
    service = SessionRuntimeLoggingService(root_logger=root, sinks=sinks)
    instance, operation, request_id = str(uuid4()), uuid4().hex, uuid4().hex
    events = []
    service.configure_diagnostics(instance_id=instance, event_sink=events.append)
    monkeypatch.setattr(provisioning_module, "logger", root)
    monkeypatch.setattr(feature_diagnostics, "logger", root)
    delivered = []
    progress = asyncio.Event()
    observer_failure = ValueError(private)
    if mode == "grouped_observer":
        observer_failure = BaseExceptionGroup(private, (asyncio.CancelledError(), observer_failure))
    statuses = []

    async def on_status(update):
        statuses.append(update)
        if update.downloaded_bytes:
            progress.set()
            if mode != "cancel":
                raise observer_failure

    request = LocalASRInstallRequest(
        backend="gpu",
        model_ids=(manifest.model_id,),
        locale="en",
        origin="manual",
        explicit_gpu_intent=True,
        preferred_source="huggingface",
    )
    task = None
    cancellation = None
    try:
        with diagnostic_operation(instance, operation, request_id):
            task = owner.start_install(
                request, on_status=on_status, result_handler=delivered.append
            )
        await asyncio.wait_for(progress.wait(), timeout=5)
        if mode == "cancel":
            cancellation = asyncio.create_task(owner.cancel_install("gpu"))
        await asyncio.wait_for(downloader.cancellation_requested.wait(), timeout=5)
        assert not task.done()
        assert downloader.active == 1
        assert owner.snapshot.state_for(manifest.model_id).operation == "downloading"
        assert owner.snapshot.activity_for("gpu") is not None
        staging = list((tmp_path / "models").glob(f"{manifest.install_dirname}.staging-*"))
        assert len(staging) == 1
        assert LocalSTTProvisioningLease.acquire(model_root=tmp_path / "models", wait=False) is None
        assert any(
            snapshot.activity_for("gpu") is not None
            and snapshot.activity_for("gpu").progress_percent > 0
            for snapshot in snapshots
        )
        assert {update.status for update in statuses} == {"downloading"}
        downloader.release.set()
        result = await asyncio.wait_for(task, timeout=5)
        if cancellation is not None:
            await cancellation
        await _wait_for_condition(
            lambda: delivered == [result] and not owner.active_result_delivery_task_names
        )
        assert downloader.active == 0
        assert len(downloader.requests) == 1
        assert not staging[0].exists()
        assert result.installed_model_ids == ()
        assert result.cancelled == (mode == "cancel")
        assert result.failed_model_ids == (() if mode == "cancel" else (manifest.model_id,))
        assert result.failure_codes == (() if mode == "cancel" else ("status_callback_failed",))
        terminal = owner.snapshot.state_for(manifest.model_id)
        assert terminal.operation == ("cancelled" if mode == "cancel" else "download_failed")
        assert terminal.integrity == "missing"
        assert owner.snapshot.activities == ()
        assert {update.status for update in statuses} == {"downloading"}
        terminal_diagnostics = [item for item in owner.diagnostics if item.event == "install"]
        assert [item.outcome for item in terminal_diagnostics] == [
            "started",
            "cancelled" if mode == "cancel" else "failed",
        ]
        if mode != "cancel":
            assert terminal_diagnostics[-1].failure_type == "_StatusCallbackError"
            assert "code=status_callback_failed" in basic.getvalue()
        cleanup_diagnostics = [
            item
            for item in owner.diagnostics
            if item.event == "cleanup" and item.origin == "manual"
        ]
        assert len(cleanup_diagnostics) == int(cleanup_fails)
        records = service.diagnostic_history({"operation_id": operation})["records"]
        cleanup_records = [record for record in records if record["stage"] == "cleanup"]
        assert len(cleanup_records) == int(cleanup_fails)
        assert len(records) == int(mode != "cancel") + int(cleanup_fails)
        if cleanup_fails:
            assert cleanup_diagnostics[0].failure_type == "PermissionError"
            secondary = cleanup_records[0]
            assert secondary["fields"]["exception_class"] == "PermissionError"
            assert secondary["fields"]["errno"] == errno.EACCES
            assert secondary["fields"]["model_id"] == manifest.model_id
            assert secondary["request_id"] == request_id
            assert secondary["disposition"] == "failed"
            assert (
                "stage=cleanup code=cleanup_failed exception_type=PermissionError"
                in basic.getvalue()
            )
            assert f"os_code={errno.EACCES}" in basic.getvalue()
        assert len(events) == len(records)
        assert private not in repr(owner.diagnostics)
        assert private not in json.dumps(records)
        assert private not in json.dumps(events)
        assert private not in basic.getvalue()
        lease = LocalSTTProvisioningLease.acquire(model_root=tmp_path / "models", wait=False)
        assert lease is not None
        lease.close()

        inspected = await owner.inspect_gpu(explicit_intent=True, verify_checksums=True)
        assert inspected.state_for(manifest.model_id).operation == "idle"
        deleted = await owner.delete_model(manifest.model_id)
        assert deleted.state_for(manifest.model_id).integrity == "missing"
        downloader.cleanup_failure = None
        retry_statuses = []
        retry = await owner.start_install(request, on_status=retry_statuses.append)
        assert retry.installed_model_ids == (manifest.model_id,)
        assert not retry.cancelled and retry.failed_model_ids == ()
        assert retry.snapshot.state_for(manifest.model_id).available
        assert retry_statuses[-1].status == "ready"
        assert downloader.active == 0
        assert not list((tmp_path / "models").glob("*.staging-*"))
    finally:
        downloader.release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        if cancellation is not None:
            await asyncio.gather(cancellation, return_exceptions=True)
        await owner.close()
        service.close()
        sinks.close(force=True)
        root.handlers.clear()
    if not records:
        assert not sinks.log_file.exists()
        return
    file_text = sinks.log_file.read_text(encoding="utf-8")
    persisted = [
        json.loads(line.partition("[Failure] ")[2])
        for line in file_text.splitlines()
        if "[Failure] {" in line
    ]
    assert persisted == records
    assert private not in file_text
    assert str(tmp_path) not in file_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retired",
    ["qwen3-asr-0.6b-int8-sherpa", "parakeet-tdt-0.6b-v3-int8-sherpa"],
)
async def test_owner_rejects_retired_install_requests_before_dispatch(tmp_path, retired) -> None:
    backend = MutableProvisioningBackend(_all_states())
    owner = _owner(backend, model_root=tmp_path)
    try:
        with pytest.raises(ValueError, match="unknown cpu Local ASR model identity"):
            owner.start_install(LocalASRInstallRequest("cpu", (retired,), "en", "manual"))
        assert backend.install_calls == []
        assert retired not in {model.model_id for model in owner.snapshot.models}
    finally:
        await owner.close()
