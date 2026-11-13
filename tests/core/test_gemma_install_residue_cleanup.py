from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.app.services.managed_gemma_translation import (
    ManagedGemmaTranslationOwner,
)
from puripuly_heart.core.diagnostics import serialize_diagnostics
from puripuly_heart.core.local_asr.local_stt_runtime_installer import (
    LocalSTTProvisioningLease,
)
from puripuly_heart.core.local_translation import assets as gemma_assets
from puripuly_heart.core.local_translation.provisioning import GemmaProvisioningCancelled
from puripuly_heart.core.local_translation.runtime import ManagedGemmaReadiness

GEMMA_E4B = gemma_assets.GEMMA_MODEL_ID
GEMMA_26B = gemma_assets.GEMMA_26B_MODEL_ID


class _StubRuntime:
    async def prepare(self, *, backend: str, **kwargs: object) -> ManagedGemmaReadiness:
        return ManagedGemmaReadiness(
            requested_backend=backend,
            effective_backend="cpu",
            source_language="ko",
            target_language="en",
            prefix_identity="cleanup-test",
        )

    async def release(self) -> None:
        return None

    async def close(self) -> None:
        return None


def _owner() -> ManagedGemmaTranslationOwner:
    return ManagedGemmaTranslationOwner(runtime=_StubRuntime())  # type: ignore[arg-type]


def _residue(root: Path, dirname: str, suffix: str, *, payload: bytes = b"x" * 16) -> Path:
    residue = root / f"{dirname}.{suffix}-{uuid.uuid4().hex}"
    (residue / "nested").mkdir(parents=True)
    (residue / "nested" / "weights.bin").write_bytes(payload)
    return residue


def _install(root: Path, dirname: str) -> Path:
    install = root / dirname
    (install / "nested").mkdir(parents=True)
    (install / "nested" / "weights.bin").write_bytes(b"live")
    return install


@pytest.fixture()
def models_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(gemma_assets, "default_models_dir", lambda: root)
    return root


@pytest.fixture()
def installed_model(models_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pinned = gemma_assets.e4b_gemma_spec()
    contents = b"verified-model"
    spec = replace(
        pinned,
        draft_filename=None,
        assets=(
            gemma_assets.GemmaAsset(
                pinned.model_filename, len(contents), hashlib.sha256(contents).hexdigest()
            ),
        ),
    )
    monkeypatch.setattr(gemma_assets, "e4b_gemma_spec", lambda: spec)
    install = models_root / spec.install_dirname
    install.mkdir()
    (install / spec.model_filename).write_bytes(contents)
    (install / gemma_assets.GEMMA_INSTALLED_MANIFEST_FILENAME).write_text(
        json.dumps(gemma_assets.InstalledGemmaManifest.expected(spec).to_dict()),
        encoding="utf-8",
    )
    return install


@pytest.fixture()
def blocked_cleanup(
    models_root: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, threading.Event, threading.Event]]:
    staging = _residue(models_root, GEMMA_E4B, "staging")
    entered = threading.Event()
    release = threading.Event()
    remove = gemma_assets.shutil.rmtree

    def gated_remove(path, *args, **kwargs):
        if Path(path) == staging:
            entered.set()
            assert release.wait(5)
        return remove(path, *args, **kwargs)

    monkeypatch.setattr(gemma_assets.shutil, "rmtree", gated_remove)
    try:
        yield staging, entered, release
    finally:
        release.set()


async def _run_operation(owner: ManagedGemmaTranslationOwner, operation: str) -> object:
    if operation == "prepare":
        return await owner.prepare(
            ManagedGemmaTranslationSelection(
                backend="cpu",
                source_language="ko",
                target_language="en",
                system_prompt="translate",
                locale="en",
            )
        )
    if operation == "install":
        return await owner.install_assets(model_id=GEMMA_E4B)
    return await owner.delete_assets(GEMMA_E4B)


def _lease_available(models_root: Path) -> bool:
    lease = LocalSTTProvisioningLease.acquire(model_root=models_root, wait=False)
    if lease is None:
        return False
    lease.close()
    return True


def test_interrupted_delete_residue_is_reconciled_on_next_inspect(
    models_root: Path,
) -> None:
    trash = _residue(models_root, GEMMA_E4B, "deleting")

    owner = _owner()
    status = owner.install_state(GEMMA_E4B).status

    assert status == "missing"
    assert not trash.exists()


def test_interrupted_provisioning_and_promote_residue_is_reconciled(
    models_root: Path,
) -> None:
    staging = _residue(models_root, GEMMA_26B, "staging")
    backup = _residue(models_root, GEMMA_26B, "backup")

    owner = _owner()
    _ = owner.install_state(GEMMA_26B)

    assert not staging.exists()
    assert not backup.exists()


def test_residue_reconcile_keeps_current_install_and_unrelated_content(
    models_root: Path,
) -> None:
    own_trash = _residue(models_root, GEMMA_E4B, "deleting")
    own_staging = _residue(models_root, GEMMA_E4B, "staging")
    other_trash = _residue(models_root, "other-model", "deleting")
    install = _install(models_root, GEMMA_E4B)
    plain_file_residue = models_root / f"{GEMMA_E4B}.deleting-file"
    plain_file_residue.write_bytes(b"residue")
    near_miss = models_root / f"{GEMMA_E4B}.stagingx-keep"
    near_miss.mkdir()
    keeper = models_root / "keep-me.txt"
    keeper.write_text("notes")

    owner = _owner()
    _ = owner.install_state(GEMMA_E4B)

    assert not own_trash.exists()
    assert not own_staging.exists()
    assert not plain_file_residue.exists()
    assert other_trash.exists()
    assert near_miss.is_dir()
    assert keeper.read_text() == "notes"
    assert (install / "nested" / "weights.bin").read_bytes() == b"live"


def test_residue_reconcile_retries_after_provisioning_lease_is_released(
    models_root: Path,
) -> None:
    trash = _residue(models_root, GEMMA_E4B, "deleting")
    owner = _owner()

    lease = LocalSTTProvisioningLease.acquire(model_root=models_root, wait=False)
    assert lease is not None
    try:
        _ = owner.install_state(GEMMA_E4B)
        assert trash.exists()
    finally:
        lease.close()

    _ = owner.install_state(GEMMA_E4B)
    assert not trash.exists()


def test_delete_assets_reconciles_preexisting_staging_residue(models_root: Path) -> None:
    install = _install(models_root, GEMMA_E4B)
    staging = _residue(models_root, GEMMA_E4B, "staging")
    owner = _owner()

    asyncio.run(owner.delete_assets(GEMMA_E4B))

    assert not install.exists()
    assert not staging.exists()


def test_cleanup_function_removes_all_three_patterns_and_reports_paths(
    models_root: Path,
) -> None:
    staging = _residue(models_root, GEMMA_E4B, "staging")
    backup = _residue(models_root, GEMMA_E4B, "backup")
    trash = _residue(models_root, GEMMA_E4B, "deleting")

    removed = gemma_assets.cleanup_gemma_install_residue(
        install_dirnames=(GEMMA_E4B,),
    )

    assert removed is not None
    assert set(removed) == {staging, backup, trash}
    assert not staging.exists()
    assert not backup.exists()
    assert not trash.exists()


def test_cleanup_function_is_noop_when_models_root_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "absent-models"

    removed = gemma_assets.cleanup_gemma_install_residue(models_dir=missing)

    assert removed == ()
    assert not missing.exists()


def test_cleanup_function_rejects_unsafe_dirnames(models_root: Path) -> None:
    for bad in (("",), (".hidden",), ("../escape",), ("a/b",)):
        with pytest.raises(ValueError):
            gemma_assets.cleanup_gemma_install_residue(install_dirnames=bad)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["prepare", "install", "delete"])
@pytest.mark.parametrize("finish", ["success", "cancel", "close", "concurrent-inspection"])
async def test_cleanup_preserves_responsiveness_and_operation_lifetime(
    models_root: Path,
    installed_model: Path,
    blocked_cleanup: tuple[Path, threading.Event, threading.Event],
    operation: str,
    finish: str,
) -> None:
    staging, entered, release = blocked_cleanup
    owner = _owner()
    inspection = None
    closing = None
    task = None
    try:
        if finish == "concurrent-inspection":
            inspection = asyncio.create_task(asyncio.to_thread(owner.install_state, GEMMA_26B))
            assert await asyncio.to_thread(entered.wait, 5)
        task = asyncio.create_task(_run_operation(owner, operation))
        assert await asyncio.to_thread(entered.wait, 5)
        for _ in range(5):
            await asyncio.sleep(0)

        assert not task.done()
        assert staging.exists()
        assert installed_model.exists()
        assert not _lease_available(models_root)

        if finish == "cancel":
            for _ in range(2):
                if operation == "prepare":
                    assert owner.cancel()
                elif operation == "install":
                    assert owner.cancel_assets(GEMMA_E4B)
                else:
                    task.cancel()
                for _ in range(5):
                    await asyncio.sleep(0)
            assert not task.done()
            assert installed_model.exists()
            assert not _lease_available(models_root)
        elif finish == "close":
            closing = asyncio.create_task(owner.close())
            for _ in range(5):
                await asyncio.sleep(0)
            assert not closing.done()
            assert owner.snapshot.state != "closed"
            assert installed_model.exists()
            assert not _lease_available(models_root)
        elif inspection is not None:
            assert not inspection.done()
    finally:
        release.set()
        if task is not None:
            result = (await asyncio.gather(task, return_exceptions=True))[0]
        if inspection is not None:
            inspection_result = await inspection
        if closing is not None:
            await closing
        await owner.close()

    assert not staging.exists()
    assert _lease_available(models_root)
    if finish == "cancel" or (finish == "close" and operation != "delete"):
        assert isinstance(result, (asyncio.CancelledError, GemmaProvisioningCancelled))
        gemma_assets.validate_gemma_install(installed_model)
    elif operation == "delete":
        assert result is None
        assert not installed_model.exists()
    elif operation == "install":
        assert result == gemma_assets.InstalledGemmaManifest.expected()
        gemma_assets.validate_gemma_install(installed_model)
    else:
        assert result.readiness.effective_backend == "cpu"
        gemma_assets.validate_gemma_install(installed_model)
    if inspection is not None:
        assert inspection_result.status == "missing"
    assert owner.snapshot.state == "closed"


@pytest.mark.parametrize("failure", [False, True], ids=["success", "failure"])
def test_cleanup_logs_success_as_info_and_failure_as_error(
    models_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: bool,
) -> None:
    staging = _residue(models_root, GEMMA_E4B, "staging")
    if failure:

        def deny_removal(path):
            raise PermissionError("synthetic deletion failure")

        monkeypatch.setattr(gemma_assets.shutil, "rmtree", deny_removal)
    owner = _owner()

    with caplog.at_level(logging.INFO, logger="puripuly_heart.core.feature_diagnostics"):
        assert owner.install_state(GEMMA_E4B).status == "missing"

    records = [
        record
        for record in caplog.records
        if record.name == "puripuly_heart.core.feature_diagnostics"
    ]
    assert len(records) == 1
    if failure:
        assert staging.exists()
        assert records[0].levelno == logging.ERROR
        assert "code=cleanup_failed" in records[0].getMessage()
        assert owner.snapshot.secondary_diagnostics[0] is owner.diagnostics[0]
        assert owner.snapshot.secondary_diagnostics[0].code == "model_assets.cleanup_failed"
        assert _lease_available(models_root)
    else:
        assert not staging.exists()
        assert records[0].levelno == logging.INFO
        assert "code=residue_removed" in records[0].getMessage()
        assert "failure" not in records[0].getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_inspection", [False, True], ids=["active", "cancelled"])
async def test_close_drains_inspection_cleanup_and_prevents_new_cleanup(
    models_root: Path,
    blocked_cleanup: tuple[Path, threading.Event, threading.Event],
    cancel_inspection: bool,
) -> None:
    staging, entered, release = blocked_cleanup
    owner = _owner()
    inspection = asyncio.create_task(asyncio.to_thread(owner.install_state, GEMMA_26B))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        if cancel_inspection:
            inspection.cancel()
            for _ in range(5):
                await asyncio.sleep(0)
        closing = asyncio.create_task(owner.close())
        for _ in range(5):
            await asyncio.sleep(0)

        assert not closing.done()
        assert owner.snapshot.state != "closed"
        assert staging.exists()
        assert not _lease_available(models_root)
    finally:
        release.set()
        result = (await asyncio.gather(inspection, return_exceptions=True))[0]
        if closing is not None:
            await closing
        await owner.close()

    if cancel_inspection:
        assert isinstance(result, asyncio.CancelledError)
    else:
        assert result.status == "missing"
    assert not staging.exists()
    assert _lease_available(models_root)
    assert owner.snapshot.state == "closed"

    later_residue = _residue(models_root, GEMMA_E4B, "staging")
    assert owner.install_state(GEMMA_26B).status == "missing"
    assert later_residue.exists()


@pytest.mark.asyncio
async def test_inspection_cleanup_failure_is_immediately_visible_without_worker_notifications(
    models_root: Path, installed_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from puripuly_heart.core import feature_diagnostics

    private = "private-residue-path-and-credential"
    staging = _residue(models_root, GEMMA_E4B, "staging")
    cleanup_calls = []
    sink_threads = []
    log_calls = []
    owner_thread = threading.get_ident()

    def deny_removal(path):
        cleanup_calls.append(threading.get_ident())
        raise PermissionError(13, private, str(staging))

    def fail_log(*args, **kwargs):
        log_calls.append(True)
        raise RuntimeError(private)

    def fail_status(snapshot):
        sink_threads.append(threading.get_ident())
        raise RuntimeError(private)

    monkeypatch.setattr(gemma_assets.shutil, "rmtree", deny_removal)
    monkeypatch.setattr(feature_diagnostics, "emit_basic_log", fail_log)
    owner = ManagedGemmaTranslationOwner(
        runtime=_StubRuntime(), status_sink=fail_status
    )
    try:
        state = await asyncio.to_thread(owner.install_state, GEMMA_E4B)
        assert state.status == "ready"
        assert state.manifest is not None
        assert installed_model.is_dir()
        assert staging.is_dir()
        assert cleanup_calls and cleanup_calls[0] != owner_thread
        assert log_calls == [True]
        assert sink_threads == []
        snapshot = owner.snapshot
        assert snapshot.state == "idle"
        assert snapshot.failure_diagnostics is None
        assert snapshot.recovery_diagnostics == ()
        assert len(snapshot.secondary_diagnostics) == 1
        cause = snapshot.secondary_diagnostics[0]
        assert cause is owner.diagnostics[0]
        assert cause.code == "model_assets.cleanup_failed"
        assert cause.operation == "cleanup"
        assert cause.fields["exception_class"] == "PermissionError"
        assert cause.fields["errno"] == 13
        assert cause.fields["cleanup_complete"] is False
        assert snapshot.secondary_diagnostics_truncated is False
        assert owner.snapshot.secondary_diagnostics == (cause,)
        assert sink_threads == []
        assert _lease_available(models_root)
        serialized = json.dumps(serialize_diagnostics(cause))
        assert private not in serialized
        assert str(models_root) not in serialized
    finally:
        await owner.close()
    assert sink_threads == [owner_thread]
