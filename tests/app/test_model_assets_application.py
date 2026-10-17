from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LOCAL_STT_MODEL_ID,
    load_local_stt_asset_manifest,
)

from puripuly_heart.app.services.model_assets import ModelAssetsApplication, ModelAssetsError
from puripuly_heart.app.services.model_assets_catalog import ModelAssetChoice

GEMMA_ID = "gemma-test"
CHOICES = (
    ModelAssetChoice(LOCAL_STT_MODEL_ID, "Qwen CPU", "d.qwen", "asr", 10),
    ModelAssetChoice(LOCAL_QWEN_GPU_MODEL_ID, "Qwen GPU", "d.gpu", "asr", 20),
    ModelAssetChoice(
        GEMMA_ID,
        "Gemma",
        "d.gemma",
        "translation",
        30,
        importable=True,
        required_files=("model.gguf", "draft.gguf"),
    ),
)


class FakeProvisioning:
    def __init__(self) -> None:
        self.requests = []
        self.cancelled = []
        self.deleted = []
        self.active_backend = None
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.result = SimpleNamespace(cancelled=False, failed_model_ids=(), failure_codes=())
        self.snapshot = SimpleNamespace(
            activity_for=lambda backend: object() if backend == self.active_backend else None
        )

    async def start_install(self, request, *, on_status=None):
        self.requests.append(request)
        on_status(SimpleNamespace(percent=42))
        self.started.set()
        await self.release.wait()
        return self.result

    async def cancel_install(self, backend, *, origin=None):
        self.cancelled.append((backend, origin))
        self.result = SimpleNamespace(cancelled=True, failed_model_ids=(), failure_codes=())
        self.release.set()

    async def delete_model(self, model_id):
        self.deleted.append(model_id)


class FakeGemma:
    def __init__(self) -> None:
        self.installs = []
        self.deleted = []
        self.status = "missing"

    def install_status(self, model_id):
        return self.status

    async def install_assets(self, **kwargs):
        self.installs.append(kwargs)
        kwargs["on_status"](SimpleNamespace(downloaded_bytes=3, total_bytes=4))
        self.status = "ready"

    async def delete_assets(self, model_id):
        self.deleted.append(model_id)
        self.status = "missing"

    def cancel_assets(self, model_id):
        return True


def _service(tmp_path, *, active=frozenset()):
    provisioning = FakeProvisioning()
    gemma = FakeGemma()
    service = ModelAssetsApplication(
        provisioning=lambda: provisioning,
        gemma=lambda: gemma,
        locale=lambda: "zh-CN",
        active_model_ids=lambda: active,
        model_root=tmp_path,
        choices=CHOICES,
    )
    return service, provisioning, gemma


def _status(service, model_id):
    return service.entry(model_id).status


@pytest.mark.asyncio
async def test_entries_report_disk_integrity_without_touching_owners(tmp_path) -> None:
    service, provisioning, gemma = _service(tmp_path)
    gemma.status = "invalid"

    entries = {entry.model_id: entry for entry in await service.refresh()}

    assert entries[LOCAL_STT_MODEL_ID].status == "missing"
    assert entries[GEMMA_ID].status == "invalid"
    assert entries[GEMMA_ID].required_files == ("model.gguf", "draft.gguf")
    assert provisioning.requests == []


@pytest.mark.asyncio
async def test_asr_download_uses_locale_order_reports_progress_and_changes_no_settings(
    tmp_path,
) -> None:
    service, provisioning, _gemma = _service(tmp_path)

    task = asyncio.create_task(service.download(LOCAL_QWEN_GPU_MODEL_ID))
    await provisioning.started.wait()

    assert _status(service, LOCAL_QWEN_GPU_MODEL_ID) == "downloading"
    assert service.entry(LOCAL_QWEN_GPU_MODEL_ID).progress_percent == 42
    with pytest.raises(ModelAssetsError) as busy:
        await service.download(LOCAL_STT_MODEL_ID)
    assert busy.value.failure_code == "task_busy"

    provisioning.release.set()
    await task
    request = provisioning.requests[0]
    assert (request.backend, request.explicit_gpu_intent) == ("gpu", True)
    assert request.locale == "zh-CN" and request.preferred_source is None
    assert request.origin == "model_assets"
    assert len(provisioning.requests) == 1


@pytest.mark.asyncio
async def test_failure_is_recorded_on_the_entry_and_cleared_by_the_next_attempt(tmp_path) -> None:
    service, provisioning, _gemma = _service(tmp_path)
    provisioning.release.set()
    provisioning.result = SimpleNamespace(
        cancelled=False,
        failed_model_ids=(LOCAL_STT_MODEL_ID,),
        failure_codes=("connection_failed",),
    )

    with pytest.raises(ModelAssetsError) as caught:
        await service.download(LOCAL_STT_MODEL_ID)

    assert caught.value.failure_code == "connection_failed"
    assert service.entry(LOCAL_STT_MODEL_ID).failure_code == "connection_failed"
    provisioning.result = SimpleNamespace(cancelled=False, failed_model_ids=(), failure_codes=())
    await service.download(LOCAL_STT_MODEL_ID)
    assert service.entry(LOCAL_STT_MODEL_ID).failure_code is None


@pytest.mark.asyncio
async def test_cancel_targets_the_model_assets_origin_and_is_not_reported_as_failure(
    tmp_path,
) -> None:
    service, provisioning, _gemma = _service(tmp_path)
    task = asyncio.create_task(service.download(LOCAL_STT_MODEL_ID))
    await provisioning.started.wait()

    assert await service.cancel(LOCAL_STT_MODEL_ID) is True
    with pytest.raises(ModelAssetsError) as caught:
        await task

    assert caught.value.failure_code == "cancelled"
    assert provisioning.cancelled == [("cpu", "model_assets")]
    assert service.entry(LOCAL_STT_MODEL_ID).failure_code is None
    assert not service.busy


@pytest.mark.asyncio
async def test_delete_refuses_models_in_use_and_otherwise_uses_the_owning_backend(
    tmp_path,
) -> None:
    service, provisioning, gemma = _service(tmp_path, active=frozenset({LOCAL_STT_MODEL_ID}))

    with pytest.raises(ModelAssetsError) as caught:
        await service.delete(LOCAL_STT_MODEL_ID)
    assert caught.value.failure_code == "model_in_use"
    assert provisioning.deleted == []

    gemma.status = "ready"
    entry = await service.delete(GEMMA_ID)
    assert gemma.deleted == [GEMMA_ID]
    assert entry.status == "missing"


@pytest.mark.asyncio
async def test_gemma_download_and_import_use_one_shot_sources(tmp_path) -> None:
    service, _provisioning, gemma = _service(tmp_path)

    with pytest.raises(ModelAssetsError) as missing:
        await service.import_directory(GEMMA_ID, tmp_path)
    assert missing.value.failure_code == "missing_file"
    assert gemma.installs == []

    for name in ("model.gguf", "draft.gguf"):
        (tmp_path / name).write_bytes(b"x")
    entry = await service.import_directory(GEMMA_ID, tmp_path)
    assert entry.status == "installed"
    assert gemma.installs[-1]["source"] == "local"
    assert {path.name for path in gemma.installs[-1]["files"]} == {"model.gguf", "draft.gguf"}

    await service.download(GEMMA_ID)
    assert gemma.installs[-1]["source"] is None
    assert gemma.installs[-1]["locale"] == "zh-CN"


@pytest.mark.asyncio
async def test_installed_asr_model_on_disk_is_reported_installed(tmp_path, monkeypatch) -> None:
    import puripuly_heart.app.services.model_assets as module

    service, _provisioning, _gemma = _service(tmp_path)
    seen = []

    def inspect(path, *, manifest):
        seen.append(path)
        return SimpleNamespace(status="ready")

    monkeypatch.setattr(module, "inspect_local_stt_install_state", inspect)

    await service.refresh()

    manifest = load_local_stt_asset_manifest(LOCAL_STT_MODEL_ID)
    assert tmp_path / manifest.install_dirname in seen
    assert _status(service, LOCAL_STT_MODEL_ID) == "installed"
