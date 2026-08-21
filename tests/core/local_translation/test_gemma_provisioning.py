from __future__ import annotations

import asyncio
import hashlib
import threading

import pytest

from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadProgress,
)
from puripuly_heart.core.local_translation import assets, provisioning


class FakeDownloader:
    def __init__(self, content_by_name: dict[str, bytes]) -> None:
        self.content_by_name = content_by_name
        self.requests = []

    async def download(self, request, *, cancel_event, on_progress):
        self.requests.append(request)
        content = self.content_by_name[request.remote_path]
        if cancel_event is not None and cancel_event.is_set():
            raise AssertionError("cancelled request reached downloader")
        path = request.local_dir / request.remote_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if on_progress is not None:
            on_progress(HuggingFaceDownloadProgress(len(content), len(content)))
        return path


class CoordinatedDownloader(FakeDownloader):
    def __init__(self, content_by_name: dict[str, bytes]) -> None:
        super().__init__(content_by_name)
        self.first_request_started = asyncio.Event()
        self.release_first_request = asyncio.Event()

    async def download(self, request, *, cancel_event, on_progress):
        if not self.requests:
            self.first_request_started.set()
            await self.release_first_request.wait()
        return await super().download(
            request,
            cancel_event=cancel_event,
            on_progress=on_progress,
        )


class BurstDownloader(FakeDownloader):
    async def download(self, request, *, cancel_event, on_progress):
        self.requests.append(request)
        content = self.content_by_name[request.remote_path]
        path = request.local_dir / request.remote_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        on_progress(HuggingFaceDownloadProgress(1, len(content)))
        on_progress(HuggingFaceDownloadProgress(len(content), len(content)))
        return path


class CancelAfterProgressDownloader(FakeDownloader):
    async def download(self, request, *, cancel_event, on_progress):
        self.requests.append(request)
        content = self.content_by_name[request.remote_path]
        path = request.local_dir / request.remote_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        on_progress(HuggingFaceDownloadProgress(len(content), len(content)))
        await asyncio.sleep(0)
        raise asyncio.CancelledError


def _pin_small_assets(monkeypatch):
    contents = {"target.gguf": b"target-model", "draft.gguf": b"draft-model"}
    pinned = tuple(
        assets.GemmaAsset(
            filename=name,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        for name, content in contents.items()
    )
    monkeypatch.setattr(assets, "GEMMA_ASSETS", pinned)
    monkeypatch.setattr(provisioning, "GEMMA_ASSETS", pinned)
    return contents


@pytest.mark.asyncio
async def test_missing_install_downloads_pinned_files_and_promotes_atomically(
    tmp_path, monkeypatch
) -> None:
    contents = _pin_small_assets(monkeypatch)
    downloader = FakeDownloader(contents)
    updates = []
    install_dir = tmp_path / "gemma"

    installed = await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
        on_status=updates.append,
    )

    assert installed == assets.InstalledGemmaManifest.expected()
    assert [request.repo_id for request in downloader.requests] == [assets.GEMMA_REPO_ID] * 2
    assert [request.revision for request in downloader.requests] == [assets.GEMMA_REVISION] * 2
    assert [request.remote_path for request in downloader.requests] == list(contents)
    assert assets.validate_gemma_install(install_dir) == installed
    assert [path.name for path in tmp_path.glob("gemma.staging-*")] == []
    assert updates[0].state == "downloading"
    assert updates[-1].state == "ready"
    assert updates[-1].percent == 100


@pytest.mark.asyncio
async def test_valid_install_is_reused_without_downloading(tmp_path, monkeypatch) -> None:
    contents = _pin_small_assets(monkeypatch)
    downloader = FakeDownloader(contents)
    install_dir = tmp_path / "gemma"
    await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
    )
    downloader.requests.clear()

    await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
    )

    assert downloader.requests == []


@pytest.mark.asyncio
async def test_valid_install_reuse_skips_checksum(tmp_path, monkeypatch) -> None:
    contents = _pin_small_assets(monkeypatch)
    downloader = FakeDownloader(contents)
    install_dir = tmp_path / "gemma"
    await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
    )

    def fail_if_hashed(_path) -> str:
        raise AssertionError("ready install must not hash files")

    monkeypatch.setattr(assets, "_sha256_file", fail_if_hashed)

    reused = await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
    )

    assert reused == assets.InstalledGemmaManifest.expected()


@pytest.mark.asyncio
async def test_failed_repair_preserves_existing_install(tmp_path, monkeypatch) -> None:
    contents = _pin_small_assets(monkeypatch)
    install_dir = tmp_path / "gemma"
    install_dir.mkdir()
    sentinel = install_dir / "keep.txt"
    sentinel.write_text("previous", encoding="utf-8")
    downloader = FakeDownloader({**contents, "draft.gguf": b"wrong-size"})

    with pytest.raises(provisioning.GemmaProvisioningError, match="size mismatch"):
        await provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
        )

    assert sentinel.read_text(encoding="utf-8") == "previous"
    assert [path for path in tmp_path.glob("gemma.staging-*")] == []


@pytest.mark.asyncio
async def test_pre_cancelled_install_does_not_mutate_disk(tmp_path, monkeypatch) -> None:
    contents = _pin_small_assets(monkeypatch)
    downloader = FakeDownloader(contents)
    cancel_event = threading.Event()
    cancel_event.set()
    install_dir = tmp_path / "gemma"

    with pytest.raises(provisioning.GemmaProvisioningCancelled):
        await provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
            cancel_event=cancel_event,
        )

    assert not install_dir.exists()
    assert downloader.requests == []


@pytest.mark.asyncio
async def test_concurrent_provisioning_serializes_and_reuses_promoted_install(
    tmp_path, monkeypatch
) -> None:
    contents = _pin_small_assets(monkeypatch)
    downloader = CoordinatedDownloader(contents)
    install_dir = tmp_path / "gemma"
    first = asyncio.create_task(
        provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
        )
    )
    await downloader.first_request_started.wait()
    second = asyncio.create_task(
        provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
        )
    )
    await asyncio.sleep(0.05)
    assert len(downloader.requests) == 0
    downloader.release_first_request.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert first_result == second_result == assets.InstalledGemmaManifest.expected()
    assert len(downloader.requests) == 2
    assert assets.validate_gemma_install(install_dir) == first_result


@pytest.mark.asyncio
async def test_cancellation_during_initial_status_removes_staging(tmp_path, monkeypatch) -> None:
    contents = _pin_small_assets(monkeypatch)
    downloader = FakeDownloader(contents)
    install_dir = tmp_path / "gemma"

    async def cancel_on_initial_status(_update):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
            on_status=cancel_on_initial_status,
        )

    assert not install_dir.exists()
    assert [path for path in tmp_path.glob("gemma.staging-*")] == []


@pytest.mark.asyncio
async def test_progress_callback_future_is_awaited(tmp_path, monkeypatch) -> None:
    contents = _pin_small_assets(monkeypatch)
    downloader = FakeDownloader(contents)
    observed = []

    async def record(update):
        await asyncio.sleep(0)
        observed.append(update)

    def on_status(update):
        return asyncio.create_task(record(update))

    await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=tmp_path / "gemma",
        on_status=on_status,
    )

    assert observed
    assert observed[-1].state == "ready"


@pytest.mark.asyncio
async def test_all_submitted_progress_failures_are_observed(tmp_path, monkeypatch) -> None:
    contents = _pin_small_assets(monkeypatch)
    asset = provisioning.GEMMA_ASSETS[0]
    downloader = BurstDownloader(contents)

    async def fail_progress(update):
        if update.downloaded_bytes:
            await asyncio.sleep(0)
            raise RuntimeError(f"progress-{update.downloaded_bytes}")

    with pytest.raises(BaseExceptionGroup) as caught:
        await provisioning._download_asset(
            downloader=downloader,
            asset=asset,
            staging_dir=tmp_path / "staging",
            completed_bytes=0,
            total_bytes=asset.size_bytes,
            cancel_event=None,
            on_status=fail_progress,
        )

    assert sorted(str(error) for error in caught.value.exceptions) == [
        "progress-1",
        f"progress-{asset.size_bytes}",
    ]


@pytest.mark.asyncio
async def test_cancellation_during_progress_drain_waits_for_accepted_callback(
    tmp_path,
    monkeypatch,
) -> None:
    contents = _pin_small_assets(monkeypatch)
    asset = provisioning.GEMMA_ASSETS[0]
    downloader = FakeDownloader(contents)
    callback_started = asyncio.Event()
    allow_callback = asyncio.Event()
    callback_finished = asyncio.Event()

    async def block_progress(_update):
        callback_started.set()
        await allow_callback.wait()
        callback_finished.set()

    operation = asyncio.create_task(
        provisioning._download_asset(
            downloader=downloader,
            asset=asset,
            staging_dir=tmp_path / "staging",
            completed_bytes=0,
            total_bytes=asset.size_bytes,
            cancel_event=None,
            on_status=block_progress,
        )
    )
    await callback_started.wait()
    operation.cancel()
    await asyncio.sleep(0)

    assert not operation.done()
    allow_callback.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert callback_finished.is_set()


@pytest.mark.asyncio
async def test_mixed_download_cancellation_and_progress_failure_removes_staging(
    tmp_path,
    monkeypatch,
) -> None:
    contents = _pin_small_assets(monkeypatch)
    downloader = CancelAfterProgressDownloader(contents)
    install_dir = tmp_path / "gemma"

    async def fail_download_progress(update):
        if update.downloaded_bytes:
            raise RuntimeError("progress failed")

    with pytest.raises(BaseExceptionGroup):
        await provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
            on_status=fail_download_progress,
        )

    assert not install_dir.exists()
    assert [path for path in tmp_path.glob("gemma.staging-*")] == []
