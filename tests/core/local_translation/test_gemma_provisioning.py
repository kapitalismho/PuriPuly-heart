from __future__ import annotations

import asyncio
import hashlib
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadProgress,
    LocalSTTDownloadPortCancelled,
    LocalSTTDownloadPortError,
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


class FailedDownloader(FakeDownloader):
    def __init__(self, contents: dict[str, bytes], failure: BaseException) -> None:
        super().__init__(contents)
        self.failure = failure

    async def download(self, request, *, cancel_event, on_progress):
        self.requests.append(request)
        raise self.failure


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


class CooperativeCancellationDownloader(FakeDownloader):
    def __init__(self, content_by_name: dict[str, bytes]) -> None:
        super().__init__(content_by_name)
        self.started = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.allow_cleanup = asyncio.Event()
        self.cleanup_finished = asyncio.Event()

    async def download(self, request, *, cancel_event, on_progress):
        self.requests.append(request)
        path = request.local_dir / request.remote_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial")
        self.started.set()
        while cancel_event is not None and not cancel_event.is_set():
            await asyncio.sleep(0)
        self.cleanup_started.set()
        await self.allow_cleanup.wait()
        self.cleanup_finished.set()
        raise LocalSTTDownloadPortCancelled("cancelled")


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
    return contents


@pytest.fixture
def modelscope_server():
    responses = {}
    requests = []
    release_body = threading.Event()
    state = SimpleNamespace(
        responses=responses,
        requests=requests,
        release_body=release_body,
        pause_filename=None,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            filename = query["FilePath"][0]
            requests.append((parsed.path, query))
            status, content = responses[filename]
            self.send_response(status)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            try:
                if filename == state.pause_filename:
                    self.wfile.write(content[:1])
                    self.wfile.flush()
                    release_body.wait(timeout=5)
                    self.wfile.write(content[1:])
                else:
                    self.wfile.write(content)
            except ConnectionError:
                pass

        def log_message(self, _format, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        release_body.set()
        server.shutdown()
        server.server_close()
        worker.join()


def _local_mirror_spec(server, contents):
    server.responses.update({name: (200, content) for name, content in contents.items()})
    spec = assets.e4b_gemma_spec()
    source = spec.sources["modelscope"]
    return replace(
        spec,
        sources={
            **spec.sources,
            "modelscope": replace(
                source,
                download_url_template=source.download_url_template.replace(
                    "https://www.modelscope.cn", server.url
                ),
            ),
        },
    )


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
    assert assets.validate_gemma_install(install_dir) == installed
    assert [path.name for path in tmp_path.glob("gemma.staging-*")] == []
    assert updates[0].state == "downloading"
    assert updates[-1].state == "ready"
    assert updates[-1].percent == 100


@pytest.mark.asyncio
async def test_drafter_less_spec_downloads_target_without_drafter(tmp_path) -> None:
    content = b"drafter-less-target"
    spec = assets.GemmaModelSpec(
        model_id="gemma-4-test-qat",
        repo_id="example/drafter-less",
        revision="abc123",
        model_filename="target.gguf",
        draft_filename=None,
        install_dirname="gemma-drafter-less",
        upstream_repo_id="example/upstream",
        license=assets.GEMMA_LICENSE,
        license_url=assets.GEMMA_LICENSE_URL,
        assets=(
            assets.GemmaAsset(
                filename="target.gguf",
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            ),
        ),
        sources={
            "huggingface": assets.GemmaAssetSource(
                name="huggingface",
                repo_id="example/drafter-less",
                revision="abc123",
            ),
        },
    )
    downloader = FakeDownloader({"target.gguf": content})
    install_dir = tmp_path / "gemma-drafter-less"

    installed = await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
        spec=spec,
    )

    assert installed == assets.InstalledGemmaManifest.expected(spec)
    assert assets.validate_gemma_install(install_dir, spec=spec) == installed


@pytest.mark.asyncio
async def test_valid_hf_install_is_reused_after_locale_and_source_changes(
    tmp_path, monkeypatch
) -> None:
    contents = _pin_small_assets(monkeypatch)
    downloader = FakeDownloader(contents)
    install_dir = tmp_path / "gemma"
    installed = await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
    )
    downloader.requests.clear()

    def reject_http(**_kwargs):
        raise AssertionError("valid install must not contact a mirror")

    monkeypatch.setattr(provisioning.network_clients, "external_client", reject_http)
    for locale, source in (("zh-Hant-HK", "modelscope"), ("en", "huggingface")):
        reused = await provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
            locale=locale,
            preferred_source=source,
        )
        assert reused == installed
    assert downloader.requests == []
    assert assets.validate_gemma_install(install_dir) == installed


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
@pytest.mark.parametrize("failure_kind", ["size", "hash", "network"])
async def test_all_sources_failed_repair_preserves_install_and_unowned_staging(
    tmp_path, monkeypatch, modelscope_server, failure_kind
) -> None:
    contents = _pin_small_assets(monkeypatch)
    spec = _local_mirror_spec(modelscope_server, contents)
    if failure_kind == "network":
        modelscope_server.responses["draft.gguf"] = (503, b"unavailable")
        downloader = FailedDownloader(contents, LocalSTTDownloadPortError("unavailable"))
    else:
        corrupt = b"wrong-size" if failure_kind == "size" else b"x" * len(contents["draft.gguf"])
        modelscope_server.responses["draft.gguf"] = (200, corrupt)
        downloader = FakeDownloader({**contents, "draft.gguf": corrupt})
    install_dir = tmp_path / "gemma"
    install_dir.mkdir()
    sentinel = install_dir / "keep.txt"
    sentinel.write_text("previous", encoding="utf-8")
    unowned = tmp_path / "gemma.staging-unowned"
    unowned.mkdir()
    (unowned / "keep.txt").write_text("unowned", encoding="utf-8")

    with pytest.raises(provisioning.GemmaProvisioningError, match="all sources"):
        await provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
            spec=spec,
        )

    assert sentinel.read_text(encoding="utf-8") == "previous"
    assert (unowned / "keep.txt").read_text(encoding="utf-8") == "unowned"
    assert list(tmp_path.glob("gemma.staging-*")) == [unowned]


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
async def test_repeated_task_cancellation_waits_for_owned_downloader_cleanup(
    tmp_path,
    monkeypatch,
) -> None:
    contents = _pin_small_assets(monkeypatch)
    downloader = CooperativeCancellationDownloader(contents)
    cancel_event = threading.Event()
    install_dir = tmp_path / "gemma"
    operation = asyncio.create_task(
        provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
            cancel_event=cancel_event,
        )
    )
    await downloader.started.wait()

    cancel_event.set()
    operation.cancel()
    await asyncio.wait_for(downloader.cleanup_started.wait(), timeout=0.5)
    operation.cancel()
    await asyncio.sleep(0)

    assert not operation.done()
    downloader.allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert downloader.cleanup_finished.is_set()
    assert not install_dir.exists()
    assert [path for path in tmp_path.glob("gemma.staging-*")] == []


@pytest.mark.asyncio
async def test_staging_cleanup_failure_preserves_cancellation_and_never_publishes_ready(
    tmp_path,
    monkeypatch,
) -> None:
    contents = _pin_small_assets(monkeypatch)
    downloader = CooperativeCancellationDownloader(contents)
    cancel_event = threading.Event()
    install_dir = tmp_path / "gemma"
    statuses = []
    real_rmtree = provisioning.shutil.rmtree

    def fail_staging_cleanup(path, *args, **kwargs):
        if ".staging-" in path.name:
            raise PermissionError(5, "denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(provisioning.shutil, "rmtree", fail_staging_cleanup)
    operation = asyncio.create_task(
        provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
            cancel_event=cancel_event,
            on_status=statuses.append,
        )
    )
    await downloader.started.wait()

    cancel_event.set()
    operation.cancel()
    await downloader.cleanup_started.wait()
    downloader.allow_cleanup.set()
    with pytest.raises(BaseExceptionGroup) as caught:
        await operation

    pending = [caught.value]
    leaves = []
    while pending:
        exception = pending.pop()
        if isinstance(exception, BaseExceptionGroup):
            pending.extend(exception.exceptions)
        else:
            leaves.append(exception)
    assert any(isinstance(exception, asyncio.CancelledError) for exception in leaves)
    assert any(isinstance(exception, PermissionError) for exception in leaves)
    assert not install_dir.exists()
    assert all(status.state != "ready" for status in statuses)


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
    spec = assets.e4b_gemma_spec()
    asset = spec.assets[0]
    downloader = BurstDownloader(contents)

    async def fail_progress(update):
        if update.downloaded_bytes:
            await asyncio.sleep(0)
            raise RuntimeError(f"progress-{update.downloaded_bytes}")

    with pytest.raises(BaseExceptionGroup) as caught:
        await provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=tmp_path / "gemma",
            on_status=fail_progress,
            spec=spec,
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
    install_dir = tmp_path / "gemma"
    downloader = FakeDownloader(contents)
    callback_started = asyncio.Event()
    allow_callback = asyncio.Event()
    callback_finished = asyncio.Event()

    async def block_progress(update):
        if update.downloaded_bytes:
            callback_started.set()
            await allow_callback.wait()
            callback_finished.set()

    operation = asyncio.create_task(
        provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
            on_status=block_progress,
        )
    )
    await callback_started.wait()
    operation.cancel()
    await asyncio.sleep(0)
    operation.cancel()
    await asyncio.sleep(0)

    assert not operation.done()
    allow_callback.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert callback_finished.is_set()
    assert not install_dir.exists()
    assert list(tmp_path.glob("gemma.staging-*")) == []


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


@pytest.mark.asyncio
async def test_chinese_install_downloads_native_modelscope_without_hf(
    tmp_path, monkeypatch, modelscope_server
) -> None:
    contents = _pin_small_assets(monkeypatch)
    spec = _local_mirror_spec(modelscope_server, contents)
    downloader = FailedDownloader(contents, AssertionError("HF must not be used"))
    install_dir = tmp_path / "gemma"

    installed = await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
        spec=spec,
        locale="zh-CN",
    )

    assert installed == assets.InstalledGemmaManifest.expected(spec)
    assert {name: (install_dir / name).read_bytes() for name in contents} == contents
    assert downloader.requests == []
    assert list(tmp_path.glob("gemma.staging-*")) == []


@pytest.mark.asyncio
async def test_non_chinese_install_uses_hf_before_mirror(
    tmp_path, monkeypatch, modelscope_server
) -> None:
    contents = _pin_small_assets(monkeypatch)
    spec = _local_mirror_spec(modelscope_server, contents)
    modelscope_server.responses.update({name: (503, b"unavailable") for name in contents})
    install_dir = tmp_path / "gemma"

    installed = await provisioning.ensure_gemma_installed(
        downloader=FakeDownloader(contents),
        install_dir=install_dir,
        spec=spec,
        locale="ko",
    )

    assert assets.validate_gemma_install(install_dir, spec=spec) == installed
    assert modelscope_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["network", "hash"])
async def test_hf_transfer_or_integrity_failure_falls_back_to_modelscope(
    tmp_path, monkeypatch, modelscope_server, failure_kind
) -> None:
    contents = _pin_small_assets(monkeypatch)
    spec = _local_mirror_spec(modelscope_server, contents)
    if failure_kind == "network":
        downloader = FailedDownloader(contents, LocalSTTDownloadPortError("unavailable"))
    else:
        downloader = FakeDownloader(
            {**contents, "target.gguf": b"x" * len(contents["target.gguf"])}
        )
    install_dir = tmp_path / "gemma"

    installed = await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
        spec=spec,
        locale="en",
    )

    assert assets.validate_gemma_install(install_dir, spec=spec) == installed
    assert downloader.requests
    assert modelscope_server.requests
    assert {name: (install_dir / name).read_bytes() for name in contents} == contents
    assert list(tmp_path.glob("gemma.staging-*")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["network", "hash", "size"])
async def test_modelscope_transfer_or_integrity_failure_falls_back_to_hf(
    tmp_path, monkeypatch, modelscope_server, failure_kind
) -> None:
    contents = _pin_small_assets(monkeypatch)
    spec = _local_mirror_spec(modelscope_server, contents)
    if failure_kind == "network":
        modelscope_server.responses["target.gguf"] = (503, b"unavailable")
    elif failure_kind == "hash":
        modelscope_server.responses["target.gguf"] = (200, b"x" * len(contents["target.gguf"]))
    else:
        modelscope_server.responses["target.gguf"] = (200, b"short")
    downloader = FakeDownloader(contents)
    install_dir = tmp_path / "gemma"

    installed = await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
        spec=spec,
        locale="zh-CN",
    )

    assert assets.validate_gemma_install(install_dir, spec=spec) == installed
    assert modelscope_server.requests
    assert downloader.requests
    assert {name: (install_dir / name).read_bytes() for name in contents} == contents
    assert list(tmp_path.glob("gemma.staging-*")) == []


@pytest.mark.asyncio
async def test_valid_modelscope_install_is_reused_after_switching_to_hf(
    tmp_path, monkeypatch, modelscope_server
) -> None:
    contents = _pin_small_assets(monkeypatch)
    spec = _local_mirror_spec(modelscope_server, contents)
    downloader = FailedDownloader(contents, AssertionError("valid install must not download"))
    install_dir = tmp_path / "gemma"
    installed = await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
        spec=spec,
        locale="zh-HK",
    )
    modelscope_server.requests.clear()

    reused = await provisioning.ensure_gemma_installed(
        downloader=downloader,
        install_dir=install_dir,
        spec=spec,
        locale="en",
        preferred_source="huggingface",
    )

    assert reused == installed
    assert downloader.requests == []
    assert modelscope_server.requests == []


@pytest.mark.asyncio
async def test_http_cancellation_without_caller_event_waits_for_worker_and_never_falls_back(
    tmp_path, monkeypatch, modelscope_server
) -> None:
    contents = _pin_small_assets(monkeypatch)
    spec = _local_mirror_spec(modelscope_server, contents)
    modelscope_server.pause_filename = "target.gguf"
    downloader = FailedDownloader(contents, AssertionError("cancellation must not fall back"))
    progress_started = asyncio.Event()
    install_dir = tmp_path / "gemma"

    def on_status(update):
        if update.state == "downloading" and update.downloaded_bytes:
            progress_started.set()

    operation = asyncio.create_task(
        provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
            spec=spec,
            locale="zh-CN",
            on_status=on_status,
        )
    )
    await asyncio.wait_for(progress_started.wait(), timeout=3)
    operation.cancel()
    await asyncio.sleep(0)
    operation.cancel()
    await asyncio.sleep(0)

    assert not operation.done()
    assert list(tmp_path.glob("gemma.staging-*"))
    modelscope_server.release_body.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=3)
    assert not install_dir.exists()
    assert downloader.requests == []
    assert list(tmp_path.glob("gemma.staging-*")) == []


@pytest.mark.asyncio
async def test_hf_download_cancellation_never_attempts_mirror(
    tmp_path, monkeypatch, modelscope_server
) -> None:
    contents = _pin_small_assets(monkeypatch)
    spec = _local_mirror_spec(modelscope_server, contents)
    downloader = FailedDownloader(contents, LocalSTTDownloadPortCancelled("cancelled"))
    install_dir = tmp_path / "gemma"

    with pytest.raises(provisioning.GemmaProvisioningCancelled):
        await provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
            spec=spec,
            locale="en",
        )

    assert not install_dir.exists()
    assert modelscope_server.requests == []
    assert list(tmp_path.glob("gemma.staging-*")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_kind",
    ["programmer", "status", "initial_status", "ready_status", "worker_protocol"],
)
async def test_programmer_and_status_failures_do_not_trigger_source_fallback(
    tmp_path, monkeypatch, modelscope_server, failure_kind
) -> None:
    contents = _pin_small_assets(monkeypatch)
    spec = _local_mirror_spec(modelscope_server, contents)
    if failure_kind == "programmer":
        downloader = FailedDownloader(contents, RuntimeError("programmer failure"))
    elif failure_kind == "worker_protocol":
        downloader = FailedDownloader(
            contents,
            LocalSTTDownloadPortError("protocol failure", failure_code="worker_protocol_failed"),
        )
    else:
        downloader = FakeDownloader(contents)
    install_dir = tmp_path / "gemma"

    def on_status(update):
        if (
            (failure_kind == "status" and update.downloaded_bytes)
            or (
                failure_kind == "initial_status"
                and update.state == "downloading"
                and not update.downloaded_bytes
            )
            or (failure_kind == "ready_status" and update.state == "ready")
        ):
            raise httpx.ReadError("status failure")

    with pytest.raises(provisioning.GemmaProvisioningError) as caught:
        await provisioning.ensure_gemma_installed(
            downloader=downloader,
            install_dir=install_dir,
            spec=spec,
            locale="en",
            on_status=on_status,
        )

    assert "failure" in str(caught.value.__cause__)
    if failure_kind == "ready_status":
        assert assets.validate_gemma_install(install_dir, spec=spec)
    else:
        assert not install_dir.exists()
    assert modelscope_server.requests == []
    assert list(tmp_path.glob("gemma.staging-*")) == []
