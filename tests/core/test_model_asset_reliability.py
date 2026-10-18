from __future__ import annotations

import asyncio
import hashlib
import threading
from dataclasses import replace

import httpx
import pytest

from puripuly_heart.app.services.managed_gemma_translation import ManagedGemmaTranslationOwner
from puripuly_heart.app.services.model_assets import ModelAssetsApplication, ModelAssetsError
from puripuly_heart.app.services.model_assets_catalog import ModelAssetChoice
from puripuly_heart.core import network_clients
from puripuly_heart.core.local_asr import local_stt_runtime_installer as installer
from puripuly_heart.core.local_asr.local_stt_assets import load_local_stt_asset_manifest
from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadProgress,
    LocalSTTDownloadPortCancelled,
    LocalSTTDownloadPortError,
)
from puripuly_heart.core.local_translation import assets, provisioning
from puripuly_heart.core.local_translation.public_download import download_public_file
from tests.app.test_managed_gemma_translation import RecordingRuntime, _readiness, _selection
from tests.core.runtime.test_local_asr_provisioning import (
    MutableProvisioningBackend,
    _all_states,
    _owner,
)

ASR_IDS = (
    "qwen3-asr-0.6b-int8-sherpa",
    "qwen3-asr-1.7b-q6-k-transcribe-vulkan",
    "parakeet-tdt-0.6b-v3-int8-sherpa",
    "parakeet-tdt-ctc-0.6b-ja-int8-sherpa",
)
SOURCE_ERRORS = (
    "download_failed",
    "download_timeout",
    "mirror_unavailable",
    "mirror_redirect_external",
    "range_invalid",
    "file_size_mismatch",
    "checksum_failed",
)


@pytest.mark.parametrize("model_id", ASR_IDS)
@pytest.mark.parametrize("redirect", [False, True])
async def test_every_asr_mirror_uses_guarded_range_and_preserves_registered_pins(
    tmp_path, monkeypatch, model_id, redirect
):
    original = load_local_stt_asset_manifest(model_id)
    data = b"aaaa"
    manifest = replace(
        original,
        files=tuple(
            replace(asset, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            for asset in original.files
        ),
    )
    requests, downloads, updates = [], [], []

    def respond(request):
        requests.append(request)
        if redirect:
            if request.url.host == "hf-mirror.net":
                return httpx.Response(302, headers={"location": "https://huggingface.co/external"})
            return httpx.Response(200, content=data)
        return httpx.Response(200, content=data)

    def factory(**kwargs):
        return httpx.Client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(network_clients, "external_client", factory)

    class PublicDownloader:
        async def download(self, request, *, cancel_event, on_progress):
            downloads.append(request)
            request.local_dir.mkdir(parents=True, exist_ok=True)
            return download_public_file(
                {
                    "endpoint": request.endpoint,
                    "repo_id": request.repo_id,
                    "revision": request.revision,
                    "remote_path": request.remote_path,
                    "local_dir": str(request.local_dir),
                    "expected_size_bytes": request.expected_size_bytes,
                },
                on_progress=lambda update: on_progress(
                    HuggingFaceDownloadProgress(update["downloaded_bytes"], update["total_bytes"])
                ),
                client_factory=factory,
            )

    arguments = dict(
        model_root=tmp_path,
        manifest=manifest,
        preferred_source="hf_mirror",
        huggingface_downloader=PublicDownloader(),
        on_status=updates.append,
    )
    if redirect:
        with pytest.raises(installer.LocalSTTRuntimeInstallError) as caught:
            await installer.ensure_local_stt_installed(**arguments)
        assert caught.value.failure_code == "mirror_redirect_external"
        assert not (tmp_path / manifest.install_dirname).exists()
        assert len(requests) == 1
    else:
        installed = await installer.ensure_local_stt_installed(**arguments)
        selected = "hf_mirror" if "hf_mirror" in original.sources else "huggingface"
        source = original.sources[selected]
        assert installed.selected_revision == source.revision
        assert installed.selected_source == selected
        assert len(downloads) == len(original.files)
        for request, asset in zip(downloads, original.files):
            assert request.endpoint == "https://hf-mirror.net"
            assert request.repo_id == source.repo_id
            assert request.revision == source.revision
            assert request.remote_path == asset.remote_path_for_source(selected)
            assert (tmp_path / manifest.install_dirname / asset.relative_path).read_bytes() == data
        assert updates[-1].status == "ready"
        installer.validate_local_stt_install(tmp_path / manifest.install_dirname, manifest=manifest)
    assert all(request.url.host == "hf-mirror.net" for request in requests)
    assert all(request.headers.get("range") == "bytes=0-" for request in requests)
    assert all("authorization" not in request.headers for request in requests)
    assert not tuple(tmp_path.glob("*.staging-*"))


def _small_gemma(monkeypatch, tmp_path, model_id=assets.GEMMA_MODEL_ID):
    pinned = assets.resolve_gemma_spec(model_id)
    contents = {
        asset.filename: bytes([97 + index]) * 4 for index, asset in enumerate(pinned.assets)
    }
    spec = replace(
        pinned,
        assets=tuple(
            assets.GemmaAsset(name, len(data), hashlib.sha256(data).hexdigest())
            for name, data in contents.items()
        ),
    )
    monkeypatch.setattr(
        assets,
        "gemma_26b_spec" if model_id == assets.GEMMA_26B_MODEL_ID else "e4b_gemma_spec",
        lambda: spec,
    )
    monkeypatch.setattr(assets, "default_models_dir", lambda: tmp_path / "models")
    return spec, contents


@pytest.mark.parametrize("failure_code", SOURCE_ERRORS)
@pytest.mark.parametrize("explicit", [False, True])
async def test_typed_source_failures_progress_automatically_but_not_for_explicit_source(
    tmp_path, monkeypatch, failure_code, explicit
):
    spec, contents = _small_gemma(monkeypatch, tmp_path)
    attempts, updates = [], []

    class Downloader:
        async def download(self, request, **kwargs):
            attempts.append("huggingface")
            raise LocalSTTDownloadPortError("controlled source failure", failure_code=failure_code)

    def respond(request):
        attempts.append("modelscope")
        return httpx.Response(200, content=contents[request.url.params["FilePath"]])

    monkeypatch.setattr(
        network_clients,
        "external_client",
        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs),
    )
    install = tmp_path / "installed"
    install.mkdir()
    (install / "previous").write_bytes(b"preserve")
    arguments = dict(
        spec=spec,
        install_dir=install,
        downloader=Downloader(),
        source="huggingface" if explicit else None,
        locale="en",
        on_status=updates.append,
    )
    if explicit:
        with pytest.raises(provisioning.GemmaProvisioningError) as caught:
            await provisioning.ensure_gemma_installed(**arguments)
        assert caught.value.failure_code == failure_code
        assert attempts == ["huggingface"]
        assert (install / "previous").read_bytes() == b"preserve"
    else:
        await provisioning.ensure_gemma_installed(**arguments)
        assert attempts == ["huggingface", *(["modelscope"] * len(spec.assets))]
        assert updates[-1].state == "ready"
        assets.validate_gemma_install(install, spec=spec)
    assert not tuple(tmp_path.glob("*.staging-*"))


@pytest.mark.parametrize("failure_code", SOURCE_ERRORS)
async def test_chinese_source_order_reaches_direct_hf_after_typed_mirror_failure(
    tmp_path, monkeypatch, failure_code
):
    spec, contents = _small_gemma(monkeypatch, tmp_path)
    attempts = []

    def respond(request):
        attempts.append("modelscope")
        return httpx.Response(503)

    monkeypatch.setattr(
        network_clients,
        "external_client",
        lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs),
    )

    class Downloader:
        async def download(self, request, **kwargs):
            if request.endpoint is not None:
                attempts.append("hf_mirror")
                raise LocalSTTDownloadPortError(
                    "controlled mirror failure", failure_code=failure_code
                )
            attempts.append("huggingface")
            path = request.local_dir / request.remote_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents[request.remote_path])
            return path

    install = tmp_path / "installed"
    await provisioning.ensure_gemma_installed(
        spec=spec, install_dir=install, downloader=Downloader(), locale="zh-CN"
    )
    assert attempts == ["modelscope", "hf_mirror", *(["huggingface"] * len(spec.assets))]
    assets.validate_gemma_install(install, spec=spec)
    assert not tuple(tmp_path.glob("*.staging-*"))


@pytest.mark.parametrize(
    "failure_code",
    [
        "network_configuration_failed",
        "worker_spawn_failed",
        "worker_protocol_failed",
        "disk_full",
        "cancelled",
    ],
)
async def test_non_source_errors_and_cancellation_never_advance_source(
    tmp_path, monkeypatch, failure_code
):
    spec, _contents = _small_gemma(monkeypatch, tmp_path)
    attempts = []

    class Downloader:
        async def download(self, request, **kwargs):
            attempts.append(request.endpoint)
            error_type = (
                LocalSTTDownloadPortCancelled
                if failure_code == "cancelled"
                else LocalSTTDownloadPortError
            )
            raise error_type("controlled non-source failure", failure_code=failure_code)

    def unexpected(**kwargs):
        raise AssertionError("Non-source failures must not try another source")

    monkeypatch.setattr(network_clients, "external_client", unexpected)
    with pytest.raises(provisioning.GemmaProvisioningError):
        await provisioning.ensure_gemma_installed(
            spec=spec, install_dir=tmp_path / "installed", downloader=Downloader(), locale="en"
        )
    assert attempts == [None]


def _asset_application(owner, spec):
    return ModelAssetsApplication(
        provisioning=lambda: None,
        gemma=lambda: owner,
        locale=lambda: "en",
        choices=(ModelAssetChoice(spec.model_id, "Gemma", "gemma", "translation", 8),),
    )


@pytest.mark.parametrize("model_id", [assets.GEMMA_MODEL_ID, assets.GEMMA_26B_MODEL_ID])
async def test_explicit_asset_repair_checks_integrity_and_preserves_cheap_reuse(
    tmp_path, monkeypatch, model_id
):
    spec, contents = _small_gemma(monkeypatch, tmp_path, model_id)
    calls = []
    corrupt_transfer = False

    class Downloader:
        async def download(self, request, *, cancel_event, on_progress):
            calls.append(request.remote_path)
            path = request.local_dir / request.remote_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"xxxx" if corrupt_transfer else contents[request.remote_path])
            return path

    downloader = Downloader()
    owner = ManagedGemmaTranslationOwner(
        runtime=RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"]))),
        downloader=downloader,
    )
    application = _asset_application(owner, spec)
    install = assets.default_gemma_install_dir(spec)
    try:
        assert (await application.download(model_id, source="huggingface")).status == "installed"
        calls.clear()
        assert (await application.download(model_id, source="huggingface")).status == "installed"
        assert calls == []
        model = install / spec.model_filename
        model.write_bytes(b"xxxx")
        assert owner.install_status(model_id) == "ready"
        await provisioning.ensure_gemma_installed(spec=spec, downloader=downloader)
        assert calls == []
        assert model.read_bytes() == b"xxxx"
        corrupt_transfer = True
        with pytest.raises(ModelAssetsError) as caught:
            await application.download(model_id, source="huggingface")
        assert caught.value.failure_code == "checksum_failed"
        assert model.read_bytes() == b"xxxx"
        corrupt_transfer = False
        entry = await application.download(model_id, source="huggingface")
        assert entry.status == "installed" and entry.failure_code is None
        assert len(calls) == 2 * len(spec.assets)
        assets.validate_gemma_install(install, spec=spec)
        assert not tuple(install.parent.glob("*.staging-*"))
    finally:
        await owner.close()


async def test_busy_asset_request_and_cancel_cannot_cancel_runtime_preparation(
    tmp_path, monkeypatch
):
    spec, _contents = _small_gemma(monkeypatch, tmp_path, assets.GEMMA_26B_MODEL_ID)
    entered, release = asyncio.Event(), asyncio.Event()
    transfers = []

    async def prepare(**kwargs):
        entered.set()
        await release.wait()
        return _readiness(str(kwargs["backend"]))

    class Downloader:
        async def download(self, request, **kwargs):
            transfers.append(request)
            raise AssertionError("Busy asset task must not start")

    owner = ManagedGemmaTranslationOwner(runtime=RecordingRuntime(prepare), downloader=Downloader())
    application = _asset_application(owner, spec)
    preparation = asyncio.create_task(owner.prepare(_selection()))
    await entered.wait()
    try:
        with pytest.raises(ModelAssetsError) as caught:
            await asyncio.wait_for(
                application.download(spec.model_id, source="huggingface"), timeout=1
            )
        assert caught.value.failure_code == "task_busy"
        assert await application.cancel(spec.model_id) is False
        assert not preparation.done()
        assert transfers == []
        release.set()
        assert (await preparation).backend == "cpu"
    finally:
        release.set()
        await asyncio.gather(preparation, return_exceptions=True)
        await owner.close()


async def test_cancel_admitted_asset_before_transfer_keeps_it_cancelled_and_allows_retry(
    tmp_path, monkeypatch
):
    spec, contents = _small_gemma(monkeypatch, tmp_path)
    transfers = []

    class Downloader:
        async def download(self, request, **kwargs):
            transfers.append(request)
            path = request.local_dir / request.remote_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents[request.remote_path])
            return path

    owner = ManagedGemmaTranslationOwner(
        runtime=RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"]))),
        downloader=Downloader(),
    )
    application = _asset_application(owner, spec)
    task = asyncio.create_task(application.download(spec.model_id, source="huggingface"))
    try:
        await asyncio.sleep(0)
        assert await application.cancel(spec.model_id) is True
        assert await application.cancel(spec.model_id) is True
        with pytest.raises(ModelAssetsError) as caught:
            await task
        assert caught.value.failure_code == "cancelled"
        assert transfers == []
        assert not application.busy
        assert application.entry(spec.model_id).failure_code is None
        assert (
            await application.download(spec.model_id, source="huggingface")
        ).status == "installed"
        assert len(transfers) == len(spec.assets)
    finally:
        await asyncio.gather(task, return_exceptions=True)
        await owner.close()


@pytest.mark.parametrize("finish", ["success", "cancel", "close"])
@pytest.mark.parametrize("linger", [False, True])
async def test_asset_lifetime_survives_demand_off_but_obeys_own_cancel_and_shutdown(
    tmp_path, monkeypatch, finish, linger
):
    spec, contents = _small_gemma(monkeypatch, tmp_path)
    started, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Downloader:
        async def download(self, request, *, cancel_event, on_progress):
            started.set()
            while not release.is_set():
                if cancel_event.is_set():
                    cancelled.set()
                    raise LocalSTTDownloadPortCancelled("cancelled")
                await asyncio.sleep(0)
            path = request.local_dir / request.remote_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents[request.remote_path])
            on_progress(HuggingFaceDownloadProgress(4, 4))
            return path

    runtime = RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"])))
    owner = ManagedGemmaTranslationOwner(
        runtime=runtime,
        downloader=Downloader(),
        idle_linger_s=0,
    )
    application = _asset_application(owner, spec)
    task = asyncio.create_task(application.download(spec.model_id, source="huggingface"))
    await started.wait()
    try:
        await asyncio.wait_for(owner.deactivate(linger=linger), timeout=1)
        if not linger:
            assert runtime.release_count == 1
        assert owner.cancel() is False
        assert owner.cancel_assets(assets.GEMMA_26B_MODEL_ID) is False
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done() and not cancelled.is_set()
        assert application.entry(spec.model_id).status == "downloading"
        if finish == "success":
            release.set()
            assert (await task).status == "installed"
            assets.validate_gemma_install(assets.default_gemma_install_dir(spec), spec=spec)
        else:
            if finish == "cancel":
                assert await application.cancel(spec.model_id) is True
                assert await application.cancel(spec.model_id) is True
            else:
                await owner.close()
            result = (await asyncio.gather(task, return_exceptions=True))[0]
            assert isinstance(result, (asyncio.CancelledError, ModelAssetsError))
            assert cancelled.is_set()
            assert not application.busy
            assert not tuple(assets.default_gemma_install_dir(spec).parent.glob("*.staging-*"))
            assert await application.cancel(spec.model_id) is False
            if finish == "cancel":
                release.set()
                assert (
                    await application.download(spec.model_id, source="huggingface")
                ).status == "installed"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await owner.close()


async def test_asset_conflicts_refuse_prepare_and_delete_then_allow_them_after_settlement(
    tmp_path, monkeypatch
):
    spec, contents = _small_gemma(monkeypatch, tmp_path, assets.GEMMA_26B_MODEL_ID)
    started, release = asyncio.Event(), asyncio.Event()
    runtime = RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"])))

    class Downloader:
        async def download(self, request, **kwargs):
            started.set()
            await release.wait()
            path = request.local_dir / request.remote_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents[request.remote_path])
            return path

    owner = ManagedGemmaTranslationOwner(runtime=runtime, downloader=Downloader())
    prior = await owner.prepare(_selection())
    application = _asset_application(owner, spec)
    task = asyncio.create_task(application.download(spec.model_id, source="huggingface"))
    await started.wait()
    try:
        for operation in (
            owner.prepare(replace(_selection("gpu"), model_id=spec.model_id)),
            owner.delete_assets(spec.model_id),
        ):
            with pytest.raises(provisioning.GemmaProvisioningError) as caught:
                await asyncio.wait_for(operation, timeout=1)
            assert caught.value.failure_code == "task_busy"
        assert owner.active_model_id == assets.GEMMA_MODEL_ID
        assert len(runtime.calls) == 1
        lease = installer.LocalSTTProvisioningLease.acquire(
            model_root=assets.default_gemma_install_dir(spec).parent, wait=False
        )
        if lease is not None:
            lease.close()
        assert lease is None
        await asyncio.wait_for(prior.release(), timeout=1)
        assert runtime.release_count == 1
        assert not task.done() and not release.is_set()
        release.set()
        assert (await task).status == "installed"
        active = await owner.prepare(replace(_selection("gpu"), model_id=spec.model_id))
        assert len(runtime.calls) == 2
        await active.release()
        await owner.delete_assets(spec.model_id)
        assert owner.install_status(spec.model_id) == "missing"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await owner.close()


@pytest.mark.parametrize("kind", ["asr", "gemma"])
@pytest.mark.parametrize("boundary", ["acquire", "delete"])
async def test_cancelled_deletion_retains_real_lease_until_worker_settles(
    tmp_path, monkeypatch, kind, boundary
):
    from puripuly_heart.core import model_install_removal
    from puripuly_heart.core.runtime import local_asr_provisioning

    if kind == "gemma":
        spec, _contents = _small_gemma(monkeypatch, tmp_path)
        owner = ManagedGemmaTranslationOwner(
            runtime=RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"])))
        )
        install = assets.default_gemma_install_dir(spec)

        def operation():
            return owner.delete_assets(spec.model_id)

    else:
        owner = _owner(MutableProvisioningBackend(_all_states()), model_root=tmp_path)
        model_id = ASR_IDS[0]
        install = tmp_path / load_local_stt_asset_manifest(model_id).install_dirname

        def operation():
            return owner.delete_model(model_id)

    install.mkdir(parents=True)
    (install / "old").write_bytes(b"old")
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    acquire = installer.LocalSTTProvisioningLease.acquire
    remove = model_install_removal.remove_model_install_dir
    acquired_leases = []

    def gated_acquire(**kwargs):
        lease = acquire(**kwargs)
        if lease is not None:
            acquired_leases.append(lease)
        if boundary == "acquire":
            entered.set()
            try:
                assert release.wait(5)
            finally:
                exited.set()
        return lease

    def gated_remove(path):
        if boundary == "delete":
            entered.set()
            try:
                assert release.wait(5)
            finally:
                exited.set()
        return remove(path)

    monkeypatch.setattr(installer.LocalSTTProvisioningLease, "acquire", gated_acquire)
    monkeypatch.setattr(model_install_removal, "remove_model_install_dir", gated_remove)
    monkeypatch.setattr(local_asr_provisioning, "remove_model_install_dir", gated_remove)
    task = asyncio.create_task(operation())
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        settled_early = task.done()
        second = acquire(model_root=install.parent, wait=False)
        if second is not None:
            second.close()
        assert not settled_early
        assert second is None
        assert install.exists()
    finally:
        release.set()
        result = (await asyncio.gather(task, return_exceptions=True))[0]
        assert await asyncio.to_thread(exited.wait, 5)
        after_settlement = acquire(model_root=install.parent, wait=False)
        if after_settlement is not None:
            after_settlement.close()
        for acquired_lease in acquired_leases:
            acquired_lease.close()
        await owner.close()
    assert isinstance(result, asyncio.CancelledError)
    assert after_settlement is not None
    lease = acquire(model_root=install.parent, wait=False)
    assert lease is not None
    try:
        install.mkdir(exist_ok=True)
        (install / "new").write_bytes(b"new")
    finally:
        lease.close()
    assert (install / "new").read_bytes() == b"new"
