import asyncio
import hashlib
import threading
from dataclasses import replace

import pytest
from puripuly_heart.core.local_asr_provisioning import LocalASRInstallRequest
from puripuly_heart.core.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LOCAL_STT_MODEL_ID,
    load_local_stt_asset_manifest,
)
from puripuly_heart.core.local_stt_download_port import LocalSTTDownloadPortError
from puripuly_heart.core.local_stt_runtime_installer import (
    LocalSTTRuntimeInstallError,
    _source_plans,
    ensure_local_stt_installed,
)

from puripuly_heart.core.model_download_sources import HF_MIRROR_ENDPOINT
from tests.core.runtime.test_local_asr_provisioning import (
    MutableProvisioningBackend,
    _all_states,
    _owner,
)


@pytest.mark.parametrize(
    "selected,expected",
    [
        ("huggingface", ("huggingface", None, "huggingface")),
        ("hf_mirror", ("huggingface", HF_MIRROR_ENDPOINT, "hf_mirror")),
        ("modelscope", ("modelscope", None, "modelscope")),
    ],
)
def test_explicit_source_never_falls_back(selected, expected):
    manifest = load_local_stt_asset_manifest(LOCAL_STT_MODEL_ID)
    assert _source_plans(manifest, preferred_source=selected, locale="zh-CN") == (expected,)


def test_unsupported_modelscope_does_not_fall_back_to_hf():
    manifest = load_local_stt_asset_manifest("parakeet-tdt-0.6b-v3-int8-sherpa")
    with pytest.raises(LocalSTTRuntimeInstallError) as caught:
        _source_plans(manifest, preferred_source="modelscope", locale=None)
    assert caught.value.failure_code == "unsupported_source"


@pytest.mark.parametrize("source,endpoint", [("hf_mirror", "https://hf-mirror.net")])
async def test_endpoint_reaches_downloader_and_pinned_checksum_is_enforced(
    tmp_path, source, endpoint
):
    manifest = load_local_stt_asset_manifest(LOCAL_STT_MODEL_ID)
    data = b"tiny valid model"
    asset = replace(
        manifest.files[0],
        relative_path="test.bin",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    manifest = replace(manifest, files=(asset,))
    requests = []

    class Downloader:
        async def download(self, request, **kwargs):
            requests.append(request)
            request.local_dir.mkdir(parents=True, exist_ok=True)
            path = request.local_dir / request.remote_path
            path.write_bytes(data)
            return path

    statuses = []
    installed = await ensure_local_stt_installed(
        model_root=tmp_path,
        manifest=manifest,
        preferred_source=source,
        huggingface_downloader=Downloader(),
        on_status=statuses.append,
    )
    assert len(requests) == 1 and requests[0].endpoint == endpoint
    assert requests[0].revision == manifest.sources["huggingface"].revision
    assert installed.selected_revision == requests[0].revision
    assert (tmp_path / manifest.install_dirname / "test.bin").read_bytes() == data
    assert statuses[-1].status == "ready"


async def test_source_error_retains_code_http_status_and_never_falls_back(tmp_path):
    manifest = load_local_stt_asset_manifest(LOCAL_STT_MODEL_ID)
    requests = []

    class Downloader:
        async def download(self, request, **kwargs):
            requests.append(request)
            raise LocalSTTDownloadPortError(
                "signed-uri-secret", failure_code="mirror_redirect_external", status_code=308
            )

    with pytest.raises(LocalSTTRuntimeInstallError) as caught:
        await ensure_local_stt_installed(
            model_root=tmp_path,
            manifest=manifest,
            preferred_source="hf_mirror",
            huggingface_downloader=Downloader(),
        )
    assert len(requests) == 1
    assert (
        caught.value.failure_code == "mirror_redirect_external" and caught.value.status_code == 308
    )
    assert "signed-uri-secret" not in str(caught.value)


async def test_owner_forwards_selected_source_and_byte_status(tmp_path):
    backend = MutableProvisioningBackend(_all_states())
    calls, statuses = [], []
    original = backend.install

    async def install(**kwargs):
        calls.append(kwargs["preferred_source"])
        return await original(**kwargs)

    owner = _owner(backend, model_root=tmp_path)
    owner._installer = install
    try:
        result = await owner.start_install(
            LocalASRInstallRequest(
                "cpu",
                (LOCAL_STT_MODEL_ID,),
                None,
                "model_assets",
                preferred_source="modelscope",
            ),
            on_status=statuses.append,
        )
        assert calls == ["modelscope"] and not result.failed_model_ids
        assert statuses[0].percent == 37 and statuses[-1].status == "ready"
    finally:
        await owner.close()


async def test_cancel_origin_does_not_cancel_other_owners_task(tmp_path):
    backend = MutableProvisioningBackend(_all_states(), release_install=asyncio.Event())
    owner = _owner(backend, model_root=tmp_path)
    try:
        task = owner.start_install(
            LocalASRInstallRequest("cpu", (LOCAL_STT_MODEL_ID,), None, "activation")
        )
        await backend.install_started.wait()
        await owner.cancel_install("cpu", origin="model_assets")
        assert not task.done()
        backend.release_install.set()
        assert not (await task).cancelled
    finally:
        await owner.close()


def _labels(manifest, locale):
    return tuple(plan[2] for plan in _source_plans(manifest, preferred_source=None, locale=locale))


def test_automatic_source_order_follows_locale_and_skips_unregistered_sources():
    qwen = load_local_stt_asset_manifest(LOCAL_STT_MODEL_ID)
    gpu = load_local_stt_asset_manifest(LOCAL_QWEN_GPU_MODEL_ID)

    assert _labels(qwen, "zh-CN") == ("modelscope", "hf_mirror", "huggingface")
    assert _labels(qwen, "ko") == ("huggingface", "modelscope", "hf_mirror")
    assert _labels(gpu, "zh-CN") == ("hf_mirror", "huggingface")
    assert _labels(gpu, "en") == ("huggingface", "hf_mirror")


async def test_cancelled_lease_acquisition_waits_and_releases_acquired_lease(tmp_path, monkeypatch):
    import puripuly_heart.core.local_asr.local_stt_runtime_installer as installer

    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    cancellation = threading.Event()

    class Lease:
        def close(self):
            closed.set()

    def acquire(**kwargs):
        entered.set()
        assert release.wait(5)
        return Lease()

    monkeypatch.setattr(installer.LocalSTTProvisioningLease, "acquire", acquire)
    task = asyncio.create_task(
        ensure_local_stt_installed(model_root=tmp_path, cancel_event=cancellation)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not closed.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert cancellation.is_set()
    assert closed.is_set()
