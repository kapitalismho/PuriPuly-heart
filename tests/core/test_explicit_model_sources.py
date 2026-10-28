import asyncio
import hashlib
import threading
from dataclasses import replace

import httpx
import pytest

from puripuly_heart.core.local_asr.local_asr_provisioning import LocalASRInstallRequest
from puripuly_heart.core.local_asr.local_stt_assets import (
    FUN_ASR_MLT_NANO_MODEL_ID,
    FUN_ASR_NANO_MODEL_ID,
    LOCAL_QWEN_GPU_MODEL_ID,
    ORUKEET_MODEL_ID,
    REQUIRED_CPU_LOCAL_STT_MODEL_IDS,
    load_local_stt_asset_manifest,
    validate_local_stt_install,
)
from puripuly_heart.core.local_asr.local_stt_download_port import LocalSTTDownloadPortError
from puripuly_heart.core.local_asr.local_stt_runtime_installer import (
    LocalSTTRuntimeInstallError,
    ensure_local_stt_installed,
)
from puripuly_heart.core.model_download_sources import HF_MIRROR_ENDPOINT
from puripuly_heart.core.runtime.local_asr_provisioning import LocalASRProvisioningOwner
from tests.core.runtime.test_local_asr_provisioning import (
    MutableProvisioningBackend,
    _all_states,
    _owner,
)
from tests.core.test_external_network import (
    isolated_network_environment as isolated_network_environment,
)
from tests.core.test_local_stt_runtime_installer import (
    file_server as file_server,
)
from tests.core.test_local_stt_runtime_installer import (
    temp_dir as temp_dir,
)

REPLACEMENT_MODELS = (ORUKEET_MODEL_ID, FUN_ASR_NANO_MODEL_ID, FUN_ASR_MLT_NANO_MODEL_ID)


def _local_manifest(model_id, source_directory, base_url):
    canonical = load_local_stt_asset_manifest(model_id)
    payload = f"synthetic pinned {model_id}".encode()
    asset = replace(
        canonical.files[0],
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    sources = {
        **canonical.sources,
        "huggingface": replace(
            canonical.sources["huggingface"],
            download_url_template=f"{base_url}/hf/{{path}}",
        ),
    }
    for prefix in ("hf", "mirror"):
        for source_name in canonical.sources:
            path = source_directory / prefix / asset.remote_path_for_source(source_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    return replace(canonical, files=(asset,), sources=sources), payload


class _LocalHuggingFaceTransport:
    def __init__(self, base_url):
        self.base_url = base_url
        self.requests = []

    async def download(self, request, **_kwargs):
        self.requests.append(request)
        prefix = {None: "hf", HF_MIRROR_ENDPOINT: "mirror"}[request.endpoint]
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(f"{self.base_url}/{prefix}/{request.remote_path}")
            response.raise_for_status()
        path = request.local_dir / request.remote_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return path


@pytest.mark.parametrize("selected", ["huggingface", "hf_mirror"])
@pytest.mark.parametrize("corrupt_selected", [False, True])
async def test_explicit_choice_is_the_only_source_and_requires_pinned_bytes(
    tmp_path, file_server, selected, corrupt_selected
):
    directory, url = file_server
    manifest, payload = _local_manifest(FUN_ASR_NANO_MODEL_ID, directory, url)
    selected_prefix = "hf" if selected == "huggingface" else "mirror"
    rejected_prefix = "mirror" if selected == "huggingface" else "hf"
    corrupted = selected_prefix if corrupt_selected else rejected_prefix
    (directory / corrupted / manifest.files[0].relative_path).write_bytes(b"x" * len(payload))
    transport = _LocalHuggingFaceTransport(url)
    statuses = []
    options = dict(
        model_root=tmp_path / "models",
        manifest=manifest,
        locale="zh-CN" if selected == "huggingface" else "en",
        preferred_source=selected,
        huggingface_downloader=transport,
        on_status=statuses.append,
    )
    install_dir = options["model_root"] / manifest.install_dirname
    if corrupt_selected:
        with pytest.raises(LocalSTTRuntimeInstallError) as rejected:
            await ensure_local_stt_installed(**options)
        assert rejected.value.failure_code == "checksum_failed"
        assert not install_dir.exists()
        assert statuses[-1].status == "download_failed"
    else:
        installed = await ensure_local_stt_installed(**options)
        assert installed.selected_source == selected
        assert installed.selected_revision == manifest.sources["huggingface"].revision
        assert validate_local_stt_install(install_dir, manifest=manifest) == installed
        assert (install_dir / manifest.files[0].relative_path).read_bytes() == payload
        assert statuses[-1].status == "ready"
    assert not list(options["model_root"].glob("*.staging-*"))
    assert len(transport.requests) == (1 if selected == "hf_mirror" else 0)


@pytest.mark.parametrize("model_id", (*REQUIRED_CPU_LOCAL_STT_MODEL_IDS, LOCAL_QWEN_GPU_MODEL_ID))
async def test_unsupported_modelscope_choice_cannot_install_from_huggingface(
    tmp_path, file_server, model_id
):
    directory, url = file_server
    manifest, _payload = _local_manifest(model_id, directory, url)
    transport = _LocalHuggingFaceTransport(url)
    with pytest.raises(LocalSTTRuntimeInstallError) as rejected:
        await ensure_local_stt_installed(
            model_id=model_id,
            model_root=tmp_path / "models",
            manifest=manifest,
            preferred_source="modelscope",
            huggingface_downloader=transport,
        )
    assert rejected.value.failure_code == "unsupported_source"
    assert transport.requests == []
    assert not (tmp_path / "models" / manifest.install_dirname).exists()


async def test_explicit_source_error_retains_safe_code_and_status_without_fallback(
    tmp_path, file_server
):
    directory, url = file_server
    manifest, _payload = _local_manifest(ORUKEET_MODEL_ID, directory, url)
    requests = []

    class Downloader:
        async def download(self, request, **_kwargs):
            requests.append(request)
            raise LocalSTTDownloadPortError(
                "signed-uri-secret", failure_code="mirror_redirect_external", status_code=308
            )

    with pytest.raises(LocalSTTRuntimeInstallError) as caught:
        await ensure_local_stt_installed(
            model_root=tmp_path / "models",
            manifest=manifest,
            preferred_source="hf_mirror",
            huggingface_downloader=Downloader(),
        )
    assert len(requests) == 1
    assert caught.value.failure_code == "mirror_redirect_external"
    assert caught.value.status_code == 308
    assert "signed-uri-secret" not in str(caught.value)
    assert not (tmp_path / "models" / manifest.install_dirname).exists()


async def test_owner_selected_source_completes_real_install_and_ready_publication(
    tmp_path, file_server
):
    directory, url = file_server
    manifest, payload = _local_manifest(FUN_ASR_NANO_MODEL_ID, directory, url)
    (directory / "hf" / manifest.files[0].relative_path).write_bytes(b"x" * len(payload))
    transport = _LocalHuggingFaceTransport(url)
    statuses = []
    owner = LocalASRProvisioningOwner(
        model_root=tmp_path / "models",
        manifest_loader=lambda model_id: (
            manifest if model_id == manifest.model_id else load_local_stt_asset_manifest(model_id)
        ),
        huggingface_downloader=transport,
    )
    try:
        result = await owner.start_install(
            LocalASRInstallRequest(
                "cpu", (FUN_ASR_NANO_MODEL_ID,), "en", "model_assets", preferred_source="hf_mirror"
            ),
            on_status=statuses.append,
        )
        assert result.installed_model_ids == (FUN_ASR_NANO_MODEL_ID,)
        assert result.failed_model_ids == ()
        assert result.failure_codes == ()
        assert result.snapshot.state_for(FUN_ASR_NANO_MODEL_ID).integrity == "ready"
        assert not result.snapshot.cpu_auto_available
        installed = validate_local_stt_install(
            tmp_path / "models" / manifest.install_dirname, manifest=manifest
        )
        assert installed.selected_source == "hf_mirror"
        assert statuses[0].status == "downloading"
        assert statuses[-1].status == "ready"
    finally:
        await owner.close()
    assert owner.is_closed
    assert owner.child_states == ()


async def test_cancel_origin_does_not_cancel_other_owners_task(tmp_path):
    backend = MutableProvisioningBackend(_all_states(), release_install=asyncio.Event())
    owner = _owner(backend, model_root=tmp_path)
    try:
        task = owner.start_install(
            LocalASRInstallRequest("cpu", (FUN_ASR_NANO_MODEL_ID,), None, "activation")
        )
        await backend.install_started.wait()
        await owner.cancel_install("cpu", origin="model_assets")
        assert not task.done()
        backend.release_install.set()
        assert not (await task).cancelled
    finally:
        await owner.close()


@pytest.mark.parametrize("model_id", REPLACEMENT_MODELS)
@pytest.mark.parametrize("locale,expected", [("zh-CN", "hf_mirror"), ("ko", "huggingface")])
async def test_automatic_choice_installs_current_manifest_from_locale_preferred_source(
    tmp_path, file_server, model_id, locale, expected
):
    directory, url = file_server
    manifest, _payload = _local_manifest(model_id, directory, url)
    installed = await ensure_local_stt_installed(
        model_id=model_id,
        model_root=tmp_path / "models",
        manifest=manifest,
        locale=locale,
        huggingface_downloader=_LocalHuggingFaceTransport(url),
    )
    assert installed.selected_source == expected
    assert (
        validate_local_stt_install(
            tmp_path / "models" / manifest.install_dirname, manifest=manifest
        )
        == installed
    )


@pytest.mark.parametrize(
    "locale,corrupt_prefix,expected",
    [("zh-CN", "mirror", "huggingface"), ("en", "hf", "hf_mirror")],
)
async def test_automatic_choice_can_recover_from_bad_first_source_using_same_pinned_bytes(
    tmp_path, file_server, locale, corrupt_prefix, expected
):
    directory, url = file_server
    manifest, payload = _local_manifest(FUN_ASR_NANO_MODEL_ID, directory, url)
    (directory / corrupt_prefix / manifest.files[0].relative_path).write_bytes(b"x" * len(payload))
    installed = await ensure_local_stt_installed(
        model_root=tmp_path / "models",
        manifest=manifest,
        locale=locale,
        huggingface_downloader=_LocalHuggingFaceTransport(url),
    )
    assert installed.selected_source == expected
    assert (
        validate_local_stt_install(
            tmp_path / "models" / manifest.install_dirname, manifest=manifest
        )
        == installed
    )


async def test_cancelled_lease_acquisition_waits_and_releases_acquired_lease(tmp_path, monkeypatch):
    import puripuly_heart.core.local_asr.local_stt_runtime_installer as installer

    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    cancellation = threading.Event()

    class Lease:
        def close(self):
            closed.set()

    def acquire(**_kwargs):
        entered.set()
        assert release.wait(5)
        return Lease()

    monkeypatch.setattr(installer.LocalSTTProvisioningLease, "acquire", acquire)
    task = asyncio.create_task(
        ensure_local_stt_installed(
            model_id=ORUKEET_MODEL_ID, model_root=tmp_path, cancel_event=cancellation
        )
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
