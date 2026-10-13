import asyncio
import errno
import hashlib
import threading
from dataclasses import replace

import httpx
import pytest

from puripuly_heart.core.local_asr.local_stt_download_port import LocalSTTDownloadPortError
from puripuly_heart.core.local_translation.assets import (
    GemmaAsset,
    e4b_gemma_spec,
    validate_gemma_install,
)
from puripuly_heart.core.local_translation.provisioning import (
    GemmaProvisioningCancelled,
    GemmaProvisioningError,
    LocalGemmaDownloadAdapter,
    ensure_gemma_installed,
)
from puripuly_heart.core.local_translation.public_download import download_public_file


def small_spec():
    return replace(
        e4b_gemma_spec(),
        assets=tuple(
            GemmaAsset(name, len(data), hashlib.sha256(data).hexdigest())
            for name, data in (("a.gguf", b"aaaa"), ("b.gguf", b"bbbb"))
        ),
    )


async def test_import_manifest_and_no_duplicate_download(tmp_path):
    spec = small_spec()
    files = tuple(tmp_path / a.filename for a in spec.assets)
    for path, data in zip(files, (b"aaaa", b"bbbb")):
        path.write_bytes(data)
    install = tmp_path / "installed"
    updates = []
    await ensure_gemma_installed(
        source="local",
        spec=spec,
        downloader=LocalGemmaDownloadAdapter(files, spec),
        install_dir=install,
        on_status=updates.append,
    )
    validate_gemma_install(install, spec=spec)
    assert {u.state for u in updates} >= {"verifying", "installing", "ready"}
    await ensure_gemma_installed(source="local", spec=spec, install_dir=install)


async def test_wrong_import_does_not_replace_existing_directory(tmp_path):
    spec = small_spec()
    old = tmp_path / "installed"
    old.mkdir()
    (old / "keep").write_bytes(b"old-model")
    files = tuple(tmp_path / a.filename for a in spec.assets)
    for path in files:
        path.write_bytes(b"xxxx")
    with pytest.raises(GemmaProvisioningError):
        await ensure_gemma_installed(
            source="local",
            spec=spec,
            downloader=LocalGemmaDownloadAdapter(files, spec),
            install_dir=old,
        )
    assert (old / "keep").read_bytes() == b"old-model"
    assert not list(tmp_path.glob("*.staging-*"))


async def test_cancel_never_opens_downloader(tmp_path):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GemmaProvisioningCancelled):
        await ensure_gemma_installed(
            source="mirror", install_dir=tmp_path / "model", cancel_event=cancel
        )


def payload(tmp_path):
    return dict(
        endpoint="https://hf-mirror.com",
        repo_id="repo/model",
        revision="pin",
        remote_path="a.gguf",
        local_dir=str(tmp_path),
        expected_size_bytes=4,
    )


def test_mirror_rejects_external_redirect_without_contacting_it(tmp_path):
    requests = []

    def respond(request):
        requests.append(request)
        assert "authorization" not in request.headers
        return httpx.Response(
            302, headers={"Location": "https://huggingface.co/file?secret=do-not-log"}
        )

    def factory(**kw):
        return httpx.Client(transport=httpx.MockTransport(respond), **kw)

    with pytest.raises(LocalSTTDownloadPortError) as error:
        download_public_file(payload(tmp_path), on_progress=lambda _: None, client_factory=factory)
    assert error.value.failure_code == "mirror_redirect_external"
    assert len(requests) == 1
    assert "do-not-log" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"aaaa"),
        httpx.Response(206, headers={"content-range": "bytes 0-3/4"}, content=b"aaaa"),
    ],
)
def test_valid_range_or_restart(tmp_path, response):
    def factory(**kw):
        return httpx.Client(transport=httpx.MockTransport(lambda _: response), **kw)

    assert (
        download_public_file(
            payload(tmp_path), on_progress=lambda _: None, client_factory=factory
        ).read_bytes()
        == b"aaaa"
    )


def test_bad_range_is_not_appended(tmp_path):
    (tmp_path / "a.gguf.partial").write_bytes(b"aa")

    def factory(**kw):
        return httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    206, headers={"content-range": "bytes 0-3/4"}, content=b"aaaa"
                )
            ),
            **kw,
        )

    with pytest.raises(LocalSTTDownloadPortError) as error:
        download_public_file(payload(tmp_path), on_progress=lambda _: None, client_factory=factory)
    assert error.value.failure_code == "range_invalid"
    assert (tmp_path / "a.gguf.partial").read_bytes() == b"aa"


def test_complete_partial_is_reused_without_invalid_range(tmp_path):
    (tmp_path / "a.gguf.partial").write_bytes(b"aaaa")
    requests = []

    def factory(**kw):
        return httpx.Client(
            transport=httpx.MockTransport(lambda request: requests.append(request)), **kw
        )

    assert (
        download_public_file(
            payload(tmp_path), on_progress=lambda _: None, client_factory=factory
        ).read_bytes()
        == b"aaaa"
    )
    assert not requests


async def test_wrong_import_is_rejected_even_when_valid_model_is_installed(tmp_path):
    spec = small_spec()
    files = tuple(tmp_path / a.filename for a in spec.assets)
    for path, data in zip(files, (b"aaaa", b"bbbb")):
        path.write_bytes(data)
    install = tmp_path / "installed"
    await ensure_gemma_installed(
        source="local",
        spec=spec,
        downloader=LocalGemmaDownloadAdapter(files, spec),
        install_dir=install,
    )
    files[0].write_bytes(b"xxxx")
    with pytest.raises(GemmaProvisioningError) as error:
        await ensure_gemma_installed(
            source="local",
            spec=spec,
            downloader=LocalGemmaDownloadAdapter(files, spec),
            install_dir=install,
        )
    assert error.value.failure_code == "checksum_failed"
    validate_gemma_install(install, spec=spec)


async def test_insufficient_disk_does_not_replace_existing_install(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import puripuly_heart.core.local_translation.provisioning as provisioning

    monkeypatch.setattr(provisioning.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    install = tmp_path / "installed"
    install.mkdir()
    (install / "keep").write_bytes(b"old")
    with pytest.raises(GemmaProvisioningError) as error:
        await ensure_gemma_installed(source="mirror", spec=small_spec(), install_dir=install)
    assert error.value.failure_code == "disk_full"
    assert error.value.__cause__.errno == errno.ENOSPC
    assert (install / "keep").read_bytes() == b"old"


def test_timeout_retries_are_bounded_and_safe(monkeypatch, tmp_path):
    import puripuly_heart.core.local_translation.public_download as public_download

    requests = []
    monkeypatch.setattr(public_download.time, "sleep", lambda _: None)

    def respond(request):
        requests.append(request)
        raise httpx.ReadTimeout("secret signed url must not appear")

    def factory(**kw):
        return httpx.Client(transport=httpx.MockTransport(respond), **kw)

    with pytest.raises(LocalSTTDownloadPortError) as error:
        download_public_file(payload(tmp_path), on_progress=lambda _: None, client_factory=factory)
    assert error.value.failure_code == "download_timeout"
    assert len(requests) == 3
    assert "secret signed url" not in str(error.value)


async def test_cancel_verification_joins_worker_before_staging_cleanup(monkeypatch, tmp_path):
    import puripuly_heart.core.local_translation.provisioning as provisioning

    spec = small_spec()
    files = tuple(tmp_path / a.filename for a in spec.assets)
    for path, data in zip(files, (b"aaaa", b"bbbb")):
        path.write_bytes(data)
    install = tmp_path / "installed"
    install.mkdir()
    (install / "keep").write_bytes(b"old")
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()

    def verify(path, **kwargs):
        entered.set()
        try:
            assert release.wait(5)
            assert path.is_dir()
            kwargs["check_cancelled"]()
        finally:
            exited.set()

    monkeypatch.setattr(provisioning, "validate_gemma_install", verify)
    cancellation = threading.Event()
    task = asyncio.create_task(
        ensure_gemma_installed(
            source="local",
            spec=spec,
            downloader=LocalGemmaDownloadAdapter(files, spec),
            install_dir=install,
            cancel_event=cancellation,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and list(tmp_path.glob("*.staging-*"))
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert exited.is_set() and cancellation.is_set()
    assert (install / "keep").read_bytes() == b"old"
    assert not list(tmp_path.glob("*.staging-*"))
