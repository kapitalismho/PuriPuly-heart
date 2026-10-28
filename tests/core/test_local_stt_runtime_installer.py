from __future__ import annotations

import asyncio
import hashlib
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from puripuly_heart.core.local_asr.local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    InstalledLocalSTTManifest,
    LocalSTTAssetFile,
    LocalSTTAssetManifest,
    LocalSTTAssetSource,
    inspect_local_stt_install_state,
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
    cleanup_local_stt_install_residue,
    ensure_local_stt_installed,
)
from tests.core.test_external_network import (
    isolated_network_environment as isolated_network_environment,
)


class _QuietHttpHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        _ = format, args


@pytest.fixture()
def temp_dir() -> Path:
    base_dir = Path(tempfile.mkdtemp(prefix="local-stt-runtime-installer-"))
    try:
        yield base_dir
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_cleanup_restores_backup_and_removes_only_known_staging(tmp_path: Path) -> None:
    backup = tmp_path / "known-model.backup"
    staging = tmp_path / "known-model.staging-abcd"
    unrelated = tmp_path / "other-model.staging-abcd"
    backup.mkdir()
    staging.mkdir()
    unrelated.mkdir()
    (backup / "model.bin").write_bytes(b"ready")

    reconciled = cleanup_local_stt_install_residue(
        model_root=tmp_path,
        install_dirnames=("known-model",),
    )

    assert {path.name for path in reconciled} == {backup.name, staging.name}
    assert (tmp_path / "known-model" / "model.bin").read_bytes() == b"ready"
    assert not backup.exists()
    assert not staging.exists()
    assert unrelated.exists()


def test_cleanup_rejects_unsafe_install_directory_names(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.mkdir()

    with pytest.raises(ValueError, match="safe directory names"):
        cleanup_local_stt_install_residue(
            model_root=tmp_path,
            install_dirnames=("../sentinel",),
        )

    assert sentinel.exists()


def test_second_process_cleanup_cannot_disturb_active_provisioning_lease(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "known-model.staging-active"
    backup = tmp_path / "known-model.backup"
    ready = tmp_path / "lease-ready"
    release = tmp_path / "lease-release"
    staging.mkdir()
    backup.mkdir()
    script = "\n".join(
        (
            "import pathlib, sys, time",
            "from puripuly_heart.core.local_asr.local_stt_runtime_installer import "
            "LocalSTTProvisioningLease",
            "root = pathlib.Path(sys.argv[1])",
            "ready = pathlib.Path(sys.argv[2])",
            "release = pathlib.Path(sys.argv[3])",
            "lease = LocalSTTProvisioningLease.acquire(model_root=root, wait=True)",
            "ready.write_text('ready')",
            "while not release.exists():",
            "    time.sleep(0.01)",
            "lease.close()",
        )
    )
    child_environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[2] / "src"
    child_environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(source_root), *sys.path, child_environment.get("PYTHONPATH", ""))
        if part
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), str(ready), str(release)],
        env=child_environment,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()

        reconciled = cleanup_local_stt_install_residue(
            model_root=tmp_path,
            install_dirnames=("known-model",),
        )

        assert reconciled is None
        assert staging.exists()
        assert backup.exists()
    finally:
        release.touch()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    reconciled = cleanup_local_stt_install_residue(
        model_root=tmp_path,
        install_dirnames=("known-model",),
    )

    assert reconciled is not None
    assert not staging.exists()
    assert not backup.exists()
    assert (tmp_path / "known-model").exists()


@pytest.fixture()
def file_server(temp_dir: Path):
    source_dir = temp_dir / "source"
    source_dir.mkdir()

    handler = lambda *args, **kwargs: _QuietHttpHandler(  # noqa: E731
        *args, directory=str(source_dir), **kwargs
    )
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield source_dir, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _build_manifest(
    *,
    huggingface_url: str,
    modelscope_url: str,
    model_bytes: bytes,
    token_bytes: bytes,
    model_id: str = "test-model",
) -> LocalSTTAssetManifest:
    return LocalSTTAssetManifest(
        manifest_version=1,
        installed_manifest_version=1,
        model_id=model_id,
        engine="sherpa-onnx",
        upstream_repo="example/repo",
        install_dirname=model_id,
        sources={
            "huggingface": LocalSTTAssetSource(
                name="huggingface",
                revision="hf-rev-1",
                download_url_template=f"{huggingface_url}/{{path}}",
            ),
            "modelscope": LocalSTTAssetSource(
                name="modelscope",
                revision="ms-rev-1",
                download_url_template=f"{modelscope_url}/{{path}}",
            ),
        },
        files=(
            LocalSTTAssetFile(
                relative_path="model.int8.onnx",
                sha256=_sha256_bytes(model_bytes),
                size_bytes=len(model_bytes),
                source_path_overrides={"modelscope": "modelscope/model.int8.onnx"},
            ),
            LocalSTTAssetFile(
                relative_path="tokenizer/tokens.txt",
                sha256=_sha256_bytes(token_bytes),
                size_bytes=len(token_bytes),
            ),
        ),
    )


def _write_manifest_file(path: Path, payload: InstalledLocalSTTManifest) -> None:
    path.write_text(json.dumps(payload.to_dict(), indent=2), encoding="utf-8")


class _HeldDownloader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.release = asyncio.Event()
        self.cancellation_requested = asyncio.Event()
        self.requests = []
        self.pending = 0
        self.cleanup_failure: Exception | None = None

    async def download(self, request, *, cancel_event, on_progress):
        self.requests.append(request)
        self.pending += 1
        target = request.local_dir / request.remote_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.payload[:1])
        on_progress(HuggingFaceDownloadProgress(1, len(self.payload)))
        try:
            try:
                while not self.release.is_set():
                    if cancel_event.is_set():
                        self.cancellation_requested.set()
                        await self.release.wait()
                        raise LocalSTTDownloadPortCancelled("controlled download cancelled")
                    await asyncio.sleep(0.005)
            except asyncio.CancelledError:
                self.cancellation_requested.set()
                await self.release.wait()
                raise
            target.write_bytes(self.payload)
            on_progress(HuggingFaceDownloadProgress(len(self.payload), len(self.payload)))
            return target
        finally:
            self.pending -= 1
            if self.cleanup_failure is not None:
                raise self.cleanup_failure


class _HeldHttpDownload:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.release = threading.Event()
        self.finished = threading.Event()
        self.requests: list[str] = []
        self.truncate = False


@pytest.fixture()
def held_http_download():
    held = _HeldHttpDownload(b"synthetic-model")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            held.requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(held.payload)))
            self.end_headers()
            try:
                self.wfile.write(held.payload[:1])
                self.wfile.flush()
                held.release.wait()
                if not held.truncate:
                    self.wfile.write(held.payload[1:])
                    self.wfile.flush()
            finally:
                held.finished.set()

        def log_message(self, format, *args):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield held, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        held.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _held_manifest(base_url: str, *, port: bool) -> LocalSTTAssetManifest:
    manifest = _build_manifest(
        huggingface_url=f"{base_url}/hf",
        modelscope_url=f"{base_url}/ms",
        model_bytes=b"synthetic-model",
        token_bytes=b"",
    )
    if port:
        manifest = replace(
            manifest,
            model_id=LOCAL_QWEN_GPU_MODEL_ID,
            sources={
                "huggingface": replace(manifest.sources["huggingface"], repo_id="example/repo"),
                "modelscope": manifest.sources["modelscope"],
            },
        )
    return replace(manifest, files=manifest.files[:1])


async def _wait_until(predicate) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


def _assert_lease_available(model_root: Path) -> None:
    lease = LocalSTTProvisioningLease.acquire(model_root=model_root, wait=False)
    assert lease is not None
    lease.close()


def _failure_leaves(failure: BaseException) -> list[BaseException]:
    if isinstance(failure, BaseExceptionGroup):
        return [leaf for member in failure.exceptions for leaf in _failure_leaves(member)]
    return [failure]


@pytest.mark.asyncio
@pytest.mark.parametrize("port", [False, True], ids=["http-thread", "huggingface-port"])
@pytest.mark.parametrize("async_callback", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("grouped_observer", [False, True], ids=["ordinary", "mixed-group"])
@pytest.mark.parametrize("cancel_during_settlement", [False, True], ids=["drain", "cancel-drain"])
async def test_status_failure_keeps_download_staging_and_lease_until_settled_then_allows_retry(
    tmp_path: Path,
    held_http_download,
    port,
    async_callback,
    cancel_during_settlement,
    grouped_observer,
) -> None:
    held, base_url = held_http_download
    downloader = _HeldDownloader(held.payload) if port else None
    if port:
        held.release.set()
    release = downloader.release if port else held.release
    requests = downloader.requests if port else held.requests
    manifest = _held_manifest(base_url, port=port)
    model_root = tmp_path / "models"
    cancellation = threading.Event()
    observer_failure = (
        LocalSTTRuntimeInstallCancelled("observer failed")
        if async_callback
        else RuntimeError("observer failed")
    )
    if grouped_observer:
        observer_failure = BaseExceptionGroup(
            "observer failure group",
            (asyncio.CancelledError(), observer_failure),
        )
    statuses = []
    failed = asyncio.Event()

    def record(update):
        statuses.append(update)
        if update.downloaded_bytes:
            failed.set()
            raise observer_failure

    async def async_record(update):
        await asyncio.sleep(0)
        record(update)

    installation = asyncio.create_task(
        ensure_local_stt_installed(
            locale="en",
            model_root=model_root,
            manifest=manifest,
            cancel_event=cancellation,
            on_status=async_record if async_callback else record,
            huggingface_downloader=downloader,
        )
    )
    try:
        await asyncio.wait_for(failed.wait(), timeout=5)
        await _wait_until(cancellation.is_set)
        if port:
            await asyncio.wait_for(downloader.cancellation_requested.wait(), timeout=5)
            assert downloader.pending == 1
        else:
            assert not held.finished.is_set()
        if cancel_during_settlement:
            for _ in range(2):
                installation.cancel()
                await asyncio.sleep(0)
        assert not installation.done()
        staging = list(model_root.glob(f"{manifest.install_dirname}.staging-*"))
        assert len(staging) == 1
        assert (staging[0] / "model.int8.onnx").exists()
        assert (
            cleanup_local_stt_install_residue(
                model_root=model_root, install_dirnames=(manifest.install_dirname,)
            )
            is None
        )
        assert staging[0].exists()
        assert len(requests) == 1
        if port:
            assert held.requests == []
        assert {update.status for update in statuses} == {"downloading"}
        release.set()
        with pytest.raises(LocalSTTRuntimeInstallError) as caught:
            await asyncio.wait_for(installation, timeout=5)
        assert caught.value.failure_code == "status_callback_failed"
        assert caught.value.__cause__ is observer_failure
        assert len(requests) == 1
        if port:
            assert held.requests == []
        assert {update.status for update in statuses} == {"downloading"}
        assert not staging[0].exists()
        assert not (model_root / manifest.install_dirname).exists()
        assert downloader.pending == 0 if port else held.finished.is_set()
        _assert_lease_available(model_root)

        retry_statuses = []
        installed = await ensure_local_stt_installed(
            locale="en",
            model_root=model_root,
            manifest=manifest,
            on_status=retry_statuses.append,
            huggingface_downloader=downloader,
        )
        assert installed.selected_source == "huggingface"
        assert retry_statuses[-1].status == "ready"
        assert len(requests) == 2
        if port:
            assert held.requests == []
        assert (
            inspect_local_stt_install_state(
                model_root / manifest.install_dirname, manifest=manifest, verify_checksums=True
            ).status
            == "ready"
        )
        assert not list(model_root.glob(f"{manifest.install_dirname}.staging-*"))
        _assert_lease_available(model_root)
    finally:
        release.set()
        await asyncio.gather(installation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("port", [False, True], ids=["http-thread", "huggingface-port"])
async def test_status_failure_preserves_initiating_and_download_cleanup_errors(
    tmp_path: Path, held_http_download, port
) -> None:
    held, base_url = held_http_download
    downloader = _HeldDownloader(held.payload) if port else None
    cleanup_failure = RuntimeError("controlled port cleanup failed")
    if port:
        downloader.cleanup_failure = cleanup_failure
        held.release.set()
    else:
        held.truncate = True
    release = downloader.release if port else held.release
    manifest = _held_manifest(base_url, port=port)
    model_root = tmp_path / "models"
    cancellation = threading.Event()
    observer_failure = RuntimeError("observer failed")

    def on_status(update):
        if update.downloaded_bytes:
            raise observer_failure

    installation = asyncio.create_task(
        ensure_local_stt_installed(
            locale="en",
            model_root=model_root,
            manifest=manifest,
            cancel_event=cancellation,
            on_status=on_status,
            huggingface_downloader=downloader,
        )
    )
    try:
        await _wait_until(cancellation.is_set)
        assert not installation.done()
        release.set()
        with pytest.raises(ExceptionGroup) as caught:
            await asyncio.wait_for(installation, timeout=5)
        primary, secondary = _failure_leaves(caught.value)
        assert isinstance(primary, LocalSTTRuntimeInstallError)
        assert primary.__cause__ is observer_failure
        assert primary.failure_code == "status_callback_failed"
        if port:
            assert secondary is cleanup_failure
            assert downloader.pending == 0
        else:
            assert isinstance(secondary, httpx.RemoteProtocolError)
            assert held.finished.is_set()
        assert not list(model_root.glob(f"{manifest.install_dirname}.staging-*"))
        assert not (model_root / manifest.install_dirname).exists()
        _assert_lease_available(model_root)
    finally:
        release.set()
        await asyncio.gather(installation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("port", [False, True], ids=["http-thread", "huggingface-port"])
@pytest.mark.parametrize("task_cancel", [False, True], ids=["event-cancel", "task-cancel"])
async def test_cancellation_drains_admitted_work_before_cleanup_and_retry(
    tmp_path: Path, held_http_download, task_cancel, port
) -> None:
    held, base_url = held_http_download
    manifest = _held_manifest(base_url, port=port)
    downloader = _HeldDownloader(held.payload) if port else None
    if port:
        held.release.set()
    release = downloader.release if port else held.release
    model_root = tmp_path / "models"
    cancellation = threading.Event()
    progress = asyncio.Event()

    def on_status(update):
        if update.downloaded_bytes:
            progress.set()

    installation = asyncio.create_task(
        ensure_local_stt_installed(
            locale="en",
            model_root=model_root,
            manifest=manifest,
            cancel_event=cancellation,
            on_status=on_status,
            huggingface_downloader=downloader,
        )
    )
    try:
        await asyncio.wait_for(progress.wait(), timeout=5)
        if task_cancel:
            installation.cancel()
        else:
            cancellation.set()
        await _wait_until(cancellation.is_set)
        if port:
            await asyncio.wait_for(downloader.cancellation_requested.wait(), timeout=5)
        else:
            assert not held.finished.is_set()
        assert not installation.done()
        staging = list(model_root.glob(f"{manifest.install_dirname}.staging-*"))
        assert len(staging) == 1
        assert (
            cleanup_local_stt_install_residue(
                model_root=model_root, install_dirnames=(manifest.install_dirname,)
            )
            is None
        )
        if task_cancel:
            for _ in range(2):
                installation.cancel()
                await asyncio.sleep(0)
        assert not installation.done()
        release.set()
        expected = asyncio.CancelledError if task_cancel else LocalSTTRuntimeInstallCancelled
        with pytest.raises(expected):
            await asyncio.wait_for(installation, timeout=5)
        assert downloader.pending == 0 if port else held.finished.is_set()
        assert not staging[0].exists()
        _assert_lease_available(model_root)
        await ensure_local_stt_installed(
            locale="en",
            model_root=model_root,
            manifest=manifest,
            huggingface_downloader=downloader,
        )
        assert (
            inspect_local_stt_install_state(
                model_root / manifest.install_dirname, manifest=manifest, verify_checksums=True
            ).status
            == "ready"
        )
    finally:
        release.set()
        await asyncio.gather(installation, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("control_flow", [SystemExit, KeyboardInterrupt])
async def test_control_flow_observer_failure_is_not_source_fallback(
    tmp_path: Path, control_flow
) -> None:
    manifest = _held_manifest("http://127.0.0.1:1", port=True)
    downloader = _HeldDownloader(b"synthetic-model")
    failure = BaseExceptionGroup("control flow", (ValueError("observer"), control_flow()))

    def on_status(_update):
        raise failure

    with pytest.raises(BaseExceptionGroup) as caught:
        await ensure_local_stt_installed(
            locale="en",
            model_root=tmp_path,
            manifest=manifest,
            on_status=on_status,
            huggingface_downloader=downloader,
        )
    assert caught.value is failure
    assert downloader.requests == []
    assert not list(tmp_path.glob(f"{manifest.install_dirname}.staging-*"))
    _assert_lease_available(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "after_download", [False, True], ids=["before-admission", "after-settlement"]
)
async def test_status_failure_outside_active_download_is_not_source_fallback(
    tmp_path: Path, after_download
) -> None:
    manifest = _held_manifest("http://127.0.0.1:1", port=True)
    downloader = _HeldDownloader(b"synthetic-model")
    downloader.release.set()
    observer_failure = RuntimeError("observer failed")
    statuses = []
    model_root = tmp_path / "models"

    def on_status(update):
        statuses.append(update)
        if update.percent == (99 if after_download else 0):
            raise observer_failure

    with pytest.raises(LocalSTTRuntimeInstallError) as caught:
        await ensure_local_stt_installed(
            locale="en",
            model_root=model_root,
            manifest=manifest,
            on_status=on_status,
            huggingface_downloader=downloader,
        )
    assert caught.value.__cause__ is observer_failure
    assert caught.value.failure_code == "status_callback_failed"
    assert len(downloader.requests) == int(after_download)
    assert {update.status for update in statuses} == {"downloading"}
    assert not list(model_root.glob(f"{manifest.install_dirname}.staging-*"))
    assert not (model_root / manifest.install_dirname).exists()
    _assert_lease_available(model_root)
    await ensure_local_stt_installed(
        locale="en",
        model_root=model_root,
        manifest=manifest,
        huggingface_downloader=downloader,
    )
    assert (
        inspect_local_stt_install_state(
            model_root / manifest.install_dirname, manifest=manifest, verify_checksums=True
        ).status
        == "ready"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["staging", "lease"])
async def test_status_failure_preserves_distinct_owned_cleanup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary
) -> None:
    manifest = _held_manifest("http://127.0.0.1:1", port=True)
    downloader = _HeldDownloader(b"synthetic-model")
    model_root = tmp_path / "models"
    cancellation = threading.Event()
    observer_failure = RuntimeError("observer failed")
    cleanup_failure = OSError("controlled cleanup failed")
    statuses = []

    def on_status(update):
        statuses.append(update)
        if update.downloaded_bytes:
            raise observer_failure

    installation = asyncio.create_task(
        ensure_local_stt_installed(
            locale="en",
            model_root=model_root,
            manifest=manifest,
            cancel_event=cancellation,
            on_status=on_status,
            huggingface_downloader=downloader,
        )
    )
    try:
        await _wait_until(cancellation.is_set)
        assert downloader.pending == 1
        staging = list(model_root.glob(f"{manifest.install_dirname}.staging-*"))
        assert len(staging) == 1
        with monkeypatch.context() as patch:
            if boundary == "staging":
                original_rmtree = shutil.rmtree

                def fail_staging_cleanup(path, *args, **kwargs):
                    if path == staging[0]:
                        assert downloader.pending == 0
                        raise cleanup_failure
                    return original_rmtree(path, *args, **kwargs)

                patch.setattr(shutil, "rmtree", fail_staging_cleanup)
            else:
                original_close = LocalSTTProvisioningLease.close

                def fail_lease_cleanup(lease):
                    assert downloader.pending == 0
                    original_close(lease)
                    raise cleanup_failure

                patch.setattr(LocalSTTProvisioningLease, "close", fail_lease_cleanup)
            downloader.release.set()
            with pytest.raises(ExceptionGroup) as caught:
                await asyncio.wait_for(installation, timeout=5)
        primary, secondary = _failure_leaves(caught.value)
        assert primary.__cause__ is observer_failure
        assert secondary is cleanup_failure
        assert {update.status for update in statuses} == {"downloading"}
        assert downloader.pending == 0
        assert not (model_root / manifest.install_dirname).exists()
        assert staging[0].exists() == (boundary == "staging")
        _assert_lease_available(model_root)
        assert (
            cleanup_local_stt_install_residue(
                model_root=model_root, install_dirnames=(manifest.install_dirname,)
            )
            is not None
        )
        await ensure_local_stt_installed(
            locale="en",
            model_root=model_root,
            manifest=manifest,
            huggingface_downloader=downloader,
        )
        assert (
            inspect_local_stt_install_state(
                model_root / manifest.install_dirname, manifest=manifest, verify_checksums=True
            ).status
            == "ready"
        )
    finally:
        downloader.release.set()
        await asyncio.gather(installation, return_exceptions=True)


@pytest.mark.asyncio
async def test_http_task_cancellation_preserves_thread_cleanup_failure(
    tmp_path: Path, held_http_download
) -> None:
    held, base_url = held_http_download
    held.truncate = True
    manifest = _held_manifest(base_url, port=False)
    cancellation = threading.Event()
    progress = asyncio.Event()
    model_root = tmp_path / "models"

    def on_status(update):
        if update.downloaded_bytes:
            progress.set()

    installation = asyncio.create_task(
        ensure_local_stt_installed(
            locale="en",
            model_root=model_root,
            manifest=manifest,
            cancel_event=cancellation,
            on_status=on_status,
        )
    )
    try:
        await asyncio.wait_for(progress.wait(), timeout=5)
        installation.cancel()
        await _wait_until(cancellation.is_set)
        assert not installation.done()
        assert not held.finished.is_set()
        held.release.set()
        with pytest.raises(BaseExceptionGroup) as caught:
            await asyncio.wait_for(installation, timeout=5)
        primary, secondary = _failure_leaves(caught.value)
        assert isinstance(primary, asyncio.CancelledError)
        assert isinstance(secondary, httpx.RemoteProtocolError)
        assert not list(model_root.glob(f"{manifest.install_dirname}.staging-*"))
        assert not (model_root / manifest.install_dirname).exists()
        _assert_lease_available(model_root)
    finally:
        held.release.set()
        await asyncio.gather(installation, return_exceptions=True)


class _MirrorDownloader:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.requests = []

    async def download(self, request, **kwargs):
        self.requests.append(request)
        assert request.endpoint == "https://hf-mirror.net"
        assert request.repo_id == "example/repo"
        assert request.revision == "hf-rev-1"
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(f"{self.base_url}/mirror/{request.remote_path}")
            response.raise_for_status()
        target = request.local_dir / request.remote_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        return target


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locale", "preferred_source", "expected_source"),
    [
        ("en", None, "huggingface"),
        ("zh-CN", None, "hf_mirror"),
        ("zh-CN", "huggingface", "huggingface"),
    ],
)
async def test_http_mirror_selection_respects_locale_and_explicit_preference(
    temp_dir: Path,
    file_server,
    locale: str,
    preferred_source: str | None,
    expected_source: str,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"model-bytes"
    token_bytes = b"token-bytes"
    for name in ("hf", "mirror"):
        source = source_dir / name
        (source / "tokenizer").mkdir(parents=True)
        (source / "model.int8.onnx").write_bytes(model_bytes)
        (source / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=f"{base_url}/hf",
        modelscope_url=f"{base_url}/unused",
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )
    manifest = replace(
        manifest,
        sources={"huggingface": replace(manifest.sources["huggingface"], repo_id="example/repo")},
    )
    model_root = temp_dir / "appdata" / "models"
    downloader = _MirrorDownloader(base_url)

    installed = await ensure_local_stt_installed(
        locale=locale,
        preferred_source=preferred_source,
        model_root=model_root,
        manifest=manifest,
        huggingface_downloader=downloader,
    )

    assert installed.selected_source == expected_source
    assert len(downloader.requests) == (2 if expected_source == "hf_mirror" else 0)
    install_dir = model_root / manifest.install_dirname
    assert (install_dir / "model.int8.onnx").read_bytes() == model_bytes
    assert (install_dir / "tokenizer" / "tokens.txt").read_bytes() == token_bytes


@pytest.mark.asyncio
async def test_http_mirror_corruption_falls_back_to_verified_huggingface_assets(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"model-bytes"
    token_bytes = b"token-bytes"
    (source_dir / "mirror").mkdir()
    (source_dir / "mirror" / "model.int8.onnx").write_bytes(b"wrong-bytes")
    (source_dir / "hf" / "tokenizer").mkdir(parents=True)
    (source_dir / "hf" / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "hf" / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=f"{base_url}/hf",
        modelscope_url=f"{base_url}/unused",
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )
    manifest = replace(
        manifest,
        sources={
            "huggingface": manifest.sources["huggingface"],
            "hf_mirror": LocalSTTAssetSource(
                name="hf_mirror",
                revision="hf-rev-1",
                repo_id="example/repo",
                download_url_template="https://hf-mirror.net/example/repo/resolve/hf-rev-1/{path}",
            ),
        },
    )
    model_root = temp_dir / "appdata" / "models"
    downloader = _MirrorDownloader(base_url)

    installed = await ensure_local_stt_installed(
        locale="zh-CN",
        model_root=model_root,
        manifest=manifest,
        huggingface_downloader=downloader,
    )

    assert installed.selected_source == "huggingface"
    assert len(downloader.requests) == 1
    assert (model_root / manifest.install_dirname / "model.int8.onnx").read_bytes() == model_bytes
    assert list(model_root.glob(f"{manifest.install_dirname}.staging-*")) == []


@pytest.mark.asyncio
async def test_existing_huggingface_install_is_reused_after_chinese_mirror_is_added(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"model-bytes"
    token_bytes = b"token-bytes"
    (source_dir / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/unused",
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )
    manifest = replace(manifest, sources={"huggingface": manifest.sources["huggingface"]})
    model_root = temp_dir / "appdata" / "models"
    installed = await ensure_local_stt_installed(
        model_root=model_root,
        manifest=manifest,
    )
    updated = replace(
        manifest,
        sources={
            "huggingface": replace(
                manifest.sources["huggingface"],
                download_url_template=f"{base_url}/missing-hf/{{path}}",
            ),
            "hf_mirror": LocalSTTAssetSource(
                name="hf_mirror",
                revision="hf-rev-1",
                download_url_template=f"{base_url}/missing-mirror/{{path}}",
            ),
        },
    )

    reused = await ensure_local_stt_installed(
        locale="zh-CN",
        model_root=model_root,
        manifest=updated,
    )

    assert reused == installed
    assert (model_root / manifest.install_dirname / "model.int8.onnx").read_bytes() == model_bytes


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_downloads_preferred_source_successfully(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"model-bytes"
    token_bytes = b"token-bytes"
    (source_dir / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/unused",
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )
    statuses: list[RuntimeLocalSTTStatusUpdate] = []

    installed = await ensure_local_stt_installed(
        preferred_source="huggingface",
        model_root=temp_dir / "appdata" / "models",
        manifest=manifest,
        on_status=statuses.append,
    )

    assert installed.selected_source == "huggingface"
    assert statuses[0] == RuntimeLocalSTTStatusUpdate(status="downloading", percent=0)
    assert statuses[-1] == RuntimeLocalSTTStatusUpdate(status="ready", percent=None)
    state = inspect_local_stt_install_state(
        (temp_dir / "appdata" / "models") / manifest.install_dirname, manifest=manifest
    )
    assert state.status == "ready"


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_falls_back_to_next_source_in_locale_order(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"model-bytes"
    token_bytes = b"token-bytes"
    (source_dir / "modelscope").mkdir()
    (source_dir / "modelscope" / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=f"{base_url}/missing",
        modelscope_url=base_url,
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )

    installed = await ensure_local_stt_installed(
        locale="en",
        model_root=temp_dir / "appdata" / "models",
        manifest=manifest,
    )

    assert installed.selected_source == "modelscope"


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_uses_source_specific_remote_paths(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"model-bytes"
    token_bytes = b"token-bytes"
    (source_dir / "modelscope").mkdir()
    (source_dir / "modelscope" / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=f"{base_url}/missing",
        modelscope_url=base_url,
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )

    installed = await ensure_local_stt_installed(
        preferred_source="modelscope",
        model_root=temp_dir / "appdata" / "models",
        manifest=manifest,
    )

    assert installed.selected_source == "modelscope"


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_raises_after_both_sources_fail(
    temp_dir: Path,
    file_server,
) -> None:
    _source_dir, base_url = file_server
    manifest = _build_manifest(
        huggingface_url=f"{base_url}/missing-a",
        modelscope_url=f"{base_url}/missing-b",
        model_bytes=b"model-bytes",
        token_bytes=b"token-bytes",
    )
    statuses: list[RuntimeLocalSTTStatusUpdate] = []

    with pytest.raises(LocalSTTRuntimeInstallError) as caught:
        await ensure_local_stt_installed(
            preferred_source="huggingface",
            model_root=temp_dir / "appdata" / "models",
            manifest=manifest,
            on_status=statuses.append,
        )

    assert statuses[0] == RuntimeLocalSTTStatusUpdate(status="downloading", percent=0)
    assert statuses[-1] == RuntimeLocalSTTStatusUpdate(status="download_failed", percent=None)
    assert not ((temp_dir / "appdata" / "models") / manifest.install_dirname).exists()
    assert isinstance(caught.value.__cause__, httpx.HTTPStatusError)
    assert caught.value.__cause__.response.status_code == 404
    assert caught.value.__cause__.request.url.path == "/missing-a/model.int8.onnx"


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_reports_monotonic_overall_progress(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"m" * 70
    token_bytes = b"t" * 30
    (source_dir / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/unused",
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )
    statuses: list[RuntimeLocalSTTStatusUpdate] = []

    await ensure_local_stt_installed(
        preferred_source="huggingface",
        model_root=temp_dir / "appdata" / "models",
        manifest=manifest,
        on_status=statuses.append,
    )

    percents = [status.percent for status in statuses if status.status == "downloading"]
    assert percents
    assert percents == sorted(percents)
    assert percents[0] == 0
    assert percents[-1] < 100


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_aborts_when_cancelled_before_download(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    (source_dir / "model.int8.onnx").write_bytes(b"model-bytes")
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(b"token-bytes")
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/unused",
        model_bytes=b"model-bytes",
        token_bytes=b"token-bytes",
    )
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(LocalSTTRuntimeInstallCancelled):
        await ensure_local_stt_installed(
            preferred_source="huggingface",
            model_root=temp_dir / "appdata" / "models",
            manifest=manifest,
            cancel_event=cancel_event,
        )

    assert not ((temp_dir / "appdata" / "models") / manifest.install_dirname).exists()


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_rejects_checksum_mismatch_before_promotion(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    (source_dir / "model.int8.onnx").write_bytes(b"corrupted-model")
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(b"token-bytes")
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/missing",
        model_bytes=b"model-bytes",
        token_bytes=b"token-bytes",
    )
    model_root = temp_dir / "appdata" / "models"
    install_dir = model_root / manifest.install_dirname
    install_dir.mkdir(parents=True)
    (install_dir / "model.int8.onnx").write_bytes(b"old-model")
    (install_dir / "tokenizer").mkdir()
    (install_dir / "tokenizer" / "tokens.txt").write_bytes(b"old-token")
    _write_manifest_file(
        install_dir / "installed-manifest.json",
        InstalledLocalSTTManifest(
            manifest_version=manifest.installed_manifest_version,
            model_id=manifest.model_id,
            engine=manifest.engine,
            install_dirname=manifest.install_dirname,
            selected_source="huggingface",
            selected_revision="old-rev",
        ),
    )

    with pytest.raises(LocalSTTRuntimeInstallError):
        await ensure_local_stt_installed(
            preferred_source="huggingface",
            model_root=model_root,
            manifest=manifest,
        )

    assert (install_dir / "model.int8.onnx").read_bytes() == b"old-model"
    assert (install_dir / "tokenizer" / "tokens.txt").read_bytes() == b"old-token"


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_redownloads_structurally_valid_but_checksum_corrupt_install(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"repaired-model"
    token_bytes = b"repaired-token"
    (source_dir / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/missing",
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )
    model_root = temp_dir / "appdata" / "models"
    install_dir = model_root / manifest.install_dirname
    install_dir.mkdir(parents=True)
    (install_dir / "model.int8.onnx").write_bytes(b"corrupt-existing-model")
    (install_dir / "tokenizer").mkdir()
    (install_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    _write_manifest_file(
        install_dir / "installed-manifest.json",
        InstalledLocalSTTManifest(
            manifest_version=manifest.installed_manifest_version,
            model_id=manifest.model_id,
            engine=manifest.engine,
            install_dirname=manifest.install_dirname,
            selected_source="huggingface",
            selected_revision=manifest.sources["huggingface"].revision,
        ),
    )

    # Cheap runtime inspection intentionally treats checksum-corrupt installs as ready.
    state = inspect_local_stt_install_state(install_dir, manifest=manifest)
    assert state.status == "ready"

    installed = await ensure_local_stt_installed(
        preferred_source="huggingface",
        model_root=model_root,
        manifest=manifest,
    )

    assert installed.selected_source == "huggingface"
    assert (install_dir / "model.int8.onnx").read_bytes() == model_bytes
    assert (install_dir / "tokenizer" / "tokens.txt").read_bytes() == token_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retired",
    ["qwen3-asr-0.6b-int8-sherpa", "parakeet-tdt-0.6b-v3-int8-sherpa"],
)
async def test_retired_install_admission_rejects_even_an_injected_manifest(tmp_path, retired):
    manifest = _build_manifest(
        huggingface_url="https://example.invalid",
        modelscope_url="https://example.invalid",
        model_bytes=b"model",
        token_bytes=b"tokens",
        model_id=retired,
    )
    with pytest.raises(LocalSTTRuntimeInstallError) as rejected:
        await ensure_local_stt_installed(model_id=retired, model_root=tmp_path, manifest=manifest)
    assert rejected.value.failure_code == "retired_model"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_failed_full_hash_replacement_preserves_previously_valid_install(
    temp_dir, file_server
):
    from puripuly_heart.core.local_asr.local_stt_assets import validate_local_stt_install

    source, url = file_server
    (source / "tokenizer").mkdir()
    (source / "model.int8.onnx").write_bytes(b"good-model")
    (source / "tokenizer" / "tokens.txt").write_bytes(b"tokens")
    original = _build_manifest(
        huggingface_url=url,
        modelscope_url=f"{url}/missing",
        model_bytes=b"good-model",
        token_bytes=b"tokens",
    )
    root = temp_dir / "models"
    before = await ensure_local_stt_installed(
        manifest=original, model_root=root, preferred_source="huggingface"
    )
    replacement = replace(
        original,
        sources={"huggingface": replace(original.sources["huggingface"], revision="next-revision")},
    )
    (source / "model.int8.onnx").write_bytes(b"evil-model")
    with pytest.raises(LocalSTTRuntimeInstallError) as failed:
        await ensure_local_stt_installed(
            manifest=replacement, model_root=root, preferred_source="huggingface"
        )
    assert failed.value.failure_code == "checksum_failed"
    assert validate_local_stt_install(root / original.install_dirname, manifest=original) == before
    assert (
        inspect_local_stt_install_state(
            root / original.install_dirname, manifest=original, verify_checksums=True
        ).status
        == "ready"
    )
    assert list(root.glob(f"{original.install_dirname}.staging-*")) == []
    assert not (root / f"{original.install_dirname}.backup").exists()


@pytest.mark.asyncio
async def test_ensure_local_stt_installed_recovers_invalid_existing_install(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_bytes = b"new-model"
    token_bytes = b"new-token"
    (source_dir / "model.int8.onnx").write_bytes(model_bytes)
    (source_dir / "tokenizer").mkdir()
    (source_dir / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
    manifest = _build_manifest(
        huggingface_url=base_url,
        modelscope_url=f"{base_url}/missing",
        model_bytes=model_bytes,
        token_bytes=token_bytes,
    )
    model_root = temp_dir / "appdata" / "models"
    install_dir = model_root / manifest.install_dirname
    install_dir.mkdir(parents=True)
    (install_dir / "model.int8.onnx").write_bytes(b"broken")
    (install_dir / "installed-manifest.json").write_text("{invalid", encoding="utf-8")

    installed = await ensure_local_stt_installed(
        preferred_source="huggingface",
        model_root=model_root,
        manifest=manifest,
    )

    assert installed.selected_source == "huggingface"
    assert (install_dir / "model.int8.onnx").read_bytes() == model_bytes
    assert (install_dir / "tokenizer" / "tokens.txt").read_bytes() == token_bytes


@pytest.mark.asyncio
async def test_independent_cpu_model_installs_survive_one_model_download_failure(
    temp_dir: Path,
    file_server,
) -> None:
    source_dir, base_url = file_server
    model_ids = ("parakeet-v3", "parakeet-ja", "qwen")
    manifests: list[LocalSTTAssetManifest] = []
    for model_id in model_ids:
        model_bytes = f"{model_id}-model".encode()
        token_bytes = f"{model_id}-tokens".encode()
        model_source = source_dir / model_id
        if model_id != "parakeet-ja":
            model_source.mkdir()
            (model_source / "model.int8.onnx").write_bytes(model_bytes)
            (model_source / "tokenizer").mkdir()
            (model_source / "tokenizer" / "tokens.txt").write_bytes(token_bytes)
        manifests.append(
            _build_manifest(
                huggingface_url=f"{base_url}/{model_id}",
                modelscope_url=f"{base_url}/{model_id}",
                model_bytes=model_bytes,
                token_bytes=token_bytes,
                model_id=model_id,
            )
        )
    model_root = temp_dir / "appdata" / "models"

    results = await asyncio.gather(
        *(
            ensure_local_stt_installed(
                model_id=manifest.model_id,
                preferred_source="huggingface",
                model_root=model_root,
                manifest=manifest,
            )
            for manifest in manifests
        ),
        return_exceptions=True,
    )

    assert isinstance(results[0], InstalledLocalSTTManifest)
    assert isinstance(results[1], LocalSTTRuntimeInstallError)
    assert isinstance(results[2], InstalledLocalSTTManifest)
    assert (
        inspect_local_stt_install_state(
            model_root / manifests[0].install_dirname,
            manifest=manifests[0],
            verify_checksums=True,
        ).status
        == "ready"
    )
    assert (
        inspect_local_stt_install_state(
            model_root / manifests[1].install_dirname,
            manifest=manifests[1],
            verify_checksums=True,
        ).status
        == "missing"
    )
    assert (
        inspect_local_stt_install_state(
            model_root / manifests[2].install_dirname,
            manifest=manifests[2],
            verify_checksums=True,
        ).status
        == "ready"
    )


@pytest.mark.asyncio
async def test_runtime_install_rejects_cross_model_manifest_target() -> None:
    manifest = _build_manifest(
        huggingface_url="https://example.invalid",
        modelscope_url="https://example.invalid",
        model_bytes=b"model",
        token_bytes=b"tokens",
        model_id="model-a",
    )

    with pytest.raises(LocalSTTRuntimeInstallError, match="does not match"):
        await ensure_local_stt_installed(model_id="model-b", manifest=manifest)
