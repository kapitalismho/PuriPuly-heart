from __future__ import annotations

import asyncio
import errno
import hashlib
import inspect
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Awaitable, Callable, Literal
from uuid import uuid4

import httpx

from puripuly_heart.core import network_clients
from puripuly_heart.core.model_download_sources import (
    HF_MIRROR_ENDPOINT,
    model_download_source_candidates,
)
from puripuly_heart.core.runtime_logging import emit_basic_log

from .local_stt_assets import (
    LOCAL_QWEN_GPU_MODEL_ID,
    LOCAL_STT_MODEL_ID,
    InstalledLocalSTTManifest,
    LocalSTTAssetError,
    LocalSTTAssetManifest,
    default_local_stt_model_root,
    inspect_local_stt_install_state,
    load_local_stt_asset_manifest,
    validate_local_stt_install,
)
from .local_stt_download_port import (
    HuggingFaceDownloadPort,
    HuggingFaceDownloadRequest,
    LocalSTTDownloadPortCancelled,
)

RuntimeLocalSTTStatus = Literal["downloading", "ready", "download_failed"]
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RuntimeLocalSTTStatusUpdate:
    status: RuntimeLocalSTTStatus
    percent: int | None = None
    downloaded_bytes: int = 0
    total_bytes: int = 0


StatusCallback = Callable[[RuntimeLocalSTTStatusUpdate], Awaitable[None] | None]


class LocalSTTProvisioningLease:
    def __init__(self, handle: IO[bytes]) -> None:
        self._handle = handle
        self._closed = False

    @classmethod
    def acquire(
        cls,
        *,
        model_root: Path,
        wait: bool,
        cancel_event: threading.Event | None = None,
    ) -> LocalSTTProvisioningLease | None:
        model_root.mkdir(parents=True, exist_ok=True)
        handle = (model_root / ".local-asr-provisioning.lock").open("a+b")
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        try:
            while True:
                if cls._try_lock(handle):
                    return cls(handle)
                if not wait:
                    handle.close()
                    return None
                _raise_if_cancelled(cancel_event)
                time.sleep(0.05)
        except BaseException:
            handle.close()
            raise

    @staticmethod
    def _try_lock(handle: IO[bytes]) -> bool:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    return False
                raise
            return True
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
        return True

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._closed = True
            self._handle.close()


def cleanup_local_stt_install_residue(
    *,
    model_root: Path | None = None,
    install_dirnames: tuple[str, ...],
    wait_for_lease: bool = False,
    cancel_event: threading.Event | None = None,
) -> tuple[Path, ...] | None:
    resolved_root = (model_root or default_local_stt_model_root()).resolve()
    normalized_dirnames = tuple(dict.fromkeys(install_dirnames))
    if not normalized_dirnames or any(
        not name
        or name.startswith(".")
        or Path(name).name != name
        or any(not (character.isalnum() or character in "._-") for character in name)
        for name in normalized_dirnames
    ):
        raise ValueError("install_dirnames must contain safe directory names")
    lease = LocalSTTProvisioningLease.acquire(
        model_root=resolved_root,
        wait=wait_for_lease,
        cancel_event=cancel_event,
    )
    if lease is None:
        return None
    try:
        reconciled: list[Path] = []
        for dirname in normalized_dirnames:
            install_dir = resolved_root / dirname
            backup_dir = resolved_root / f"{dirname}.backup"
            if backup_dir.exists() or backup_dir.is_symlink():
                if not install_dir.exists() and backup_dir.is_dir() and not backup_dir.is_symlink():
                    backup_dir.rename(install_dir)
                elif backup_dir.is_symlink() or not backup_dir.is_dir():
                    backup_dir.unlink(missing_ok=True)
                else:
                    shutil.rmtree(backup_dir)
                reconciled.append(backup_dir)
            for staging_dir in resolved_root.glob(f"{dirname}.staging-*"):
                if staging_dir.is_symlink() or not staging_dir.is_dir():
                    staging_dir.unlink(missing_ok=True)
                else:
                    shutil.rmtree(staging_dir)
                reconciled.append(staging_dir)
        return tuple(reconciled)
    finally:
        lease.close()


class LocalSTTRuntimeInstallError(LocalSTTAssetError):
    """Raised when runtime local STT provisioning fails."""

    def __init__(self, message: str, *, failure_code: str = "unknown", status_code=None):
        super().__init__(message)
        self.failure_code = failure_code
        self.status_code = status_code


class LocalSTTRuntimeInstallCancelled(LocalSTTAssetError):
    """Raised when runtime local STT provisioning is cancelled."""


async def _emit_status(
    on_status: StatusCallback | None,
    status: RuntimeLocalSTTStatus,
    *,
    percent: int | None = None,
    downloaded_bytes: int = 0,
    total_bytes: int = 0,
) -> None:
    if on_status is None:
        return
    result = on_status(
        RuntimeLocalSTTStatusUpdate(
            status=status,
            percent=percent,
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
        )
    )
    if inspect.isawaitable(result):
        await result


class _DownloadProgress:
    def __init__(self, total_bytes: int) -> None:
        self._total_bytes = max(total_bytes, 0)
        self._downloaded_bytes = 0
        self._lock = threading.Lock()

    def add(self, size_bytes: int) -> None:
        if size_bytes <= 0:
            return
        with self._lock:
            self._downloaded_bytes += size_bytes

    def set_downloaded(self, size_bytes: int) -> None:
        with self._lock:
            self._downloaded_bytes = max(self._downloaded_bytes, size_bytes)

    def percent(self) -> int:
        if self._total_bytes <= 0:
            return 0
        with self._lock:
            downloaded_bytes = self._downloaded_bytes
        return min(99, int(downloaded_bytes * 100 / self._total_bytes))

    def bytes(self) -> int:
        with self._lock:
            return self._downloaded_bytes


def _source_plans(
    manifest: LocalSTTAssetManifest,
    *,
    preferred_source: str | None,
    locale: str | None,
) -> tuple[tuple[str, str | None, str], ...]:
    names = model_download_source_candidates(
        tuple(manifest.sources),
        locale=locale,
        explicit=preferred_source,
    )
    if not names:
        raise LocalSTTRuntimeInstallError(
            "Selected source is unsupported", failure_code="unsupported_source"
        )
    return tuple(
        (
            (name if name in manifest.sources else "huggingface", HF_MIRROR_ENDPOINT, name)
            if name == "hf_mirror"
            else (name, None, name)
        )
        for name in names
    )


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise LocalSTTRuntimeInstallCancelled("runtime local STT install cancelled")


async def _owned_io(function, *, io_cancel_event: threading.Event, **kwargs):
    worker = asyncio.create_task(asyncio.to_thread(function, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        io_cancel_event.set()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        await asyncio.gather(worker, return_exceptions=True)
        raise


def _download_source_into_staging(
    *,
    source_name: str,
    staging_dir: Path,
    manifest: LocalSTTAssetManifest,
    cancel_event: threading.Event | None = None,
    progress: _DownloadProgress | None = None,
) -> InstalledLocalSTTManifest:
    try:
        _raise_if_cancelled(cancel_event)
        source = manifest.sources[source_name]
        with network_clients.external_client(
            timeout=httpx.Timeout(30.0, connect=15.0), follow_redirects=True
        ) as client:
            for asset in manifest.files:
                _raise_if_cancelled(cancel_event)
                asset_path = staging_dir / asset.relative_path
                asset_path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size_bytes = 0
                url = source.download_url_template.format(
                    path=asset.remote_path_for_source(source_name)
                )
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with asset_path.open("wb") as handle:
                        for chunk in response.iter_bytes():
                            _raise_if_cancelled(cancel_event)
                            if not chunk:
                                continue
                            handle.write(chunk)
                            digest.update(chunk)
                            size_bytes += len(chunk)
                            if asset.size_bytes is not None and size_bytes > asset.size_bytes:
                                raise LocalSTTRuntimeInstallError(
                                    "Downloaded file too large", failure_code="file_size_mismatch"
                                )
                            if progress is not None:
                                progress.add(len(chunk))
                if digest.hexdigest() != asset.sha256:
                    raise LocalSTTRuntimeInstallError(
                        f"checksum mismatch for required model file: {asset.relative_path}",
                        failure_code="checksum_failed",
                    )
                if asset.size_bytes is not None and size_bytes != asset.size_bytes:
                    raise LocalSTTRuntimeInstallError(
                        f"size mismatch for required model file: {asset.relative_path}",
                        failure_code="file_size_mismatch",
                    )

        installed = InstalledLocalSTTManifest(
            manifest_version=manifest.installed_manifest_version,
            model_id=manifest.model_id,
            engine=manifest.engine,
            install_dirname=manifest.install_dirname,
            selected_source=source_name,
            selected_revision=source.revision,
        )
        (staging_dir / manifest.installed_manifest_filename).write_text(
            json.dumps(installed.to_dict(), indent=2),
            encoding="utf-8",
        )
        validate_local_stt_install(staging_dir, manifest=manifest)
        return installed
    except LocalSTTRuntimeInstallCancelled:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _uses_huggingface_xet(
    *,
    source_name: str,
    manifest: LocalSTTAssetManifest,
) -> bool:
    source = manifest.sources[source_name]
    return (
        manifest.model_id == LOCAL_QWEN_GPU_MODEL_ID
        and source_name == "huggingface"
        and bool(source.repo_id)
    )


def _validate_downloaded_asset(
    *,
    asset_path: Path,
    relative_path: str,
    expected_sha256: str,
    expected_size_bytes: int | None,
    cancel_event: threading.Event | None = None,
) -> None:
    digest = hashlib.sha256()
    size_bytes = 0
    with asset_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            _raise_if_cancelled(cancel_event)
            digest.update(chunk)
            size_bytes += len(chunk)
    if digest.hexdigest() != expected_sha256:
        raise LocalSTTRuntimeInstallError(
            f"checksum mismatch for required model file: {relative_path}",
            failure_code="checksum_failed",
        )
    if expected_size_bytes is not None and size_bytes != expected_size_bytes:
        raise LocalSTTRuntimeInstallError(
            f"size mismatch for required model file: {relative_path}",
            failure_code="file_size_mismatch",
        )


async def _download_huggingface_xet_source_into_staging(
    *,
    source_name: str,
    staging_dir: Path,
    manifest: LocalSTTAssetManifest,
    downloader: HuggingFaceDownloadPort,
    cancel_event: threading.Event | None,
    progress: _DownloadProgress | None,
    endpoint: str | None = None,
) -> InstalledLocalSTTManifest:
    source = manifest.sources[source_name]
    completed_bytes = 0
    try:
        for asset in manifest.files:
            _raise_if_cancelled(cancel_event)
            remote_path = asset.remote_path_for_source(source_name)

            def on_progress(update, *, offset: int = completed_bytes) -> None:
                if progress is not None:
                    progress.set_downloaded(offset + update.downloaded_bytes)

            downloaded_path = await downloader.download(
                HuggingFaceDownloadRequest(
                    repo_id=source.repo_id,
                    revision=source.revision,
                    remote_path=remote_path,
                    local_dir=staging_dir,
                    expected_size_bytes=asset.size_bytes,
                    endpoint=endpoint,
                ),
                cancel_event=cancel_event,
                on_progress=on_progress,
            )
            asset_path = staging_dir / asset.relative_path
            if downloaded_path.resolve() != asset_path.resolve():
                asset_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded_path.replace(asset_path)
            await _owned_io(
                _validate_downloaded_asset,
                asset_path=asset_path,
                relative_path=asset.relative_path,
                expected_sha256=asset.sha256,
                expected_size_bytes=asset.size_bytes,
                cancel_event=cancel_event,
                io_cancel_event=cancel_event or threading.Event(),
            )
            completed_bytes += asset.size_bytes or asset_path.stat().st_size
            if progress is not None:
                progress.set_downloaded(completed_bytes)

        shutil.rmtree(staging_dir / ".cache", ignore_errors=True)
        installed = InstalledLocalSTTManifest(
            manifest_version=manifest.installed_manifest_version,
            model_id=manifest.model_id,
            engine=manifest.engine,
            install_dirname=manifest.install_dirname,
            selected_source=source_name,
            selected_revision=source.revision,
        )
        (staging_dir / manifest.installed_manifest_filename).write_text(
            json.dumps(installed.to_dict(), indent=2),
            encoding="utf-8",
        )
        await _owned_io(
            validate_local_stt_install,
            model_dir=staging_dir,
            manifest=manifest,
            io_cancel_event=cancel_event or threading.Event(),
        )
        return installed
    except LocalSTTDownloadPortCancelled as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise LocalSTTRuntimeInstallCancelled("runtime local STT install cancelled") from exc


async def _download_source_into_staging_async(
    *,
    source_name: str,
    staging_dir: Path,
    manifest: LocalSTTAssetManifest,
    cancel_event: threading.Event | None,
    progress: _DownloadProgress | None,
    huggingface_downloader: HuggingFaceDownloadPort | None,
    endpoint: str | None = None,
) -> InstalledLocalSTTManifest:
    if endpoint is not None or _uses_huggingface_xet(source_name=source_name, manifest=manifest):
        if huggingface_downloader is None:
            from puripuly_heart.core.local_stt_huggingface_xet_adapter import (
                HuggingFaceXetDownloadAdapter,
            )

            huggingface_downloader = HuggingFaceXetDownloadAdapter()
        return await _download_huggingface_xet_source_into_staging(
            source_name=source_name,
            staging_dir=staging_dir,
            manifest=manifest,
            downloader=huggingface_downloader,
            cancel_event=cancel_event,
            progress=progress,
            endpoint=endpoint,
        )
    worker_cancel = cancel_event or threading.Event()
    return await _owned_io(
        lambda: _download_source_into_staging(
            source_name=source_name,
            staging_dir=staging_dir,
            manifest=manifest,
            cancel_event=worker_cancel,
            progress=progress,
        ),
        io_cancel_event=worker_cancel,
    )


def _promote_staging_install(
    *,
    staging_dir: Path,
    install_dir: Path,
    cancel_event: threading.Event | None = None,
) -> None:
    _raise_if_cancelled(cancel_event)
    backup_dir = install_dir.with_name(f"{install_dir.name}.backup")
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(backup_dir, ignore_errors=True)

    had_existing_install = install_dir.exists()
    if had_existing_install:
        install_dir.rename(backup_dir)

    try:
        staging_dir.rename(install_dir)
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir, ignore_errors=True)
        if had_existing_install and backup_dir.exists():
            backup_dir.rename(install_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)


async def ensure_local_stt_installed(
    *,
    model_id: str | None = None,
    preferred_source: str | None = None,
    locale: str | None = None,
    model_root: Path | None = None,
    manifest: LocalSTTAssetManifest | None = None,
    on_status: StatusCallback | None = None,
    cancel_event: threading.Event | None = None,
    huggingface_downloader: HuggingFaceDownloadPort | None = None,
) -> InstalledLocalSTTManifest:
    resolved_root = (model_root or default_local_stt_model_root()).resolve()
    cancellation = cancel_event or threading.Event()
    _raise_if_cancelled(cancellation)
    lease = None

    def acquire_lease() -> None:
        nonlocal lease
        lease = LocalSTTProvisioningLease.acquire(
            model_root=resolved_root,
            wait=True,
            cancel_event=cancellation,
        )

    try:
        await _owned_io(acquire_lease, io_cancel_event=cancellation)
        if lease is None:
            raise LocalSTTRuntimeInstallError("local STT provisioning lease is unavailable")
        return await _ensure_local_stt_installed_with_lease(
            model_id=model_id,
            preferred_source=preferred_source,
            locale=locale,
            model_root=resolved_root,
            manifest=manifest,
            on_status=on_status,
            cancel_event=cancellation,
            huggingface_downloader=huggingface_downloader,
        )
    finally:
        if lease is not None:
            await _owned_io(lease.close, io_cancel_event=cancellation)


async def _ensure_local_stt_installed_with_lease(
    *,
    model_id: str | None = None,
    preferred_source: str | None = None,
    locale: str | None = None,
    model_root: Path | None = None,
    manifest: LocalSTTAssetManifest | None = None,
    on_status: StatusCallback | None = None,
    cancel_event: threading.Event | None = None,
    huggingface_downloader: HuggingFaceDownloadPort | None = None,
) -> InstalledLocalSTTManifest:
    if manifest is not None and model_id is not None and manifest.model_id != model_id:
        raise LocalSTTRuntimeInstallError(
            "runtime local STT model_id does not match the supplied manifest"
        )
    resolved_manifest = manifest or load_local_stt_asset_manifest(model_id or LOCAL_STT_MODEL_ID)
    resolved_root = model_root or default_local_stt_model_root()
    install_dir = resolved_root / resolved_manifest.install_dirname
    total_bytes = sum(asset.size_bytes or 0 for asset in resolved_manifest.files)

    _raise_if_cancelled(cancel_event)
    state = inspect_local_stt_install_state(install_dir, manifest=resolved_manifest)
    if state.status == "ready" and state.installed_manifest is not None:
        try:
            return await _owned_io(
                validate_local_stt_install,
                model_dir=install_dir,
                manifest=resolved_manifest,
                io_cancel_event=cancel_event or threading.Event(),
            )
        except LocalSTTAssetError:
            # Cheap runtime inspection is allowed to say "ready" without checksums.
            # Repair/download should only skip when the full install contract passes.
            pass

    _raise_if_cancelled(cancel_event)
    if total_bytes and shutil.disk_usage(resolved_root).free < total_bytes + 64 * 1024**2:
        raise LocalSTTRuntimeInstallError("Insufficient staging space", failure_code="disk_full")
    failures: list[str] = []
    first_error: Exception | None = None
    last_progress_percent: int | None = None
    last_error: Exception | None = None

    for source_name, endpoint, source_label in _source_plans(
        resolved_manifest,
        preferred_source=preferred_source,
        locale=locale,
    ):
        _raise_if_cancelled(cancel_event)
        emit_basic_log(
            logger,
            f"[ModelDownload] source_attempt model={resolved_manifest.model_id} "
            f"source={source_label}",
            level=logging.INFO,
        )
        staging_dir = resolved_root / f"{resolved_manifest.install_dirname}.staging-{uuid4().hex}"
        shutil.rmtree(staging_dir, ignore_errors=True)
        progress = _DownloadProgress(total_bytes)
        download_task: asyncio.Task[InstalledLocalSTTManifest] | None = None
        try:
            current_percent = 0 if last_progress_percent is None else last_progress_percent
            if current_percent != last_progress_percent:
                last_progress_percent = current_percent
                await _emit_status(on_status, "downloading", percent=current_percent)

            download_task = asyncio.create_task(
                _download_source_into_staging_async(
                    source_name=source_name,
                    staging_dir=staging_dir,
                    manifest=resolved_manifest,
                    cancel_event=cancel_event,
                    progress=progress,
                    huggingface_downloader=huggingface_downloader,
                    endpoint=endpoint,
                )
            )
            last_report = 0.0
            while not download_task.done():
                _raise_if_cancelled(cancel_event)
                current_percent = max(last_progress_percent or 0, progress.percent())
                if (
                    current_percent != last_progress_percent
                    or time.monotonic() - last_report >= 0.25
                ):
                    last_report = time.monotonic()
                    last_progress_percent = current_percent
                    await _emit_status(
                        on_status,
                        "downloading",
                        percent=current_percent,
                        downloaded_bytes=progress.bytes(),
                        total_bytes=total_bytes,
                    )
                await asyncio.sleep(0.05)

            installed = await download_task
            current_percent = max(last_progress_percent or 0, progress.percent())
            if current_percent != last_progress_percent:
                last_progress_percent = current_percent
                await _emit_status(on_status, "downloading", percent=current_percent)
            await _owned_io(
                _promote_staging_install,
                staging_dir=staging_dir,
                install_dir=install_dir,
                cancel_event=cancel_event,
                io_cancel_event=cancel_event or threading.Event(),
            )
            emit_basic_log(
                logger,
                f"[ModelDownload] source_completed model={resolved_manifest.model_id} "
                f"source={source_label}",
                level=logging.INFO,
            )
            await _emit_status(on_status, "ready", percent=None)
            return installed
        except asyncio.CancelledError:
            if download_task is not None and not download_task.done():
                download_task.cancel()
                await asyncio.gather(download_task, return_exceptions=True)
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        except LocalSTTRuntimeInstallCancelled:
            if download_task is not None and not download_task.done():
                await asyncio.gather(download_task, return_exceptions=True)
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        except Exception as exc:
            if first_error is None:
                first_error = exc
            failures.append(f"{source_label}: {exc}")
            last_error = exc
            emit_basic_log(
                logger,
                f"[ModelDownload] source_failed model={resolved_manifest.model_id} "
                f"source={source_label} cause={type(exc).__name__}",
                level=logging.WARNING,
            )
            shutil.rmtree(staging_dir, ignore_errors=True)

    await _emit_status(on_status, "download_failed", percent=None)
    code = getattr(last_error, "failure_code", "unknown")
    if isinstance(last_error, httpx.TimeoutException):
        code = "download_timeout"
    elif isinstance(last_error, httpx.ProxyError):
        code = "proxy_failed"
    elif isinstance(last_error, httpx.ConnectError):
        code = "connection_failed"
    elif isinstance(last_error, httpx.HTTPStatusError):
        code = "source_unavailable"
    elif isinstance(last_error, OSError) and last_error.errno == errno.ENOSPC:
        code = "disk_full"
    status_code = getattr(last_error, "status_code", None)
    if isinstance(last_error, httpx.HTTPStatusError):
        status_code = last_error.response.status_code
    if preferred_source is not None:
        message = "checksum mismatch" if code == "checksum_failed" else "Model installation failed"
    else:
        message = "; ".join(failures) or "runtime local STT install failed"
    raise LocalSTTRuntimeInstallError(
        message,
        failure_code=code,
        status_code=status_code,
    ) from first_error
