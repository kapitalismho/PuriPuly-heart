from __future__ import annotations

import asyncio
import errno
import hashlib
import inspect
import json
import logging
import shutil
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar
from urllib.parse import quote
from uuid import uuid4

import httpx

from puripuly_heart.core import network_clients
from puripuly_heart.core.feature_diagnostics import log_feature
from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadPort,
    HuggingFaceDownloadProgress,
    HuggingFaceDownloadRequest,
    LocalSTTDownloadPortCancelled,
    LocalSTTDownloadPortError,
)
from puripuly_heart.core.local_asr.local_stt_runtime_installer import (
    LocalSTTProvisioningLease,
    LocalSTTRuntimeInstallCancelled,
)
from puripuly_heart.core.local_translation.assets import (
    GEMMA_INSTALLED_MANIFEST_FILENAME,
    GemmaAsset,
    GemmaAssetSource,
    GemmaInstallInvalidError,
    GemmaModelSpec,
    InstalledGemmaManifest,
    default_gemma_install_dir,
    e4b_gemma_spec,
    inspect_gemma_install,
    validate_gemma_install,
)
from puripuly_heart.core.model_download_sources import (
    HF_MIRROR_ENDPOINT,
    MODEL_DOWNLOAD_SOURCES,
    model_download_source_candidates,
)


class GemmaProvisioningError(RuntimeError):
    def __init__(self, message: str, *, failure_code: str = "install_failed") -> None:
        super().__init__(message)
        self.failure_code = failure_code


class GemmaProvisioningCancelled(GemmaProvisioningError):
    pass


class _GemmaSourceError(GemmaProvisioningError):
    pass


_WorkerResult = TypeVar("_WorkerResult")


class _HTTPDownloadAdapter:
    def __init__(self, source: GemmaAssetSource) -> None:
        self._source = source

    async def download(
        self,
        request: HuggingFaceDownloadRequest,
        *,
        cancel_event: threading.Event | None,
        on_progress: Callable[[HuggingFaceDownloadProgress], None] | None,
    ) -> Path:
        return await asyncio.to_thread(
            self._download,
            request,
            cancel_event=cancel_event,
            on_progress=on_progress,
        )

    def _download(
        self,
        request: HuggingFaceDownloadRequest,
        *,
        cancel_event: threading.Event | None,
        on_progress: Callable[[HuggingFaceDownloadProgress], None] | None,
    ) -> Path:
        _raise_if_cancelled(cancel_event)
        destination = request.local_dir / request.remote_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = self._source.download_url_template.format(path=quote(request.remote_path, safe="/"))
        downloaded_bytes = 0
        reported_percent = -1
        try:
            with network_clients.external_client(timeout=30.0, follow_redirects=True) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with destination.open("wb") as handle:
                        for chunk in response.iter_bytes():
                            _raise_if_cancelled(cancel_event)
                            if not chunk:
                                continue
                            handle.write(chunk)
                            downloaded_bytes += len(chunk)
                            if on_progress is not None:
                                percent = (
                                    min(100, downloaded_bytes * 100 // request.expected_size_bytes)
                                    if request.expected_size_bytes
                                    else None
                                )
                                if percent is None or percent != reported_percent:
                                    reported_percent = percent
                                    on_progress(
                                        HuggingFaceDownloadProgress(
                                            downloaded_bytes, request.expected_size_bytes
                                        )
                                    )
        except httpx.HTTPError as exc:
            response = getattr(exc, "response", None)
            raise LocalSTTDownloadPortError(
                "Gemma model download failed",
                cause_type=type(exc).__name__,
                status_code=response.status_code if response is not None else None,
            ) from exc
        _raise_if_cancelled(cancel_event)
        return destination


@dataclass(frozen=True, slots=True)
class GemmaProvisioningUpdate:
    state: str
    downloaded_bytes: int
    total_bytes: int
    percent: int
    filename: str | None = None


GemmaProvisioningCallback = Callable[
    [GemmaProvisioningUpdate],
    Awaitable[None] | None,
]


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise GemmaProvisioningCancelled("Gemma model provisioning cancelled")


async def _emit(
    callback: GemmaProvisioningCallback | None,
    *,
    state: str,
    downloaded_bytes: int,
    total_bytes: int,
    filename: str | None = None,
) -> None:
    if callback is None:
        return
    update = GemmaProvisioningUpdate(
        state=state,
        downloaded_bytes=downloaded_bytes,
        total_bytes=total_bytes,
        percent=min(100, (downloaded_bytes * 100) // total_bytes) if total_bytes else 100,
        filename=filename,
    )
    result = callback(update)
    if inspect.isawaitable(result):
        await result


def _remove_owned_staging(
    staging_dir: Path,
    *,
    primary_failure: BaseException,
) -> None:
    try:
        shutil.rmtree(staging_dir)
    except FileNotFoundError:
        pass
    except BaseException as cleanup_failure:
        raise BaseExceptionGroup(
            "Gemma provisioning and staging cleanup failed",
            (primary_failure, cleanup_failure),
        ) from primary_failure


async def _await_download_cleanup(
    download: Awaitable[_WorkerResult],
    *,
    cancel_event: threading.Event | None,
) -> _WorkerResult:
    task = asyncio.create_task(download)
    settled = asyncio.Event()
    task.add_done_callback(lambda _task: settled.set())
    try:
        await settled.wait()
    except asyncio.CancelledError as cancellation:
        current = asyncio.current_task()
        if current is None or not current.cancelling():
            raise
        if cancel_event is not None:
            cancel_event.set()
        else:
            task.cancel()
        while not task.done():
            try:
                await settled.wait()
            except asyncio.CancelledError:
                continue
        try:
            task.result()
        except (
            asyncio.CancelledError,
            LocalSTTDownloadPortCancelled,
            LocalSTTRuntimeInstallCancelled,
            GemmaProvisioningCancelled,
        ):
            pass
        except BaseException as cleanup_failure:
            raise BaseExceptionGroup(
                "Gemma provisioning cancellation and cleanup failed",
                (cancellation, cleanup_failure),
            ) from None
        raise cancellation
    return task.result()


async def _owned_io(function, *args, cancel_event=None, **kwargs):
    return await _await_download_cleanup(
        asyncio.to_thread(function, *args, **kwargs),
        cancel_event=cancel_event,
    )


async def _download_asset(
    *,
    downloader: HuggingFaceDownloadPort,
    source: GemmaAssetSource,
    asset: GemmaAsset,
    staging_dir: Path,
    completed_bytes: int,
    total_bytes: int,
    cancel_event: threading.Event | None,
    on_status: GemmaProvisioningCallback | None,
    endpoint: str | None = None,
) -> int:
    loop = asyncio.get_running_loop()
    progress_futures: list[Future[None]] = []
    progress_lock = threading.Lock()
    accepting_progress = True

    def on_progress(update) -> None:
        if on_status is None:
            return
        downloaded = min(asset.size_bytes, max(0, update.downloaded_bytes))
        with progress_lock:
            if not accepting_progress:
                return
            future = asyncio.run_coroutine_threadsafe(
                _emit(
                    on_status,
                    state="downloading",
                    downloaded_bytes=completed_bytes + downloaded,
                    total_bytes=total_bytes,
                    filename=asset.filename,
                ),
                loop,
            )
            progress_futures.append(future)

    async def drain_progress() -> tuple[BaseException, ...]:
        nonlocal accepting_progress
        with progress_lock:
            accepting_progress = False
            pending = tuple(progress_futures)
        if not pending:
            return ()
        gathered = asyncio.gather(
            *(asyncio.wrap_future(future) for future in pending),
            return_exceptions=True,
        )
        cancellation = None
        while not gathered.done():
            try:
                await asyncio.shield(gathered)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
                if cancel_event is not None:
                    cancel_event.set()
        results = gathered.result()
        failures = tuple(result for result in results if isinstance(result, BaseException))
        if cancellation is not None:
            if failures:
                raise BaseExceptionGroup(
                    "Gemma progress drain cancelled with callback failures",
                    (cancellation, *failures),
                )
            raise cancellation
        return failures

    try:
        downloaded_path = await _await_download_cleanup(
            downloader.download(
                HuggingFaceDownloadRequest(
                    repo_id=source.repo_id,
                    revision=source.revision,
                    remote_path=asset.filename,
                    local_dir=staging_dir,
                    expected_size_bytes=asset.size_bytes,
                    endpoint=endpoint,
                ),
                cancel_event=cancel_event,
                on_progress=on_progress,
            ),
            cancel_event=cancel_event,
        )
    except BaseException as download_failure:
        progress_failures = await drain_progress()
        if progress_failures:
            raise BaseExceptionGroup(
                "Gemma download and progress callbacks failed",
                (download_failure, *progress_failures),
            )
        if isinstance(download_failure, httpx.HTTPError) or (
            isinstance(download_failure, LocalSTTDownloadPortError)
            and not isinstance(download_failure, LocalSTTDownloadPortCancelled)
            and download_failure.failure_code
            in {
                "download_failed",
                "download_timeout",
                "mirror_unavailable",
                "mirror_redirect_external",
                "range_invalid",
                "file_size_mismatch",
                "checksum_failed",
            }
        ):
            _raise_if_cancelled(cancel_event)
            raise _GemmaSourceError(
                str(download_failure),
                failure_code=getattr(download_failure, "failure_code", "download_failed"),
            ) from download_failure
        raise
    progress_failures = await drain_progress()
    if len(progress_failures) == 1:
        raise progress_failures[0]
    if progress_failures:
        raise BaseExceptionGroup("Gemma progress callbacks failed", progress_failures)
    destination = staging_dir / asset.filename
    if downloaded_path.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path.replace(destination)
    return completed_bytes + asset.size_bytes


def _promote(staging_dir: Path, install_dir: Path) -> None:
    backup_dir = install_dir.with_name(f"{install_dir.name}.backup-{uuid4().hex}")
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    existing = install_dir.exists()
    if existing:
        install_dir.rename(backup_dir)
    try:
        staging_dir.rename(install_dir)
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir, ignore_errors=True)
        if existing and backup_dir.exists():
            backup_dir.rename(install_dir)
        raise
    else:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _source_plans(
    spec: GemmaModelSpec,
    *,
    source: str | None,
    locale: str | None,
) -> tuple[tuple[str, str | None, str], ...]:
    if source == "local":
        return (("local", None, "local"),)
    names = model_download_source_candidates(tuple(spec.sources), locale=locale, explicit=source)
    if source is not None and not names:
        raise GemmaProvisioningError(
            "Selected source is unsupported", failure_code="unsupported_source"
        )
    return tuple(
        (
            ("huggingface", HF_MIRROR_ENDPOINT, name)
            if name == "hf_mirror" and name not in spec.sources
            else (name, None, name)
        )
        for name in names
    )


async def ensure_gemma_installed(
    *,
    downloader: HuggingFaceDownloadPort | None = None,
    install_dir: Path | None = None,
    cancel_event: threading.Event | None = None,
    on_status: GemmaProvisioningCallback | None = None,
    spec: GemmaModelSpec | None = None,
    source: str | None = None,
    locale: str | None = None,
    verify_existing_checksums: bool = False,
) -> InstalledGemmaManifest:
    if cancel_event is None:
        cancel_event = threading.Event()
    resolved_spec = spec or e4b_gemma_spec()
    resolved = (install_dir or default_gemma_install_dir(resolved_spec)).resolve()
    cancellation = cancel_event if cancel_event is not None else threading.Event()
    _raise_if_cancelled(cancellation)
    lease = None

    async def acquire_lease() -> LocalSTTProvisioningLease | None:
        nonlocal lease
        lease = await asyncio.to_thread(
            LocalSTTProvisioningLease.acquire,
            model_root=resolved.parent,
            wait=True,
            cancel_event=cancellation,
        )
        return lease

    try:
        try:
            await _await_download_cleanup(acquire_lease(), cancel_event=cancellation)
        except LocalSTTRuntimeInstallCancelled as exc:
            raise GemmaProvisioningCancelled("Gemma model provisioning cancelled") from exc
        if lease is None:
            raise GemmaProvisioningError("Gemma model provisioning lease is unavailable")
        return await _ensure_gemma_installed_with_lease(
            downloader=downloader,
            install_dir=resolved,
            cancel_event=cancellation,
            on_status=on_status,
            spec=resolved_spec,
            source=source,
            locale=locale,
            verify_existing_checksums=verify_existing_checksums,
        )
    finally:
        if lease is not None:
            await _await_download_cleanup(
                asyncio.to_thread(lease.close),
                cancel_event=cancellation,
            )


async def _ensure_gemma_installed_with_lease(
    *,
    downloader: HuggingFaceDownloadPort | None,
    install_dir: Path,
    cancel_event: threading.Event,
    on_status: GemmaProvisioningCallback | None,
    spec: GemmaModelSpec,
    source: str | None,
    locale: str | None,
    verify_existing_checksums: bool,
) -> InstalledGemmaManifest:
    resolved = install_dir
    if source is not None and source not in {*MODEL_DOWNLOAD_SOURCES, "local"}:
        raise ValueError("Unsupported model source")
    total_bytes = sum(asset.size_bytes for asset in spec.assets)
    _raise_if_cancelled(cancel_event)
    if isinstance(downloader, LocalGemmaDownloadAdapter):
        await _emit(on_status, state="verifying", downloaded_bytes=0, total_bytes=total_bytes)
        try:
            await downloader.validate_inputs(cancel_event=cancel_event)
        except GemmaProvisioningCancelled:
            raise
        except Exception as error:
            code = (
                "checksum_failed"
                if isinstance(error, GemmaInstallInvalidError)
                else getattr(error, "failure_code", "import_failed")
            )
            log_feature("ModelDownload", "preflight", code, error=error)
            raise GemmaProvisioningError(
                "Imported files do not match the pinned model", failure_code=code
            ) from error
    state = await _owned_io(
        inspect_gemma_install,
        resolved,
        spec=spec,
        verify_checksums=verify_existing_checksums or source == "local",
        check_cancelled=lambda: _raise_if_cancelled(cancel_event),
        cancel_event=cancel_event,
    )
    if state.status == "ready" and state.manifest is not None:
        await _emit(
            on_status,
            state="ready",
            downloaded_bytes=total_bytes,
            total_bytes=total_bytes,
        )
        return state.manifest

    if source == "local" and not isinstance(downloader, LocalGemmaDownloadAdapter):
        log_feature("ModelDownload", "preflight", "missing_file")
        raise GemmaProvisioningError("Import the pinned files first", failure_code="missing_file")
    source_plans = _source_plans(spec, source=source, locale=locale)
    if not source_plans:
        raise GemmaProvisioningError("Gemma model has no registered download sources")

    staging_dir = None
    completed_bytes = 0
    failures: list[_GemmaSourceError] = []
    stage = "preflight"
    try:
        for source_name, endpoint, source_label in source_plans:
            _raise_if_cancelled(cancel_event)
            log_feature("ModelDownload", "source", "attempt_" + source_label, level=logging.INFO)
            asset_source = (
                GemmaAssetSource(name="local", repo_id=spec.repo_id, revision=spec.revision)
                if source_name == "local"
                else spec.sources[source_name]
            )
            source_downloader: HuggingFaceDownloadPort
            if source_name in {"huggingface", "local"}:
                if downloader is None:
                    from puripuly_heart.core.local_asr.local_stt_huggingface_xet_adapter import (
                        HuggingFaceXetDownloadAdapter,
                    )

                    downloader = HuggingFaceXetDownloadAdapter()
                source_downloader = downloader
            else:
                source_downloader = _HTTPDownloadAdapter(asset_source)
            candidate = resolved.with_name(f"{resolved.name}.staging-{uuid4().hex}")
            candidate.mkdir(parents=True, exist_ok=False)
            staging_dir = candidate
            completed_bytes = 0
            stage = "preflight"
            if shutil.disk_usage(staging_dir).free < total_bytes + 64 * 1024**2:
                raise OSError(errno.ENOSPC, "Not enough space for verified model installation")
            await _emit(
                on_status,
                state="downloading",
                downloaded_bytes=0,
                total_bytes=total_bytes,
            )
            try:
                for asset in spec.assets:
                    _raise_if_cancelled(cancel_event)
                    stage = "download"
                    completed_bytes = await _download_asset(
                        downloader=source_downloader,
                        source=asset_source,
                        asset=asset,
                        staging_dir=staging_dir,
                        completed_bytes=completed_bytes,
                        total_bytes=total_bytes,
                        cancel_event=cancel_event,
                        on_status=on_status,
                        endpoint=endpoint,
                    )
                    await _emit(
                        on_status,
                        state="downloading",
                        downloaded_bytes=completed_bytes,
                        total_bytes=total_bytes,
                    )
                _raise_if_cancelled(cancel_event)
                manifest = InstalledGemmaManifest.expected(spec)
                stage = "verifying"
                await _emit(
                    on_status,
                    state="verifying",
                    downloaded_bytes=completed_bytes,
                    total_bytes=total_bytes,
                )
                (staging_dir / GEMMA_INSTALLED_MANIFEST_FILENAME).write_text(
                    json.dumps(manifest.to_dict(), indent=2),
                    encoding="utf-8",
                )
                try:
                    await _owned_io(
                        validate_gemma_install,
                        staging_dir,
                        spec=spec,
                        check_cancelled=lambda: _raise_if_cancelled(cancel_event),
                        cancel_event=cancel_event,
                    )
                except GemmaInstallInvalidError as exc:
                    raise _GemmaSourceError(str(exc), failure_code="checksum_failed") from exc
            except _GemmaSourceError as exc:
                _raise_if_cancelled(cancel_event)
                if source is not None:
                    raise
                failed_staging = staging_dir
                staging_dir = None
                _remove_owned_staging(failed_staging, primary_failure=exc)
                failures.append(exc)
                log_feature(
                    "ModelDownload",
                    "source",
                    "failed_" + source_label,
                    error=exc,
                    level=logging.WARNING,
                )
                continue
            _raise_if_cancelled(cancel_event)
            stage = "installation"
            await _emit(
                on_status,
                state="installing",
                downloaded_bytes=completed_bytes,
                total_bytes=total_bytes,
            )
            await _owned_io(_promote, staging_dir, resolved, cancel_event=cancel_event)
            log_feature("ModelDownload", "source", "completed_" + source_label, level=logging.INFO)
            await _emit(
                on_status,
                state="ready",
                downloaded_bytes=total_bytes,
                total_bytes=total_bytes,
            )
            return manifest
        raise GemmaProvisioningError(
            "Gemma model provisioning failed for all sources: "
            + "; ".join(str(failure) for failure in failures),
            failure_code=failures[-1].failure_code,
        ) from failures[-1]
    except BaseException as exc:
        if isinstance(exc, LocalSTTDownloadPortCancelled):
            failure = GemmaProvisioningCancelled("Gemma model provisioning cancelled")
            failure.__cause__ = exc
        elif isinstance(exc, Exception) and not isinstance(
            exc, (GemmaProvisioningCancelled, BaseExceptionGroup)
        ):
            code = (
                "checksum_failed"
                if isinstance(exc, GemmaInstallInvalidError)
                else (
                    "disk_full"
                    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC
                    else getattr(exc, "failure_code", "install_failed")
                )
            )
            log_feature(
                "ModelDownload",
                "verifying" if code == "checksum_failed" else stage,
                code,
                error=exc,
                http_status=getattr(exc, "status_code", None),
            )
            failure = (
                exc
                if isinstance(exc, GemmaProvisioningError)
                and not (source is not None and isinstance(exc, _GemmaSourceError))
                else GemmaProvisioningError(
                    f"Gemma model provisioning failed ({code})", failure_code=code
                )
            )
            if failure is not exc:
                failure.__cause__ = exc
        else:
            failure = exc
        if staging_dir is not None:
            _remove_owned_staging(staging_dir, primary_failure=failure)
        if isinstance(failure, Exception) and not isinstance(
            failure, (GemmaProvisioningCancelled, BaseExceptionGroup)
        ):
            await _emit(
                on_status,
                state="failed",
                downloaded_bytes=completed_bytes,
                total_bytes=total_bytes,
            )
        if failure is exc:
            raise
        raise failure


__all__ = [
    "GemmaProvisioningCallback",
    "GemmaProvisioningCancelled",
    "GemmaProvisioningError",
    "GemmaProvisioningUpdate",
    "ensure_gemma_installed",
]


class LocalGemmaDownloadAdapter:
    def __init__(self, files: tuple[Path, ...], spec: GemmaModelSpec) -> None:
        self.files = {path.name: path for path in files}
        self.spec = spec

    async def validate_inputs(self, *, cancel_event):
        def validate():
            if set(self.files) != {asset.filename for asset in self.spec.assets}:
                raise GemmaProvisioningError(
                    "Select the exact pinned files", failure_code="missing_file"
                )
            for asset in self.spec.assets:
                source = self.files[asset.filename]
                if not source.is_file():
                    raise GemmaProvisioningError("Missing pinned file", failure_code="missing_file")
                if source.stat().st_size != asset.size_bytes:
                    raise GemmaProvisioningError(
                        "Wrong file size", failure_code="file_size_mismatch"
                    )
                digest = hashlib.sha256()
                with source.open("rb") as reader:
                    while chunk := reader.read(4 * 1024 * 1024):
                        _raise_if_cancelled(cancel_event)
                        digest.update(chunk)
                if digest.hexdigest() != asset.sha256:
                    raise GemmaInstallInvalidError("Import checksum mismatch")

        task = asyncio.create_task(asyncio.to_thread(validate))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if cancel_event is not None:
                cancel_event.set()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def download(self, request, *, cancel_event, on_progress):
        asset = next(asset for asset in self.spec.assets if asset.filename == request.remote_path)
        source = self.files.get(asset.filename)
        if source is None or not source.is_file():
            raise GemmaProvisioningError("Missing pinned model file", failure_code="missing_file")
        if source.stat().st_size != asset.size_bytes:
            raise GemmaProvisioningError("Wrong file size", failure_code="file_size_mismatch")
        destination = request.local_dir / asset.filename

        def copy_verified():
            from puripuly_heart.core.local_asr.local_stt_download_port import (
                HuggingFaceDownloadProgress,
            )

            digest = hashlib.sha256()
            copied = 0
            last_update = 0.0
            with source.open("rb") as reader, destination.open("wb") as writer:
                while chunk := reader.read(4 * 1024 * 1024):
                    _raise_if_cancelled(cancel_event)
                    writer.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                    if on_progress is not None and (
                        time.monotonic() - last_update >= 0.25 or copied == asset.size_bytes
                    ):
                        on_progress(HuggingFaceDownloadProgress(copied, asset.size_bytes))
                        last_update = time.monotonic()
            if copied != asset.size_bytes or digest.hexdigest() != asset.sha256:
                from puripuly_heart.core.local_translation.assets import GemmaInstallInvalidError

                raise GemmaInstallInvalidError("Imported file does not match the pinned model")

        task = asyncio.create_task(asyncio.to_thread(copy_verified))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if cancel_event is not None:
                cancel_event.set()
            await asyncio.gather(task, return_exceptions=True)
            raise
        return destination
