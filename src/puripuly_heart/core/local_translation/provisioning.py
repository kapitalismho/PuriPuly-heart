from __future__ import annotations

import asyncio
import inspect
import json
import shutil
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar
from urllib.parse import quote
from uuid import uuid4

import httpx

from puripuly_heart.core import network_clients
from puripuly_heart.core.local_asr.local_stt_assets import default_local_stt_source_for_locale
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


class GemmaProvisioningError(RuntimeError):
    pass


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
) -> None:
    if callback is None:
        return
    update = GemmaProvisioningUpdate(
        state=state,
        downloaded_bytes=downloaded_bytes,
        total_bytes=total_bytes,
        percent=min(100, (downloaded_bytes * 100) // total_bytes) if total_bytes else 100,
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
            and download_failure.failure_code == "download_failed"
        ):
            _raise_if_cancelled(cancel_event)
            raise _GemmaSourceError(str(download_failure)) from download_failure
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


def _source_order(
    spec: GemmaModelSpec,
    *,
    preferred_source: str | None,
    locale: str | None,
) -> tuple[str, ...]:
    selected = preferred_source or default_local_stt_source_for_locale(locale)
    if selected not in spec.sources:
        return tuple(spec.sources)
    return (selected, *(name for name in spec.sources if name != selected))


async def ensure_gemma_installed(
    *,
    downloader: HuggingFaceDownloadPort | None = None,
    install_dir: Path | None = None,
    cancel_event: threading.Event | None = None,
    on_status: GemmaProvisioningCallback | None = None,
    spec: GemmaModelSpec | None = None,
    locale: str | None = None,
    preferred_source: str | None = None,
) -> InstalledGemmaManifest:
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
            locale=locale,
            preferred_source=preferred_source,
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
    locale: str | None,
    preferred_source: str | None,
) -> InstalledGemmaManifest:
    resolved = install_dir
    total_bytes = sum(asset.size_bytes for asset in spec.assets)
    _raise_if_cancelled(cancel_event)
    state = inspect_gemma_install(resolved, spec=spec)
    if state.status == "ready" and state.manifest is not None:
        await _emit(
            on_status,
            state="ready",
            downloaded_bytes=total_bytes,
            total_bytes=total_bytes,
        )
        return state.manifest

    staging_dir = None
    completed_bytes = 0
    failures = []
    try:
        source_names = _source_order(spec, preferred_source=preferred_source, locale=locale)
        if not source_names:
            raise GemmaProvisioningError("Gemma model has no registered download sources")
        for source_name in source_names:
            _raise_if_cancelled(cancel_event)
            source = spec.sources[source_name]
            source_downloader: HuggingFaceDownloadPort
            if source_name == "huggingface":
                if downloader is None:
                    from puripuly_heart.core.local_asr.local_stt_huggingface_xet_adapter import (
                        HuggingFaceXetDownloadAdapter,
                    )

                    downloader = HuggingFaceXetDownloadAdapter()
                source_downloader = downloader
            else:
                source_downloader = _HTTPDownloadAdapter(source)
            candidate = resolved.with_name(f"{resolved.name}.staging-{uuid4().hex}")
            candidate.mkdir(parents=True, exist_ok=False)
            staging_dir = candidate
            completed_bytes = 0
            await _emit(
                on_status,
                state="downloading",
                downloaded_bytes=0,
                total_bytes=total_bytes,
            )
            try:
                for asset in spec.assets:
                    _raise_if_cancelled(cancel_event)
                    completed_bytes = await _download_asset(
                        downloader=source_downloader,
                        source=source,
                        asset=asset,
                        staging_dir=staging_dir,
                        completed_bytes=completed_bytes,
                        total_bytes=total_bytes,
                        cancel_event=cancel_event,
                        on_status=on_status,
                    )
                    await _emit(
                        on_status,
                        state="downloading",
                        downloaded_bytes=completed_bytes,
                        total_bytes=total_bytes,
                    )
                _raise_if_cancelled(cancel_event)
                manifest = InstalledGemmaManifest.expected(spec)
                (staging_dir / GEMMA_INSTALLED_MANIFEST_FILENAME).write_text(
                    json.dumps(manifest.to_dict(), indent=2),
                    encoding="utf-8",
                )
                try:
                    await _await_download_cleanup(
                        asyncio.to_thread(validate_gemma_install, staging_dir, spec=spec),
                        cancel_event=cancel_event,
                    )
                except GemmaInstallInvalidError as exc:
                    raise _GemmaSourceError(str(exc)) from exc
            except _GemmaSourceError as exc:
                _raise_if_cancelled(cancel_event)
                failed_staging = staging_dir
                staging_dir = None
                _remove_owned_staging(failed_staging, primary_failure=exc)
                failures.append(exc)
                continue
            _raise_if_cancelled(cancel_event)
            await _await_download_cleanup(
                asyncio.to_thread(_promote, staging_dir, resolved),
                cancel_event=cancel_event,
            )
            await _emit(
                on_status,
                state="ready",
                downloaded_bytes=total_bytes,
                total_bytes=total_bytes,
            )
            return manifest
        raise GemmaProvisioningError(
            "Gemma model provisioning failed for all sources: "
            + "; ".join(str(failure) for failure in failures)
        ) from failures[-1]
    except BaseException as exc:
        if isinstance(exc, LocalSTTDownloadPortCancelled):
            failure = GemmaProvisioningCancelled("Gemma model provisioning cancelled")
            failure.__cause__ = exc
        elif isinstance(exc, Exception) and not isinstance(
            exc, (GemmaProvisioningError, BaseExceptionGroup)
        ):
            failure = GemmaProvisioningError(f"Gemma model provisioning failed: {exc}")
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
