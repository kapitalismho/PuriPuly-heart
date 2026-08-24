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
from uuid import uuid4

from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadPort,
    HuggingFaceDownloadRequest,
    LocalSTTDownloadPortCancelled,
)
from puripuly_heart.core.local_asr.local_stt_runtime_installer import (
    LocalSTTProvisioningLease,
    LocalSTTRuntimeInstallCancelled,
)
from puripuly_heart.core.local_translation.assets import (
    GEMMA_INSTALLED_MANIFEST_FILENAME,
    GemmaAsset,
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


async def _download_asset(
    *,
    downloader: HuggingFaceDownloadPort,
    spec: GemmaModelSpec,
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
        try:
            results = await asyncio.shield(gathered)
        except asyncio.CancelledError as cancellation:
            results = await gathered
            failures = tuple(result for result in results if isinstance(result, BaseException))
            if failures:
                raise BaseExceptionGroup(
                    "Gemma progress drain cancelled with callback failures",
                    (cancellation, *failures),
                )
            raise
        return tuple(result for result in results if isinstance(result, BaseException))

    try:
        downloaded_path = await downloader.download(
            HuggingFaceDownloadRequest(
                repo_id=spec.repo_id,
                revision=spec.revision,
                remote_path=asset.filename,
                local_dir=staging_dir,
                expected_size_bytes=asset.size_bytes,
            ),
            cancel_event=cancel_event,
            on_progress=on_progress,
        )
    except BaseException as download_failure:
        progress_failures = await drain_progress()
        if progress_failures:
            raise BaseExceptionGroup(
                "Gemma download and progress callbacks failed",
                (download_failure, *progress_failures),
            )
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


async def ensure_gemma_installed(
    *,
    downloader: HuggingFaceDownloadPort | None = None,
    install_dir: Path | None = None,
    cancel_event: threading.Event | None = None,
    on_status: GemmaProvisioningCallback | None = None,
    spec: GemmaModelSpec | None = None,
) -> InstalledGemmaManifest:
    resolved_spec = spec or e4b_gemma_spec()
    resolved = (install_dir or default_gemma_install_dir(resolved_spec)).resolve()
    _raise_if_cancelled(cancel_event)
    try:
        lease = await asyncio.to_thread(
            LocalSTTProvisioningLease.acquire,
            model_root=resolved.parent,
            wait=True,
            cancel_event=cancel_event,
        )
    except LocalSTTRuntimeInstallCancelled as exc:
        raise GemmaProvisioningCancelled("Gemma model provisioning cancelled") from exc
    if lease is None:
        raise GemmaProvisioningError("Gemma model provisioning lease is unavailable")
    try:
        return await _ensure_gemma_installed_with_lease(
            downloader=downloader,
            install_dir=resolved,
            cancel_event=cancel_event,
            on_status=on_status,
            spec=resolved_spec,
        )
    finally:
        await asyncio.to_thread(lease.close)


async def _ensure_gemma_installed_with_lease(
    *,
    downloader: HuggingFaceDownloadPort | None,
    install_dir: Path,
    cancel_event: threading.Event | None,
    on_status: GemmaProvisioningCallback | None,
    spec: GemmaModelSpec,
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

    if downloader is None:
        from puripuly_heart.core.local_asr.local_stt_huggingface_xet_adapter import (
            HuggingFaceXetDownloadAdapter,
        )

        downloader = HuggingFaceXetDownloadAdapter()

    staging_dir = resolved.with_name(f"{resolved.name}.staging-{uuid4().hex}")
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=False)
    completed_bytes = 0
    try:
        await _emit(
            on_status,
            state="downloading",
            downloaded_bytes=0,
            total_bytes=total_bytes,
        )
        for asset in spec.assets:
            _raise_if_cancelled(cancel_event)
            completed_bytes = await _download_asset(
                downloader=downloader,
                spec=spec,
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
        await asyncio.to_thread(validate_gemma_install, staging_dir, spec=spec)
        _raise_if_cancelled(cancel_event)
        await asyncio.to_thread(_promote, staging_dir, resolved)
        await _emit(
            on_status,
            state="ready",
            downloaded_bytes=total_bytes,
            total_bytes=total_bytes,
        )
        return manifest
    except asyncio.CancelledError:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    except LocalSTTDownloadPortCancelled as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise GemmaProvisioningCancelled("Gemma model provisioning cancelled") from exc
    except GemmaProvisioningCancelled:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        await _emit(
            on_status,
            state="failed",
            downloaded_bytes=completed_bytes,
            total_bytes=total_bytes,
        )
        if isinstance(exc, GemmaProvisioningError):
            raise
        raise GemmaProvisioningError(f"Gemma model provisioning failed: {exc}") from exc
    except BaseExceptionGroup:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


__all__ = [
    "GemmaProvisioningCallback",
    "GemmaProvisioningCancelled",
    "GemmaProvisioningError",
    "GemmaProvisioningUpdate",
    "ensure_gemma_installed",
]
