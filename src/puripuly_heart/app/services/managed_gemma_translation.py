from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadPort,
    LocalSTTDownloadPortError,
)
from puripuly_heart.core.local_asr.local_stt_runtime_installer import LocalSTTProvisioningLease
from puripuly_heart.core.local_translation.assets import (
    InstalledGemmaManifest,
    default_gemma_install_dir,
    inspect_gemma_install,
    resolve_gemma_spec,
)
from puripuly_heart.core.local_translation.provisioning import (
    GemmaProvisioningCallback,
    GemmaProvisioningCancelled,
    GemmaProvisioningError,
    GemmaProvisioningUpdate,
    LocalGemmaDownloadAdapter,
    ensure_gemma_installed,
)
from puripuly_heart.core.local_translation.runtime import (
    ManagedGemmaReadiness,
    ManagedGemmaRuntimeError,
    ManagedGemmaRuntimeMissingError,
    ManagedGemmaRuntimeOwner,
    ManagedGemmaRuntimeVersionError,
)
from puripuly_heart.core.observability import DiagnosticEvent
from puripuly_heart.core.owned_thread import run_owned_thread_call


@dataclass(frozen=True, slots=True)
class ManagedGemmaTranslationSnapshot:
    state: str
    backend: str | None
    progress_percent: int | None
    error_type: str | None = None
    failure_phase: str | None = None
    failure_code: str | None = None
    cause_type: str | None = None
    worker_exit_code: int | None = None
    status_code: int | None = None
    os_error_code: int | None = None
    model_id: str | None = None


ManagedGemmaTranslationStatusSink = Callable[[ManagedGemmaTranslationSnapshot], None]
ManagedGemmaLifecycleDiagnosticSink = Callable[[DiagnosticEvent], None]


class _ManagedGemmaDiagnosticsSink:
    def __init__(self, sink: ManagedGemmaLifecycleDiagnosticSink | None) -> None:
        self._sink = sink

    async def emit_diagnostic(self, event: DiagnosticEvent) -> None:
        if self._sink is not None:
            self._sink(event)


@dataclass(frozen=True, slots=True)
class ManagedGemmaActivation:
    readiness: ManagedGemmaReadiness
    runtime: ManagedGemmaRuntimeOwner
    backend: str
    release: Callable[[], Awaitable[None]]


class ManagedGemmaTranslationOwner:
    def __init__(
        self,
        *,
        runtime: ManagedGemmaRuntimeOwner,
        downloader: HuggingFaceDownloadPort | None = None,
        status_sink: ManagedGemmaTranslationStatusSink | None = None,
        lifecycle_diagnostic_sink: ManagedGemmaLifecycleDiagnosticSink | None = None,
        idle_linger_s: float = 10.0,
    ) -> None:
        self._runtime = runtime
        self._downloader = downloader
        self._status_sink = status_sink
        self._idle_linger_s = max(0.0, float(idle_linger_s))
        self._snapshot = ManagedGemmaTranslationSnapshot("idle", None, None)
        self._lock = asyncio.Lock()
        self._task_scope = LifecycleScope(
            "ManagedGemmaTranslationOwner",
            diagnostics_sink=_ManagedGemmaDiagnosticsSink(lifecycle_diagnostic_sink),
        )
        self._generation = 0
        self._cancel_event: threading.Event | None = None
        self._prepare_task: asyncio.Task[ManagedGemmaReadiness] | None = None
        self._asset_task: asyncio.Task[InstalledGemmaManifest] | None = None
        self._asset_cancel_event: threading.Event | None = None
        self._asset_model_id: str | None = None
        self._linger_task: asyncio.Task[None] | None = None
        self._active_generation: int | None = None
        self._selected_model_id: str | None = None
        self._closed = False

    @property
    def runtime(self) -> ManagedGemmaRuntimeOwner:
        return self._runtime

    @property
    def snapshot(self) -> ManagedGemmaTranslationSnapshot:
        return self._snapshot

    @property
    def active_model_id(self) -> str | None:
        return self._selected_model_id if self._active_generation is not None else None

    def install_status(self, model_id: str) -> str:
        return inspect_gemma_install(spec=resolve_gemma_spec(model_id)).status

    async def prepare(
        self,
        selection: ManagedGemmaTranslationSelection,
    ) -> ManagedGemmaActivation:
        backend = selection.backend
        if self._asset_task is not None:
            raise GemmaProvisioningError("Another model task is active", failure_code="task_busy")
        lingering = self._cancel_linger()
        await self._await_cancelled_linger(lingering)
        async with self._lock:
            if self._closed:
                raise RuntimeError("managed Gemma translation owner is closed")
            if self._asset_task is not None:
                raise GemmaProvisioningError(
                    "Another model task is active", failure_code="task_busy"
                )
            self._generation += 1
            generation = self._generation
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            self._selected_model_id = selection.model_id
            self._publish("checking", backend=backend, progress_percent=None)

            def on_status(update: GemmaProvisioningUpdate) -> None:
                if generation != self._generation:
                    return
                state = "downloading" if update.state == "downloading" else "preparing"
                self._publish(
                    state,
                    backend=backend,
                    progress_percent=update.percent,
                )

            spec = resolve_gemma_spec(selection.model_id)
            task = start_lifecycle_task(
                self._task_scope,
                self._runtime.prepare(
                    backend=backend,
                    source_language=selection.source_language,
                    target_language=selection.target_language,
                    system_prompt=selection.system_prompt,
                    vulkan_device=selection.vulkan_device,
                    spec=spec,
                    provision_kwargs={
                        "downloader": self._downloader,
                        "cancel_event": cancel_event,
                        "on_status": on_status,
                        "spec": spec,
                        "locale": selection.locale,
                    },
                ),
                name=f"prepare-{generation}",
            )
            self._prepare_task = task
            try:
                readiness = await task
            except asyncio.CancelledError as exc:
                if asyncio.current_task() is not None and asyncio.current_task().cancelling():
                    raise
                self._publish("cancelled", backend=backend, progress_percent=None)
                raise GemmaProvisioningCancelled("Gemma model provisioning cancelled") from exc
            except GemmaProvisioningCancelled:
                self._publish("cancelled", backend=backend, progress_percent=None)
                raise
            except BaseExceptionGroup as exc:
                failure = _failure_metadata(exc)
                self._publish(
                    "failed",
                    backend=backend,
                    progress_percent=None,
                    error_type=type(exc).__name__,
                    failure_phase=failure["phase"],
                    failure_code=failure["code"],
                    cause_type=failure["cause_type"],
                    worker_exit_code=failure["worker_exit_code"],
                    status_code=failure["status_code"],
                    os_error_code=failure["os_error_code"],
                )
                raise
            except Exception as exc:
                failure = _failure_metadata(exc)
                self._publish(
                    "failed",
                    backend=backend,
                    progress_percent=None,
                    error_type=type(exc).__name__,
                    failure_phase=failure["phase"],
                    failure_code=failure["code"],
                    cause_type=failure["cause_type"],
                    worker_exit_code=failure["worker_exit_code"],
                    status_code=failure["status_code"],
                    os_error_code=failure["os_error_code"],
                )
                raise
            finally:
                if self._prepare_task is task:
                    self._prepare_task = None
                if self._cancel_event is cancel_event:
                    self._cancel_event = None
            if generation != self._generation or self._closed:
                await self._runtime.release()
                raise GemmaProvisioningCancelled("Gemma model activation was superseded")
            self._active_generation = generation
            self._publish("ready", backend=backend, progress_percent=100)
            return ManagedGemmaActivation(
                readiness=readiness,
                runtime=self._runtime,
                backend=backend,
                release=lambda: self._release_generation(generation),
            )

    def cancel(self) -> bool:
        cancel_event = self._cancel_event
        task = self._prepare_task
        if cancel_event is None and task is None:
            return False
        if cancel_event is not None:
            cancel_event.set()
        if task is not None and not task.done():
            task.cancel()
        return True

    def cancel_assets(self, model_id: str) -> bool:
        if model_id != self._asset_model_id:
            return False
        cancellation = self._asset_cancel_event
        task = self._asset_task
        if cancellation is None or task is None:
            return False
        cancellation.set()
        if not task.done():
            task.cancel()
        return True

    async def install_assets(
        self,
        *,
        source: str | None = None,
        model_id: str | None = None,
        files: tuple[Path, ...] = (),
        on_status: GemmaProvisioningCallback | None = None,
        locale: str | None = None,
    ) -> InstalledGemmaManifest:
        from puripuly_heart.core.feature_diagnostics import log_feature

        if self._closed:
            raise RuntimeError("Model owner closed")
        if self._lock.locked() or self._asset_task is not None:
            raise GemmaProvisioningError("Another model task is active", failure_code="task_busy")
        spec = resolve_gemma_spec(model_id)
        cancellation = threading.Event()
        downloader = (
            LocalGemmaDownloadAdapter(files, spec)
            if source == "local" and files
            else None if source == "local" else self._downloader
        )

        task = start_lifecycle_task(
            self._task_scope,
            ensure_gemma_installed(
                source=source,
                spec=spec,
                downloader=downloader,
                cancel_event=cancellation,
                on_status=on_status,
                locale=locale,
                verify_existing_checksums=True,
            ),
            name="install-assets",
        )
        self._asset_task = task
        self._asset_cancel_event = cancellation
        self._asset_model_id = spec.model_id
        try:
            return await task
        except asyncio.CancelledError as exc:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            raise GemmaProvisioningCancelled("Gemma asset installation cancelled") from exc
        except GemmaProvisioningCancelled:
            raise
        except Exception as error:
            log_feature(
                "ModelDownload",
                "installation",
                getattr(error, "failure_code", "install_failed"),
                error=error,
            )
            raise
        finally:
            if self._asset_task is task:
                self._asset_task = None
                self._asset_cancel_event = None
                self._asset_model_id = None

    async def delete_assets(self, model_id: str) -> None:
        from puripuly_heart.core.model_install_removal import (
            ModelInstallInUseError,
            remove_model_install_dir,
        )

        if self._asset_task is not None:
            raise GemmaProvisioningError("Another model task is active", failure_code="task_busy")
        lingering = self._cancel_linger()
        await self._await_cancelled_linger(lingering)
        async with self._lock:
            if self._closed:
                raise RuntimeError("Model owner closed")
            if self._asset_task is not None:
                raise GemmaProvisioningError(
                    "Another model task is active", failure_code="task_busy"
                )
            if self._prepare_task is not None and not self._prepare_task.done():
                raise GemmaProvisioningError(
                    "Another model task is active", failure_code="task_busy"
                )
            if self.active_model_id == model_id:
                raise ModelInstallInUseError("Model is loaded for translation")
            if self._active_generation is None:
                await self._runtime.release()
            install_dir = default_gemma_install_dir(resolve_gemma_spec(model_id))

            def remove_install() -> None:
                lease = LocalSTTProvisioningLease.acquire(
                    model_root=install_dir.parent,
                    wait=False,
                )
                if lease is None:
                    raise GemmaProvisioningError(
                        "Another model task is active", failure_code="task_busy"
                    )
                try:
                    remove_model_install_dir(install_dir)
                finally:
                    lease.close()

            await run_owned_thread_call(remove_install)

    def schedule_demand_sync(
        self,
        *,
        desired: bool,
        selection: ManagedGemmaTranslationSelection | None = None,
    ) -> None:
        async def run() -> None:
            if self._closed:
                return
            if desired:
                if selection is None:
                    return
                await self.prepare(selection)
                return
            await self.deactivate(linger=True)

        start_lifecycle_task(self._task_scope, run(), name="demand-sync")

    async def deactivate(self, *, linger: bool = False) -> None:
        self._generation += 1
        self._active_generation = None
        self.cancel()
        task = self._prepare_task
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)
        if linger and not self._closed:
            self._publish("idle", backend=None, progress_percent=None)
            self._schedule_linger(self._idle_linger_s)
            return
        lingering = self._cancel_linger()
        await self._await_cancelled_linger(lingering)
        async with self._lock:
            await self._runtime.release()
            self._publish("idle", backend=None, progress_percent=None)

    async def close(self) -> None:
        lingering = self._cancel_linger()
        if self._closed and self._prepare_task is None and self._asset_task is None:
            await self._await_cancelled_linger(lingering)
            await self._runtime.close()
            return
        self._closed = True
        self._generation += 1
        self._active_generation = None
        self.cancel()
        if self._asset_model_id is not None:
            self.cancel_assets(self._asset_model_id)
        tasks = tuple(
            task
            for task in (self._prepare_task, self._asset_task)
            if task is not None and task is not asyncio.current_task()
        )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._await_cancelled_linger(lingering)
        try:
            await self._task_scope.close()
        finally:
            async with self._lock:
                await self._runtime.close()
                self._publish("closed", backend=None, progress_percent=None)

    def _cancel_linger(self) -> asyncio.Task[None] | None:
        task = self._linger_task
        self._linger_task = None
        if task is not None and not task.done():
            task.cancel()
            return task
        return None

    async def _await_cancelled_linger(self, task: asyncio.Task[None] | None) -> None:
        if task is None or task is asyncio.current_task():
            return
        await asyncio.gather(task, return_exceptions=True)

    def _schedule_linger(self, delay_s: float) -> None:
        self._cancel_linger()
        generation = self._generation
        self._linger_task = start_lifecycle_task(
            self._task_scope,
            self._linger_release(generation, delay_s),
            name=f"linger-{generation}",
        )

    async def _linger_release(self, generation: int, delay_s: float) -> None:
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            return
        if generation != self._generation or self._closed:
            return
        async with self._lock:
            if generation != self._generation or self._closed:
                return
            try:
                await self._runtime.release()
            except asyncio.CancelledError:
                if self._closed:
                    return
                raise
            if not self._closed:
                self._publish("idle", backend=None, progress_percent=None)

    async def _release_generation(self, generation: int) -> None:
        async with self._lock:
            if self._active_generation != generation:
                return
            self._active_generation = None
            await self._runtime.release()
            if not self._closed:
                self._publish("idle", backend=None, progress_percent=None)

    def _publish(
        self,
        state: str,
        *,
        backend: str | None,
        progress_percent: int | None,
        error_type: str | None = None,
        failure_phase: str | None = None,
        failure_code: str | None = None,
        cause_type: str | None = None,
        worker_exit_code: int | None = None,
        status_code: int | None = None,
        os_error_code: int | None = None,
    ) -> None:
        snapshot = ManagedGemmaTranslationSnapshot(
            state=state,
            backend=backend,
            progress_percent=progress_percent,
            error_type=error_type,
            failure_phase=failure_phase,
            failure_code=failure_code,
            cause_type=cause_type,
            worker_exit_code=worker_exit_code,
            status_code=status_code,
            os_error_code=os_error_code,
            model_id=self._selected_model_id if backend is not None else None,
        )
        self._snapshot = snapshot
        if self._status_sink is not None:
            with contextlib.suppress(Exception):
                self._status_sink(snapshot)


def _exception_tree(exception: BaseException) -> tuple[BaseException, ...]:
    found: list[BaseException] = []
    pending = [exception]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        found.append(current)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        linked = current.__cause__ or current.__context__
        if linked is not None:
            pending.append(linked)
    return tuple(found)


def _failure_metadata(exception: BaseException) -> dict[str, str | int | None]:
    chain = _exception_tree(exception)
    filesystem_failure = next((item for item in chain if isinstance(item, OSError)), None)
    if isinstance(exception, BaseExceptionGroup) and filesystem_failure is not None:
        return {
            "phase": "provisioning",
            "code": "filesystem_failed",
            "cause_type": type(filesystem_failure).__name__,
            "worker_exit_code": None,
            "status_code": None,
            "os_error_code": getattr(filesystem_failure, "winerror", None)
            or filesystem_failure.errno,
        }
    download_failure = next(
        (item for item in chain if isinstance(item, LocalSTTDownloadPortError)),
        None,
    )
    if download_failure is not None:
        return {
            "phase": "download",
            "code": download_failure.failure_code,
            "cause_type": download_failure.cause_type or type(chain[-1]).__name__,
            "worker_exit_code": download_failure.worker_exit_code,
            "status_code": download_failure.status_code,
            "os_error_code": download_failure.os_error_code,
        }
    runtime_files_failure = next(
        (
            item
            for item in chain
            if isinstance(item, ManagedGemmaRuntimeMissingError | ManagedGemmaRuntimeVersionError)
        ),
        None,
    )
    if runtime_files_failure is not None:
        return {
            "phase": "runtime_files",
            "code": (
                "runtime_missing"
                if isinstance(runtime_files_failure, ManagedGemmaRuntimeMissingError)
                else "runtime_incompatible"
            ),
            "cause_type": type(runtime_files_failure).__name__,
            "worker_exit_code": None,
            "status_code": None,
            "os_error_code": None,
        }
    if isinstance(exception, GemmaProvisioningError):
        phase = "provisioning"
        code = "provisioning_failed"
    elif isinstance(exception, ManagedGemmaRuntimeError):
        phase = "runtime"
        code = "runtime_failed"
    else:
        phase = "prepare"
        code = "prepare_failed"
    root = chain[-1]
    os_error_code = None
    if isinstance(root, OSError):
        phase = "provisioning"
        code = "filesystem_failed"
        os_error_code = getattr(root, "winerror", None) or root.errno
    return {
        "phase": phase,
        "code": code,
        "cause_type": type(root).__name__,
        "worker_exit_code": None,
        "status_code": None,
        "os_error_code": os_error_code,
    }


__all__ = [
    "ManagedGemmaActivation",
    "ManagedGemmaLifecycleDiagnosticSink",
    "ManagedGemmaTranslationOwner",
    "ManagedGemmaTranslationSnapshot",
    "ManagedGemmaTranslationStatusSink",
]
