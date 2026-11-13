from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.core.diagnostics import current_diagnostic_operation, sanitize_diagnostics
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.local_asr.local_stt_download_port import (
    HuggingFaceDownloadPort,
    LocalSTTDownloadPortError,
)
from puripuly_heart.core.local_asr.local_stt_runtime_installer import LocalSTTProvisioningLease
from puripuly_heart.core.local_translation.assets import (
    GemmaInstallState,
    InstalledGemmaManifest,
    cleanup_gemma_install_residue,
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
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.model_asset_diagnostics import model_asset_failure_diagnostics
from puripuly_heart.core.network_diagnostics import transport_exception_chain
from puripuly_heart.core.observability import DiagnosticEvent
from puripuly_heart.core.owned_thread import run_owned_thread_call


@dataclass(frozen=True, slots=True)
class ManagedGemmaTranslationSnapshot:
    state: str
    backend: str | None
    progress_percent: int | None
    failure_diagnostics: ErrorDiagnostics | None = None
    recovery_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    secondary_diagnostics: tuple[ErrorDiagnostics, ...] = ()
    recovery_diagnostics_truncated: bool = False
    secondary_diagnostics_truncated: bool = False
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
        self._diagnostics: deque[ErrorDiagnostics] = deque(maxlen=8)
        self._diagnostics_truncated = False
        self._residue_cleanup_lock = threading.Lock()
        self._residue_cleanup_complete = False

    @property
    def diagnostics(self) -> tuple[ErrorDiagnostics, ...]:
        return tuple(self._diagnostics)

    def _record_failure(
        self, error: BaseException, *, identity: dict[str, str | None] | None = None
    ) -> ErrorDiagnostics:
        if isinstance(error, ManagedGemmaRuntimeVersionError):
            operation, code = "runtime_version", "model_assets.runtime_incompatible"
        elif isinstance(error, ManagedGemmaRuntimeMissingError):
            operation, code = "runtime_files", "model_assets.runtime_missing"
        elif isinstance(error, ManagedGemmaRuntimeError):
            operation, code = "runtime", "model_assets.runtime_failed"
        elif not isinstance(error, BaseExceptionGroup) and any(
            isinstance(item, LocalSTTDownloadPortError) for item in transport_exception_chain(error)
        ):
            operation, code = "download", "model_assets.install_failed"
        else:
            operation, code = "gemma_install", "model_assets.install_failed"
        diagnostics = model_asset_failure_diagnostics(error, operation=operation, code=code)
        if identity is not None:
            diagnostics = sanitize_diagnostics(
                replace(
                    diagnostics,
                    operation_id=identity.get("operation_id") or diagnostics.operation_id,
                    request_id=identity.get("request_id") or diagnostics.request_id,
                )
            )
        error.failure_diagnostics = diagnostics
        if (
            not self._diagnostics
            or self._diagnostics[-1].diagnostic_id != diagnostics.diagnostic_id
        ):
            self._diagnostics_truncated |= len(self._diagnostics) == self._diagnostics.maxlen
            self._diagnostics.append(diagnostics)
        return diagnostics

    def _retain_settled_results(self, results: list[object]) -> None:
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                self._record_failure(result)

    @property
    def runtime(self) -> ManagedGemmaRuntimeOwner:
        return self._runtime

    @property
    def snapshot(self) -> ManagedGemmaTranslationSnapshot:
        snapshot = self._snapshot
        excluded = {
            value.diagnostic_id
            for value in snapshot.recovery_diagnostics + snapshot.secondary_diagnostics
        }
        if snapshot.failure_diagnostics is not None:
            excluded.add(snapshot.failure_diagnostics.diagnostic_id)
        secondary = snapshot.secondary_diagnostics + tuple(
            value for value in self.diagnostics if value.diagnostic_id not in excluded
        )
        truncated = snapshot.secondary_diagnostics_truncated or self._diagnostics_truncated
        if (
            secondary == snapshot.secondary_diagnostics
            and truncated == snapshot.secondary_diagnostics_truncated
        ):
            return snapshot
        return replace(
            snapshot,
            secondary_diagnostics=secondary,
            secondary_diagnostics_truncated=truncated,
        )

    @property
    def active_model_id(self) -> str | None:
        return self._selected_model_id if self._active_generation is not None else None

    def install_state(self, model_id: str) -> GemmaInstallState:
        self._ensure_startup_cleanup()
        return inspect_gemma_install(spec=resolve_gemma_spec(model_id))

    def _ensure_startup_cleanup(self) -> None:
        from puripuly_heart.core.feature_diagnostics import log_feature

        with self._residue_cleanup_lock:
            if self._closed or self._residue_cleanup_complete:
                return
            try:
                removed = cleanup_gemma_install_residue()
            except Exception as error:
                self._residue_cleanup_complete = True
                error.failure_diagnostics = model_asset_failure_diagnostics(
                    error,
                    operation="cleanup",
                    code="model_assets.cleanup_failed",
                    fields={"cleanup_complete": False},
                )
                self._record_failure(error)
                log_feature(
                    "ModelDownload",
                    "cleanup",
                    "cleanup_failed",
                    error=error,
                )
                return
            if removed is None:
                return
            self._residue_cleanup_complete = True
            if removed:
                log_feature(
                    "ModelDownload",
                    "cleanup",
                    "residue_removed",
                    level=logging.INFO,
                )

    def _stop_residue_cleanup(self) -> None:
        with self._residue_cleanup_lock:
            self._residue_cleanup_complete = True

    async def prepare(
        self,
        selection: ManagedGemmaTranslationSelection,
    ) -> ManagedGemmaActivation:
        backend = selection.backend
        identity = current_diagnostic_operation()
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

            async def prepare_runtime() -> ManagedGemmaReadiness:
                await run_owned_thread_call(self._ensure_startup_cleanup)
                return await self._runtime.prepare(
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
                )

            task = start_lifecycle_task(
                self._task_scope,
                prepare_runtime(),
                name=f"prepare-{generation}",
            )
            self._prepare_task = task
            try:
                readiness = await task
            except asyncio.CancelledError as exc:
                if asyncio.current_task() is not None and asyncio.current_task().cancelling():
                    raise
                history = getattr(exc, "secondary_diagnostics", ())
                self._publish(
                    "cancelled",
                    backend=backend,
                    progress_percent=None,
                    secondary_diagnostics=history,
                )
                cancelled = GemmaProvisioningCancelled("Gemma model provisioning cancelled")
                cancelled.secondary_diagnostics = history
                raise cancelled from exc
            except GemmaProvisioningCancelled as exc:
                self._publish(
                    "cancelled",
                    backend=backend,
                    progress_percent=None,
                    secondary_diagnostics=exc.secondary_diagnostics,
                )
                raise
            except (Exception, BaseExceptionGroup) as exc:
                diagnostics = self._record_failure(exc, identity=identity)
                self._publish(
                    "failed",
                    backend=backend,
                    progress_percent=None,
                    failure_diagnostics=diagnostics,
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
            self._publish(
                "ready",
                backend=backend,
                progress_percent=100,
                recovery_diagnostics=readiness.recovery_diagnostics,
                secondary_diagnostics=readiness.secondary_diagnostics,
                recovery_diagnostics_truncated=readiness.recovery_diagnostics_truncated,
                secondary_diagnostics_truncated=readiness.secondary_diagnostics_truncated,
            )
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

        identity = current_diagnostic_operation()

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

        async def install() -> InstalledGemmaManifest:
            await run_owned_thread_call(self._ensure_startup_cleanup)
            return await ensure_gemma_installed(
                source=source,
                spec=spec,
                downloader=downloader,
                cancel_event=cancellation,
                on_status=on_status,
                locale=locale,
                verify_existing_checksums=True,
            )

        task = start_lifecycle_task(
            self._task_scope,
            install(),
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
            cancelled = GemmaProvisioningCancelled("Gemma asset installation cancelled")
            cancelled.secondary_diagnostics = getattr(exc, "secondary_diagnostics", ())
            raise cancelled from exc
        except GemmaProvisioningCancelled:
            raise
        except (Exception, BaseExceptionGroup) as error:
            self._record_failure(error, identity=identity)
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
            await run_owned_thread_call(self._ensure_startup_cleanup)
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
        task = self._prepare_task
        preparing = task is not None and not task.done()
        self.cancel()
        if task is not None and task is not asyncio.current_task():
            self._retain_settled_results(await asyncio.gather(task, return_exceptions=True))
        if linger and not preparing and not self._closed:
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
            await run_owned_thread_call(self._stop_residue_cleanup)
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
            self._retain_settled_results(await asyncio.gather(*tasks, return_exceptions=True))
        await self._await_cancelled_linger(lingering)
        failures: list[BaseException] = []
        try:
            await self._task_scope.close()
        except BaseException as error:
            failures.append(error)
            if not isinstance(error, asyncio.CancelledError):
                self._record_failure(error)
        async with self._lock:
            try:
                await run_owned_thread_call(self._stop_residue_cleanup)
                await self._runtime.close()
            except BaseException as error:
                failures.append(error)
                if not isinstance(error, asyncio.CancelledError):
                    self._record_failure(error)
            else:
                self._publish("closed", backend=None, progress_percent=None)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Managed Gemma owner shutdown failed", failures)

    def _cancel_linger(self) -> asyncio.Task[None] | None:
        task = self._linger_task
        self._linger_task = None
        if task is not None and not task.done():
            task.cancel()
        return task

    async def _await_cancelled_linger(self, task: asyncio.Task[None] | None) -> None:
        if task is None or task is asyncio.current_task():
            return
        self._retain_settled_results(await asyncio.gather(task, return_exceptions=True))

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
        failure_diagnostics: ErrorDiagnostics | None = None,
        recovery_diagnostics: tuple[ErrorDiagnostics, ...] = (),
        secondary_diagnostics: tuple[ErrorDiagnostics, ...] = (),
        recovery_diagnostics_truncated: bool = False,
        secondary_diagnostics_truncated: bool = False,
    ) -> None:
        snapshot = ManagedGemmaTranslationSnapshot(
            state=state,
            backend=backend,
            progress_percent=progress_percent,
            failure_diagnostics=failure_diagnostics,
            recovery_diagnostics=recovery_diagnostics,
            secondary_diagnostics=secondary_diagnostics,
            recovery_diagnostics_truncated=recovery_diagnostics_truncated,
            secondary_diagnostics_truncated=secondary_diagnostics_truncated,
            model_id=self._selected_model_id if backend is not None else None,
        )
        self._snapshot = snapshot
        if self._status_sink is not None:
            try:
                self._status_sink(self.snapshot)
            except Exception as error:
                diagnostics = model_asset_failure_diagnostics(
                    error,
                    operation="state_delivery",
                    code="model_assets.state_delivery_failed",
                )
                self._diagnostics_truncated |= len(self._diagnostics) == self._diagnostics.maxlen
                self._diagnostics.append(diagnostics)


__all__ = [
    "ManagedGemmaActivation",
    "ManagedGemmaLifecycleDiagnosticSink",
    "ManagedGemmaTranslationOwner",
    "ManagedGemmaTranslationSnapshot",
    "ManagedGemmaTranslationStatusSink",
]
