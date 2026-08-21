from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.core.lifecycle import LifecycleScope, start_lifecycle_task
from puripuly_heart.core.local_asr.local_stt_download_port import HuggingFaceDownloadPort
from puripuly_heart.core.local_translation.provisioning import (
    GemmaProvisioningCancelled,
    GemmaProvisioningUpdate,
)
from puripuly_heart.core.local_translation.runtime import (
    ManagedGemmaReadiness,
    ManagedGemmaRuntimeOwner,
)
from puripuly_heart.core.observability import DiagnosticEvent


@dataclass(frozen=True, slots=True)
class ManagedGemmaTranslationSnapshot:
    state: str
    backend: str | None
    progress_percent: int | None
    error_type: str | None = None


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
        self._linger_task: asyncio.Task[None] | None = None
        self._active_generation: int | None = None
        self._closed = False

    @property
    def runtime(self) -> ManagedGemmaRuntimeOwner:
        return self._runtime

    @property
    def snapshot(self) -> ManagedGemmaTranslationSnapshot:
        return self._snapshot

    async def prepare(
        self,
        selection: ManagedGemmaTranslationSelection,
    ) -> ManagedGemmaActivation:
        backend = selection.backend
        lingering = self._cancel_linger()
        await self._await_cancelled_linger(lingering)
        async with self._lock:
            if self._closed:
                raise RuntimeError("managed Gemma translation owner is closed")
            self._generation += 1
            generation = self._generation
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
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

            task = start_lifecycle_task(
                self._task_scope,
                self._runtime.prepare(
                    backend=backend,
                    source_language=selection.source_language,
                    target_language=selection.target_language,
                    system_prompt=selection.system_prompt,
                    provision_kwargs={
                        "downloader": self._downloader,
                        "cancel_event": cancel_event,
                        "on_status": on_status,
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
            except Exception as exc:
                self._publish(
                    "failed",
                    backend=backend,
                    progress_percent=None,
                    error_type=type(exc).__name__,
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
        if self._closed and self._prepare_task is None:
            await self._await_cancelled_linger(lingering)
            await self._runtime.close()
            return
        self._closed = True
        self._generation += 1
        self._active_generation = None
        self.cancel()
        task = self._prepare_task
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)
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
    ) -> None:
        snapshot = ManagedGemmaTranslationSnapshot(
            state=state,
            backend=backend,
            progress_percent=progress_percent,
            error_type=error_type,
        )
        self._snapshot = snapshot
        if self._status_sink is not None:
            with contextlib.suppress(Exception):
                self._status_sink(snapshot)


__all__ = [
    "ManagedGemmaActivation",
    "ManagedGemmaLifecycleDiagnosticSink",
    "ManagedGemmaTranslationOwner",
    "ManagedGemmaTranslationSnapshot",
    "ManagedGemmaTranslationStatusSink",
]
