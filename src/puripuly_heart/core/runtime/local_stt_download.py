from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from puripuly_heart.core.local_stt_runtime_installer import RuntimeLocalSTTStatusUpdate

LocalSTTDownloadRunner = Callable[[threading.Event, int], Awaitable[object]]
LocalSTTDownloadStateChanged = Callable[["LocalSTTDownloadRuntime"], None]
LocalSTTStatusHandler = Callable[[RuntimeLocalSTTStatusUpdate], Awaitable[None] | None]


class LocalSTTDownloadRuntime:
    """Owns the local STT model download/install task and progress callbacks."""

    resource_fields = (
        "_download_task",
        "_cancel_event",
        "_origin",
        "_generation",
    )
    stop_ingress = "reject new install/start commands"
    shutdown_policy = "cancel active install task and set installer cancel event"
    late_callback_rule = "late progress ignored after generation change"

    def __init__(
        self,
        *,
        cancel_timeout_s: float = 2.0,
        state_changed: LocalSTTDownloadStateChanged | None = None,
    ) -> None:
        self._cancel_timeout_s = max(0.0, float(cancel_timeout_s))
        self._state_changed = state_changed
        self._download_task: asyncio.Task[object] | None = None
        self._cancel_event: threading.Event | None = None
        self._origin: str | None = None
        self._generation = 0
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def owner_name(self) -> str:
        return "LocalSTTDownloadRuntime"

    @property
    def download_task(self) -> asyncio.Task[object] | None:
        return self._download_task

    @property
    def cancel_event(self) -> threading.Event | None:
        return self._cancel_event

    @property
    def origin(self) -> str | None:
        return self._origin

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def is_closing(self) -> bool:
        return self._closing

    @property
    def is_closed(self) -> bool:
        return self._closed

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    def start(
        self,
        *,
        origin: str,
        run_download: LocalSTTDownloadRunner,
    ) -> asyncio.Task[object]:
        if self._closing or self._closed:
            state = "closing" if self._closing else "closed"
            raise RuntimeError(f"LocalSTTDownloadRuntime is {state} to new install tasks")
        if self._download_task is not None:
            if not self._download_task.done():
                raise RuntimeError("LocalSTTDownloadRuntime already owns an active install task")
            self._observe_task_exception(self._download_task)
            self._download_task = None

        self._generation += 1
        generation = self._generation
        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        self._origin = origin
        task = asyncio.create_task(
            run_download(cancel_event, generation),
            name=f"{self.owner_name}:install",
        )
        self._download_task = task
        task.add_done_callback(self._on_download_task_done)
        self._notify_state_changed()
        return task

    def is_current_generation(self, generation: int) -> bool:
        return not self._closing and not self._closed and generation == self._generation

    async def dispatch_status_update(
        self,
        update: RuntimeLocalSTTStatusUpdate,
        *,
        generation: int,
        on_status: LocalSTTStatusHandler,
    ) -> None:
        if not self.is_current_generation(generation):
            return
        result = on_status(update)
        if inspect.isawaitable(result):
            await result

    async def cancel(self) -> None:
        self._generation += 1
        task = self._download_task
        cancel_event = self._cancel_event
        if cancel_event is not None:
            cancel_event.set()
        self._notify_state_changed()
        await self._cancel_task_bounded(task)
        self._release_download_task(task)

    async def close(self) -> None:
        if self._closed and not self._has_resources():
            return
        async with self._close_lock:
            if self._closed and not self._has_resources():
                return
            self._closing = True
            self._closed = True
            self._notify_state_changed()
            try:
                await self.cancel()
            finally:
                self._closing = False
                self._notify_state_changed()

    def _has_resources(self) -> bool:
        return self._download_task is not None or self._cancel_event is not None

    async def _cancel_task_bounded(self, task: asyncio.Task[object] | None) -> None:
        if task is None or task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        done, pending = await asyncio.wait({task}, timeout=self._cancel_timeout_s)
        for completed in done:
            self._observe_task_exception(completed)
        if pending:
            raise TimeoutError("Local STT download task cancellation timed out")

    def _on_download_task_done(self, task: asyncio.Task[object]) -> None:
        self._observe_task_exception(task)
        self._release_download_task(task)

    def _release_download_task(self, task: asyncio.Task[object] | None) -> None:
        if task is None or self._download_task is not task or not task.done():
            return
        self._download_task = None
        self._cancel_event = None
        self._origin = None
        self._generation += 1
        self._notify_state_changed()

    @staticmethod
    def _observe_task_exception(task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            try:
                task.exception()
            except asyncio.CancelledError:
                pass

    def _notify_state_changed(self) -> None:
        if self._state_changed is not None:
            self._state_changed(self)


__all__ = ["LocalSTTDownloadRuntime"]
