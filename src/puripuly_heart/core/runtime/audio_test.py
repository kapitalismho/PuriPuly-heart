from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from puripuly_heart.core.owned_thread import run_owned_thread_call

_TaskResultT = TypeVar("_TaskResultT")
AudioTestSessionRunner = Callable[[int], Awaitable[None]]
AudioTestCaptureRuntimeStateChanged = Callable[["AudioTestCaptureRuntime"], None]


class AudioTestCaptureRuntime:
    """Owns one audio-test channel's capture source and frame/session tasks."""

    resource_fields = (
        "_session_task",
        "_source",
        "_pending_frame_task",
        "_generation",
    )
    stop_ingress = "stop capture and reject new tests"
    shutdown_policy = "cancel/gather test task, cancel frame task, close source"
    late_callback_rule = "late result cannot update disposed UI snapshot"

    def __init__(
        self,
        *,
        cancel_timeout_s: float = 2.0,
        state_changed: AudioTestCaptureRuntimeStateChanged | None = None,
    ) -> None:
        self._cancel_timeout_s = max(0.0, float(cancel_timeout_s))
        self._state_changed = state_changed
        self._session_task: asyncio.Task[None] | None = None
        self._source: object | None = None
        self._pending_frame_task: asyncio.Task[Any] | None = None
        self._generation = 0
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._source_close_lock = asyncio.Lock()

    @property
    def owner_name(self) -> str:
        return "AudioTestCaptureRuntime"

    @property
    def session_task(self) -> asyncio.Task[None] | None:
        return self._session_task

    @property
    def source(self) -> object | None:
        return self._source

    @property
    def pending_frame_task(self) -> asyncio.Task[Any] | None:
        return self._pending_frame_task

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def is_closing(self) -> bool:
        return self._closing

    @property
    def has_resources(self) -> bool:
        return self._has_active_capture_resources()

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    def start(self, run_session: AudioTestSessionRunner) -> asyncio.Task[None]:
        if self._closing or self._closed:
            state = "closing" if self._closing else "closed"
            raise RuntimeError(f"{self.owner_name} is {state} to new capture sessions")
        if self._has_active_capture_resources():
            raise RuntimeError(f"{self.owner_name} already owns active capture resources")
        if self._session_task is not None:
            if not self._session_task.done():
                raise RuntimeError(f"{self.owner_name} already owns an active capture session")
            self._observe_task_exception(self._session_task)
            self._session_task = None

        self._generation += 1
        generation = self._generation
        task = asyncio.create_task(
            self._run_session_guarded(run_session=run_session, generation=generation),
            name=f"{self.owner_name}:session",
        )
        self._session_task = task
        task.add_done_callback(self._on_session_task_done)
        self._notify_state_changed()
        return task

    def is_current_generation(self, generation: int) -> bool:
        return not self._closing and not self._closed and generation == self._generation

    def retain_source(self, source: object) -> None:
        if self._source is not None and self._source is not source:
            raise RuntimeError(f"{self.owner_name} already owns a source")
        self._source = source
        self._notify_state_changed()

    def attach_source(self, source: object, *, generation: int) -> bool:
        if not self.is_current_generation(generation):
            return False
        self._source = source
        self._notify_state_changed()
        return True

    async def close_source(self, source: object | None) -> None:
        async with self._source_close_lock:
            if source is None or self._source is not source:
                return
            frame_task = self._pending_frame_task
            if frame_task is not None and not frame_task.done():
                return
            await _call_close(source)
            if self._source is source:
                self._source = None
                self._notify_state_changed()

    def create_frame_task(
        self,
        coroutine: Coroutine[Any, Any, _TaskResultT],
        *,
        generation: int,
    ) -> asyncio.Task[_TaskResultT]:
        if not self.is_current_generation(generation):
            coroutine.close()
            raise RuntimeError(f"{self.owner_name} generation is stale to new frame tasks")
        task = asyncio.create_task(coroutine, name=f"{self.owner_name}:frame")
        self._pending_frame_task = task
        task.add_done_callback(self._on_frame_task_done)
        self._notify_state_changed()
        return task

    async def cancel_frame_task(self, task: asyncio.Task[Any] | None = None) -> None:
        frame_task = task or self._pending_frame_task
        if frame_task is None:
            return
        await self._cancel_task_bounded(frame_task)

    async def stop(self) -> None:
        self._generation += 1
        task = self._session_task
        frame_task = self._pending_frame_task
        self._notify_state_changed()
        await self._cancel_task_bounded(frame_task)
        await self._cancel_task_bounded(task)
        if any(item is not None and not item.done() for item in (frame_task, task)):
            return
        await self.close_source(self._source)

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
                await self.stop()
            finally:
                self._closing = False
                self._notify_state_changed()

    async def _run_session_guarded(
        self,
        *,
        run_session: AudioTestSessionRunner,
        generation: int,
    ) -> None:
        if self.is_current_generation(generation):
            await run_session(generation)

    def _has_resources(self) -> bool:
        return (
            self._session_task is not None
            or self._pending_frame_task is not None
            or self._source is not None
        )

    def _has_active_capture_resources(self) -> bool:
        return (
            self._source is not None
            or (self._pending_frame_task is not None and not self._pending_frame_task.done())
            or (self._session_task is not None and not self._session_task.done())
        )

    async def _cancel_task_bounded(self, task: asyncio.Task[Any] | None) -> None:
        if task is None or task is asyncio.current_task():
            return
        if not task.done() and not task.cancelling():
            task.cancel()
        done, _pending = await asyncio.wait({task}, timeout=self._cancel_timeout_s)
        for completed in done:
            self._observe_task_exception(completed)

    def _on_session_task_done(self, task: asyncio.Task[None]) -> None:
        self._observe_task_exception(task)
        if self._session_task is task:
            self._session_task = None
            self._generation += 1
            self._notify_state_changed()

    def _on_frame_task_done(self, task: asyncio.Task[Any]) -> None:
        self._observe_task_exception(task)
        if self._pending_frame_task is task:
            self._pending_frame_task = None
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


async def _call_close(resource: object) -> None:
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    if inspect.iscoroutinefunction(close):
        await close()
    else:
        result = await run_owned_thread_call(close)
        if inspect.isawaitable(result):
            await result


__all__ = ["AudioTestCaptureRuntime"]
