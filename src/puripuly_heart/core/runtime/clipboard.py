from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextvars import copy_context
from dataclasses import replace
from typing import Any

from puripuly_heart.core.diagnostics import emit_failure, exception_diagnostics
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.core.owned_thread import wait_for_owned_future

logger = logging.getLogger(__name__)

ClipboardWatcherFactory = Callable[[Callable[[str], None]], object]
ClipboardSubmitHandler = Callable[[str], Awaitable[None]]
ClipboardRuntimeStateChanged = Callable[["ClipboardRuntime"], None]


class ClipboardRuntime:
    """Owns clipboard watcher ingress and clipboard submit tasks."""

    resource_fields = (
        "_watcher",
        "_retiring_watcher",
        "_loop",
        "_generation",
        "_submit_tasks",
    )
    stop_ingress = "stop watcher before runtime shutdown"
    shutdown_policy = "join watcher; gather submit tasks with bounded timeout"
    late_callback_rule = "generation guard rejects late clipboard callbacks"

    def __init__(
        self,
        *,
        watcher_factory: ClipboardWatcherFactory,
        submit_handler: ClipboardSubmitHandler,
        submit_gather_timeout_s: float = 2.0,
        state_changed: ClipboardRuntimeStateChanged | None = None,
        failure_sink: Callable[[ErrorDiagnostics], None] | None = None,
    ) -> None:
        self._watcher_factory = watcher_factory
        self._submit_handler = submit_handler
        self._submit_gather_timeout_s = max(0.0, float(submit_gather_timeout_s))
        self._state_changed = state_changed
        self._failure_sink = failure_sink
        self.failure_diagnostics: ErrorDiagnostics | None = None
        self._watcher: object | None = None
        self._retiring_watcher: object | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._generation = 0
        self._submit_tasks: set[asyncio.Task[Any]] = set()
        self._closing = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._submit_gather_timeouts = 0

    @property
    def owner_name(self) -> str:
        return "ClipboardRuntime"

    @property
    def watcher(self) -> object | None:
        return self._watcher

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def active_submit_task_count(self) -> int:
        return len(self._submit_tasks)

    @property
    def submit_gather_timeouts(self) -> int:
        return self._submit_gather_timeouts

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    async def sync(
        self, *, enabled: bool, strict_runtime_errors: bool = False
    ) -> ErrorDiagnostics | None:
        if not enabled:
            return await self.stop(strict_runtime_errors=strict_runtime_errors)
        return await self.start(strict_runtime_errors=strict_runtime_errors)

    async def start(self, *, strict_runtime_errors: bool = False) -> ErrorDiagnostics | None:
        async with self._lock:
            if self._watcher is not None:
                return
            if self._closing or self._closed:
                if strict_runtime_errors:
                    raise RuntimeError("ClipboardRuntime is closed to new watcher work")
                return
            if self._retiring_watcher is not None:
                try:
                    await self._stop_watcher(self._retiring_watcher)
                except Exception as exc:
                    diagnostics = self.record_failure(exc, "watcher_stop")
                    if strict_runtime_errors:
                        raise
                    return diagnostics
            self._generation += 1
            generation = self._generation
            self._loop = asyncio.get_running_loop()
            try:
                watcher = self._watcher_factory(
                    lambda text, *, _generation=generation: self.on_text_from_thread(
                        text,
                        generation=_generation,
                    )
                )
                set_failure_sink = getattr(watcher, "set_failure_sink", None)
                if callable(set_failure_sink):
                    loop = asyncio.get_running_loop()
                    set_failure_sink(
                        lambda diagnostics: loop.call_soon_threadsafe(
                            self._accept_failure, diagnostics
                        )
                    )
            except Exception as exc:
                self._loop = None
                diagnostics = self.record_failure(exc, "watcher_start")
                if strict_runtime_errors:
                    raise
                return diagnostics
            try:
                starting = asyncio.get_running_loop().run_in_executor(
                    None, copy_context().run, watcher.start
                )
                await wait_for_owned_future(starting, on_cancel=self._revoke_ingress)
            except asyncio.CancelledError:
                if not starting.cancelled() and isinstance(
                    error := starting.exception(), Exception
                ):
                    self.record_failure(error, "watcher_start")
                try:
                    await self._stop_watcher(watcher)
                except Exception as exc:
                    self.record_failure(exc, "watcher_stop")
                raise
            except Exception as exc:
                diagnostics = exception_diagnostics(
                    exc, component="clipboard", operation="watcher_start", code="clipboard_failed"
                )
                self._revoke_ingress()
                self._notify_state_changed()
                try:
                    await self._stop_watcher(watcher)
                except Exception as cleanup_exc:
                    cleanup = exception_diagnostics(
                        cleanup_exc,
                        component="clipboard",
                        operation="watcher_stop",
                        code="clipboard_failed",
                    )
                    cleanup = replace(cleanup, cause_id=diagnostics.diagnostic_id)
                    cleanup_exc.failure_diagnostics = cleanup
                    diagnostics = replace(diagnostics, causes=(*diagnostics.causes, cleanup))
                self._retain_failure(exc, diagnostics)
                if strict_runtime_errors:
                    raise
                return diagnostics
            if generation != self._generation or self._closing or self._closed:
                await self._stop_watcher(watcher)
                return
            self._watcher = watcher
            self._notify_state_changed()

    async def _stop_watcher(self, watcher: object) -> None:
        self._retiring_watcher = watcher
        stopping = asyncio.get_running_loop().run_in_executor(
            None, copy_context().run, watcher.stop
        )
        try:
            await wait_for_owned_future(stopping)
        except asyncio.CancelledError:
            if not stopping.cancelled():
                error = stopping.exception()
                if error is None:
                    self._retiring_watcher = None
                elif isinstance(error, Exception):
                    self.record_failure(error, "watcher_stop")
            raise
        self._retiring_watcher = None

    async def stop(self, *, strict_runtime_errors: bool = False) -> ErrorDiagnostics | None:
        stop_failure: Exception | None = None
        stop_diagnostics: ErrorDiagnostics | None = None
        async with self._lock:
            self._revoke_ingress()
            watcher = self._watcher if self._watcher is not None else self._retiring_watcher
            self._watcher = None
            self._notify_state_changed()
            if watcher is not None:
                try:
                    await self._stop_watcher(watcher)
                except Exception as exc:
                    stop_failure = exc
                    stop_diagnostics = exception_diagnostics(
                        exc,
                        component="clipboard",
                        operation="watcher_stop",
                        code="clipboard_failed",
                    )
        try:
            drain_diagnostics = await self._gather_submit_tasks_bounded()
        except asyncio.CancelledError:
            if stop_failure is not None and stop_diagnostics is not None:
                self._retain_failure(stop_failure, stop_diagnostics)
            raise
        if stop_failure is not None and stop_diagnostics is not None:
            if drain_diagnostics is not None:
                stop_diagnostics = replace(
                    stop_diagnostics, causes=(*stop_diagnostics.causes, drain_diagnostics)
                )
            self._retain_failure(stop_failure, stop_diagnostics)
        self._notify_state_changed()
        if stop_failure is not None and strict_runtime_errors:
            raise stop_failure
        return stop_diagnostics or drain_diagnostics

    async def close(self) -> ErrorDiagnostics | None:
        if (
            self._closed
            and self._watcher is None
            and self._retiring_watcher is None
            and not self._submit_tasks
        ):
            return
        self._closing = True
        try:
            return await self.stop(strict_runtime_errors=True)
        finally:
            self._closed = True
            self._closing = False

    def on_text_from_thread(self, text: str, *, generation: int | None = None) -> None:
        trimmed = text.strip()
        if not trimmed or len(trimmed) > 300:
            return
        expected_generation = self._generation if generation is None else generation
        if not self._is_current_generation(expected_generation):
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(
            self.submit_from_loop,
            trimmed,
            expected_generation,
        )

    def submit_from_loop(self, text: str, generation: int | None = None) -> None:
        expected_generation = self._generation if generation is None else generation
        if not self._is_current_generation(expected_generation):
            return
        coroutine = self._run_submit(text, expected_generation)
        try:
            task = asyncio.create_task(coroutine, name="ClipboardRuntime:submit")
        except Exception:
            coroutine.close()
            raise
        self._submit_tasks.add(task)
        task.add_done_callback(self._on_submit_done)
        self._notify_state_changed()

    def _revoke_ingress(self) -> None:
        self._generation += 1
        self._loop = None

    def _is_current_generation(self, generation: int) -> bool:
        return not self._closing and not self._closed and generation == self._generation

    async def _run_submit(self, text: str, generation: int) -> None:
        if not self._is_current_generation(generation):
            return
        await self._submit_handler(text)

    async def _gather_submit_tasks_bounded(self) -> ErrorDiagnostics | None:
        current_task = asyncio.current_task()
        tasks = tuple(task for task in self._submit_tasks if task is not current_task)
        if not tasks:
            return

        done, pending = await asyncio.wait(tasks, timeout=self._submit_gather_timeout_s)
        for task in done:
            self._on_submit_done(task)
        if pending:
            self._submit_gather_timeouts += 1
            for task in pending:
                task.cancel()
            done_after_cancel, still_pending = await asyncio.wait(
                pending,
                timeout=self._submit_gather_timeout_s,
            )
            for task in done_after_cancel:
                self._on_submit_done(task)
            return self.record_failure(
                TimeoutError(),
                "submit_drain",
                code="clipboard_submit_timeout",
                cleanup_confirmed=not still_pending,
            )

    def _on_submit_done(self, task: asyncio.Task[Any]) -> None:
        if task not in self._submit_tasks:
            return
        self._submit_tasks.discard(task)
        self._observe_submit_task_exception(task)
        self._notify_state_changed()

    def _observe_submit_task_exception(self, task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            exception = task.exception()
            if exception is not None:
                self.record_failure(exception, "clipboard_submit")

    def record_failure(
        self,
        exception: BaseException,
        operation: str,
        *,
        code: str = "clipboard_failed",
        cleanup_confirmed: bool | None = None,
    ) -> ErrorDiagnostics:
        diagnostics = exception_diagnostics(
            exception,
            component="clipboard",
            operation=operation,
            code=code,
            fields=(
                {"timed_out": True, "cleanup_confirmed": cleanup_confirmed}
                if cleanup_confirmed is not None
                else {}
            ),
        )
        return self._retain_failure(exception, diagnostics)

    def _retain_failure(
        self, exception: BaseException, diagnostics: ErrorDiagnostics
    ) -> ErrorDiagnostics:
        exception.failure_diagnostics = diagnostics
        emit_failure(logger, diagnostics)
        self._accept_failure(diagnostics)
        return diagnostics

    def _accept_failure(self, diagnostics: ErrorDiagnostics) -> None:
        self.failure_diagnostics = diagnostics
        if self._failure_sink is not None:
            try:
                self._failure_sink(diagnostics)
            except Exception:
                pass

    def _notify_state_changed(self) -> None:
        if self._state_changed is not None:
            try:
                self._state_changed(self)
            except Exception as exc:
                self.record_failure(exc, "clipboard_callback")


__all__ = ["ClipboardRuntime"]
