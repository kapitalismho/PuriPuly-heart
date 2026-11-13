from __future__ import annotations

import asyncio
import inspect
import logging
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import flet as ft

from puripuly_heart.app.services.application_shutdown import (
    ApplicationIntentRejectedError,
    ApplicationShutdownCallback,
    ApplicationShutdownCoordinator,
    application_shutdown_callback,
)
from puripuly_heart.core.diagnostics import emit_failure, exception_diagnostics
from puripuly_heart.core.messages import ErrorDiagnostics
from puripuly_heart.ui.foundation.adapter import FletFoundationAdapter
from puripuly_heart.ui.foundation.resources import (
    DEFAULT_FOUNDATION_RESOURCES,
    FoundationResourceLocator,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FoundationRuntimeSnapshot:
    lifecycle_bound: bool
    accepting_tasks: bool
    tracked_task_count: int
    close_completed: bool


class FletFoundationRuntime:
    owner_name = "FletFoundationRuntime"

    def __init__(
        self,
        page: ft.Page,
        adapter: FletFoundationAdapter,
        *,
        resources: FoundationResourceLocator = DEFAULT_FOUNDATION_RESOURCES,
    ) -> None:
        self._page = page
        self._adapter = adapter
        self._resources = resources
        self._application_lifecycle: ApplicationShutdownCoordinator | None = None
        self._tracked_tasks: set[object] = set()
        self._application_shutdown_tasks: set[object] = set()
        self._close_completed = False
        self.failure_diagnostics: ErrorDiagnostics | None = None

    @property
    def snapshot(self) -> FoundationRuntimeSnapshot:
        lifecycle = self._application_lifecycle
        return FoundationRuntimeSnapshot(
            lifecycle_bound=lifecycle is not None,
            accepting_tasks=(
                lifecycle is not None and lifecycle.accepting_intents and not self._close_completed
            ),
            tracked_task_count=len(self._tracked_tasks),
            close_completed=self._close_completed,
        )

    @property
    def adapter(self) -> FletFoundationAdapter:
        return self._adapter

    @property
    def resources(self) -> FoundationResourceLocator:
        return self._resources

    def application_shutdown_callbacks(self) -> tuple[ApplicationShutdownCallback, ...]:
        return (
            application_shutdown_callback(
                phase="stop_external_producers",
                owner_name=self.owner_name,
                callback_name="cancel_page_tasks",
                callback=self.close,
            ),
        )

    def bind_application_lifecycle(
        self,
        lifecycle: ApplicationShutdownCoordinator,
    ) -> None:
        current = self._application_lifecycle
        if current is not None and current is not lifecycle:
            raise RuntimeError("Flet foundation runtime is already bound to another lifecycle")
        self._application_lifecycle = lifecycle

    def run_page_task(
        self,
        coroutine: Any,
        *args: object,
        intent_name: str | None = None,
    ) -> object | None:
        lifecycle = self._application_lifecycle
        if lifecycle is None:
            raise RuntimeError("Flet foundation runtime lifecycle is not bound")
        try:
            lifecycle.admit_intent(intent_name or getattr(coroutine, "__name__", "page_task"))
        except ApplicationIntentRejectedError:
            close = getattr(coroutine, "close", None)
            if callable(close):
                close()
            return None
        task = self._page.run_task(coroutine, *args)
        if task is None:
            return None
        self._tracked_tasks.add(task)
        add_done_callback = getattr(task, "add_done_callback", None)
        if callable(add_done_callback):
            add_done_callback(lambda done: self._on_task_done(done, "page_task"))
        return task

    def run_application_shutdown_task(
        self,
        coroutine: Any,
        *args: object,
    ) -> object | None:
        if self._application_lifecycle is None:
            raise RuntimeError("Flet foundation runtime lifecycle is not bound")
        task = self._page.run_task(coroutine, *args)
        if task is None:
            return None
        self._application_shutdown_tasks.add(task)
        add_done_callback = getattr(task, "add_done_callback", None)
        if callable(add_done_callback):
            add_done_callback(lambda done: self._on_task_done(done, "shutdown_task"))
        return task

    async def close(self) -> None:
        if self._close_completed:
            return
        current_task = asyncio.current_task()
        tasks = tuple(
            task
            for task in self._tracked_tasks
            if task is not current_task and not getattr(task, "done", lambda: True)()
        )
        for task in tasks:
            cancel = getattr(task, "cancel", None)
            if callable(cancel):
                cancel()
        awaitables = tuple(
            asyncio.wrap_future(task) if isinstance(task, Future) else task
            for task in tasks
            if isinstance(task, Future) or inspect.isawaitable(task)
        )
        if awaitables:
            await asyncio.gather(*awaitables, return_exceptions=True)
        for task in tuple(self._tracked_tasks):
            if getattr(task, "done", lambda: False)():
                self._on_task_done(task, "page_task")
        self._close_completed = True

    def _on_task_done(self, task: object, operation: str) -> None:
        tasks = (
            self._application_shutdown_tasks
            if operation == "shutdown_task"
            else self._tracked_tasks
        )
        if task not in tasks:
            return
        tasks.discard(task)
        if getattr(task, "cancelled", lambda: False)():
            return
        exception = getattr(task, "exception", lambda: None)()
        if exception is None:
            return
        diagnostics = exception_diagnostics(
            exception, component="ui.foundation", operation=operation, code="ui_task_failed"
        )
        self.failure_diagnostics = diagnostics
        emit_failure(logger, diagnostics)


__all__ = ["FletFoundationRuntime", "FoundationRuntimeSnapshot"]
