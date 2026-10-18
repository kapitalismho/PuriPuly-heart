from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

from puripuly_heart.core.openrouter_pkce import OpenRouterPKCEExchangeResult

_TaskResultT = TypeVar("_TaskResultT")
ExternalTaskRunner = Callable[[Callable[[], Awaitable[Any]]], object]


@dataclass(slots=True)
class _ExternalAuthTask:
    handle: object | None = None
    task: asyncio.Task[Any] | None = None
    cancel_requested: bool = False


class OAuthRuntime:
    """Owns OAuth loopback listeners and auth-related background tasks."""

    resource_fields = (
        "_openrouter_pkce_client",
        "_auth_tasks",
        "_external_task_handles",
        "_loopback_listeners",
    )
    stop_ingress = "cancel auth task / close listener"
    shutdown_policy = "listener close unblocks wait; browser side effects are not retried"
    late_callback_rule = (
        "second close is a no-op; late callback maps to cancelled or expired result"
    )

    def __init__(self, *, auth_task_timeout_s: float = 2.0) -> None:
        self._auth_task_timeout_s = max(0.0, float(auth_task_timeout_s))
        self._auth_tasks: dict[str, asyncio.Task[Any]] = {}
        self._task_names: dict[asyncio.Task[Any], str] = {}
        self._external_task_handles: dict[str, _ExternalAuthTask] = {}
        self._loopback_listeners: dict[str, object] = {}
        self._openrouter_pkce_client: object | None = None
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def owner_name(self) -> str:
        return "OAuthRuntime"

    @property
    def active_task_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._auth_tasks))

    @property
    def external_task_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._external_task_handles))

    @property
    def is_closing(self) -> bool:
        return self._closing

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def active_listener_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._loopback_listeners))

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": self.stop_ingress,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
        }

    def create_auth_task(
        self,
        coroutine: Coroutine[Any, Any, _TaskResultT],
        *,
        task_name: str,
    ) -> asyncio.Task[_TaskResultT]:
        if self._closing or self._closed:
            coroutine.close()
            state = "closing" if self._closing else "closed"
            raise RuntimeError(f"OAuthRuntime is {state} to new auth tasks")
        existing = self._auth_tasks.get(task_name)
        if existing is not None and not existing.done():
            coroutine.close()
            raise RuntimeError(f"OAuthRuntime already owns auth task {task_name!r}")

        task = asyncio.create_task(coroutine, name=f"{self.owner_name}:{task_name}")
        self._auth_tasks[task_name] = task
        self._task_names[task] = task_name
        task.add_done_callback(self._on_auth_task_done)
        return task

    def attach_loopback_listener(self, listener: object, *, listener_name: str) -> None:
        if self._closing or self._closed:
            raise RuntimeError("OAuthRuntime is closed to new loopback listeners")
        self._loopback_listeners[listener_name] = listener

    async def close_loopback_listener(self, listener: object, *, listener_name: str) -> None:
        current = self._loopback_listeners.get(listener_name)
        await _close_listener(listener)
        if current is listener:
            self._loopback_listeners.pop(listener_name, None)

    def detach_loopback_listener(self, listener: object, *, listener_name: str) -> None:
        current = self._loopback_listeners.get(listener_name)
        if current is listener:
            self._loopback_listeners.pop(listener_name, None)

    async def run_openrouter_pkce_flow(
        self,
        client: object,
    ) -> OpenRouterPKCEExchangeResult:
        run_desktop_flow = getattr(client, "run_desktop_flow")
        task = self.create_auth_task(
            run_desktop_flow(),
            task_name="openrouter-pkce",
        )
        self._openrouter_pkce_client = client
        try:
            return await task
        finally:
            if self._openrouter_pkce_client is client:
                self._openrouter_pkce_client = None

    def reopen_openrouter_pkce_authorization_url(self) -> bool:
        client = self._openrouter_pkce_client
        if client is None:
            return False
        reopen = getattr(client, "reopen_authorization_url", None)
        if not callable(reopen):
            return False
        return bool(reopen())

    def start_external_task(
        self,
        *,
        task_runner: ExternalTaskRunner,
        task_factory: Callable[[], Awaitable[Any]],
        task_name: str,
        generation: int | None = None,
    ) -> object:
        _ = generation
        if self._closing or self._closed:
            raise RuntimeError("OAuthRuntime is closed to new external auth tasks")
        if task_name in self._external_task_handles:
            raise RuntimeError(f"OAuthRuntime already owns external task {task_name!r}")
        owned = _ExternalAuthTask()
        self._external_task_handles[task_name] = owned

        async def run_owned() -> Any:
            if self._external_task_handles.get(task_name) is not owned:
                return None
            owned.task = asyncio.current_task()
            try:
                if not owned.cancel_requested:
                    return await task_factory()
                return None
            finally:
                if self._external_task_handles.get(task_name) is owned:
                    self._external_task_handles.pop(task_name, None)

        try:
            handle = task_runner(run_owned)
            if handle is None:
                raise RuntimeError("OAuthRuntime external task was not admitted")
        except BaseException:
            if self._external_task_handles.get(task_name) is owned:
                self._external_task_handles.pop(task_name, None)
            raise
        owned.handle = handle
        add_done_callback = getattr(handle, "add_done_callback", None)
        if callable(add_done_callback):
            add_done_callback(lambda finished: self._clear_external_task(task_name, finished))
        return handle

    def cancel_external_task(
        self,
        handle: object | None = None,
        *,
        task_name: str | None = None,
    ) -> None:
        owned = self._external_task_handles.get(task_name) if task_name is not None else None
        if owned is None:
            owned = next(
                (item for item in self._external_task_handles.values() if item.handle is handle),
                None,
            )
        if owned is None or owned.cancel_requested:
            return
        owned.cancel_requested = True
        if owned.task is not None:
            owned.task.get_loop().call_soon_threadsafe(owned.task.cancel)
        else:
            cancel = getattr(owned.handle, "cancel", None)
            if callable(cancel):
                cancel()

    def clear_external_task(self, task_name: str, handle: object | None = None) -> None:
        owned = self._external_task_handles.get(task_name)
        if owned is not None:
            self._clear_external_task(task_name, owned.handle if handle is None else handle)

    async def close(self) -> None:
        if self._closed and not self._has_resources():
            return

        async with self._close_lock:
            if self._closed and not self._has_resources():
                return
            self._closing = True
            self._closed = True
            failures: list[Exception] = []
            try:
                failures.extend(await self._close_loopback_listeners())
                failures.extend(await self._cancel_external_tasks())
                failures.extend(await self._cancel_and_gather_auth_tasks())
                self._openrouter_pkce_client = None
            finally:
                self._closing = False
            _raise_cleanup_failures("OAuthRuntime close failed", failures)

    def _has_resources(self) -> bool:
        return bool(
            self._auth_tasks
            or self._external_task_handles
            or self._loopback_listeners
            or self._openrouter_pkce_client is not None
        )

    async def _close_loopback_listeners(self) -> list[Exception]:
        failures: list[Exception] = []
        for listener_name, listener in tuple(self._loopback_listeners.items()):
            try:
                await _close_listener(listener)
            except Exception as exc:
                failures.append(exc)
            else:
                if self._loopback_listeners.get(listener_name) is listener:
                    self._loopback_listeners.pop(listener_name, None)
        return failures

    async def _cancel_external_tasks(self) -> list[Exception]:
        failures: list[Exception] = []
        for task_name, owned in tuple(self._external_task_handles.items()):
            try:
                self.cancel_external_task(owned.handle, task_name=task_name)
            except Exception as exc:
                failures.append(exc)
        await asyncio.sleep(0)
        current_task = asyncio.current_task()
        tasks = {
            owned.task
            for owned in self._external_task_handles.values()
            if owned.task is not None and owned.task is not current_task
        }
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=self._auth_task_timeout_s)
            for task in done:
                _observe_task_exception(task)
            if pending:
                failures.append(RuntimeError("OAuthRuntime external auth tasks did not settle"))
        for task_name, owned in tuple(self._external_task_handles.items()):
            self._clear_external_task(task_name, owned.handle)
            if (
                self._external_task_handles.get(task_name) is owned
                and owned.task is not current_task
                and owned.task not in tasks
            ):
                failures.append(RuntimeError("OAuthRuntime external task settlement is unobserved"))
        return failures

    async def _cancel_and_gather_auth_tasks(self) -> list[Exception]:
        failures: list[Exception] = []
        current_task = asyncio.current_task()
        tasks = tuple(task for task in self._task_names if task is not current_task)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=self._auth_task_timeout_s)
            for task in done:
                _observe_task_exception(task)
            if pending:
                for task in pending:
                    task.cancel()
                done_after_cancel, _still_pending = await asyncio.wait(
                    pending,
                    timeout=self._auth_task_timeout_s,
                )
                for task in done_after_cancel:
                    _observe_task_exception(task)
        if any(not task.done() for task in tasks):
            failures.append(RuntimeError("OAuthRuntime auth tasks did not settle"))
        return failures

    def _on_auth_task_done(self, task: asyncio.Task[Any]) -> None:
        task_name = self._task_names.pop(task, None)
        if task_name is not None and self._auth_tasks.get(task_name) is task:
            self._auth_tasks.pop(task_name, None)
        _observe_task_exception(task)

    def _clear_external_task(self, task_name: str, handle: object) -> None:
        owned = self._external_task_handles.get(task_name)
        if owned is None or owned.handle is not handle:
            return
        if owned.task is not None:
            settled = owned.task.done()
        else:
            done = getattr(handle, "done", None)
            settled = callable(done) and bool(done())
        if settled:
            self._external_task_handles.pop(task_name, None)


async def _close_listener(listener: object) -> None:
    close = getattr(listener, "close", None)
    if not callable(close):
        return
    if inspect.iscoroutinefunction(close):
        result = close()
    else:
        result = await asyncio.to_thread(close)
    if inspect.isawaitable(result):
        await result


def _observe_task_exception(task: asyncio.Task[Any]) -> Exception | None:
    if task.cancelled():
        return None
    try:
        exception = task.exception()
    except asyncio.CancelledError:
        return None
    if exception is not None:
        return exception
    return None


def _raise_cleanup_failures(message: str, failures: list[Exception]) -> None:
    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]
    raise ExceptionGroup(message, failures)


__all__ = ["OAuthRuntime"]
