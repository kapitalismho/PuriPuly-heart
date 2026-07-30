from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any

ProviderEventHandler = Callable[[object], Awaitable[None]]
ProviderExceptionHandler = Callable[[Exception], Awaitable[None] | None]
ProviderStateChanged = Callable[["ProviderRuntimeHandle"], None]
logger = logging.getLogger(__name__)


class ProviderRuntimeHandle:
    """Owns one provider resource and its optional provider event loop task."""

    resource_fields = ("provider", "event_task", "idle_release_task", "generation")
    toggle_off_policy = (
        "STT toggle-off drains final transcript by awaiting provider.close() before "
        "keeping provider event ingress active for later toggle-on; configured idle release, "
        "app shutdown, and replacement use backend close."
    )
    shutdown_policy = (
        "stop ingress, cancel the provider event task, await provider.close(), then close "
        "backend-level resources when the provider exposes them"
    )
    late_callback_rule = "provider event generation rejects stale STT callbacks"

    def __init__(
        self,
        *,
        name: str,
        provider: object | None = None,
        event_handler: ProviderEventHandler | None = None,
        retired_event_handler: ProviderEventHandler | None = None,
        exception_handler: ProviderExceptionHandler | None = None,
        state_changed: ProviderStateChanged | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._name = name
        self._provider = provider
        self._event_handler = event_handler
        self._retired_event_handler = retired_event_handler
        self._exception_handler = exception_handler
        self._state_changed = state_changed
        self._sleep = sleep
        self._event_task: asyncio.Task[None] | None = None
        self._idle_release_task: asyncio.Task[None] | None = None
        self._generation = 0
        self._running = False
        self._closed = False
        self._retired_providers: list[object] = []
        self._draining_event_tasks: list[tuple[object, asyncio.Task[None]]] = []
        self._retirement_tasks: set[asyncio.Task[None]] = set()
        self._pending_handoff: tuple[object, bool, asyncio.Future[object | None]] | None = None
        self._lock = asyncio.Lock()

    @property
    def owner_name(self) -> str:
        return f"ProviderRuntimeHandle:{self._name}"

    @property
    def provider(self) -> object | None:
        return self._provider

    @property
    def event_task(self) -> asyncio.Task[None] | None:
        return self._event_task

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def has_resources(self) -> bool:
        return (
            self._provider is not None
            or self._event_task is not None
            or self._idle_release_task is not None
            or bool(self._retired_providers)
            or bool(self._draining_event_tasks)
            or bool(self._retirement_tasks)
            or self._pending_handoff is not None
        )

    def current_provider_generation(self) -> tuple[object | None, int]:
        """Capture the current provider identity and generation for an in-flight call."""

        return self._provider, self._generation

    def is_current_provider_generation(self, *, provider: object, generation: int) -> bool:
        """Return whether a captured provider/generation is still current."""

        return generation == self._generation and provider is self._provider and not self._closed

    def lifecycle_owner_snapshot(self) -> dict[str, object]:
        return {
            "owner": self.owner_name,
            "resource_fields": self.resource_fields,
            "stop_ingress": "detach from channel owner/runtime coordinator",
            "toggle_off_policy": self.toggle_off_policy,
            "shutdown_policy": self.shutdown_policy,
            "late_callback_rule": self.late_callback_rule,
            "pending_handoff": self._pending_handoff is not None,
            "retiring_provider_count": len(self._retirement_tasks),
        }

    def attach_provider_reference(self, provider: object | None) -> None:
        """Synchronize a compatibility field assignment without closing resources."""

        self._provider = provider
        self._closed = False
        self._notify_state_changed()

    async def start(self) -> None:
        async with self._lock:
            await self._cancel_idle_release_task()
            self._running = True
            self._closed = False
            self._start_event_loop_if_needed()

    async def start_if_provider(self, expected_provider: object) -> bool:
        async with self._lock:
            if self._provider is not expected_provider:
                return False
            await self._cancel_idle_release_task()
            self._running = True
            self._closed = False
            self._start_event_loop_if_needed()
            return True

    async def replace_provider(self, provider: object | None, *, start: bool) -> object | None:
        async with self._lock:
            await self._cancel_idle_release_task()
            old_provider = self._provider
            self._generation += 1
            await self._cancel_event_task()
            self._provider = provider
            self._closed = False
            self._notify_state_changed()
            if start:
                self._running = True
                self._start_event_loop_if_needed()
            if old_provider is not None and old_provider is not provider:
                try:
                    await self._close_provider_for_shutdown(old_provider)
                except Exception:
                    self._retain_retired_provider(old_provider)
                    raise
            return old_provider

    async def handoff_provider_at_boundary(
        self,
        provider: object,
        *,
        start: bool,
    ) -> object | None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object | None] = loop.create_future()
        superseded: object | None = None
        async with self._lock:
            await self._cancel_idle_release_task()
            pending = self._pending_handoff
            if pending is not None:
                superseded, _pending_start, pending_future = pending
                if not pending_future.done():
                    pending_future.set_exception(RuntimeError("provider handoff superseded"))
            self._pending_handoff = (provider, start, future)
            current = self._provider
            boundary_open = current is None or bool(
                getattr(current, "is_at_utterance_boundary", True)
            )
            if boundary_open:
                self._commit_pending_handoff_locked()
        if superseded is not None and superseded is not provider:
            self._schedule_provider_retirement(superseded, event_task=None)
        return await asyncio.shield(future)

    async def commit_pending_handoff(self) -> object | None:
        async with self._lock:
            return self._commit_pending_handoff_locked()

    async def cancel_pending_handoff(self, provider: object) -> bool:
        async with self._lock:
            pending = self._pending_handoff
            if pending is None or pending[0] is not provider:
                return False
            _pending_provider, _start, future = pending
            self._pending_handoff = None
            if not future.done():
                future.cancel()
            return True

    def _commit_pending_handoff_locked(self) -> object | None:
        pending = self._pending_handoff
        if pending is None:
            return None
        provider, start, future = pending
        old_provider = self._provider
        old_event_task = self._event_task
        self._pending_handoff = None
        self._generation += 1
        self._event_task = None
        if old_provider is not None and old_provider is not provider and old_event_task is not None:
            self._draining_event_tasks.append((old_provider, old_event_task))
        self._provider = provider
        self._closed = False
        self._running = bool(start)
        self._notify_state_changed()
        if start:
            self._start_event_loop_if_needed()
        if old_provider is not None and old_provider is not provider:
            self._schedule_provider_retirement(old_provider, event_task=old_event_task)
        if not future.done():
            future.set_result(old_provider)
        return old_provider

    async def drain_for_toggle_off(self, *, release_backend_after: float | None = None) -> None:
        async with self._lock:
            if release_backend_after is None:
                await self._cancel_idle_release_task()
            provider = self._provider
            if provider is not None:
                await _call_async_method(provider, "close")
            self._start_event_loop_if_needed()
            if (
                provider is not None
                and release_backend_after is not None
                and self._idle_release_task is None
            ):
                self._schedule_idle_release_locked(provider, release_backend_after)

    async def schedule_idle_release(self, *, release_backend_after: float) -> None:
        async with self._lock:
            provider = self._provider
            if provider is not None and self._idle_release_task is None:
                self._schedule_idle_release_locked(provider, release_backend_after)

    async def retire_for_dormant_reuse(self, provider: object) -> None:
        async with self._lock:
            if self._provider is not provider:
                return
            self._running = False
            self._generation += 1
            await self._cancel_event_task()
            await _call_async_method(provider, "close")
            await _call_async_method(provider, "discard_pending_events")

    async def stop_ingress(self) -> None:
        async with self._lock:
            self._running = False
            self._generation += 1
            await self._cancel_event_task()

    async def abort_and_release(self) -> None:
        async with self._lock:
            self._running = False
            self._generation += 1
            await self._cancel_idle_release_task()
            await self._cancel_event_task()
            failures: list[Exception] = []
            pending = self._pending_handoff
            self._pending_handoff = None
            if pending is not None:
                pending_provider, _pending_start, pending_future = pending
                if not pending_future.done():
                    pending_future.set_result(None)
                try:
                    await self._close_provider_for_shutdown(pending_provider)
                except Exception as exc:
                    failures.append(exc)
                    self._retain_retired_provider(pending_provider)

            provider = self._provider
            if provider is not None:
                provider_failed = False
                abort = getattr(provider, "abort_for_toggle_off", None)
                if callable(abort):
                    try:
                        result = abort()
                        if inspect.isawaitable(result):
                            await result
                    except Exception as exc:
                        failures.append(exc)
                        provider_failed = True
                try:
                    await self._close_provider_for_shutdown(provider)
                except Exception as exc:
                    failures.append(exc)
                    provider_failed = True
                self._provider = None
                if provider_failed:
                    self._retain_retired_provider(provider)
                self._notify_state_changed()
                logger.info(
                    "%s toggle-off release provider=%s backend_released=%s failures=%s",
                    self.owner_name,
                    type(provider).__name__,
                    not provider_failed,
                    len(failures),
                )
            _raise_close_failures(failures, f"{self.owner_name} toggle-off release failed")

    async def close(self) -> None:
        async with self._lock:
            if self._closed and not self.has_resources:
                return
            self._closed = True
            self._running = False
            self._generation += 1
            await self._cancel_idle_release_task()
            await self._cancel_event_task()
            failures: list[Exception] = []
            pending = self._pending_handoff
            self._pending_handoff = None
            if pending is not None:
                pending_provider, _pending_start, pending_future = pending
                if not pending_future.done():
                    pending_future.set_result(None)
                try:
                    await self._close_provider_for_shutdown(pending_provider)
                except Exception as exc:
                    failures.append(exc)
            provider = self._provider
            if provider is not None:
                try:
                    await self._close_provider_for_shutdown(provider)
                except Exception as exc:
                    failures.append(exc)
                else:
                    if self._provider is provider:
                        self._provider = None
                        self._notify_state_changed()
            failures.extend(await self._close_retired_providers())
            retirement_tasks = tuple(self._retirement_tasks)
        if retirement_tasks:
            results = await asyncio.gather(*retirement_tasks, return_exceptions=True)
            failures.extend(result for result in results if isinstance(result, Exception))
        async with self._lock:
            draining_tasks = tuple(task for _provider, task in self._draining_event_tasks)
            self._draining_event_tasks.clear()
        for task in draining_tasks:
            if not task.done():
                task.cancel()
        if draining_tasks:
            await asyncio.gather(*draining_tasks, return_exceptions=True)
        async with self._lock:
            _raise_close_failures(failures, f"{self.owner_name} provider close failed")

    def _start_event_loop_if_needed(self) -> None:
        if not self._running or self._event_handler is None or self._provider is None:
            self._notify_state_changed()
            return
        if self._event_task is not None and not self._event_task.done():
            self._notify_state_changed()
            return
        generation = self._generation
        provider = self._provider
        self._event_task = self._create_task(
            self._run_event_loop(provider=provider, generation=generation),
            task_name="events",
        )
        self._event_task.add_done_callback(self._on_event_task_done)
        self._notify_state_changed()

    async def _run_event_loop(self, *, provider: object, generation: int) -> None:
        try:
            async for event in provider.events():  # type: ignore[attr-defined]
                handler = self._event_handler_for(provider=provider, generation=generation)
                if handler is None:
                    continue
                await handler(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._exception_handler is not None:
                result = self._exception_handler(exc)
                if inspect.isawaitable(result):
                    await result
            raise

    def _event_handler_for(
        self,
        *,
        provider: object,
        generation: int,
    ) -> ProviderEventHandler | None:
        if not self._running:
            return None
        if self.is_current_provider_generation(provider=provider, generation=generation):
            return self._event_handler
        if self._is_draining_provider(provider):
            return self._retired_event_handler
        return None

    def _is_draining_provider(self, provider: object) -> bool:
        return any(candidate is provider for candidate, _task in self._draining_event_tasks)

    async def _cancel_event_task(self) -> None:
        task = self._event_task
        if task is None:
            self._notify_state_changed()
            return
        self._event_task = None
        self._notify_state_changed()
        if task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _release_dormant_provider_after(
        self,
        *,
        provider: object,
        generation: int,
        delay_seconds: float,
    ) -> None:
        await self._sleep(delay_seconds)
        async with self._lock:
            if provider is not self._provider or generation != self._generation or self._closed:
                return
            self._running = False
            self._generation += 1
            await self._cancel_event_task()
            try:
                await self._close_provider_for_shutdown(provider)
            except Exception as exc:
                if self._provider is provider:
                    self._provider = None
                    self._retain_retired_provider(provider)
                    self._notify_state_changed()
                if self._exception_handler is not None:
                    result = self._exception_handler(exc)
                    if inspect.isawaitable(result):
                        await result
                return
            if self._provider is provider:
                self._provider = None
                self._notify_state_changed()

    def _schedule_idle_release_locked(self, provider: object, delay_seconds: float) -> None:
        self._idle_release_task = self._create_task(
            self._release_dormant_provider_after(
                provider=provider,
                generation=self._generation,
                delay_seconds=delay_seconds,
            ),
            task_name="idle-release",
        )
        self._idle_release_task.add_done_callback(self._on_idle_release_task_done)

    async def _cancel_idle_release_task(self) -> None:
        task = self._idle_release_task
        if task is None:
            return
        self._idle_release_task = None
        if task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _on_idle_release_task_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()
        if self._idle_release_task is task:
            self._idle_release_task = None
        self._notify_state_changed()

    def _create_task(self, coroutine: Awaitable[None], *, task_name: str) -> asyncio.Task[None]:
        return asyncio.create_task(coroutine, name=f"{self.owner_name}:{task_name}")

    def _on_event_task_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            try:
                task.exception()
            except asyncio.CancelledError:
                pass
        if self._event_task is task:
            self._event_task = None
            self._notify_state_changed()

    async def _close_provider_for_shutdown(self, provider: object | None) -> None:
        if provider is None:
            return
        close_backend = getattr(provider, "close_backend", None)
        if callable(close_backend):
            result = close_backend()
            if inspect.isawaitable(result):
                await result
            return
        await _call_async_method(provider, "close")

    def _schedule_provider_retirement(
        self,
        provider: object,
        *,
        event_task: asyncio.Task[None] | None,
    ) -> None:
        task = self._create_task(
            self._retire_handed_off_provider(provider, event_task=event_task),
            task_name="retire-provider",
        )
        self._retirement_tasks.add(task)
        task.add_done_callback(self._on_retirement_task_done)

    async def _retire_handed_off_provider(
        self,
        provider: object,
        *,
        event_task: asyncio.Task[None] | None,
    ) -> None:
        try:
            await _call_async_method(provider, "close")
            wait_for_ingress = getattr(provider, "wait_for_event_ingress_drain", None)
            if callable(wait_for_ingress) and event_task is not None and not event_task.done():
                result = wait_for_ingress()
                if inspect.isawaitable(result):
                    await result
        finally:
            if event_task is not None and not event_task.done():
                event_task.cancel()
                await asyncio.gather(event_task, return_exceptions=True)
            self._draining_event_tasks = [
                entry for entry in self._draining_event_tasks if entry[0] is not provider
            ]
        await self._close_provider_for_shutdown(provider)

    def _on_retirement_task_done(self, task: asyncio.Task[None]) -> None:
        self._retirement_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            pass
        self._notify_state_changed()

    def _retain_retired_provider(self, provider: object) -> None:
        if provider is self._provider:
            return
        if any(retired_provider is provider for retired_provider in self._retired_providers):
            return
        self._retired_providers.append(provider)

    async def _close_retired_providers(self) -> list[Exception]:
        failures: list[Exception] = []
        still_retired: list[object] = []
        for provider in self._retired_providers:
            try:
                await self._close_provider_for_shutdown(provider)
            except Exception as exc:
                failures.append(exc)
                still_retired.append(provider)
        self._retired_providers = still_retired
        return failures

    def _notify_state_changed(self) -> None:
        if self._state_changed is not None:
            self._state_changed(self)


async def _call_async_method(resource: object, method_name: str) -> None:
    method = getattr(resource, method_name, None)
    if not callable(method):
        return
    result: Any = method()
    if inspect.isawaitable(result):
        await result


def _raise_close_failures(failures: list[Exception], message: str) -> None:
    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]
    raise ExceptionGroup(message, failures)


__all__ = ["ProviderRuntimeHandle"]
