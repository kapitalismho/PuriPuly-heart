from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle


class ExceptionObservingTask(asyncio.Task):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.exception_requests = 0

    def exception(self) -> BaseException | None:
        self.exception_requests += 1
        return super().exception()


class RaisingEventsProvider:
    def __init__(self, failure: Exception) -> None:
        self.failure = failure

    async def events(self):
        raise self.failure
        yield object()


class RetriableCloseProvider:
    def __init__(self, *, close_failures: int = 1, label: str = "provider") -> None:
        self.close_failures = close_failures
        self.label = label
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_failures > 0:
            self.close_failures -= 1
            raise RuntimeError(f"{self.label} close failed")


class QueueEventsProvider(RetriableCloseProvider):
    def __init__(self, *, close_failures: int = 0, label: str = "provider") -> None:
        super().__init__(close_failures=close_failures, label=label)
        self.queue: asyncio.Queue[object | None] = asyncio.Queue()

    async def emit(self, event: object) -> None:
        await self.queue.put(event)

    async def close(self) -> None:
        await super().close()
        await self.queue.put(None)

    async def events(self):
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item


class DormantBackendProvider:
    def __init__(self, *, close_backend_failures: int = 0) -> None:
        self.close_calls = 0
        self.close_backend_calls = 0
        self.close_backend_failures = close_backend_failures

    async def close(self) -> None:
        self.close_calls += 1

    async def close_backend(self) -> None:
        self.close_backend_calls += 1
        if self.close_backend_failures > 0:
            self.close_backend_failures -= 1
            raise RuntimeError("backend close failed")


class AbortableBackendProvider(QueueEventsProvider):
    def __init__(self) -> None:
        super().__init__()
        self.abort_calls = 0
        self.close_backend_calls = 0

    async def abort_for_toggle_off(self) -> None:
        self.abort_calls += 1
        await self.queue.put(None)

    async def close_backend(self) -> None:
        self.close_backend_calls += 1


async def wait_until(predicate: Callable[[], bool], *, timeout_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_abort_and_release_stops_ingress_and_releases_backend() -> None:
    provider = AbortableBackendProvider()
    events: list[object] = []

    async def handle_event(event: object) -> None:
        events.append(event)

    handle = ProviderRuntimeHandle(
        name="self_stt",
        provider=provider,
        event_handler=handle_event,
    )
    await handle.start()

    await handle.abort_and_release()
    await provider.emit("late")
    await asyncio.sleep(0)

    assert provider.abort_calls == 1
    assert provider.close_backend_calls == 1
    assert handle.provider is None
    assert handle.event_task is None
    assert events == []


@pytest.mark.asyncio
async def test_dormant_reuse_aborts_publication_without_releasing_backend() -> None:
    provider = AbortableBackendProvider()
    events: list[object] = []

    async def handle_event(event: object) -> None:
        events.append(event)

    handle = ProviderRuntimeHandle(
        name="peer_stt",
        provider=provider,
        event_handler=handle_event,
    )
    await handle.start()
    await handle.retire_for_dormant_reuse(provider)

    assert provider.abort_calls == 1
    assert provider.close_backend_calls == 0
    assert handle.provider is provider
    assert handle.event_task is None
    assert events == []

    await handle.close()
    assert provider.close_backend_calls == 1


@pytest.mark.asyncio
async def test_close_failure_retains_provider_for_retry_and_clears_after_success() -> None:
    provider = RetriableCloseProvider(close_failures=1, label="self provider")
    notifications: list[object | None] = []
    handle = ProviderRuntimeHandle(
        name="self_stt",
        provider=provider,
        state_changed=lambda changed: notifications.append(changed.provider),
    )

    with pytest.raises(RuntimeError, match="self provider close failed"):
        await handle.close()

    assert handle.provider is provider
    assert provider.close_calls == 1

    await handle.close()

    assert provider.close_calls == 2
    assert handle.provider is None
    assert notifications[-1] is None


@pytest.mark.asyncio
async def test_replace_provider_starts_new_ingress_when_old_close_fails() -> None:
    processed_events: list[object] = []
    old_provider = QueueEventsProvider(close_failures=1, label="old provider")
    new_provider = QueueEventsProvider(label="new provider")

    async def handle_event(event: object) -> None:
        processed_events.append(event)

    handle = ProviderRuntimeHandle(
        name="self_stt",
        provider=old_provider,
        event_handler=handle_event,
    )

    await handle.start()
    old_event_task = handle.event_task
    assert old_event_task is not None

    with pytest.raises(RuntimeError, match="old provider close failed"):
        await handle.replace_provider(new_provider, start=True)

    assert handle.provider is new_provider
    assert handle.event_task is not None
    assert handle.event_task is not old_event_task

    await old_provider.emit("old stale event")
    await new_provider.emit("new current event")
    await wait_until(lambda: processed_events == ["new current event"])

    await handle.close()

    assert old_provider.close_calls == 2
    assert new_provider.close_calls == 1
    assert handle.provider is None
    assert handle.event_task is None
    assert processed_events == ["new current event"]


@pytest.mark.asyncio
async def test_failed_event_task_exception_is_retrieved_by_owner_done_callback() -> None:
    observed_tasks: list[ExceptionObservingTask] = []
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()

    def task_factory(
        task_loop: asyncio.AbstractEventLoop,
        coro: Coroutine[Any, Any, None],
        **kwargs: object,
    ) -> ExceptionObservingTask:
        task = ExceptionObservingTask(coro, loop=task_loop, **kwargs)
        observed_tasks.append(task)
        return task

    loop.set_task_factory(task_factory)
    try:
        handled_exceptions: list[Exception] = []
        failure = RuntimeError("provider events failed")

        async def handle_event(_event: object) -> None:
            raise AssertionError("provider should fail before yielding events")

        async def handle_exception(exc: Exception) -> None:
            handled_exceptions.append(exc)

        handle = ProviderRuntimeHandle(
            name="self_stt",
            provider=RaisingEventsProvider(failure),
            event_handler=handle_event,
            exception_handler=handle_exception,
        )

        await handle.start()
        task = handle.event_task

        assert isinstance(task, ExceptionObservingTask)
        await wait_until(task.done)
        await asyncio.sleep(0)
        owner_exception_requests = task.exception_requests
    finally:
        loop.set_task_factory(previous_factory)
        for observed_task in observed_tasks:
            if observed_task.done() and observed_task.exception_requests == 0:
                _ = observed_task.exception()

    assert handled_exceptions == [failure]
    assert handle.event_task is None
    assert owner_exception_requests == 1


@pytest.mark.asyncio
async def test_dormant_provider_backend_is_released_after_idle_ttl() -> None:
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()
    delays: list[float] = []

    async def controlled_sleep(delay: float) -> None:
        delays.append(delay)
        sleep_started.set()
        await release_sleep.wait()

    provider = DormantBackendProvider()
    handle = ProviderRuntimeHandle(name="self_stt", provider=provider, sleep=controlled_sleep)

    await handle.drain_for_toggle_off(release_backend_after=600.0)
    await sleep_started.wait()
    await handle.drain_for_toggle_off(release_backend_after=600.0)

    assert provider.close_calls == 2
    assert provider.close_backend_calls == 0
    assert handle.provider is provider
    assert delays == [600.0]

    release_sleep.set()
    await wait_until(lambda: not handle.has_resources)

    assert provider.close_backend_calls == 1
    assert handle.has_resources is False


@pytest.mark.asyncio
async def test_reenable_cancels_dormant_backend_release() -> None:
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()

    async def controlled_sleep(_delay: float) -> None:
        sleep_started.set()
        await release_sleep.wait()

    provider = DormantBackendProvider()
    handle = ProviderRuntimeHandle(name="self_stt", provider=provider, sleep=controlled_sleep)

    await handle.drain_for_toggle_off(release_backend_after=600.0)
    await sleep_started.wait()
    assert await handle.start_if_provider(provider) is True
    release_sleep.set()
    await asyncio.sleep(0)

    assert handle.provider is provider
    assert provider.close_backend_calls == 0

    await handle.close()
    assert provider.close_backend_calls == 1


@pytest.mark.asyncio
async def test_shutdown_cancels_idle_timer_and_closes_backend_once() -> None:
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()

    async def controlled_sleep(_delay: float) -> None:
        sleep_started.set()
        await release_sleep.wait()

    provider = DormantBackendProvider()
    handle = ProviderRuntimeHandle(name="self_stt", provider=provider, sleep=controlled_sleep)

    await handle.drain_for_toggle_off(release_backend_after=600.0)
    await sleep_started.wait()
    await handle.close()
    release_sleep.set()
    await asyncio.sleep(0)

    assert provider.close_backend_calls == 1
    assert handle.has_resources is False


@pytest.mark.asyncio
async def test_provider_replacement_closes_dormant_backend_immediately() -> None:
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()

    async def controlled_sleep(_delay: float) -> None:
        sleep_started.set()
        await release_sleep.wait()

    old_provider = DormantBackendProvider()
    new_provider = DormantBackendProvider()
    handle = ProviderRuntimeHandle(name="self_stt", provider=old_provider, sleep=controlled_sleep)

    await handle.drain_for_toggle_off(release_backend_after=600.0)
    await sleep_started.wait()
    await handle.replace_provider(new_provider, start=False)

    assert old_provider.close_backend_calls == 1
    assert new_provider.close_backend_calls == 0
    assert handle.provider is new_provider

    release_sleep.set()
    await handle.close()


@pytest.mark.asyncio
async def test_failed_idle_release_detaches_provider_and_retries_on_shutdown() -> None:
    release_sleep = asyncio.Event()
    observed: list[Exception] = []

    async def controlled_sleep(_delay: float) -> None:
        await release_sleep.wait()

    async def handle_exception(exc: Exception) -> None:
        observed.append(exc)

    provider = DormantBackendProvider(close_backend_failures=1)
    handle = ProviderRuntimeHandle(
        name="self_stt",
        provider=provider,
        exception_handler=handle_exception,
        sleep=controlled_sleep,
    )

    await handle.drain_for_toggle_off(release_backend_after=600.0)
    release_sleep.set()
    await wait_until(lambda: handle.provider is None)

    assert provider.close_backend_calls == 1
    assert len(observed) == 1
    assert handle.has_resources is True

    await handle.close()

    assert provider.close_backend_calls == 2
    assert handle.has_resources is False


@pytest.mark.asyncio
async def test_handoff_keeps_unhashable_retired_event_ingress_until_final_drain() -> None:
    current_events: list[str] = []
    retired_events: list[str] = []

    class HandoffProvider:
        __hash__ = None

        def __init__(self, name: str, *, active: bool) -> None:
            self.name = name
            self.active = active
            self.queue: asyncio.Queue[object] = asyncio.Queue()
            self.close_calls = 0
            self.close_backend_calls = 0

        @property
        def is_at_utterance_boundary(self) -> bool:
            return not self.active

        async def events(self):
            while True:
                item = await self.queue.get()
                if isinstance(item, asyncio.Event):
                    item.set()
                    continue
                yield item

        async def close(self) -> None:
            self.close_calls += 1
            await self.queue.put(f"{self.name}-final")

        async def wait_for_event_ingress_drain(self) -> None:
            reached = asyncio.Event()
            await self.queue.put(reached)
            await reached.wait()

        async def close_backend(self) -> None:
            self.close_backend_calls += 1

    async def handle_event(event: object) -> None:
        current_events.append(str(event))

    async def handle_retired_event(event: object) -> None:
        retired_events.append(str(event))

    old = HandoffProvider("old", active=True)
    new = HandoffProvider("new", active=False)
    handle = ProviderRuntimeHandle(
        name="self_stt",
        provider=old,
        event_handler=handle_event,
        retired_event_handler=handle_retired_event,
    )
    await handle.start()

    handoff = asyncio.create_task(handle.handoff_provider_at_boundary(new, start=True))
    await asyncio.sleep(0)
    assert handle.provider is old
    assert handoff.done() is False

    old.active = False
    retired = await handle.commit_pending_handoff()
    assert retired is old
    assert await handoff is old
    assert handle.provider is new
    await new.queue.put("new-event")

    for _ in range(20):
        if old.close_backend_calls and "new-event" in current_events:
            break
        await asyncio.sleep(0)

    assert retired_events == ["old-final"]
    assert current_events == ["new-event"]
    assert old.close_calls == 1
    assert old.close_backend_calls == 1
    await handle.close()
