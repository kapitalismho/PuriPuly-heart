from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable

import pytest

from puripuly_heart.core.runtime import ClipboardRuntime


class FakeClipboardWatcher:
    def __init__(self, on_text: Callable[[str], None]) -> None:
        self.on_text = on_text
        self.started = False
        self.stopped = False
        self.start_calls = 0
        self.stop_calls = 0
        self.stopped_event = threading.Event()
        self.on_stop: Callable[[], None] | None = None

    def start(self) -> None:
        self.started = True
        self.start_calls += 1

    def stop(self) -> None:
        self.stopped = True
        self.stop_calls += 1
        if self.on_stop is not None:
            self.on_stop()
        self.stopped_event.set()


class FailingStopClipboardWatcher(FakeClipboardWatcher):
    def stop(self) -> None:
        super().stop()
        raise RuntimeError("watcher stop failed")


async def _flush_loop() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_clipboard_runtime_stops_watcher_before_gathering_submit_tasks() -> None:
    watchers: list[FakeClipboardWatcher] = []
    order: list[str] = []
    submit_started = asyncio.Event()
    submit_release = asyncio.Event()

    def watcher_factory(on_text: Callable[[str], None]) -> FakeClipboardWatcher:
        watcher = FakeClipboardWatcher(on_text)
        watcher.on_stop = lambda: order.append("watcher-stop")
        watchers.append(watcher)
        return watcher

    async def submit(text: str) -> None:
        assert text == "clipboard text"
        order.append("submit-start")
        submit_started.set()
        await submit_release.wait()
        order.append("submit-finish")

    runtime = ClipboardRuntime(
        watcher_factory=watcher_factory,
        submit_handler=submit,
        submit_gather_timeout_s=0.2,
    )
    await runtime.sync(enabled=True)
    watcher = watchers[0]

    watcher.on_text(" clipboard text ")
    await submit_started.wait()

    async def stop_and_record() -> None:
        await runtime.close()
        order.append("runtime-close-returned")

    close_task = asyncio.create_task(stop_and_record())

    assert await asyncio.to_thread(watcher.stopped_event.wait, 1.0) is True
    assert order == ["submit-start", "watcher-stop"]

    submit_release.set()
    await close_task

    assert order == [
        "submit-start",
        "watcher-stop",
        "submit-finish",
        "runtime-close-returned",
    ]
    assert runtime.active_submit_task_count == 0


@pytest.mark.asyncio
async def test_late_clipboard_callback_from_old_generation_is_ignored() -> None:
    watchers: list[FakeClipboardWatcher] = []
    submitted: list[str] = []

    def watcher_factory(on_text: Callable[[str], None]) -> FakeClipboardWatcher:
        watcher = FakeClipboardWatcher(on_text)
        watchers.append(watcher)
        return watcher

    async def submit(text: str) -> None:
        submitted.append(text)

    runtime = ClipboardRuntime(watcher_factory=watcher_factory, submit_handler=submit)
    await runtime.sync(enabled=True)
    old_watcher = watchers[0]
    await runtime.stop()

    old_watcher.on_text(" stale clipboard text ")
    await _flush_loop()

    await runtime.sync(enabled=True)
    watchers[1].on_text(" fresh clipboard text ")
    await _flush_loop()

    assert submitted == ["fresh clipboard text"]


@pytest.mark.asyncio
async def test_clipboard_runtime_submit_gather_timeout_is_bounded() -> None:
    watchers: list[FakeClipboardWatcher] = []
    submit_started = asyncio.Event()

    def watcher_factory(on_text: Callable[[str], None]) -> FakeClipboardWatcher:
        watcher = FakeClipboardWatcher(on_text)
        watchers.append(watcher)
        return watcher

    async def submit(_text: str) -> None:
        submit_started.set()
        await asyncio.sleep(999)

    runtime = ClipboardRuntime(
        watcher_factory=watcher_factory,
        submit_handler=submit,
        submit_gather_timeout_s=0.01,
    )
    await runtime.sync(enabled=True)
    watchers[0].on_text("slow clipboard text")
    await submit_started.wait()

    await runtime.close()

    assert watchers[0].stopped is True
    assert runtime.active_submit_task_count == 0
    assert runtime.submit_gather_timeouts == 1


@pytest.mark.asyncio
async def test_clipboard_runtime_close_returns_when_submit_suppresses_cancellation() -> None:
    watchers: list[FakeClipboardWatcher] = []
    submit_started = asyncio.Event()
    cancel_seen = asyncio.Event()
    release_stubborn_submit = asyncio.Event()

    def watcher_factory(on_text: Callable[[str], None]) -> FakeClipboardWatcher:
        watcher = FakeClipboardWatcher(on_text)
        watchers.append(watcher)
        return watcher

    async def submit(_text: str) -> None:
        submit_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancel_seen.set()
            await release_stubborn_submit.wait()

    runtime = ClipboardRuntime(
        watcher_factory=watcher_factory,
        submit_handler=submit,
        submit_gather_timeout_s=0.01,
    )
    await runtime.sync(enabled=True)
    watchers[0].on_text("stubborn clipboard text")
    await submit_started.wait()

    close_task = asyncio.create_task(runtime.close())
    try:
        await asyncio.wait_for(asyncio.shield(close_task), timeout=0.2)
        assert runtime.active_submit_task_count == 1
        assert runtime.failure_diagnostics.code == "clipboard_submit_timeout"
        assert runtime.failure_diagnostics.fields["cleanup_confirmed"] is False
    finally:
        release_stubborn_submit.set()
        with contextlib.suppress(Exception):
            await close_task
        await _flush_loop()

    assert cancel_seen.is_set() is True
    assert watchers[0].stopped is True
    assert runtime.active_submit_task_count == 0
    assert runtime.submit_gather_timeouts == 1


@pytest.mark.asyncio
async def test_clipboard_runtime_gathers_submit_tasks_when_watcher_stop_fails() -> None:
    watchers: list[FailingStopClipboardWatcher] = []
    submit_started = asyncio.Event()
    submit_release = asyncio.Event()
    submitted: list[str] = []

    def watcher_factory(on_text: Callable[[str], None]) -> FailingStopClipboardWatcher:
        watcher = FailingStopClipboardWatcher(on_text)
        watchers.append(watcher)
        return watcher

    async def submit(text: str) -> None:
        submit_started.set()
        await submit_release.wait()
        submitted.append(text)

    runtime = ClipboardRuntime(
        watcher_factory=watcher_factory,
        submit_handler=submit,
        submit_gather_timeout_s=0.2,
    )
    await runtime.sync(enabled=True)
    watchers[0].on_text("pending clipboard text")
    await submit_started.wait()

    stop_task = asyncio.create_task(runtime.stop(strict_runtime_errors=True))
    assert await asyncio.to_thread(watchers[0].stopped_event.wait, 1.0) is True
    submit_release.set()

    with pytest.raises(RuntimeError, match="watcher stop failed"):
        await stop_task
    assert submitted == ["pending clipboard text"]
    assert runtime.active_submit_task_count == 0


@pytest.mark.asyncio
async def test_clipboard_runtime_close_surfaces_stop_failure_after_gathering_submit_tasks() -> None:
    watchers: list[FailingStopClipboardWatcher] = []
    submit_started = asyncio.Event()
    submit_release = asyncio.Event()
    submitted: list[str] = []

    def watcher_factory(on_text: Callable[[str], None]) -> FailingStopClipboardWatcher:
        watcher = FailingStopClipboardWatcher(on_text)
        watchers.append(watcher)
        return watcher

    async def submit(text: str) -> None:
        submit_started.set()
        await submit_release.wait()
        submitted.append(text)

    runtime = ClipboardRuntime(
        watcher_factory=watcher_factory,
        submit_handler=submit,
        submit_gather_timeout_s=0.2,
    )
    await runtime.sync(enabled=True)
    watchers[0].on_text("pending close clipboard text")
    await submit_started.wait()

    close_task = asyncio.create_task(runtime.close())
    assert await asyncio.to_thread(watchers[0].stopped_event.wait, 1.0) is True
    submit_release.set()

    with pytest.raises(RuntimeError, match="watcher stop failed"):
        await close_task
    assert submitted == ["pending close clipboard text"]
    assert runtime.active_submit_task_count == 0
    assert runtime.watcher is None


@pytest.mark.asyncio
async def test_submit_failure_retains_safe_original_cause_after_timeout() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    failures = []

    async def submit(_text: str) -> None:
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        raise PermissionError(13, "private clipboard text")

    runtime = ClipboardRuntime(
        watcher_factory=FakeClipboardWatcher,
        submit_handler=submit,
        submit_gather_timeout_s=0.001,
        failure_sink=failures.append,
    )
    await runtime.start()
    runtime.submit_from_loop("private clipboard text")
    await started.wait()
    try:
        await runtime.close()
        assert runtime.active_submit_task_count == 1
    finally:
        release.set()
        await asyncio.gather(*tuple(runtime._submit_tasks), return_exceptions=True)
        await _flush_loop()
    assert runtime.active_submit_task_count == 0
    assert runtime.failure_diagnostics.fields["errno"] == 13
    assert [item.code for item in failures] == ["clipboard_submit_timeout", "clipboard_failed"]
    assert "private" not in repr(failures)


@pytest.mark.asyncio
async def test_non_strict_start_returns_connected_start_and_cleanup_failures():
    class FailedWatcher(FakeClipboardWatcher):
        def start(self):
            raise PermissionError(13, "private startup content")

        def stop(self):
            raise OSError(5, "private cleanup content")

    async def submit(_text):
        return None

    runtime = ClipboardRuntime(watcher_factory=FailedWatcher, submit_handler=submit)
    diagnostics = await runtime.start()
    assert diagnostics is runtime.failure_diagnostics
    assert diagnostics.operation == "watcher_start"
    assert diagnostics.fields["errno"] == 13
    cleanup = diagnostics.causes[-1]
    assert cleanup.operation == "watcher_stop"
    assert cleanup.fields["errno"] == 5
    assert cleanup.cause_id == diagnostics.diagnostic_id
    assert runtime.watcher is None
    assert "private" not in repr(diagnostics)
