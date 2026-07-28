from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from puripuly_heart.app.services.clipboard_auto_translation import (
    ClipboardAutoTranslationOwner,
)


class FakeClipboardWatcher:
    def __init__(self, on_text: Callable[[str], None]) -> None:
        self.on_text = on_text
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_owner_starts_submits_and_stops_windows_watcher() -> None:
    watchers: list[FakeClipboardWatcher] = []
    submitted: list[str] = []
    submitted_event = asyncio.Event()

    def watcher_factory(on_text: Callable[[str], None]) -> FakeClipboardWatcher:
        watcher = FakeClipboardWatcher(on_text)
        watchers.append(watcher)
        return watcher

    async def submit_text(text: str) -> None:
        submitted.append(text)
        submitted_event.set()

    owner = ClipboardAutoTranslationOwner(
        watcher_factory=watcher_factory,
        submit_text=submit_text,
        failure_sink=lambda _message: None,
        platform_provider=lambda: "win32",
    )

    await owner.sync(enabled=True)
    watchers[0].on_text("  clipboard text  ")
    await asyncio.wait_for(submitted_event.wait(), timeout=1.0)
    await owner.stop()

    assert watchers[0].started is True
    assert watchers[0].stopped is True
    assert submitted == ["clipboard text"]


@pytest.mark.asyncio
async def test_owner_does_not_create_watcher_when_disabled_or_not_windows() -> None:
    factory_calls = 0

    def watcher_factory(on_text: Callable[[str], None]) -> FakeClipboardWatcher:
        nonlocal factory_calls
        factory_calls += 1
        return FakeClipboardWatcher(on_text)

    async def submit_text(_text: str) -> None:
        return None

    owner = ClipboardAutoTranslationOwner(
        watcher_factory=watcher_factory,
        submit_text=submit_text,
        failure_sink=lambda _message: None,
        platform_provider=lambda: "linux",
    )

    await owner.sync(enabled=True)
    await owner.sync(enabled=False)

    assert factory_calls == 0
    assert owner.runtime is None


@pytest.mark.asyncio
async def test_owner_contains_or_propagates_start_failure_by_policy() -> None:
    failures: list[str] = []

    class FailingWatcher(FakeClipboardWatcher):
        def start(self) -> None:
            raise RuntimeError("start failed")

    async def submit_text(_text: str) -> None:
        return None

    def make_owner() -> ClipboardAutoTranslationOwner:
        return ClipboardAutoTranslationOwner(
            watcher_factory=FailingWatcher,
            submit_text=submit_text,
            failure_sink=failures.append,
            platform_provider=lambda: "win32",
        )

    await make_owner().sync(enabled=True)
    with pytest.raises(RuntimeError, match="start failed"):
        await make_owner().sync(enabled=True, strict_runtime_errors=True)

    assert failures == ["Clipboard watcher failed to start"]


@pytest.mark.asyncio
async def test_owner_contains_submit_failure_and_reports_diagnostic() -> None:
    failures: list[str] = []

    async def submit_text(_text: str) -> None:
        raise OSError("delivery failed")

    owner = ClipboardAutoTranslationOwner(
        watcher_factory=FakeClipboardWatcher,
        submit_text=submit_text,
        failure_sink=failures.append,
        platform_provider=lambda: "win32",
    )

    await owner.submit_now("text")

    assert failures == ["Clipboard submit failed: delivery failed"]


@pytest.mark.asyncio
async def test_owner_shutdown_close_propagates_stop_failure_when_strict() -> None:
    failures: list[str] = []

    class FailingStopWatcher(FakeClipboardWatcher):
        def stop(self) -> None:
            self.stopped = True
            raise RuntimeError("stop failed")

    async def submit_text(_text: str) -> None:
        return None

    owner = ClipboardAutoTranslationOwner(
        watcher_factory=FailingStopWatcher,
        submit_text=submit_text,
        failure_sink=failures.append,
        platform_provider=lambda: "win32",
    )
    await owner.sync(enabled=True)

    with pytest.raises(RuntimeError, match="stop failed"):
        await owner.close(strict_runtime_errors=True)

    assert failures == ["Clipboard runtime failed to close"]
