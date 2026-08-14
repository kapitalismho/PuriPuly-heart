from __future__ import annotations

import asyncio
import threading

import pytest
from puripuly_heart.core.local_stt_runtime_installer import RuntimeLocalSTTStatusUpdate

from puripuly_heart.core.runtime import LocalSTTDownloadRuntime
from tests.helpers.lifecycle import assert_lifecycle_structure


def test_local_stt_download_runtime_exposes_lifecycle_inventory_and_policy() -> None:
    runtime = LocalSTTDownloadRuntime()

    snapshot = runtime.lifecycle_owner_snapshot()

    assert_lifecycle_structure(snapshot)
    assert snapshot["owner"] == "LocalSTTDownloadRuntime"
    assert len(snapshot["resource_fields"]) == 4


@pytest.mark.asyncio
async def test_local_stt_download_runtime_close_sets_cancel_event_and_cancels_task() -> None:
    runtime = LocalSTTDownloadRuntime(cancel_timeout_s=0.05)
    started = asyncio.Event()
    cancel_events: list[threading.Event] = []

    async def run_download(cancel_event: threading.Event, generation: int) -> object:
        assert runtime.is_current_generation(generation)
        cancel_events.append(cancel_event)
        started.set()
        await asyncio.sleep(999)
        return object()

    task = runtime.start(origin="manual", run_download=run_download)
    await started.wait()

    await runtime.close()

    assert task.done()
    assert cancel_events[0].is_set() is True
    assert runtime.download_task is None
    assert runtime.cancel_event is None
    assert runtime.origin is None
    assert runtime.is_closed is True


@pytest.mark.asyncio
async def test_close_retains_timed_out_download_task_for_retry() -> None:
    runtime = LocalSTTDownloadRuntime(cancel_timeout_s=0.01)
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_download(_cancel_event: threading.Event, _generation: int) -> object:
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        return object()

    task = runtime.start(origin="manual", run_download=run_download)
    await started.wait()

    with pytest.raises(TimeoutError, match="cancellation timed out"):
        await runtime.close()

    assert task.done() is False
    assert runtime.download_task is task
    assert runtime.cancel_event is not None
    assert runtime.cancel_event.is_set() is True
    assert runtime.origin == "manual"
    assert runtime.is_closed is True

    release.set()
    await runtime.close()
    await task

    assert runtime.download_task is None
    assert runtime.cancel_event is None
    assert runtime.origin is None


@pytest.mark.asyncio
async def test_local_stt_download_runtime_rejects_start_while_closing_or_closed() -> None:
    runtime = LocalSTTDownloadRuntime(cancel_timeout_s=0.01)
    await runtime.close()

    async def never_started(_cancel_event: threading.Event, _generation: int) -> object:
        await asyncio.sleep(999)
        return object()

    with pytest.raises(RuntimeError, match="closed"):
        runtime.start(origin="manual", run_download=never_started)


@pytest.mark.asyncio
async def test_late_local_stt_progress_from_old_generation_is_ignored() -> None:
    runtime = LocalSTTDownloadRuntime(cancel_timeout_s=0.01)
    started = asyncio.Event()
    seen_updates: list[RuntimeLocalSTTStatusUpdate] = []

    async def run_download(_cancel_event: threading.Event, _generation: int) -> object:
        started.set()
        await asyncio.sleep(999)
        return object()

    runtime.start(origin="manual", run_download=run_download)
    generation = runtime.generation
    await started.wait()

    await runtime.dispatch_status_update(
        RuntimeLocalSTTStatusUpdate(status="downloading", percent=25),
        generation=generation,
        on_status=seen_updates.append,
    )
    await runtime.close()
    await runtime.dispatch_status_update(
        RuntimeLocalSTTStatusUpdate(status="downloading", percent=99),
        generation=generation,
        on_status=seen_updates.append,
    )

    assert seen_updates == [RuntimeLocalSTTStatusUpdate(status="downloading", percent=25)]


@pytest.mark.asyncio
async def test_late_local_stt_progress_after_normal_completion_is_ignored() -> None:
    runtime = LocalSTTDownloadRuntime(cancel_timeout_s=0.01)
    seen_updates: list[RuntimeLocalSTTStatusUpdate] = []

    async def run_download(_cancel_event: threading.Event, _generation: int) -> object:
        return object()

    task = runtime.start(origin="manual", run_download=run_download)
    generation = runtime.generation
    await task
    await asyncio.sleep(0)

    await runtime.dispatch_status_update(
        RuntimeLocalSTTStatusUpdate(status="downloading", percent=99),
        generation=generation,
        on_status=seen_updates.append,
    )

    assert seen_updates == []
