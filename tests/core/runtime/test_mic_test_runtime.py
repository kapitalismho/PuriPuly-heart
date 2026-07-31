from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.core.runtime import MicTestRuntime
from tests.helpers.lifecycle import assert_lifecycle_structure


class RecordingSource:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class FailsOnceSource(RecordingSource):
    async def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("source close failed")


def test_mic_test_runtime_exposes_lifecycle_inventory_and_policy() -> None:
    runtime = MicTestRuntime()

    snapshot = runtime.lifecycle_owner_snapshot()

    assert_lifecycle_structure(snapshot)
    assert snapshot["owner"] == "MicTestRuntime"
    assert len(snapshot["resource_fields"]) == 5


@pytest.mark.asyncio
async def test_mic_test_runtime_stop_cancels_session_frame_and_closes_source() -> None:
    runtime = MicTestRuntime(cancel_timeout_s=0.01)
    source = RecordingSource()
    session_started = asyncio.Event()
    frame_cancelled = asyncio.Event()

    async def pending_frame() -> object:
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            frame_cancelled.set()
            raise

    async def run_session(generation: int) -> None:
        runtime.attach_source(source, generation=generation)
        runtime.create_frame_task(pending_frame(), generation=generation)
        session_started.set()
        await asyncio.sleep(999)

    task = runtime.start(run_session)
    await session_started.wait()

    await runtime.stop()

    assert task.done()
    assert frame_cancelled.is_set() is True
    assert source.close_calls == 1
    assert runtime.session_task is None
    assert runtime.pending_frame_task is None
    assert runtime.source is None


@pytest.mark.asyncio
async def test_mic_test_runtime_source_close_failure_retains_source_for_retry() -> None:
    runtime = MicTestRuntime(cancel_timeout_s=0.01)
    source = FailsOnceSource()
    generation = runtime.begin_direct_capture()
    assert runtime.attach_source(source, generation=generation) is True

    with pytest.raises(RuntimeError, match="source close failed"):
        await runtime.close_source(source)

    assert runtime.source is source
    assert source.close_calls == 1

    await runtime.close_source(source)

    assert runtime.source is None
    assert source.close_calls == 2


@pytest.mark.asyncio
async def test_mic_test_runtime_stop_source_close_failure_keeps_retry_after_task_cleanup() -> None:
    runtime = MicTestRuntime(cancel_timeout_s=0.01)
    source = FailsOnceSource()
    session_started = asyncio.Event()
    frame_cancelled = asyncio.Event()
    session_cancelled = asyncio.Event()

    async def pending_frame() -> object:
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            frame_cancelled.set()
            raise

    async def run_session(generation: int) -> None:
        runtime.attach_source(source, generation=generation)
        runtime.create_frame_task(pending_frame(), generation=generation)
        session_started.set()
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            session_cancelled.set()
            raise

    task = runtime.start(run_session)
    await session_started.wait()

    with pytest.raises(RuntimeError, match="source close failed"):
        await runtime.stop()

    assert task.done()
    assert frame_cancelled.is_set() is True
    assert session_cancelled.is_set() is True
    assert runtime.session_task is None
    assert runtime.pending_frame_task is None
    assert runtime.source is source
    assert source.close_calls == 1

    await runtime.stop()

    assert runtime.source is None
    assert source.close_calls == 2


@pytest.mark.asyncio
async def test_mic_test_runtime_rejects_new_session_after_close() -> None:
    runtime = MicTestRuntime(cancel_timeout_s=0.01)
    await runtime.close()

    async def never_started(_generation: int) -> None:
        await asyncio.sleep(999)

    with pytest.raises(RuntimeError, match="closed"):
        runtime.start(never_started)


@pytest.mark.asyncio
async def test_mic_test_runtime_generation_changes_ignore_late_results() -> None:
    runtime = MicTestRuntime(cancel_timeout_s=0.01)
    session_started = asyncio.Event()

    async def run_session(generation: int) -> None:
        assert runtime.is_current_generation(generation)
        session_started.set()
        await asyncio.sleep(999)

    runtime.start(run_session)
    generation = runtime.generation
    await session_started.wait()

    await runtime.stop()

    assert runtime.is_current_generation(generation) is False


@pytest.mark.asyncio
async def test_mic_test_runtime_direct_capture_rejects_active_session_without_generation_change() -> (
    None
):
    runtime = MicTestRuntime(cancel_timeout_s=0.01)
    session_started = asyncio.Event()

    async def run_session(generation: int) -> None:
        assert runtime.is_current_generation(generation)
        session_started.set()
        await asyncio.sleep(999)

    runtime.start(run_session)
    await session_started.wait()
    active_generation = runtime.generation

    with pytest.raises(RuntimeError, match="active capture"):
        runtime.begin_direct_capture()

    assert runtime.generation == active_generation
    assert runtime.is_current_generation(active_generation) is True
    assert runtime.session_task is not None

    await runtime.stop()
