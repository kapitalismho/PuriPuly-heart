from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from puripuly_heart.providers.stt.local_decode import (
    LocalDecodeBacklog,
    LocalDecodeCompletion,
    LocalDecodeCoordinator,
    LocalDecodeFailure,
)


@pytest.mark.asyncio
async def test_local_decode_coordinator_preserves_fifo_and_empty_boundaries() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    decoded: list[np.ndarray] = []
    completions: list[LocalDecodeCompletion] = []
    failures: list[Exception] = []

    async def decode(samples_f32: np.ndarray) -> str:
        decoded.append(samples_f32.copy())
        if len(decoded) == 1:
            started.set()
            await release.wait()
        return str(int(samples_f32[0]))

    async def on_completion(completion: LocalDecodeCompletion) -> None:
        completions.append(completion)

    async def on_failure(failure: LocalDecodeFailure) -> None:
        failures.append(failure.error)

    coordinator = LocalDecodeCoordinator(
        owner_name="test-local-decode",
        sample_rate_hz=16000,
        decode=decode,
        on_completion=on_completion,
        on_failure=on_failure,
    )

    assert coordinator.enqueue(np.ones(160, dtype=np.float32)) is True
    await asyncio.wait_for(started.wait(), timeout=0.1)
    assert coordinator.enqueue(np.empty((0,), dtype=np.float32)) is True
    assert coordinator.enqueue(np.full(320, 2.0, dtype=np.float32)) is True
    assert coordinator.pending_jobs == 3

    stop_task = asyncio.create_task(coordinator.stop())
    await asyncio.sleep(0)
    assert not stop_task.done()
    assert coordinator.enqueue(np.ones(1, dtype=np.float32)) is False

    release.set()
    await asyncio.wait_for(stop_task, timeout=1.0)

    assert failures == []
    assert [completion.job.sequence for completion in completions] == [1, 2, 3]
    assert [completion.text for completion in completions] == ["1", "", "2"]
    assert len(decoded) == 2
    assert coordinator.pending_jobs == 0
    assert coordinator._worker_task is not None
    assert coordinator._worker_task.done()
    await coordinator.close()


@pytest.mark.asyncio
async def test_local_decode_coordinator_warns_once_per_backlog_excursion() -> None:
    release = asyncio.Event()
    started = asyncio.Event()
    completions: list[LocalDecodeCompletion] = []
    warnings: list[LocalDecodeBacklog] = []

    async def decode(samples_f32: np.ndarray) -> str:
        _ = samples_f32
        started.set()
        await release.wait()
        return ""

    async def on_completion(completion: LocalDecodeCompletion) -> None:
        completions.append(completion)

    async def on_failure(failure: LocalDecodeFailure) -> None:
        raise AssertionError("unexpected failure") from failure.error

    coordinator = LocalDecodeCoordinator(
        owner_name="test-local-decode-backlog",
        sample_rate_hz=16000,
        decode=decode,
        on_completion=on_completion,
        on_failure=on_failure,
        on_backlog_warning=warnings.append,
    )

    for _ in range(9):
        assert coordinator.enqueue(np.ones(160, dtype=np.float32)) is True
    await asyncio.wait_for(started.wait(), timeout=0.1)
    assert len(warnings) == 1
    assert warnings[0].pending_jobs == 9
    assert warnings[0].buffered_audio_ms == pytest.approx(90.0)

    assert coordinator.enqueue(np.ones(160, dtype=np.float32)) is True
    assert len(warnings) == 1
    release.set()
    for _ in range(100):
        if coordinator.pending_jobs == 0:
            break
        await asyncio.sleep(0)
    assert coordinator.pending_jobs == 0

    release.clear()
    started.clear()
    for _ in range(9):
        assert coordinator.enqueue(np.ones(160, dtype=np.float32)) is True
    await asyncio.wait_for(started.wait(), timeout=0.1)
    assert len(warnings) == 2

    release.set()
    await coordinator.stop()
    assert len(completions) == 19
    await coordinator.close()


@pytest.mark.asyncio
async def test_local_decode_coordinator_failure_discards_remaining_jobs() -> None:
    attempted: list[np.ndarray] = []
    completions: list[LocalDecodeCompletion] = []
    failures: list[LocalDecodeFailure] = []
    failed = asyncio.Event()

    async def decode(samples_f32: np.ndarray) -> str:
        attempted.append(samples_f32.copy())
        raise RuntimeError("decode failed")

    async def on_completion(completion: LocalDecodeCompletion) -> None:
        completions.append(completion)

    async def on_failure(failure: LocalDecodeFailure) -> None:
        failures.append(failure)
        failed.set()

    coordinator = LocalDecodeCoordinator(
        owner_name="test-local-decode-failure",
        sample_rate_hz=16000,
        decode=decode,
        on_completion=on_completion,
        on_failure=on_failure,
    )

    assert coordinator.enqueue(np.ones(160, dtype=np.float32)) is True
    assert coordinator.enqueue(np.full(160, 2.0, dtype=np.float32)) is True
    await asyncio.wait_for(failed.wait(), timeout=0.1)

    assert len(attempted) == 1
    assert completions == []
    assert len(failures) == 1
    assert str(failures[0].error) == "decode failed"
    assert failures[0].job.sequence == 1
    assert [job.sequence for job in failures[0].discarded_jobs] == [2]
    assert coordinator.accepting is False
    assert coordinator.pending_jobs == 0
    assert coordinator.enqueue(np.ones(1, dtype=np.float32)) is False
    await coordinator.close()


@pytest.mark.asyncio
async def test_local_decode_coordinator_close_cancels_active_worker() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    completions: list[LocalDecodeCompletion] = []

    async def decode(samples_f32: np.ndarray) -> str:
        _ = samples_f32
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return ""

    async def on_completion(completion: LocalDecodeCompletion) -> None:
        completions.append(completion)

    async def on_failure(failure: LocalDecodeFailure) -> None:
        raise AssertionError("unexpected failure") from failure.error

    coordinator = LocalDecodeCoordinator(
        owner_name="test-local-decode-close",
        sample_rate_hz=16000,
        decode=decode,
        on_completion=on_completion,
        on_failure=on_failure,
    )

    assert coordinator.enqueue(np.ones(160, dtype=np.float32)) is True
    assert coordinator.enqueue(np.full(160, 2.0, dtype=np.float32)) is True
    await asyncio.wait_for(started.wait(), timeout=0.1)

    await asyncio.wait_for(coordinator.close(), timeout=0.1)

    assert cancelled.is_set()
    assert completions == []
    assert coordinator.accepting is False
    assert coordinator.pending_jobs == 0


@pytest.mark.asyncio
async def test_local_decode_coordinator_stop_and_close_can_race() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def decode(samples_f32: np.ndarray) -> str:
        _ = samples_f32
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return ""

    async def on_completion(completion: LocalDecodeCompletion) -> None:
        _ = completion

    async def on_failure(failure: LocalDecodeFailure) -> None:
        raise AssertionError("unexpected failure") from failure.error

    coordinator = LocalDecodeCoordinator(
        owner_name="test-local-decode-stop-close-race",
        sample_rate_hz=16000,
        decode=decode,
        on_completion=on_completion,
        on_failure=on_failure,
    )

    assert coordinator.enqueue(np.ones(160, dtype=np.float32)) is True
    await asyncio.wait_for(started.wait(), timeout=0.1)

    stop_task = asyncio.create_task(coordinator.stop())
    await asyncio.sleep(0)
    close_task = asyncio.create_task(coordinator.close())
    await asyncio.wait_for(asyncio.gather(stop_task, close_task), timeout=1.0)

    assert cancelled.is_set()
    assert coordinator.accepting is False
    assert coordinator.pending_jobs == 0


@pytest.mark.asyncio
async def test_local_decode_coordinator_contains_failure_callback_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def decode(samples_f32: np.ndarray) -> str:
        _ = samples_f32
        raise RuntimeError("decode failed")

    async def on_completion(completion: LocalDecodeCompletion) -> None:
        _ = completion

    async def on_failure(failure: LocalDecodeFailure) -> None:
        raise RuntimeError("failure callback failed") from failure.error

    coordinator = LocalDecodeCoordinator(
        owner_name="test-local-decode-callback-failure",
        sample_rate_hz=16000,
        decode=decode,
        on_completion=on_completion,
        on_failure=on_failure,
    )

    with caplog.at_level(
        logging.ERROR,
        logger="puripuly_heart.providers.stt.local_decode",
    ):
        assert coordinator.enqueue(np.ones(160, dtype=np.float32)) is True
        await asyncio.wait_for(coordinator.stop(), timeout=1.0)
        await asyncio.wait_for(coordinator.close(), timeout=1.0)

    assert coordinator.accepting is False
    assert coordinator.pending_jobs == 0
    assert any("decode failure callback failed" in message for message in caplog.messages)
