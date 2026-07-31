from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.core.runtime import GithubStarPromptRuntime
from tests.helpers.lifecycle import assert_lifecycle_structure


async def _flush_loop() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def test_github_star_prompt_runtime_exposes_lifecycle_inventory() -> None:
    runtime = GithubStarPromptRuntime()

    snapshot = runtime.lifecycle_owner_snapshot()

    assert_lifecycle_structure(snapshot)
    assert snapshot["owner"] == "GithubStarPromptRuntime"
    assert len(snapshot["resource_fields"]) == 3


@pytest.mark.asyncio
async def test_github_star_prompt_runtime_idle_close_clears_closing_state() -> None:
    runtime = GithubStarPromptRuntime()

    runtime.stop_ingress()
    assert runtime.is_closed is True
    assert runtime.is_closing is True

    await runtime.close()
    assert runtime.is_closed is True
    assert runtime.is_closing is False

    await runtime.close()
    assert runtime.is_closed is True
    assert runtime.is_closing is False


@pytest.mark.asyncio
async def test_github_star_prompt_runtime_close_cancels_and_gathers_launch_timer() -> None:
    runtime = GithubStarPromptRuntime(cancel_timeout_s=0.01)
    prompt_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    opened: list[str] = []

    async def run_prompt(generation: int) -> bool:
        assert runtime.is_current_generation(generation)
        prompt_started.set()
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            cancellation_seen.set()
            raise
        opened.append("opened")
        return True

    task = runtime.start_launch_prompt(run_prompt)
    await prompt_started.wait()

    await runtime.close()

    assert cancellation_seen.is_set() is True
    assert task.done() is True
    assert runtime.launch_prompt_task is None
    assert runtime.is_closed is True
    assert opened == []


@pytest.mark.asyncio
async def test_github_star_prompt_runtime_stops_ingress_before_async_close() -> None:
    runtime = GithubStarPromptRuntime()
    started = asyncio.Event()

    async def observe_success() -> bool:
        started.set()
        await asyncio.sleep(999)
        return True

    task = runtime.start_translation_success_observation(observe_success())
    await started.wait()

    runtime.stop_ingress()

    assert runtime.is_closed is True
    assert task.cancelling() > 0
    rejected = observe_success()
    with pytest.raises(RuntimeError, match="clos"):
        runtime.start_translation_success_observation(rejected)

    await runtime.close()
    assert task.cancelled() is True


@pytest.mark.asyncio
async def test_github_star_prompt_runtime_late_prompt_generation_is_not_current_after_close() -> (
    None
):
    runtime = GithubStarPromptRuntime(cancel_timeout_s=0.01)
    generation_seen: list[int] = []
    started = asyncio.Event()

    async def run_prompt(generation: int) -> bool:
        generation_seen.append(generation)
        started.set()
        await asyncio.sleep(999)
        return True

    runtime.start_launch_prompt(run_prompt)
    await started.wait()
    generation = generation_seen[0]

    await runtime.close()

    assert runtime.is_current_generation(generation) is False


@pytest.mark.asyncio
async def test_github_star_prompt_runtime_translation_success_task_clears_when_done() -> None:
    runtime = GithubStarPromptRuntime()

    async def observe_success() -> bool:
        return True

    task = runtime.start_translation_success_observation(observe_success())
    assert runtime.translation_success_task is task

    assert await task is True
    await _flush_loop()

    assert runtime.translation_success_task is None


@pytest.mark.asyncio
async def test_github_star_prompt_runtime_close_timeout_keeps_translation_task_for_retry() -> None:
    runtime = GithubStarPromptRuntime(cancel_timeout_s=0.01)
    started = asyncio.Event()
    release = asyncio.Event()
    cancellations: list[str] = []

    async def suppress_cancellation() -> bool:
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellations.append("cancelled")
        return True

    task = runtime.start_translation_success_observation(suppress_cancellation())
    await started.wait()

    try:
        with pytest.raises(TimeoutError, match="translation_success"):
            await asyncio.wait_for(runtime.close(), timeout=0.2)

        assert cancellations == ["cancelled"]
        assert task.done() is False
        assert runtime.translation_success_task is task

        with pytest.raises(TimeoutError, match="translation_success"):
            await asyncio.wait_for(runtime.close(), timeout=0.2)

        assert runtime.translation_success_task is task
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=0.2)
        await _flush_loop()

    assert runtime.translation_success_task is None


@pytest.mark.asyncio
async def test_github_star_prompt_runtime_state_callback_error_does_not_abort_close_cleanup() -> (
    None
):
    callback_calls = 0

    def state_changed(_runtime: GithubStarPromptRuntime) -> None:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls > 1:
            raise RuntimeError("state callback failed")

    runtime = GithubStarPromptRuntime(cancel_timeout_s=0.01, state_changed=state_changed)
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()

    async def run_prompt(_generation: int) -> bool:
        started.set()
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            cancellation_seen.set()
            raise
        return True

    task = runtime.start_launch_prompt(run_prompt)
    await started.wait()

    try:
        await runtime.close()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert cancellation_seen.is_set() is True
    assert task.done() is True
    assert runtime.launch_prompt_task is None


@pytest.mark.asyncio
async def test_github_star_prompt_runtime_rejects_duplicate_active_launch_schedule() -> None:
    runtime = GithubStarPromptRuntime(cancel_timeout_s=0.01)
    started = asyncio.Event()

    async def run_prompt(_generation: int) -> bool:
        started.set()
        await asyncio.sleep(999)
        return True

    runtime.start_launch_prompt(run_prompt)
    await started.wait()

    with pytest.raises(RuntimeError, match="launch prompt task"):
        runtime.start_launch_prompt(run_prompt)

    await runtime.close()
