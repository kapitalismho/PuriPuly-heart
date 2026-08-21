from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from puripuly_heart.app.ports.managed_gemma_translation import (
    ManagedGemmaTranslationSelection,
)
from puripuly_heart.app.services.managed_gemma_translation import (
    ManagedGemmaTranslationOwner,
)
from puripuly_heart.core.local_translation.provisioning import (
    GemmaProvisioningCancelled,
    GemmaProvisioningUpdate,
)
from puripuly_heart.core.local_translation.runtime import ManagedGemmaReadiness


def _selection(backend: str = "cpu") -> ManagedGemmaTranslationSelection:
    return ManagedGemmaTranslationSelection(
        backend=backend,
        source_language="ko",
        target_language="en",
        system_prompt="translate {source_language} to {target_language}",
    )


def _readiness(backend: str) -> ManagedGemmaReadiness:
    effective = "vulkan" if backend == "gpu" else "cpu"
    return ManagedGemmaReadiness(
        requested_backend=backend,
        effective_backend=effective,
        source_language="ko",
        target_language="en",
        prefix_identity=f"prefix-{backend}",
    )


class RecordingRuntime:
    def __init__(self, prepare: Callable[..., Any]) -> None:
        self._prepare = prepare
        self.calls: list[dict[str, object]] = []
        self.release_count = 0
        self.close_count = 0

    async def prepare(self, **kwargs: object) -> ManagedGemmaReadiness:
        self.calls.append(kwargs)
        result = self._prepare(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def release(self) -> None:
        self.release_count += 1

    async def close(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_prepare_reports_progress_and_gates_activation_on_runtime_readiness() -> None:
    ready = asyncio.Event()
    progress_emitted = asyncio.Event()
    statuses = []

    async def prepare(**kwargs: object) -> ManagedGemmaReadiness:
        callback = kwargs["provision_kwargs"]["on_status"]
        callback(GemmaProvisioningUpdate("downloading", 25, 100, 25))
        progress_emitted.set()
        await ready.wait()
        callback(GemmaProvisioningUpdate("ready", 100, 100, 100))
        return _readiness("gpu")

    runtime = RecordingRuntime(prepare)
    owner = ManagedGemmaTranslationOwner(runtime=runtime, status_sink=statuses.append)

    task = asyncio.create_task(owner.prepare(_selection("gpu")))
    await progress_emitted.wait()

    assert task.done() is False
    assert [snapshot.state for snapshot in statuses] == ["checking", "downloading"]
    assert statuses[-1].progress_percent == 25

    ready.set()
    activation = await task

    assert activation.backend == "gpu"
    assert activation.readiness.effective_backend == "vulkan"
    assert [snapshot.state for snapshot in statuses] == [
        "checking",
        "downloading",
        "preparing",
        "ready",
    ]
    assert runtime.calls[0]["backend"] == "gpu"
    assert runtime.calls[0]["source_language"] == "ko"
    assert runtime.calls[0]["target_language"] == "en"


@pytest.mark.asyncio
async def test_cancel_interrupts_prepare_and_exposes_retryable_cancelled_state() -> None:
    started = asyncio.Event()

    async def prepare(**_kwargs: object) -> ManagedGemmaReadiness:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    runtime = RecordingRuntime(prepare)
    owner = ManagedGemmaTranslationOwner(runtime=runtime)
    task = asyncio.create_task(owner.prepare(_selection()))
    await started.wait()

    assert owner.cancel() is True
    with pytest.raises(GemmaProvisioningCancelled):
        await task

    assert owner.snapshot.state == "cancelled"
    assert owner.cancel() is False


@pytest.mark.asyncio
async def test_failed_prepare_can_be_retried_and_activated() -> None:
    attempts = 0

    def prepare(**kwargs: object) -> ManagedGemmaReadiness:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("download failed")
        return _readiness(str(kwargs["backend"]))

    runtime = RecordingRuntime(prepare)
    owner = ManagedGemmaTranslationOwner(runtime=runtime)

    with pytest.raises(RuntimeError, match="download failed"):
        await owner.prepare(_selection())
    assert owner.snapshot.state == "failed"

    activation = await owner.prepare(_selection())

    assert activation.backend == "cpu"
    assert owner.snapshot.state == "ready"
    assert attempts == 2


@pytest.mark.asyncio
async def test_failed_prepare_diagnostic_does_not_prevent_terminal_runtime_close() -> None:
    diagnostics = []

    def prepare(**_kwargs: object) -> ManagedGemmaReadiness:
        raise RuntimeError("download failed")

    runtime = RecordingRuntime(prepare)
    owner = ManagedGemmaTranslationOwner(
        runtime=runtime,
        lifecycle_diagnostic_sink=diagnostics.append,
    )

    with pytest.raises(RuntimeError, match="download failed"):
        await owner.prepare(_selection())

    await owner.close()

    assert runtime.close_count == 1
    assert len(diagnostics) == 1
    assert diagnostics[0].fields["exception_class"] == "RuntimeError"


@pytest.mark.asyncio
async def test_lifecycle_diagnostic_sink_failure_still_closes_runtime() -> None:
    def prepare(**_kwargs: object) -> ManagedGemmaReadiness:
        raise RuntimeError("download failed")

    def fail_diagnostic(_event: object) -> None:
        raise RuntimeError("diagnostic sink failed")

    runtime = RecordingRuntime(prepare)
    owner = ManagedGemmaTranslationOwner(
        runtime=runtime,
        lifecycle_diagnostic_sink=fail_diagnostic,
    )

    with pytest.raises(RuntimeError, match="download failed"):
        await owner.prepare(_selection())
    with pytest.raises(RuntimeError, match="diagnostic sink failed"):
        await owner.close()

    assert runtime.close_count == 1


@pytest.mark.asyncio
async def test_retired_activation_cannot_release_newer_shared_runtime() -> None:
    runtime = RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"])))
    owner = ManagedGemmaTranslationOwner(runtime=runtime)

    first = await owner.prepare(_selection("cpu"))
    second = await owner.prepare(_selection("gpu"))

    await first.release()
    assert runtime.release_count == 0

    await second.release()
    assert runtime.release_count == 1
    assert owner.snapshot.state == "idle"


@pytest.mark.asyncio
async def test_linger_deactivate_keeps_runtime_until_timer_elapses() -> None:
    runtime = RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"])))
    owner = ManagedGemmaTranslationOwner(runtime=runtime, idle_linger_s=0.05)
    await owner.prepare(_selection())

    await owner.deactivate(linger=True)

    assert runtime.release_count == 0
    assert owner.snapshot.state == "idle"
    await asyncio.sleep(0.12)
    assert runtime.release_count == 1


@pytest.mark.asyncio
async def test_prepare_during_linger_cancels_shutdown_and_reuses_runtime() -> None:
    runtime = RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"])))
    owner = ManagedGemmaTranslationOwner(runtime=runtime, idle_linger_s=0.2)
    await owner.prepare(_selection())
    await owner.deactivate(linger=True)

    await owner.prepare(_selection())

    assert runtime.release_count == 0
    assert owner.snapshot.state == "ready"
    await asyncio.sleep(0.25)
    assert runtime.release_count == 0


@pytest.mark.asyncio
async def test_immediate_deactivate_and_close_skip_linger() -> None:
    runtime = RecordingRuntime(lambda **kwargs: _readiness(str(kwargs["backend"])))
    owner = ManagedGemmaTranslationOwner(runtime=runtime, idle_linger_s=1.0)
    await owner.prepare(_selection())

    await owner.deactivate()
    assert runtime.release_count == 1

    await owner.prepare(_selection())
    await owner.deactivate(linger=True)
    await owner.close()

    assert runtime.release_count == 1
    assert runtime.close_count == 1
    assert owner.snapshot.state == "closed"
