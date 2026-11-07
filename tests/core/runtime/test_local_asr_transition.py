from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.runtime import local_asr_transition
from puripuly_heart.core.runtime.local_asr_transition import (
    LocalASRSessionOptions,
    LocalASRTransitionCoordinator,
    LocalASRTransitionRequest,
    PreparedLocalASRTransition,
)


@dataclass
class Candidate:
    name: str
    close_calls: int = 0

    async def close_backend(self) -> None:
        self.close_calls += 1


def request(model_id: str) -> LocalASRTransitionRequest:
    return LocalASRTransitionRequest(
        channel="self",
        requested_provider="local_cpu_auto",
        actual_provider="local_cpu_auto",
        model_id=model_id,
        session_options=LocalASRSessionOptions(source_language="en"),
        trigger="settings",
    )


@pytest.mark.parametrize(
    ("admission", "first_status", "expected_models"),
    [
        ("during_stabilization", "superseded", ["qwen"]),
        ("after_first_commit", "applied", ["parakeet-v3", "qwen"]),
    ],
)
@pytest.mark.asyncio
async def test_stabilization_respects_second_request_admission_boundary(
    monkeypatch: pytest.MonkeyPatch,
    admission: str,
    first_status: str,
    expected_models: list[str],
) -> None:
    clock = FakeClock()
    stabilization_entered = asyncio.Event()
    expire = asyncio.Event()

    async def controlled_expiry(waiter, *, timeout: float) -> None:
        stabilization_entered.set()
        try:
            await expire.wait()
            clock.advance(timeout)
        finally:
            waiter.close()
        raise TimeoutError

    controlled_asyncio = SimpleNamespace(**vars(asyncio))
    controlled_asyncio.wait_for = controlled_expiry
    monkeypatch.setattr(local_asr_transition, "asyncio", controlled_asyncio)
    coordinator = LocalASRTransitionCoordinator(
        channel="self", stabilization_s=0.02, clock=clock.now
    )
    prepared_models: list[str] = []
    committed_models: list[str] = []

    async def prepare(
        value: LocalASRTransitionRequest,
        generation: int,
    ) -> PreparedLocalASRTransition:
        prepared_models.append(value.model_id)
        return PreparedLocalASRTransition(
            request=value,
            provider=Candidate(str(value.model_id)),
            generation=generation,
        )

    async def commit(value: PreparedLocalASRTransition) -> None:
        committed_models.append(str(value.request.model_id))

    first = asyncio.create_task(
        coordinator.request_transition(request("parakeet-v3"), prepare=prepare, commit=commit)
    )
    try:
        await stabilization_entered.wait()
        assert prepared_models == committed_models == []
        if admission == "during_stabilization":
            second = asyncio.create_task(
                coordinator.request_transition(request("qwen"), prepare=prepare, commit=commit)
            )
            first_outcome = await first
            assert prepared_models == committed_models == []
            expire.set()
            second_outcome = await second
        else:
            expire.set()
            first_outcome = await first
            assert committed_models == ["parakeet-v3"]
            second_outcome = await coordinator.request_transition(
                request("qwen"), prepare=prepare, commit=commit
            )

        assert first_outcome.status == first_status
        assert second_outcome.status == "applied"
        assert prepared_models == expected_models
        assert committed_models == expected_models
    finally:
        expire.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_native_load_finishes_then_superseded_candidate_is_discarded() -> None:
    coordinator = LocalASRTransitionCoordinator(channel="self", stabilization_s=0)
    entered = asyncio.Event()
    release = asyncio.Event()
    candidates: dict[str, Candidate] = {}

    async def prepare(
        value: LocalASRTransitionRequest,
        generation: int,
    ) -> PreparedLocalASRTransition:
        candidate = Candidate(str(value.model_id))
        candidates[str(value.model_id)] = candidate
        if value.model_id == "parakeet-v3":
            entered.set()
            await release.wait()
        return PreparedLocalASRTransition(
            request=value,
            provider=candidate,
            generation=generation,
        )

    async def commit(_value: PreparedLocalASRTransition) -> None:
        return None

    first = asyncio.create_task(
        coordinator.request_transition(request("parakeet-v3"), prepare=prepare, commit=commit)
    )
    await entered.wait()
    second = asyncio.create_task(
        coordinator.request_transition(request("qwen"), prepare=prepare, commit=commit)
    )
    release.set()
    first_outcome, second_outcome = await asyncio.gather(first, second)
    await asyncio.sleep(0)

    assert first_outcome.status == "superseded"
    assert second_outcome.status == "applied"
    assert candidates["parakeet-v3"].close_calls == 1
    assert candidates["qwen"].close_calls == 0
    snapshot = coordinator.lifecycle_snapshot()
    assert snapshot["phase"] == "idle"
    assert snapshot["active_generation"] == second_outcome.generation
    assert snapshot["temporary_candidate_count"] == 0
    await coordinator.close()


@pytest.mark.asyncio
async def test_failed_prepare_diagnostic_preserves_failure_type() -> None:
    diagnostics: list[dict[str, object]] = []
    coordinator = LocalASRTransitionCoordinator(
        channel="self",
        stabilization_s=0,
        diagnostic_sink=diagnostics.append,
    )

    async def prepare(
        _value: LocalASRTransitionRequest,
        _generation: int,
    ) -> PreparedLocalASRTransition:
        raise RuntimeError("private model path")

    async def commit(_value: PreparedLocalASRTransition) -> None:
        raise AssertionError("commit must not run")

    outcome = await coordinator.request_transition(
        request("qwen"),
        prepare=prepare,
        commit=commit,
    )

    assert outcome.status == "failed"
    assert diagnostics[-1]["outcome"] == "failed"
    assert diagnostics[-1]["failure_type"] == "RuntimeError"
    assert "private model path" not in str(diagnostics[-1])
    await coordinator.close()
