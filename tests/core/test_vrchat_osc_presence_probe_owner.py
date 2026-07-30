from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.core.runtime.vrchat_osc_presence import (
    VrchatOscPresenceProbeOwner,
)


class PresencePort:
    def __init__(self, results: list[bool | None | BaseException]) -> None:
        self.results = list(results)
        self.ports: list[int] = []

    async def should_prompt_enable_osc(self, *, port: int) -> bool | None:
        self.ports.append(port)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _owner(
    presence: object | None,
    *,
    interval_seconds: float = 60,
) -> tuple[
    VrchatOscPresenceProbeOwner,
    list[bool],
    list[tuple[str, dict[str, object], BaseException | None]],
]:
    notices: list[bool] = []
    diagnostics: list[tuple[str, dict[str, object], BaseException | None]] = []
    owner = VrchatOscPresenceProbeOwner(
        presence_provider=lambda: presence,
        port_provider=lambda: 9001,
        publish_notice=notices.append,
        diagnostics_sink=lambda event, metadata, exception: diagnostics.append(
            (event, dict(metadata), exception)
        ),
        interval_seconds=interval_seconds,
    )
    return owner, notices, diagnostics


@pytest.mark.asyncio
async def test_owner_publishes_probe_result_and_cancels_owned_task() -> None:
    presence = PresencePort([True])
    owner, notices, _ = _owner(presence)

    owner.schedule(force=True)
    for _ in range(20):
        if notices:
            break
        await asyncio.sleep(0)

    assert presence.ports == [9001]
    assert notices == [True]
    assert owner.task is not None
    assert owner.task.get_name() == "VrchatOscPresenceProbeOwner:vrchat-osc-presence-1"

    await owner.cancel()

    assert owner.task is None
    assert owner.notice_active is False
    assert notices == [True, False]


@pytest.mark.asyncio
async def test_owner_generation_drops_stale_result() -> None:
    release = asyncio.Event()

    class BlockingPresence:
        async def should_prompt_enable_osc(self, *, port: int) -> bool:
            _ = port
            await release.wait()
            return True

    owner, notices, _ = _owner(BlockingPresence())
    generation = owner.generation
    task = asyncio.create_task(owner.run(generation))
    await asyncio.sleep(0)

    owner.stop_ingress()
    release.set()
    await task

    assert notices == []


@pytest.mark.asyncio
async def test_owner_contains_probe_failure_and_reports_diagnostics() -> None:
    error = RuntimeError("boom")
    owner, notices, diagnostics = _owner(
        PresencePort([error]),
        interval_seconds=0.01,
    )
    generation = owner.generation
    task = asyncio.create_task(owner.run(generation))
    for _ in range(20):
        if diagnostics:
            break
        await asyncio.sleep(0)
    owner.stop_ingress()
    await task

    assert notices == []
    assert diagnostics == [
        (
            "vrchat_osc_presence_probe_failed",
            {"error_type": "RuntimeError"},
            error,
        )
    ]


@pytest.mark.asyncio
async def test_owner_without_presence_port_clears_existing_notice() -> None:
    owner, notices, _ = _owner(None)
    owner.publish(True)

    owner.schedule()

    assert owner.task is None
    assert notices == [True, False]


@pytest.mark.asyncio
async def test_owner_rejects_new_probes_after_ingress_stop_and_cancel() -> None:
    presence = PresencePort([True])
    owner, notices, _ = _owner(presence)

    owner.stop_ingress()
    owner.schedule()

    assert owner.accepting_ingress is False
    assert owner.task is None
    assert presence.ports == []
    assert notices == []

    await owner.cancel()
    owner.schedule()

    assert owner.task is None
    assert presence.ports == []
