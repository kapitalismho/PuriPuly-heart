from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.core.runtime.overlay_session_fallback import (
    OverlaySessionFallbackOwner,
)


def _owner(
    *,
    can_start=lambda: True,
    start_overlay=None,
) -> tuple[
    OverlaySessionFallbackOwner,
    list[bool],
    list[tuple[str, dict[str, object], BaseException | None]],
]:
    notices: list[bool] = []
    diagnostics: list[tuple[str, dict[str, object], BaseException | None]] = []

    async def default_start() -> None:
        return None

    owner = OverlaySessionFallbackOwner(
        can_start=can_start,
        start_overlay=start_overlay or default_start,
        publish_notice=notices.append,
        diagnostics_sink=lambda event, metadata, exception: diagnostics.append(
            (event, dict(metadata), exception)
        ),
    )
    return owner, notices, diagnostics


def test_owner_applies_session_fallback_policy_without_changing_configured_target() -> None:
    owner, _, _ = _owner()

    assert owner.should_fallback(
        reason="steamvr_not_running",
        active_target=None,
        configured_enabled=True,
        configured_target="steamvr",
        desktop_target="desktop",
        steamvr_target="steamvr",
    )
    assert not owner.should_fallback(
        reason="runtime_crashed",
        active_target=None,
        configured_enabled=True,
        configured_target="steamvr",
        desktop_target="desktop",
        steamvr_target="steamvr",
    )

    owner.activate()

    assert not owner.should_fallback(
        reason="steamvr_not_running",
        active_target=None,
        configured_enabled=True,
        configured_target="steamvr",
        desktop_target="desktop",
        steamvr_target="steamvr",
    )


@pytest.mark.asyncio
async def test_owner_schedules_one_generation_checked_fallback_start() -> None:
    starts: list[str] = []

    async def start() -> None:
        starts.append("desktop")

    owner, notices, _ = _owner(start_overlay=start)
    owner.activate()
    owner.publish(True)
    owner.schedule()
    task = owner.task

    assert task is not None
    assert task.get_name() == "OverlaySessionFallbackOwner:overlay-session-desktop-fallback-1"
    await task

    assert starts == ["desktop"]
    assert notices == [True]
    assert owner.task is None


@pytest.mark.asyncio
async def test_owner_clear_cancels_stale_start_and_notice() -> None:
    starts: list[str] = []

    async def start() -> None:
        starts.append("desktop")

    owner, notices, _ = _owner(start_overlay=start)
    owner.activate()
    owner.publish(True)
    owner.schedule()

    owner.clear()
    await asyncio.sleep(0)

    assert starts == []
    assert owner.active is False
    assert notices == [True, False]


@pytest.mark.asyncio
async def test_owner_reports_task_factory_failure_and_closes_coroutine() -> None:
    error = RuntimeError("closed")
    owner, _, diagnostics = _owner()

    def fail(_coroutine, _name):
        raise error

    owner.task_factory = fail
    owner.activate()
    owner.schedule()

    assert owner.task is None
    assert diagnostics == [
        (
            "overlay_session_fallback_schedule_failed",
            {"error_type": "RuntimeError"},
            error,
        )
    ]


@pytest.mark.asyncio
async def test_owner_close_cancels_task_and_clears_session_state() -> None:
    release = asyncio.Event()

    async def start() -> None:
        await release.wait()

    owner, notices, _ = _owner(start_overlay=start)
    owner.activate()
    owner.publish(True)
    owner.schedule()
    await asyncio.sleep(0)

    await owner.close()

    assert owner.task is None
    assert owner.active is False
    assert notices == [True, False]

    owner.activate()
    owner.schedule()

    assert owner.accepting_ingress is False
    assert owner.active is False
    assert owner.task is None


@pytest.mark.asyncio
async def test_owner_clear_invalidates_pending_start_without_freezing_ingress() -> None:
    starts: list[str] = []

    async def start() -> None:
        starts.append("desktop")

    owner, _, _ = _owner(start_overlay=start)
    owner.activate()
    owner.schedule()
    owner.clear()
    owner.activate()
    owner.schedule()
    task = owner.task

    assert owner.accepting_ingress is True
    assert task is not None
    await task
    assert starts == ["desktop"]
