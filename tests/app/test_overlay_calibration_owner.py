from __future__ import annotations

import logging

import pytest

from puripuly_heart.app.services.overlay_calibration import OverlayCalibrationOwner
from puripuly_heart.config.overlay_calibration import OverlayCalibration


def _owner(
    *,
    can_persist=lambda: True,
    can_emit=lambda: True,
) -> tuple[
    OverlayCalibrationOwner,
    list[object],
    list[OverlayCalibration],
    list[OverlayCalibration],
    list[tuple[str, dict[str, object]]],
]:
    tasks: list[object] = []
    persisted: list[OverlayCalibration] = []
    emitted: list[OverlayCalibration] = []
    logs: list[tuple[str, dict[str, object]]] = []

    def schedule(task) -> bool:
        tasks.append(task)
        return True

    async def persist(calibration: OverlayCalibration) -> None:
        persisted.append(calibration)

    async def emit(calibration: OverlayCalibration) -> None:
        emitted.append(calibration)

    owner = OverlayCalibrationOwner(
        schedule_task=schedule,
        persist=persist,
        emit=emit,
        can_persist=can_persist,
        can_emit=can_emit,
        log_detailed=lambda message, **kwargs: logs.append((message, kwargs)),
    )
    return owner, tasks, persisted, emitted, logs


@pytest.mark.asyncio
async def test_owner_commits_draft_then_schedules_persistence_and_runtime_emission() -> None:
    owner, tasks, persisted, emitted, _ = _owner()

    owner.begin()
    updated = owner.set_field("distance", 1.2)
    applied = owner.apply()

    assert updated.distance == 1.2
    assert applied.distance == 1.2
    assert owner.current.distance == 1.2
    assert owner.draft is None
    assert len(tasks) == 2

    for task in tasks:
        await task()

    assert [value.distance for value in persisted] == [1.2]
    assert [value.distance for value in emitted] == [1.2]


def test_owner_syncs_committed_value_without_clobbering_open_draft() -> None:
    owner, _, _, _, _ = _owner()
    owner.replace_current(OverlayCalibration(distance=0.9))
    owner.begin()
    owner.set_field("distance", 1.2)

    owner.replace_current(OverlayCalibration(distance=0.8))

    assert owner.current.distance == 0.8
    assert owner.begin().distance == 1.2
    assert owner.cancel().distance == 0.8


def test_owner_validates_fields_before_committing() -> None:
    owner, tasks, _, _, _ = _owner()

    with pytest.raises(ValueError, match="unknown overlay calibration field"):
        owner.set_field("missing", 1)
    with pytest.raises(ValueError, match="distance must be > 0"):
        owner.set_field("distance", 0)

    assert tasks == []


def test_owner_reports_unavailable_or_failed_ui_task_scheduler() -> None:
    owner, _, _, _, logs = _owner(can_emit=lambda: False)
    owner.schedule_task = lambda _task: False

    owner.schedule_persistence(OverlayCalibration())

    assert logs == [
        (
            "[Overlay] Calibration persistence skipped reason=page_run_task_unavailable",
            {"level": logging.WARNING},
        )
    ]

    error = RuntimeError("boom")

    def fail(_task) -> bool:
        raise error

    owner.schedule_task = fail
    owner.can_emit = lambda: True
    owner.schedule_emit()

    assert logs[-1][0] == ("[Overlay] Failed to schedule calibration update via page.run_task")
    assert logs[-1][1] == {"level": logging.WARNING, "exception": error}


@pytest.mark.asyncio
async def test_owner_rechecks_runtime_availability_before_emitting_scheduled_value() -> None:
    available = True
    owner, tasks, _, emitted, _ = _owner(can_emit=lambda: available)

    owner.schedule_emit()
    available = False
    await tasks[0]()

    assert emitted == []
