from __future__ import annotations

import pytest

from puripuly_heart.core.runtime.desktop_overlay_bounds import (
    DesktopOverlayBoundsOwner,
)


def _bounds(x: int = 100) -> dict[str, int | float]:
    return {"x": x, "y": 200, "width": 800, "height": 220}


def _owner(
    persisted: list[dict[str, int | float]],
    *,
    debounce_seconds: float = 0,
    diagnostics: list[tuple[str, dict[str, object]]] | None = None,
) -> DesktopOverlayBoundsOwner:
    async def persist(bounds: dict[str, int | float]) -> None:
        persisted.append(bounds)

    return DesktopOverlayBoundsOwner(
        persist_bounds=persist,
        debounce_seconds=lambda: debounce_seconds,
        minimum_width=640,
        minimum_height=160,
        diagnostics_sink=(
            None
            if diagnostics is None
            else lambda event, metadata: diagnostics.append((event, dict(metadata)))
        ),
    )


def test_owner_validates_bounds_event_contract_and_tracks_programmatic_echoes() -> None:
    owner = _owner([])
    payload = {
        "event": "window_bounds_changed",
        "source": "user",
        "persist": True,
        **_bounds(),
    }

    assert owner.is_valid_event_payload(payload) is True
    assert owner.bounds_from_payload({**payload, "width": 639}) is None
    assert owner.is_valid_event_payload({**payload, "persist": False}) is False
    assert owner.is_valid_event_payload({**payload, "x": True}) is False

    owner.track_apply_control({"command": "apply_window_bounds", **_bounds()})

    assert owner.consume_suppressed(_bounds()) is True
    assert owner.consume_suppressed(_bounds()) is False


@pytest.mark.asyncio
async def test_owner_debounces_to_latest_bounds_and_clears_task_state() -> None:
    persisted: list[dict[str, int | float]] = []
    owner = _owner(persisted, debounce_seconds=0.01)

    owner.schedule_persistence(_bounds(100))
    first = owner.persist_task
    owner.schedule_persistence(_bounds(300))
    second = owner.persist_task

    assert first is not None
    assert second is not None
    assert first is not second
    await second

    assert first.cancelled() is True
    assert persisted == [_bounds(300)]
    assert owner.pending_bounds is None
    assert owner.persist_task is None


@pytest.mark.asyncio
async def test_owner_cancel_gathers_task_and_discards_pending_bounds() -> None:
    owner = _owner([], debounce_seconds=60)
    owner.schedule_persistence(_bounds())
    task = owner.persist_task

    await owner.cancel()

    assert task is not None
    assert task.cancelled() is True
    assert owner.persist_task is None
    assert owner.pending_bounds is None


@pytest.mark.asyncio
async def test_owner_reports_persistence_failure_without_leaking_task_state() -> None:
    diagnostics: list[tuple[str, dict[str, object]]] = []

    async def fail(_bounds: dict[str, int | float]) -> None:
        raise RuntimeError("boom")

    owner = DesktopOverlayBoundsOwner(
        persist_bounds=fail,
        debounce_seconds=lambda: 0,
        minimum_width=640,
        minimum_height=160,
        diagnostics_sink=lambda event, metadata: diagnostics.append((event, dict(metadata))),
    )
    owner.schedule_persistence(_bounds())
    task = owner.persist_task

    assert task is not None
    await task

    assert diagnostics == [
        (
            "desktop_overlay_bounds_persistence_failed",
            {"error_type": "RuntimeError"},
        )
    ]
    assert owner.persist_task is None
    assert owner.pending_bounds is None


def test_owner_discard_cancels_without_requiring_running_loop() -> None:
    owner = _owner([])
    owner.replace_pending_bounds(_bounds())

    owner.discard()

    assert owner.pending_bounds is None
