from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from uuid import UUID

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.domain.models import Transcript
from puripuly_heart.ui.overlay_calibration import OverlayCalibration


@dataclass(slots=True)
class RecordingTraceBridge:
    snapshots: list[object] = field(default_factory=list)

    async def replace_snapshot(self, snapshot: object) -> None:
        self.snapshots.append(snapshot)

    async def broadcast_shutdown(self) -> None:
        return None


async def _wait_for(predicate) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("trace condition did not become true")


def _snapshot_payloads(bridge: RecordingTraceBridge) -> list[dict[str, object]]:
    return [snapshot.to_dict() for snapshot in bridge.snapshots]


async def _natural_trace(channel: str, anchor: str) -> dict[str, object]:
    bridge = RecordingTraceBridge()
    clock = FakeClock(_now=10.0)
    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        if delay != 0.1:
            await asyncio.Event().wait()
            return
        sleep_delays.append(delay)
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(anchor=anchor),
        clock=clock,
        sleep=fake_sleep,
    )
    adapter = OverlayEventAdapter(clock=clock)
    turn_id = UUID(
        "11111111-1111-1111-1111-111111111111"
        if channel == "peer"
        else "22222222-2222-2222-2222-222222222222"
    )
    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=turn_id,
                    channel=channel,
                    text=f"{channel} source",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="en",
                target_language="ko",
            )
        )
        if channel == "peer":
            await presenter.emit(
                adapter.translation_final(
                    utterance_id=turn_id,
                    channel="peer",
                    text="peer translation",
                    source_language="en",
                    target_language="ko",
                    applied_context_mode=None,
                    created_at=10.1,
                )
            )
        task_name = f"_{channel}_presentation_refresh_burst_task"
        await _wait_for(lambda: getattr(presenter, task_name) is None)
        return {
            "sleep_delays": sleep_delays,
            "snapshots": _snapshot_payloads(bridge),
        }
    finally:
        await presenter.clear_for_runtime_detach()


async def _lifecycle_trace() -> dict[str, object]:
    bridge = RecordingTraceBridge()
    clock = FakeClock(_now=20.0)
    sleep_delays: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        if delay != 0.1:
            await asyncio.Event().wait()
            return
        sleep_delays.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(anchor="spatial_locked"),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    first = UUID("33333333-3333-3333-3333-333333333333")
    second = UUID("44444444-4444-4444-4444-444444444444")

    async def publish(turn_id: UUID, text: str, created_at: float) -> None:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=turn_id,
                    channel="self",
                    text=text,
                    is_final=True,
                    created_at=created_at,
                ),
                source_language="ko",
                target_language="en",
            )
        )

    try:
        await publish(first, "first self source", 20.0)
        await _wait_for(lambda: len(sleep_events) == 1)
        sleep_events[-1].set()
        await _wait_for(
            lambda: presenter.snapshot().blocks[0].session_scope == "self_presentation_refresh=1"
        )
        sleep_count_before_replacement = len(sleep_events)
        await publish(second, "second self source", 20.1)
        await _wait_for(lambda: len(sleep_events) > sleep_count_before_replacement)
        sleep_events[-1].set()
        await _wait_for(
            lambda: any(
                block.id == f"self:{second}"
                and block.session_scope == "self_presentation_refresh=1"
                for block in presenter.snapshot().blocks
            )
        )
        await presenter.update_calibration(OverlayCalibration(anchor="head_locked"))
        await presenter.update_calibration(
            OverlayCalibration(anchor="spatial_locked", offset_x=0.35)
        )
        await presenter.update_self_presentation_refresh_burst(False)
        return {
            "sleep_delays": sleep_delays,
            "snapshots": _snapshot_payloads(bridge),
        }
    finally:
        await presenter.clear_for_runtime_detach()


async def _ownership_trace() -> dict[str, object]:
    bridge = RecordingTraceBridge()
    clock = FakeClock(_now=30.0)
    sleep_delays: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        if delay != 0.1:
            await asyncio.Event().wait()
            return
        sleep_delays.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(anchor="spatial_locked"),
        clock=clock,
        sleep=fake_sleep,
    )
    adapter = OverlayEventAdapter(clock=clock)
    turn_id = UUID("55555555-5555-5555-5555-555555555555")
    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=turn_id,
                    channel="peer",
                    text="ownership peer source",
                    is_final=True,
                    created_at=30.0,
                ),
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=turn_id,
                channel="peer",
                text="ownership peer translation",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=30.1,
            )
        )
        await _wait_for(lambda: len(sleep_events) == 1)
        sleep_events[-1].set()
        await _wait_for(
            lambda: presenter.snapshot().blocks[0].session_scope == "peer_presentation_refresh=1"
        )
        await presenter.update_native_retry_ownership(True)
        sleep_count_before_release = len(sleep_events)
        await presenter.update_native_retry_ownership(False)
        await _wait_for(lambda: len(sleep_events) > sleep_count_before_release)
        sleep_events[-1].set()
        await _wait_for(
            lambda: presenter.snapshot().blocks[0].session_scope == "peer_presentation_refresh=1"
        )
        await presenter.update_peer_presentation_refresh_burst(False)
        await presenter.update_self_presentation_refresh_burst(False)
        return {
            "sleep_delays": sleep_delays,
            "snapshots": _snapshot_payloads(bridge),
        }
    finally:
        await presenter.clear_for_runtime_detach()


async def build_overlay_refresh_trace_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "traces": {
            "peer_head_natural": await _natural_trace("peer", "head_locked"),
            "peer_spatial_natural": await _natural_trace("peer", "spatial_locked"),
            "self_head_natural": await _natural_trace("self", "head_locked"),
            "self_spatial_natural": await _natural_trace("self", "spatial_locked"),
            "spatial_lifecycle": await _lifecycle_trace(),
            "spatial_ownership": await _ownership_trace(),
        },
    }


def main() -> None:
    print(
        json.dumps(
            asyncio.run(build_overlay_refresh_trace_contract()),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
