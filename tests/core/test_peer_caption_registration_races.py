from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.runtime.output import OutputRuntime
from tests.core.runtime.test_output_runtime import RecordingChatbox, _publish_peer_overlay
from tests.core.test_peer_caption_fragments import fragment_id, identity, own_speech
from tests.core.test_peer_overlay_policy import Surface, Timeline, caption


class HoldingSurface(Surface):
    def __init__(self):
        super().__init__()
        self.hold = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def replace_snapshot(self, snapshot, *, block_expirations=None):
        if self.hold:
            self.started.set()
            await self.release.wait()
        await super().replace_snapshot(snapshot, block_expirations=block_expirations)


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", ["retirement", "revocation"])
async def test_registration_rechecks_authority_after_waiting_for_scene_transport_lock(fence):
    timeline, surface = Timeline(), HoldingSurface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    unit = identity(4070)
    authorized = True
    surface.hold = True
    scene_task = asyncio.create_task(presenter.emit(own_speech()))
    pending = None
    try:
        await asyncio.wait_for(surface.started.wait(), 0.5)
        event = caption(1, unit, "가" * 64)
        pending = asyncio.create_task(
            presenter.register_peer_caption(
                event,
                publication_generation=1,
                is_current=lambda: authorized,
            )
        )
        await timeline.advance(0)
        assert not pending.done()
        if fence == "retirement":
            presenter.retire_peer_generation(1)
        else:
            authorized = False
        surface.release.set()
        await asyncio.wait_for(scene_task, 0.5)
        receipt = await asyncio.wait_for(pending, 0.5)
        assert receipt.outcome == "not_applied" and receipt.cause == "publication_revoked"
        await timeline.advance(1)
        assert presenter.peer_fragment_states(unit) == ()
        assert all(block.channel == "self" for block in presenter.snapshot().blocks)
        assert presenter.snapshot().native_quiet_tail_episodes.peer is None
    finally:
        surface.release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await asyncio.gather(scene_task, return_exceptions=True)
        await presenter.close()


@pytest.mark.asyncio
async def test_retirement_during_root_transport_wait_cannot_later_admit_pending_tail():
    timeline, surface = Timeline(), HoldingSurface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    output = OutputRuntime(
        chatbox=RecordingChatbox(),
        clock=timeline.clock,
        overlay_sink=presenter,
    )
    unit = identity(4170)
    event = caption(1, unit, "가" * 64)
    try:
        await presenter.emit(own_speech())
        surface.hold = True
        await _publish_peer_overlay(output, event, source_order=1)
        await asyncio.wait_for(surface.started.wait(), 0.5)
        assert presenter.peer_fragment_state(unit, 1).pending
        receipt = presenter.application_receipt(event.event_id)
        assert receipt.outcome == "applied"
        episode = presenter.snapshot().native_quiet_tail_episodes.peer
        await timeline.advance(0.2)
        output.retire_peer_generation(1)
        await asyncio.wait_for(output.wait_for_peer_output_idle(), 0.5)
        surface.release.set()
        await timeline.advance(0.8)
        assert presenter.peer_fragment_state(unit, 1) is None
        assert all(block.id != fragment_id(unit, 1) for block in presenter.snapshot().blocks)
        assert presenter.snapshot().native_quiet_tail_episodes.peer == episode
        assert presenter.application_receipt(event.event_id) is receipt
        late = await presenter.register_peer_caption(
            replace(event, event_id="retired-revision", seq=2, peer_caption_revision=2),
            publication_generation=1,
        )
        assert late.outcome == "not_applied" and late.cause == "publication_revoked"
        assert presenter.snapshot().native_quiet_tail_episodes.peer == episode
    finally:
        surface.release.set()
        await output.close()
        await presenter.close()
