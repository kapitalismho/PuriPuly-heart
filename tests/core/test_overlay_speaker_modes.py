from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.domain.models import Transcript
from puripuly_heart.ui.overlay_calibration import OverlayCalibration


def _peer_event(
    adapter: OverlayEventAdapter,
    turn_id: UUID,
    text: str,
    comparison: str,
):
    return adapter.translation_final(
        utterance_id=turn_id,
        channel="peer",
        text=text,
        source_text=text,
        source_language="en",
        target_language="ko",
        applied_context_mode="integrated",
        logical_turn_key=f"peer:{turn_id}",
        speaker_transition=comparison,
        speaker_transition_claim_id=f"peer:{turn_id}",
    )


def _self_event(adapter: OverlayEventAdapter, turn_id: UUID, text: str):

    return adapter.transcript_final(
        Transcript(
            utterance_id=turn_id,
            text=text,
            is_final=True,
            channel="self",
        ),
        source_language="en",
        target_language="ko",
        logical_turn_key=f"self:{turn_id}",
    )


@pytest.mark.asyncio
async def test_mode_c_assigns_stable_run_colors_once_across_revisions_and_uncertainty() -> None:
    clock = FakeClock(_now=10.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        speaker_transition_mode="C",
    )
    first, changed, uncertain = uuid4(), uuid4(), uuid4()

    await presenter.emit(_peer_event(adapter, first, "same", "context_reset"))
    await presenter.emit(_peer_event(adapter, changed, "same", "transition"))
    assert [(block.id, block.speaker_style) for block in presenter.snapshot().blocks] == [
        (f"peer:{first}", "gold"),
        (f"peer:{changed}", "cyan"),
    ]

    await presenter.emit(_peer_event(adapter, changed, "revised", "transition"))
    assert presenter.snapshot().blocks[-1].speaker_style == "cyan"

    await presenter.emit(_peer_event(adapter, uncertain, "new", "context_reset"))
    assert presenter.snapshot().blocks[-1].speaker_style == "cyan"
    assert all(not block.speaker_boundary for block in presenter.snapshot().blocks)


@pytest.mark.asyncio
async def test_late_transition_claim_on_readable_revision_is_withheld_without_replay() -> None:
    clock = FakeClock(_now=15.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        speaker_transition_mode="C",
    )
    turn_id = uuid4()

    await presenter.emit(_peer_event(adapter, turn_id, "first", "unavailable"))
    entry = presenter._entries[("peer", turn_id)]
    visible_since = entry.visible_since
    await presenter.emit(_peer_event(adapter, turn_id, "revision", "transition"))

    block = presenter.snapshot().blocks[-1]
    assert block.speaker_style == "gold"
    assert block.speaker_boundary is False
    assert entry.speaker_transition == "unavailable"
    assert entry.visible_since == visible_since


@pytest.mark.asyncio
async def test_mode_e_expires_on_next_distinct_readable_turn_not_revision() -> None:
    clock = FakeClock(_now=20.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        speaker_transition_mode="E",
    )
    changed, self_turn = uuid4(), uuid4()

    await presenter.emit(_peer_event(adapter, changed, "changed", "transition"))
    entry = presenter._entries[("peer", changed)]
    visible_since = entry.visible_since
    assert presenter.snapshot().blocks[-1].speaker_style == "cyan"
    assert presenter.snapshot().blocks[-1].speaker_boundary is True

    await presenter.emit(_peer_event(adapter, changed, "changed revision", "transition"))
    assert presenter.snapshot().blocks[-1].speaker_style == "cyan"
    assert entry.visible_since == visible_since

    await presenter.emit(_self_event(adapter, self_turn, "self"))
    peer = next(block for block in presenter.snapshot().blocks if block.id == f"peer:{changed}")
    assert peer.speaker_style == "gold"
    assert peer.speaker_boundary is True
    assert entry.visible_since == visible_since


@pytest.mark.asyncio
async def test_entering_mode_e_reprojects_boundary_without_replaying_emphasis() -> None:
    clock = FakeClock(_now=30.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    turn_id = uuid4()

    await presenter.emit(_peer_event(adapter, turn_id, "changed", "transition"))
    entry = presenter._entries[("peer", turn_id)]
    visible_since = entry.visible_since
    await presenter.update_speaker_transition_mode("E")

    block = presenter.snapshot().blocks[-1]
    assert block.speaker_boundary is True
    assert block.speaker_style == "gold"
    assert entry.visible_since == visible_since
