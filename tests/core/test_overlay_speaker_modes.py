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
        Transcript(turn_id, text, True, channel="self"),
        source_language="en",
        target_language="ko",
        logical_turn_key=f"self:{turn_id}",
    )


@pytest.mark.asyncio
async def test_transition_emphasis_expires_on_next_distinct_readable_turn_not_revision() -> None:
    clock = FakeClock(_now=20.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    changed, self_turn = uuid4(), uuid4()

    await presenter.emit(_peer_event(adapter, changed, "changed", "transition"))
    entry = presenter._entries[("peer", changed)]
    visible_since = entry.visible_since
    assert presenter.snapshot().blocks[-1].speaker_style == "cyan"

    await presenter.emit(_peer_event(adapter, changed, "changed revision", "transition"))
    assert presenter.snapshot().blocks[-1].speaker_style == "cyan"
    assert entry.visible_since == visible_since

    await presenter.emit(_self_event(adapter, self_turn, "self"))
    peer = next(block for block in presenter.snapshot().blocks if block.id == f"peer:{changed}")
    self_block = next(
        block for block in presenter.snapshot().blocks if block.id == f"self:{self_turn}"
    )
    assert peer.speaker_style == "gold"
    assert self_block.speaker_style is None
    assert entry.visible_since == visible_since


@pytest.mark.asyncio
async def test_consecutive_transitions_move_emphasis_to_incoming_turn() -> None:
    clock = FakeClock(_now=30.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    first, second = uuid4(), uuid4()

    await presenter.emit(_peer_event(adapter, first, "first", "transition"))
    await presenter.emit(_peer_event(adapter, second, "second", "transition"))

    blocks = {block.id: block for block in presenter.snapshot().blocks}
    assert blocks[f"peer:{first}"].speaker_style == "gold"
    assert blocks[f"peer:{second}"].speaker_style == "cyan"


@pytest.mark.asyncio
async def test_runtime_detach_clears_stale_emphasis_without_disabling_next_transition() -> None:
    clock = FakeClock(_now=35.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    reused, next_changed = uuid4(), uuid4()

    await presenter.emit(_peer_event(adapter, reused, "changed", "transition"))
    assert presenter.snapshot().blocks[-1].speaker_style == "cyan"

    await presenter.clear_for_runtime_detach()
    await presenter.emit(_peer_event(adapter, reused, "reused after detach", "context_reset"))
    assert presenter.snapshot().blocks[-1].speaker_style == "gold"

    await presenter.emit(_peer_event(adapter, next_changed, "next transition", "transition"))
    blocks = {block.id: block for block in presenter.snapshot().blocks}
    assert blocks[f"peer:{reused}"].speaker_style == "gold"
    assert blocks[f"peer:{next_changed}"].speaker_style == "cyan"


@pytest.mark.asyncio
async def test_uncertainty_and_late_transition_revision_do_not_emphasize() -> None:
    clock = FakeClock(_now=40.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    turn_id = uuid4()

    await presenter.emit(_peer_event(adapter, turn_id, "first", "unavailable"))
    entry = presenter._entries[("peer", turn_id)]
    visible_since = entry.visible_since
    await presenter.emit(_peer_event(adapter, turn_id, "revision", "transition"))

    block = presenter.snapshot().blocks[-1]
    assert block.speaker_style == "gold"
    assert entry.speaker_transition == "unavailable"
    assert entry.visible_since == visible_since


@pytest.mark.asyncio
async def test_source_only_claim_is_not_reattached_by_later_translation_revision() -> None:
    clock = FakeClock(_now=50.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    turn_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(turn_id, "peer source", True, channel="peer"),
            source_language="en",
            target_language="ko",
            logical_turn_key=f"peer:{turn_id}",
            speaker_transition="transition",
            speaker_transition_claim_id="source-claim",
        )
    )
    entry = presenter._entries[("peer", turn_id)]
    visible_since = entry.visible_since
    assert presenter.snapshot().blocks[-1].speaker_style == "cyan"

    await presenter.emit(_self_event(adapter, uuid4(), "self"))
    await presenter.emit(
        adapter.translation_final(
            utterance_id=turn_id,
            channel="peer",
            text="translated revision",
            source_text="peer source",
            source_language="en",
            target_language="ko",
            applied_context_mode="integrated",
            logical_turn_key=f"peer:{turn_id}",
            speaker_transition="transition",
            speaker_transition_claim_id="late-claim",
        )
    )

    block = next(block for block in presenter.snapshot().blocks if block.id == f"peer:{turn_id}")
    assert block.speaker_style == "gold"
    assert entry.speaker_transition_claim_id == "source-claim"
    assert entry.visible_since == visible_since
