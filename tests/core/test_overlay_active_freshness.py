from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.domain.models import Transcript
from puripuly_heart.ui.desktop_overlay import build_desktop_caption_plan
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.helpers.overlay_active_freshness import projected_self_lifecycle


@pytest.mark.asyncio
async def test_active_self_projection_keeps_stream_episode_until_semantic_final() -> None:
    snapshots = await projected_self_lifecycle()
    blocks = [snapshot["blocks"][0] for snapshot in snapshots[:-1]]
    assert {block["id"] for block in blocks} == {blocks[0]["id"]}
    assert {block["occupant_key"] for block in blocks} == {blocks[0]["occupant_key"]}
    assert {block["appearance_seq"] for block in blocks} == {blocks[0]["appearance_seq"]}
    assert [block["block_variant"] for block in blocks] == ["active_self"] * 4 + ["finalized"] * 2
    assert [
        snapshot["native_fresh_render_generations"]["self"] for snapshot in snapshots[:-1]
    ] == list(range(1, 7))
    episodes = [snapshot["native_quiet_tail_episodes"]["self"] for snapshot in snapshots[:-1]]
    assert episodes[:4] == [{"phase": "stream", "generation": 1}] * 4
    assert episodes[4:] == [
        {"phase": "final", "generation": 2},
        {"phase": "final", "generation": 3},
    ]
    assert snapshots[-1]["blocks"] == []
    assert "native_quiet_tail_episodes" not in snapshots[-1]


@pytest.mark.asyncio
async def test_active_self_noop_and_hidden_secondary_do_not_renew_freshness_or_age() -> None:
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=clock, native_retry_enabled=True
    )
    adapter = OverlayEventAdapter(clock=clock)
    turn = uuid4()
    try:
        await presenter.emit(
            adapter.self_active_update(
                text="source", utterance_id=turn, occupant_key=f"self:{turn}"
            )
        )
        initial = presenter.snapshot()
        entry = presenter._entries[("self", turn)]
        visible_at = entry.last_meaningful_visible_at
        clock.advance(1.0)
        await presenter.emit(
            adapter.self_active_update(
                text="source",
                utterance_id=turn,
                occupant_key=f"self:{turn}",
                update_id="metadata-only",
            )
        )
        assert (
            presenter.snapshot().native_fresh_render_generations
            == initial.native_fresh_render_generations
        )
        assert presenter.snapshot().native_quiet_tail_episodes == initial.native_quiet_tail_episodes
        assert entry.last_meaningful_visible_at == visible_at
        await presenter.update_display_preferences(show_translation=False, show_peer_original=True)
        visible_at = entry.last_meaningful_visible_at
        await presenter.emit(
            adapter.self_active_update(
                text="source",
                secondary_text="hidden",
                utterance_id=turn,
                occupant_key=f"self:{turn}",
            )
        )
        assert (
            presenter.snapshot().native_fresh_render_generations
            == initial.native_fresh_render_generations
        )
        assert entry.last_meaningful_visible_at == visible_at
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [True, False])
async def test_mixed_self_peer_pacing_and_late_translation_preserve_shared_window(
    active: bool,
) -> None:
    clock = FakeClock(_now=10.0)
    waits = []

    async def paced_sleep(delay: float) -> None:
        if delay <= 1.0:
            waits.append(delay)
            clock.advance(delay)
            await asyncio.sleep(0)
        else:
            await asyncio.Event().wait()

    presenter = OverlayPresenter(
        calibration=OverlayCalibration(anchor="spatial_locked"),
        clock=clock,
        sleep=paced_sleep,
        native_retry_enabled=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    turn = uuid4()
    peers = [uuid4(), uuid4()]
    try:
        if active:
            await presenter.emit(
                adapter.self_active_update(
                    text="source", utterance_id=turn, occupant_key=f"self:{turn}"
                )
            )
        else:
            await presenter.emit(
                adapter.transcript_final(
                    Transcript(
                        utterance_id=turn,
                        channel="self",
                        text="source",
                        is_final=True,
                        created_at=clock.now(),
                    ),
                    source_language="en",
                    target_language="ko",
                )
            )
        initial = presenter.snapshot().blocks[0]
        clock.advance(0.1)
        for peer in peers:
            await presenter.emit_peer_when_admissible(
                adapter.transcript_final(
                    Transcript(
                        utterance_id=peer,
                        channel="peer",
                        text="peer source",
                        is_final=True,
                        created_at=clock.now(),
                    ),
                    source_language="en",
                    target_language="ko",
                )
            )
            receipt = await presenter.emit_peer_when_admissible(
                adapter.translation_final(
                    utterance_id=peer,
                    channel="peer",
                    text="peer translation",
                    source_language="en",
                    target_language="ko",
                    applied_context_mode=None,
                )
            )
            assert receipt.outcome == "applied"
        assert waits == pytest.approx([1.0])
        assert clock.now() == pytest.approx(11.1)
        before_translation = presenter.snapshot()
        assert len(before_translation.blocks) == 2
        receipt = await presenter.emit(
            adapter.translation_final(
                utterance_id=turn,
                channel="self",
                text="late translation",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
            )
        )
        if active:
            assert receipt.outcome == "applied"
            blocks = presenter.snapshot().blocks
            assert [block.id for block in blocks] == [initial.id, f"peer:{peers[1]}"]
            translated = blocks[0]
            assert (translated.id, translated.occupant_key, translated.appearance_seq) == (
                initial.id,
                initial.occupant_key,
                initial.appearance_seq,
            )
            assert translated.block_variant == "active_self"
            assert translated.secondary_text == "late translation"
        else:
            assert receipt.outcome == "stale"
            assert [block.id for block in presenter.snapshot().blocks] == [
                f"peer:{peer}" for peer in peers
            ]
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_actual_desktop_plan_is_independent_of_native_active_freshness() -> None:
    clock = FakeClock(_now=10.0)
    turn = uuid4()
    presenters = [
        OverlayPresenter(
            calibration=OverlayCalibration(), clock=clock, native_retry_enabled=enabled
        )
        for enabled in (False, True)
    ]
    adapter = OverlayEventAdapter(clock=clock)
    try:
        for text, secondary in (("source", ""), ("source extension", "sticky translation")):
            event = adapter.self_active_update(
                text=text, secondary_text=secondary, utterance_id=turn, occupant_key=f"self:{turn}"
            )
            for presenter in presenters:
                await presenter.emit(event)
            plans = [build_desktop_caption_plan(presenter.snapshot()) for presenter in presenters]
            assert plans[0] == plans[1]
            assert plans[1].lines[0].role == "active_self_source"
            assert plans[1].lines[0].priority == 100
        event = adapter.transcript_final(
            Transcript(
                utterance_id=turn,
                channel="self",
                text="source extension",
                is_final=True,
                created_at=clock.now(),
            ),
            source_language="en",
            target_language="ko",
        )
        for presenter in presenters:
            await presenter.emit(event)
        assert build_desktop_caption_plan(presenters[0].snapshot()) == build_desktop_caption_plan(
            presenters[1].snapshot()
        )
        assert presenters[0].snapshot().native_fresh_render_generations is None
    finally:
        for presenter in presenters:
            await presenter.close()
