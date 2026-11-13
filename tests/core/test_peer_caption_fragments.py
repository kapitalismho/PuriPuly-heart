from __future__ import annotations

import asyncio
from uuid import UUID, uuid5

import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.overlay import presenter_acceptance
from puripuly_heart.core.overlay.peer_caption_layout import peer_translation_fragments
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import SelfActiveUpdate
from puripuly_heart.core.overlay.state import (
    PEER_FRAGMENT_DEPARTURE_BYTES,
    PEER_FRAGMENT_REGISTRATION_BYTES,
)
from puripuly_heart.core.runtime.output import OutputRuntime
from tests.core.runtime.test_output_runtime import RecordingChatbox, _publish_peer_overlay
from tests.core.test_peer_overlay_policy import Surface, Timeline, caption


def identity(number: int) -> UUID:
    return UUID(int=number)


def fragment_id(unit: UUID, index: int) -> str:
    return f"peer:{unit if index == 0 else uuid5(unit, f'fragment:{index}')}"


def own_speech(seq: int = 100) -> SelfActiveUpdate:
    unit = identity(900)
    return SelfActiveUpdate(
        event_id=f"self-{seq}",
        seq=seq,
        utterance_id=unit,
        channel="self",
        created_at=0,
        text="Self source",
        secondary_text="Self translation",
        occupant_key=f"self:{unit}",
        caption_revision=seq,
        caption_source_closed=False,
    )


@pytest.mark.asyncio
async def test_g1_real_output_visible_revision_bypasses_waiter_and_coalesces_pending():
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep, bridge=surface
    )
    output = OutputRuntime(chatbox=RecordingChatbox(), clock=timeline.clock, overlay_sink=presenter)
    a, b, c, d = map(identity, (1, 2, 3, 4))
    try:
        for event, order in ((caption(1, a, "A"), 1), (caption(2, b, "B preview", final=False), 2)):
            await _publish_peer_overlay(output, event, source_order=order)
            await output.wait_for_peer_output_idle()
        await timeline.advance(0.001)
        await _publish_peer_overlay(output, caption(3, c, "C preview", final=False), source_order=3)
        await timeline.advance(0)
        c_entry = presenter._entries[("peer", c)]
        registered = c_entry.peer_registered_at, c_entry.appearance_seq
        await timeline.advance(0.335)
        await _publish_peer_overlay(output, caption(4, b, "B final", revision=2), source_order=2)
        await timeline.advance(0)
        assert [(block.id, block.primary_text) for block in presenter.snapshot().blocks] == [
            (fragment_id(a, 0), "A"),
            (fragment_id(b, 0), "B final"),
        ]
        assert timeline.clock.now() == pytest.approx(0.336)
        await _publish_peer_overlay(output, caption(5, c, "C final", revision=2), source_order=3)
        await _publish_peer_overlay(output, caption(6, d, "D"), source_order=4)
        await timeline.advance(0)
        assert (c_entry.peer_registered_at, c_entry.appearance_seq) == registered
        assert presenter._entries[("peer", d)].appearance_seq > c_entry.appearance_seq
        await timeline.advance(0.665)
        assert [block.primary_text for block in presenter.snapshot().blocks] == [
            "B final",
            "C final",
        ]
        assert presenter._entries[("peer", c)].visible_since == pytest.approx(1.001)
        await timeline.advance(0.335)
        await asyncio.wait_for(output.wait_for_peer_output_idle(), 1)
        assert [block.primary_text for block in presenter.snapshot().blocks] == ["C final", "D"]
        assert all(
            block.primary_text != "B preview"
            for snapshot in surface.snapshots[2:]
            for block in snapshot.blocks
        )
    finally:
        await output.close()
        await presenter.close()


@pytest.mark.asyncio
async def test_l1_pattern_replay_records_candidate_application_not_historical_lifetime():
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep, bridge=surface
    )
    output = OutputRuntime(chatbox=RecordingChatbox(), clock=timeline.clock, overlay_sink=presenter)
    a, b, c = map(identity, (11, 12, 13))
    try:
        await _publish_peer_overlay(output, caption(1, a, "A"), source_order=1)
        await output.wait_for_peer_output_idle()
        await _publish_peer_overlay(output, caption(2, b, "preview", final=False), source_order=2)
        await output.wait_for_peer_output_idle()
        await timeline.advance(0.001)
        await _publish_peer_overlay(output, caption(3, c, "waiting C"), source_order=3)
        await timeline.advance(0)
        assert presenter._entries[("peer", c)].peer_registered_at == 0.001
        await timeline.advance(0.981)
        await _publish_peer_overlay(
            output, caption(4, b, "same-unit final", revision=2), source_order=2
        )
        await timeline.advance(0.002)
        assert timeline.clock.now() == 0.984
        assert presenter._entries[("peer", b)].last_meaningful_visible_at == 0.984
        assert presenter.snapshot().blocks[-1].primary_text == "same-unit final"
        await timeline.advance(0.017)
        await asyncio.wait_for(output.wait_for_peer_output_idle(), 1)
        assert presenter._entries[("peer", b)].last_meaningful_visible_at == 0.984
        await timeline.advance(7.983)
        assert all(block.id != fragment_id(b, 0) for block in presenter.snapshot().blocks)
    finally:
        await output.close()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["lower", "upper", "self-promotion"])
@pytest.mark.parametrize("successor", [False, True])
async def test_g2_short_correction_has_eight_second_idle_ttl_without_metadata_renewal(
    placement, successor
):
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep, bridge=surface
    )
    unit, other = identity(21), identity(22)
    try:
        if placement == "lower":
            await presenter.emit(caption(1, other, "other"))
        await presenter.emit(caption(2, unit, "preview", final=False))
        if placement == "upper":
            await presenter.emit(caption(3, other, "other"))
        if placement == "self-promotion":
            await presenter.emit(own_speech())
        await timeline.advance(1.8 if placement != "self-promotion" else 2.5)
        old = presenter._entries[("peer", unit)].expiration_revision
        await presenter.emit(caption(101, unit, "short final", revision=2))
        deadline = timeline.clock.now() + 8
        assert surface.expirations[fragment_id(unit, 0)] == deadline
        await presenter._expire_entry_after_ttl(("peer", unit), old)
        await timeline.advance(0.5)
        await presenter.emit(caption(102, unit, "short final", revision=3))
        assert surface.expirations[fragment_id(unit, 0)] == deadline
        if successor:
            await presenter.emit(caption(103, identity(23), "following C"))
            await presenter.emit(caption(104, identity(24), "following D"))
            await timeline.advance(2)
            assert all(block.id != fragment_id(unit, 0) for block in presenter.snapshot().blocks)
            assert timeline.clock.now() < deadline
            return
        await timeline.advance(1.5)
        assert any(block.id == fragment_id(unit, 0) for block in presenter.snapshot().blocks)
        await timeline.advance(6)
        assert all(block.id != fragment_id(unit, 0) for block in presenter.snapshot().blocks)
        assert (
            await presenter.emit(caption(103, unit, "late repair", revision=4))
        ).outcome == "stale"
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_g4_full_revision_updates_survivors_appends_after_waiters_and_removes_tail():
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        native_retry_enabled=True,
    )
    a, b = identity(31), identity(32)
    try:
        await presenter.emit(caption(101, a, "가" * 64))
        await presenter.emit(own_speech(102))
        await timeline.advance(0)
        second = presenter.peer_fragment_state(a, 1)
        original = second.registered_at, second.registration_order
        await timeline.advance(0.1)
        await presenter.emit(caption(102, b, "B"))
        await timeline.advance(0.1)
        await presenter.emit(caption(103, a, "나" * 128, revision=2))
        pending = sorted(
            (entry for entry in presenter.peer_fragment_states() if entry.pending),
            key=lambda entry: entry.registration_order,
        )
        assert [(entry.logical_id, entry.index) for entry in pending] == [(a, 1), (b, 0), (a, 2)]
        assert (pending[0].registered_at, pending[0].registration_order) == original
        assert all(entry.revision == 2 for entry in presenter.peer_fragment_states(a))
        assert presenter.snapshot().native_fresh_render_targets.peer in {
            block.id for block in presenter.snapshot().blocks
        }
        await presenter.emit(caption(104, a, "다" * 24, revision=3))
        assert [entry.index for entry in presenter.peer_fragment_states(a)] == [0]
        assert presenter._entries[("peer", b)].peer_pending
        assert presenter.snapshot().blocks[0].primary_text == "다" * 24
        await timeline.advance(0.9)
        assert [block.primary_text for block in presenter.snapshot().blocks] == ["Self source", "B"]
        assert (
            await presenter.emit(caption(105, a, "cannot revive", revision=4))
        ).outcome == "stale"
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_pending_continuation_survives_first_sibling_eviction_without_replaying_departed_index():
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        native_retry_enabled=True,
    )
    unit = identity(41)
    try:
        await presenter.emit(caption(101, unit, "가" * 128, final=False))
        await presenter.emit(own_speech(102))
        await timeline.advance(0)
        assert sorted(
            entry.index for entry in presenter.peer_fragment_states() if entry.pending
        ) == [1, 2]
        await timeline.advance(1)
        assert presenter._entries[("peer", unit)].peer_departed
        await presenter.emit(caption(102, unit, "나" * 128, revision=2))
        await timeline.advance(0)
        blocks = [block for block in presenter.snapshot().blocks if block.channel == "peer"]
        assert blocks and all(block.id != fragment_id(unit, 0) for block in blocks)
        assert presenter.snapshot().native_fresh_render_targets.peer == blocks[-1].id
        assert all(entry.revision == 2 for entry in presenter.peer_fragment_states(unit))
        await presenter.emit(caption(103, identity(42), "replacement", superseded=(unit,)))
        assert presenter.peer_fragment_states(unit) == ()
        assert (await presenter.emit(caption(104, unit, "late", revision=3))).outcome == "stale"
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_same_phase_drawable_changes_rearm_but_duplicate_and_retired_updates_do_not():
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        native_retry_enabled=True,
    )
    unit = identity(51)
    try:
        previous = None
        for seq, text, final in (
            (1, "preview", False),
            (2, "preview changed", False),
            (3, "final", True),
            (4, "repair", True),
        ):
            await presenter.emit(caption(seq, unit, text, revision=seq, final=final))
            current = presenter.snapshot().native_quiet_tail_episodes.peer
            assert previous is None or current.generation > previous.generation
            previous = current
        await presenter.emit(caption(5, unit, "repair", revision=5))
        assert presenter.snapshot().native_quiet_tail_episodes.peer == previous
        assert (await presenter.emit(caption(6, unit, "stale", revision=3))).outcome == "stale"
        assert presenter.snapshot().native_quiet_tail_episodes.peer == previous
        await presenter.emit(caption(7, unit, "", revision=6, source=""))
        targets = presenter.snapshot().native_fresh_render_targets
        assert targets is None or targets.peer is None
        assert (await presenter.emit(caption(8, unit, "late", revision=7))).outcome == "stale"
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["generation-retirement", "output-shutdown"])
async def test_waiting_revisioned_output_cancellation_cannot_apply_or_revive(ending):
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        native_retry_enabled=True,
    )
    output = OutputRuntime(chatbox=RecordingChatbox(), clock=timeline.clock, overlay_sink=presenter)
    a, b, pending = map(identity, (61, 62, 63))
    event = caption(3, pending, "cancelled pending")
    try:
        for seq, unit in enumerate((a, b), 1):
            await _publish_peer_overlay(
                output, caption(seq, unit, f"visible {seq}"), source_order=seq
            )
            await output.wait_for_peer_output_idle()
        await _publish_peer_overlay(output, event, source_order=3)
        await timeline.advance(0)
        assert presenter._entries[("peer", pending)].peer_pending
        receipt = presenter.application_receipt(event.event_id)
        assert receipt.outcome == "registered" and receipt.scene_revision is None
        before = presenter.snapshot().native_quiet_tail_episodes.peer
        if ending == "generation-retirement":
            output.retire_peer_generation(1)
            await output.wait_for_peer_output_idle()
        else:
            await output.close()
        await timeline.advance(1.1)
        assert all(block.id != fragment_id(pending, 0) for block in presenter.snapshot().blocks)
        assert presenter.snapshot().native_quiet_tail_episodes.peer == before
        assert not any(
            decision.publication_id == event.event_id and decision.reason == "application_applied"
            for decision in output.routing_decisions
        )
        assert output._peer_active_batch is None
    finally:
        await output.close()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("final", [False, True], ids=["preview", "final"])
@pytest.mark.parametrize("native_retry", [False, True], ids=["desktop", "vr"])
@pytest.mark.parametrize("show_original", [False, True])
async def test_translated_fragments_project_complete_original_only_on_head(
    final, native_retry, show_original
):
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=native_retry,
        show_peer_original=show_original,
    )
    unit = identity(42101)
    source, text = "Complete original sentence. " * 8, "가" * 56 + "나" * 20
    try:
        await presenter.emit(caption(1, unit, text, source=source, final=final))
        await timeline.advance(0)
        blocks = presenter.snapshot().blocks
        assert [block.primary_text for block in blocks] == list(peer_translation_fragments(text))
        assert [block.id for block in blocks] == [fragment_id(unit, 0), fragment_id(unit, 1)]
        assert [block.secondary_text for block in blocks] == [source.strip(), ""]
        assert [block.secondary_enabled for block in blocks] == [show_original, False]
        assert [block.secondary_language for block in blocks] == [
            "en" if show_original else None,
            None,
        ]
        assert surface.snapshots[-1] == presenter.snapshot()
        if native_retry:
            episode = presenter.snapshot().native_quiet_tail_episodes.peer
            assert episode.phase == ("final" if final else "stream")
            assert episode.generation == 2
            assert presenter.snapshot().native_fresh_render_targets.peer == blocks[-1].id
        before = presenter.snapshot()
        deadlines = dict(surface.expirations)
        await timeline.advance(0.2)
        await presenter.emit(caption(2, unit, text, source=source, revision=2, final=final))
        assert presenter.snapshot() is before
        assert surface.expirations == deadlines
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_one_two_one_revision_keeps_original_on_head_and_shrink_does_not_revive_tail():
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep
    )
    unit = identity(42102)
    try:
        for revision, text, count in (
            (1, "가" * 24, 1),
            (2, "가" * 56 + "나" * 20, 2),
            (3, "다" * 24, 1),
        ):
            source = f"Complete original revision {revision}. " * 3
            await presenter.emit(
                caption(revision, unit, text, revision=revision, source=source, final=revision == 3)
            )
            await timeline.advance(0)
            blocks = presenter.snapshot().blocks
            assert len(blocks) == count
            assert [block.primary_text for block in blocks] == list(
                peer_translation_fragments(text)
            )
            assert [block.secondary_text for block in blocks] == [source.strip()] + [""] * (
                count - 1
            )
            assert [block.secondary_enabled for block in blocks] == [True] + [False] * (count - 1)
            assert blocks[0].id == fragment_id(unit, 0)
        await presenter.update_display_preferences(show_translation=True, show_peer_original=False)
        await presenter.update_display_preferences(show_translation=True, show_peer_original=True)
        assert [block.id for block in presenter.snapshot().blocks] == [fragment_id(unit, 0)]
        assert presenter.snapshot().blocks[0].secondary_text == source.strip()
        assert [state.index for state in presenter.peer_fragment_states(unit)] == [0]
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_original_toggle_changes_only_head_without_repartitioning_or_renewing_tail():
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    unit = identity(42103)
    source, text = "Complete original sentence.", "가" * 56 + "나" * 20
    try:
        await presenter.emit(caption(1, unit, text, source=source))
        await timeline.advance(0)
        parent = presenter._entries[("peer", unit)]
        fragments = parent.peer_fragments
        registration = [
            (state.id, state.registered_at, state.registration_order)
            for state in presenter.peer_fragment_states(unit)
        ]
        for show_original in (False, True):
            before = presenter.snapshot()
            tail = before.blocks[1]
            await timeline.advance(0.2)
            await presenter.update_display_preferences(
                show_translation=True, show_peer_original=show_original
            )
            after = presenter.snapshot()
            assert parent.peer_fragments is fragments
            assert [
                (state.id, state.registered_at, state.registration_order)
                for state in presenter.peer_fragment_states(unit)
            ] == registration
            assert after.blocks[1] == tail
            assert after.blocks[0].secondary_text == source
            assert after.blocks[0].secondary_enabled is show_original
            assert surface.expirations[tail.id] == 8
            assert surface.expirations[fragment_id(unit, 0)] == timeline.clock.now() + 8
            assert after.revision == before.revision + 1
            assert (
                after.native_quiet_tail_episodes.peer.generation
                == before.native_quiet_tail_episodes.peer.generation + 1
            )
            assert after.native_fresh_render_targets.peer == fragment_id(unit, 0)
            await presenter.update_display_preferences(
                show_translation=True, show_peer_original=show_original
            )
            assert presenter.snapshot() is after
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("departure", ["expiry", "eviction"])
async def test_departed_head_original_never_moves_on_toggle_correction_or_shrink(departure):
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    unit = identity(42104)
    text, source = "가" * 56 + "나" * 20, "Complete original sentence."
    revision = 1
    try:
        await presenter.emit(caption(1, unit, text, source=source))
        await timeline.advance(0)
        if departure == "expiry":
            await timeline.advance(1)
            text = text[:-1] + "다"
            revision += 1
            before = presenter.snapshot()
            await presenter.emit(caption(2, unit, text, revision=revision, source=source))
            assert surface.expirations[fragment_id(unit, 0)] == 8
            assert surface.expirations[fragment_id(unit, 1)] == 9
            assert presenter.snapshot().revision == before.revision + 1
            assert (
                presenter.snapshot().native_quiet_tail_episodes.peer.generation
                == before.native_quiet_tail_episodes.peer.generation + 1
            )
            await timeline.advance(7)
        else:
            await timeline.advance(0.1)
            await presenter.emit(caption(2, identity(42105), "successor", source=""))
            await timeline.advance(1)
        assert presenter.peer_fragment_state(unit, 0).departed
        tail = next(
            block for block in presenter.snapshot().blocks if block.id == fragment_id(unit, 1)
        )
        assert (tail.secondary_text, tail.secondary_enabled, tail.secondary_language) == (
            "",
            False,
            None,
        )
        observed_before = len(surface.snapshots)
        before = presenter.snapshot()
        deadline = surface.expirations[tail.id]
        for show_original in (False, True):
            await presenter.update_display_preferences(
                show_translation=True, show_peer_original=show_original
            )
            assert presenter.snapshot() is before
            assert surface.expirations[tail.id] == deadline
        revision += 1
        await presenter.emit(
            caption(3, unit, text, revision=revision, source="Corrected complete original.")
        )
        assert presenter.snapshot() is before
        assert surface.expirations[tail.id] == deadline
        await timeline.advance(0.1)
        revision += 1
        await presenter.emit(caption(4, unit, text[:-1] + "라", revision=revision, source=source))
        corrected = next(block for block in presenter.snapshot().blocks if block.id == tail.id)
        assert corrected.primary_text != tail.primary_text
        assert corrected.secondary_text == "" and not corrected.secondary_enabled
        assert presenter.snapshot().native_fresh_render_targets.peer == tail.id
        assert (
            presenter.snapshot().native_quiet_tail_episodes.peer.generation
            == before.native_quiet_tail_episodes.peer.generation + 1
        )
        assert surface.expirations[tail.id] == timeline.clock.now() + 8
        revision += 1
        await presenter.emit(caption(5, unit, "다" * 24, revision=revision, source=source))
        assert presenter.peer_fragment_states(unit) == ()
        assert (
            await presenter.emit(caption(6, unit, text, revision=revision + 1, source=source))
        ).outcome == "stale"
        assert all(
            block.id != fragment_id(unit, 0)
            for snapshot in surface.snapshots[observed_before:]
            for block in snapshot.blocks
        )
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_original_uses_current_translation_position_not_immutable_fragment_index():
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        translation_enabled=False,
    )
    unit = identity(42106)
    source = "가" * 56 + "나" * 20
    try:
        await presenter.emit(own_speech())
        await presenter.emit(caption(101, unit, "short translation", source=source))
        await timeline.advance(1)
        assert presenter.peer_fragment_state(unit, 0).departed
        await presenter.update_translation_enabled(True)
        block = presenter.snapshot().blocks[-1]
        assert block.id == fragment_id(unit, 1)
        assert block.primary_text == "short translation"
        assert block.secondary_text == source
        assert block.secondary_enabled
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("show_original", [False, True])
async def test_retained_bound_counts_full_original_once_even_after_head_departure(
    monkeypatch, show_original
):
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        show_peer_original=show_original,
    )
    unit = identity(42107)
    text, source = "가" * 56 + "나" * 20, "Complete retained original. " * 24
    event = caption(1, unit, text, source=source)
    fragments = peer_translation_fragments(text)
    retained = (
        256
        + len(source.encode("utf-8"))
        + len(text.encode("utf-8"))
        + len("enko")
        + len(fragment_id(unit, 0))
        + sum(len(fragment.encode("utf-8")) for fragment in fragments)
        + 2 * PEER_FRAGMENT_REGISTRATION_BYTES
    )
    monkeypatch.setattr(
        presenter_acceptance, "PRESENTER_AGGREGATE_BYTE_LIMIT", retained + len(event.event_id)
    )
    try:
        assert (await presenter.emit(event)).outcome == "applied"
        await timeline.advance(0)
        parent = presenter._entries[("peer", unit)]
        assert presenter._acceptance.entry_payload_bytes(parent) == retained
        monkeypatch.setattr(presenter_acceptance, "PRESENTER_AGGREGATE_BYTE_LIMIT", retained)
        await presenter.update_calibration(OverlayCalibration(text_scale=0.8))
        assert parent.original_text == source
        assert presenter._acceptance.entry_payload_bytes(parent) == retained
        await timeline.advance(1)
        text = text[:-1] + "다"
        assert (
            await presenter.emit(caption(2, unit, text, revision=2, source=source))
        ).outcome == "applied"
        await timeline.advance(7)
        assert presenter.peer_fragment_state(unit, 0).departed
        retained += -PEER_FRAGMENT_REGISTRATION_BYTES + 2 * PEER_FRAGMENT_DEPARTURE_BYTES
        assert parent.original_text == source
        assert presenter._acceptance.entry_payload_bytes(parent) == retained
        monkeypatch.setattr(presenter_acceptance, "PRESENTER_AGGREGATE_BYTE_LIMIT", retained)
        await presenter.update_calibration(OverlayCalibration())
        for enabled in (False, True):
            await presenter.update_display_preferences(
                show_translation=True, show_peer_original=enabled
            )
            block = presenter.snapshot().blocks[0]
            assert block.id == fragment_id(unit, 1)
            assert block.secondary_text == "" and not block.secondary_enabled
            assert presenter._acceptance.entry_payload_bytes(parent) == retained
        before = presenter.snapshot()
        rejected = await presenter.emit(caption(3, unit, text, revision=3, source=source + "x"))
        assert rejected.cause == "presenter_payload_exhausted"
        assert presenter.snapshot() is before
        assert parent.original_text == source
    finally:
        await presenter.close()
