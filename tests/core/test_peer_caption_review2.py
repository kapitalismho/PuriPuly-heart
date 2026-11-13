from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.overlay import presenter_acceptance
from puripuly_heart.core.overlay.peer_caption_layout import peer_translation_fragments
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter, OverlayPublicationScope
from puripuly_heart.core.runtime.output import OutputRuntime
from tests.core.runtime.test_output_runtime import RecordingChatbox
from tests.core.test_peer_caption_fragments import fragment_id, identity, own_speech
from tests.core.test_peer_caption_lifecycle import final, idle, partial, setup, transcript
from tests.core.test_peer_overlay_policy import Surface, Timeline, caption


@pytest.mark.asyncio
async def test_r1_real_language_reset_revokes_pending_preview_and_allows_same_capture_successor():
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    harness, backend, _, _ = setup(overlay_sink=presenter, clock=timeline.clock)
    neighbor = identity(4201)
    old = transcript("Old language unfinished", 1)
    current = transcript("Current language ended.", 2)
    backend.translations[old.text] = "Pending old preview"
    backend.translations[current.text] = "Current new final"
    try:
        await harness.start()
        await presenter.emit(own_speech())
        await presenter.emit(caption(100, neighbor, "나" * 40))
        await partial(harness, old)
        await timeline.advance(4.9)
        await presenter.emit(caption(101, neighbor, "다" * 40, revision=2))
        await timeline.advance(0.1)
        await harness.peer_owner.cloud_captions.check_previews()
        await harness.peer_owner.cloud_captions.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        old_unit = backend.calls[-1].utterance_id
        old_state = presenter.peer_fragment_state(old_unit, 0)
        assert old_state.pending and old_state.registered_at == 5
        assert old_state.visible_since is None
        before = presenter.snapshot().native_quiet_tail_episodes.peer
        neighbor_deadline = surface.expirations[fragment_id(neighbor, 0)]
        assert harness.output_runtime._peer_generation == 1
        await harness.peer_owner.clear_language_runtime_state(channel="peer")
        assert presenter.peer_fragment_state(old_unit, 0) is None
        assert harness.output_runtime._peer_generation == 1
        await timeline.advance(1)
        assert all(
            block.primary_text != "Pending old preview" for block in presenter.snapshot().blocks
        )
        assert presenter.snapshot().native_quiet_tail_episodes.peer == before
        assert surface.expirations[fragment_id(neighbor, 0)] == neighbor_deadline
        await final(harness, current)
        await idle(harness)
        await timeline.advance(1)
        await harness.output_runtime.wait_for_peer_output_idle()
        assert any(
            block.primary_text == "Current new final" for block in presenter.snapshot().blocks
        )
        assert harness.output_runtime._peer_generation == 1
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [current.text]
        assert not any(
            block.primary_text == "Pending old preview"
            for snapshot in surface.snapshots
            for block in snapshot.blocks
        )
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("local", [False, True])
async def test_r1_reset_revokes_settled_pending_final_without_retiring_capture_generation(local):
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        native_retry_enabled=True,
    )
    output = OutputRuntime(chatbox=RecordingChatbox(), clock=timeline.clock, overlay_sink=presenter)
    adapter = OverlayEventAdapter(clock=timeline.clock)
    output.activate_peer_generation(1)
    old, current = identity(4211), identity(4212)
    try:
        await presenter.emit(own_speech())
        await presenter.emit(caption(100, identity(4210), "나" * 40))
        scope = None
        if local:
            assert await output.admit_translation_parent(
                parent_id=str(identity(4311)),
                channel="peer",
                origin="peer",
                turn_generation=1,
                turn_order=1,
                retained_payloads=("Local old source.",),
                destination_targets={"overlay": frozenset({0})},
            ) == frozenset({"overlay"})
            scope = OverlayPublicationScope(
                turn_kind="peer",
                parent_utterance_id=identity(4311),
                turn_generation=1,
                turn_order=1,
            )
        event = adapter.translation_final(
            utterance_id=old,
            channel="peer",
            text="Pending old final",
            source_text="Local old source." if local else "Cloud old source.",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            peer_caption_revision=0 if local else 1,
            output_scope=scope,
        )
        await output.publish_overlay_event(event, publication_generation=1, source_order=1)
        if local:
            await output.publish_overlay_event(
                adapter.utterance_closed(utterance_id=old, channel="peer", output_scope=scope),
                publication_generation=1,
                source_order=1,
            )
        await output.wait_for_peer_output_idle()
        assert presenter.peer_fragment_state(old, 0).pending
        assert presenter.application_receipt(event.event_id).outcome == "registered"
        before = presenter.snapshot().native_quiet_tail_episodes.peer
        await output.reset_peer_caption_admission()
        await output.reset_peer_caption_admission()
        assert output._peer_generation == 1
        assert presenter.peer_fragment_state(old, 0) is None
        await timeline.advance(1)
        assert presenter.snapshot().native_quiet_tail_episodes.peer == before
        successor = caption(101, current, "Current same-capture final")
        await output.publish_overlay_event(successor, publication_generation=1, source_order=2)
        await output.wait_for_peer_output_idle()
        await timeline.advance(1)
        assert any(block.id == fragment_id(current, 0) for block in presenter.snapshot().blocks)
        assert all(block.id != fragment_id(old, 0) for block in presenter.snapshot().blocks)
    finally:
        await output.close()
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("self_active", [False, True])
async def test_r2_toggle_reconciles_effective_source_and_cached_translation_without_duplicate_tail(
    self_active,
):
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    unit = identity(4221)
    text, source = "가" * 64, "One source sentence."
    try:
        if self_active:
            await presenter.emit(own_speech())
        await presenter.emit(caption(101, unit, text, source=source))
        await timeline.advance(0)
        await timeline.advance(0.2)
        await presenter.update_translation_enabled(False)
        peers = [block for block in presenter.snapshot().blocks if block.channel == "peer"]
        assert [block.primary_text for block in peers] == [source]
        assert all(not block.secondary_text and not block.secondary_enabled for block in peers)
        assert surface.expirations[peers[0].id] == pytest.approx(8.2)
        episode = presenter.snapshot().native_quiet_tail_episodes.peer
        await timeline.advance(1)
        assert [
            block.primary_text for block in presenter.snapshot().blocks if block.channel == "peer"
        ] == [source]
        assert presenter.snapshot().native_quiet_tail_episodes.peer == episode
        await presenter.update_translation_enabled(True)
        await timeline.advance(0)
        peers = [block for block in presenter.snapshot().blocks if block.channel == "peer"]
        assert [block.primary_text for block in peers] == (
            [peer_translation_fragments(text)[1]]
            if self_active
            else list(peer_translation_fragments(text))
        )
        assert [block.secondary_text for block in peers] == ([""] if self_active else [source, ""])
        assert [block.secondary_enabled for block in peers] == (
            [False] if self_active else [True, False]
        )
        assert presenter.snapshot().native_fresh_render_targets.peer in {
            block.id for block in peers
        }
        assert presenter._entries[("peer", unit)].translation_text == text
        if self_active:
            assert all(block.id != fragment_id(unit, 0) for block in peers)
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_r2_new_source_overflow_waits_after_existing_waiter_and_retains_view_coordinates():
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    unit, waiter = identity(4231), identity(4232)
    source = "나" * 64
    try:
        await presenter.emit(own_speech())
        await presenter.emit(caption(101, unit, "short translation", source=source))
        await timeline.advance(0.1)
        await presenter.emit(caption(102, waiter, "B waiter", source="B source"))
        waiting = presenter.peer_fragment_state(waiter, 0)
        await timeline.advance(0.1)
        await presenter.update_translation_enabled(False)
        tail = presenter.peer_fragment_state(unit, 1)
        assert tail.pending and not tail.dormant and tail.registered_at == 0.2
        assert tail.registration_order > waiting.registration_order
        assert presenter.snapshot().blocks[-1].primary_text == peer_translation_fragments(source)[0]
        await timeline.advance(0.2)
        await presenter.update_translation_enabled(True)
        dormant = presenter.peer_fragment_state(unit, 1)
        assert dormant.dormant
        assert presenter.snapshot().blocks[-1].primary_text == "short translation"
        await timeline.advance(0.1)
        await presenter.update_translation_enabled(False)
        restored = presenter.peer_fragment_state(unit, 1)
        assert (restored.registered_at, restored.registration_order) == (
            tail.registered_at,
            tail.registration_order,
        )
        revision = presenter.snapshot().revision
        deadline = surface.expirations[fragment_id(unit, 0)]
        await presenter.update_translation_enabled(False)
        assert presenter.snapshot().revision == revision
        assert surface.expirations[fragment_id(unit, 0)] == deadline
        await timeline.advance(0.6)
        assert presenter.snapshot().blocks[-1].id == fragment_id(waiter, 0)
        assert presenter.snapshot().blocks[-1].primary_text == "B source"
        await timeline.advance(0.1)
        assert presenter.snapshot().blocks[-1].primary_text == peer_translation_fragments(source)[1]
        assert presenter.peer_fragment_state(unit, 0).departed
        assert presenter.snapshot().native_fresh_render_targets.peer == fragment_id(unit, 1)
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_r2_departed_root_source_view_uses_current_drawable_without_replaying_prefix():
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    unit, source, text = identity(4241), "One source sentence.", "가" * 64
    try:
        await presenter.emit(own_speech())
        await presenter.emit(caption(101, unit, text, source=source))
        await timeline.advance(1)
        assert presenter.peer_fragment_state(unit, 0).departed
        assert presenter.snapshot().blocks[-1].id == fragment_id(unit, 1)
        observed_before = len(surface.snapshots)
        await timeline.advance(0.2)
        await presenter.update_translation_enabled(False)
        block = presenter.snapshot().blocks[-1]
        assert (block.id, block.primary_text) == (fragment_id(unit, 1), source)
        assert surface.expirations[block.id] == pytest.approx(9.2)
        episode = presenter.snapshot().native_quiet_tail_episodes.peer
        await timeline.advance(0.1)
        await presenter.update_translation_enabled(False)
        assert presenter.snapshot().native_quiet_tail_episodes.peer == episode
        assert surface.expirations[block.id] == pytest.approx(9.2)
        await timeline.advance(0.1)
        await presenter.update_translation_enabled(True)
        block = presenter.snapshot().blocks[-1]
        assert (block.id, block.primary_text) == (
            fragment_id(unit, 1),
            peer_translation_fragments(text)[1],
        )
        assert presenter.snapshot().native_fresh_render_targets.peer == block.id
        assert all(
            candidate.id != fragment_id(unit, 0)
            for snapshot in surface.snapshots[observed_before:]
            for candidate in snapshot.blocks
        )
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_r2_full_revision_removes_obsolete_dormant_tail_and_never_restores_old_revision():
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    unit, source = identity(4251), "One source sentence."
    try:
        await presenter.emit(caption(101, unit, "가" * 64, source=source))
        await timeline.advance(0)
        await timeline.advance(0.2)
        await presenter.update_translation_enabled(False)
        assert presenter.peer_fragment_state(unit, 1).dormant
        deadline = surface.expirations[fragment_id(unit, 0)]
        episode = presenter.snapshot().native_quiet_tail_episodes.peer
        await timeline.advance(0.2)
        await presenter.emit(
            caption(102, unit, "new fitting translation", source=source, revision=2)
        )
        assert presenter.peer_fragment_state(unit, 1) is None
        assert surface.expirations[fragment_id(unit, 0)] == deadline
        assert presenter.snapshot().native_quiet_tail_episodes.peer == episode
        await timeline.advance(0.2)
        await presenter.update_translation_enabled(True)
        assert [block.primary_text for block in presenter.snapshot().blocks] == [
            "new fitting translation"
        ]
        assert presenter._entries[("peer", unit)].peer_caption_revision == 2
        assert presenter.peer_fragment_state(unit, 1) is None
        assert surface.expirations[fragment_id(unit, 0)] == pytest.approx(8.6)
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_r2_dormant_tail_does_not_keep_expired_parent_alive_or_rearm_on_reenable():
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        native_retry_enabled=True,
    )
    unit = identity(4261)
    try:
        await presenter.emit(caption(101, unit, "가" * 64, source="One source sentence."))
        await timeline.advance(0)
        await timeline.advance(0.2)
        await presenter.update_translation_enabled(False)
        assert presenter.peer_fragment_state(unit, 1).dormant
        await timeline.advance(8.1)
        assert presenter.peer_fragment_states(unit) == ()
        assert presenter.snapshot().blocks == []
        await presenter.update_translation_enabled(True)
        assert presenter.snapshot().blocks == []
        episodes = presenter.snapshot().native_quiet_tail_episodes
        assert episodes is None or episodes.peer is None
        assert (await presenter.emit(caption(102, unit, "late", revision=2))).outcome == "stale"
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_r2_real_owner_toggle_does_not_translate_or_append_history_again():
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        native_retry_enabled=True,
    )
    harness, backend, _, _ = setup(overlay_sink=presenter, clock=timeline.clock)
    source = transcript("One source sentence.", 1)
    backend.translations[source.text] = "가" * 64
    try:
        await harness.start()
        await final(harness, source)
        await idle(harness)
        await timeline.advance(0)
        assert len(presenter.snapshot().blocks) == 2
        history = tuple(harness.peer_runtime.translation_history)
        requests = dict(harness.peer_owner.cloud_captions.request_counts)
        await timeline.advance(0.2)
        await presenter.update_translation_enabled(False)
        assert [block.primary_text for block in presenter.snapshot().blocks] == [source.text]
        revision = presenter.snapshot().revision
        await presenter.update_translation_enabled(False)
        assert presenter.snapshot().revision == revision
        await presenter.update_translation_enabled(True)
        await timeline.advance(0)
        assert [block.primary_text for block in presenter.snapshot().blocks] == list(
            peer_translation_fragments("가" * 64)
        )
        assert len(backend.calls) == 1
        assert tuple(harness.peer_runtime.translation_history) == history
        assert harness.peer_owner.cloud_captions.request_counts == requests
    finally:
        await harness.stop()
        await presenter.close()


@pytest.mark.asyncio
async def test_r1_reset_cancels_output_owned_final_before_scene_lock_and_recovers_same_capture():
    from tests.core.test_peer_caption_registration_races import HoldingSurface

    timeline, surface = Timeline(), HoldingSurface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
    )
    output = OutputRuntime(chatbox=RecordingChatbox(), clock=timeline.clock, overlay_sink=presenter)
    output.activate_peer_generation(1)
    old, current = identity(4271), identity(4272)
    surface.hold = True
    scene = asyncio.create_task(presenter.emit(own_speech()))
    try:
        await asyncio.wait_for(surface.started.wait(), 0.5)
        event = caption(101, old, "Old final behind lock")
        await output.publish_overlay_event(event, publication_generation=1, source_order=1)
        await timeline.advance(0)
        assert output._peer_active_batch.delivery_task is not None
        assert presenter.peer_fragment_state(old, 0) is None
        await output.reset_peer_caption_admission()
        await asyncio.wait_for(output.wait_for_peer_output_idle(), 0.5)
        assert output._peer_generation == 1
        assert any(
            decision.publication_id == event.event_id and decision.reason == "cancelled_local"
            for decision in output.routing_decisions
        )
        assert presenter.peer_fragment_state(old, 0) is None
        await output.publish_overlay_event(
            caption(102, current, "Current final after reset"),
            publication_generation=1,
            source_order=2,
        )
        surface.release.set()
        await asyncio.wait_for(scene, 0.5)
        await asyncio.wait_for(output.wait_for_peer_output_idle(), 0.5)
        assert any(block.id == fragment_id(current, 0) for block in presenter.snapshot().blocks)
        assert all(block.id != fragment_id(old, 0) for block in presenter.snapshot().blocks)
    finally:
        surface.release.set()
        await asyncio.gather(scene, return_exceptions=True)
        await output.close()
        await presenter.close()


@pytest.mark.asyncio
async def test_r4_source_expansion_over_budget_retires_caption_and_preserves_self_progress():
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=surface,
        native_retry_enabled=True,
    )
    unit = identity(4401)
    event = caption(1, unit, "short translation", source="가\n" * 90000)
    try:
        assert (
            presenter._acceptance.event_payload_bytes(event)
            < presenter_acceptance.PRESENTER_EVENT_BYTE_LIMIT
        )
        assert (await presenter.emit(event)).outcome == "applied"
        await presenter.update_translation_enabled(False)
        assert (
            sum(
                presenter._acceptance.entry_payload_bytes(entry)
                for entry in presenter._entries.values()
            )
            <= presenter_acceptance.PRESENTER_AGGREGATE_BYTE_LIMIT
        )
        assert presenter.peer_fragment_states(unit) == ()
        assert presenter.snapshot().blocks == []
        episodes = presenter.snapshot().native_quiet_tail_episodes
        assert episodes is None or episodes.peer is None
        assert (await presenter.emit(own_speech())).outcome == "applied"
        assert any(block.channel == "self" for block in surface.snapshots[-1].blocks)
        await presenter.update_translation_enabled(True)
        stale = await presenter.emit(caption(2, unit, "late old translation", revision=2))
        assert (stale.outcome, stale.cause) == ("stale", "stale")
        successor = identity(4402)
        assert (
            await presenter.emit(caption(3, successor, "current translation"))
        ).outcome == "applied"
        await timeline.advance(1)
        assert {block.id for block in presenter.snapshot().blocks} == {
            f"self:{identity(900)}",
            fragment_id(successor, 0),
        }
        assert presenter.peer_fragment_states(unit) == ()
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_r4_source_view_budget_accounts_for_other_captions_without_truncating_survivors(
    monkeypatch,
):
    monkeypatch.setattr(presenter_acceptance, "PRESENTER_AGGREGATE_BYTE_LIMIT", 3500)
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep
    )
    first, second = identity(4411), identity(4412)
    source = "가\n" * 10
    try:
        for seq, unit in enumerate((first, second), start=1):
            assert (
                await presenter.emit(caption(seq, unit, "short translation", source=source))
            ).outcome == "applied"
        await presenter.update_translation_enabled(False)
        assert presenter._entries[("peer", first)].peer_fragments == peer_translation_fragments(
            source
        )
        assert presenter.peer_fragment_states(second) == ()
        assert (
            sum(
                presenter._acceptance.entry_payload_bytes(entry)
                for entry in presenter._entries.values()
            )
            <= presenter_acceptance.PRESENTER_AGGREGATE_BYTE_LIMIT
        )
        assert (await presenter.emit(own_speech())).outcome == "applied"
        assert any(block.channel == "self" for block in presenter.snapshot().blocks)
        assert any(block.channel == "peer" for block in presenter.snapshot().blocks)
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_r4_calibration_reflow_uses_the_same_retained_budget_and_terminal_retirement(
    monkeypatch,
):
    monkeypatch.setattr(presenter_acceptance, "PRESENTER_AGGREGATE_BYTE_LIMIT", 1700)
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        translation_enabled=False,
        native_retry_enabled=True,
    )
    unit = identity(4421)
    try:
        assert (
            await presenter.emit(caption(1, unit, "short translation", source="가" * 64))
        ).outcome == "applied"
        assert len(presenter.peer_fragment_states(unit)) == 2
        await presenter.update_calibration(OverlayCalibration(text_scale=2))
        assert presenter.peer_fragment_states(unit) == ()
        assert presenter.snapshot().blocks == []
        episodes = presenter.snapshot().native_quiet_tail_episodes
        assert episodes is None or episodes.peer is None
        await presenter.update_calibration(OverlayCalibration())
        assert presenter.snapshot().blocks == []
        assert (await presenter.emit(own_speech())).outcome == "applied"
        assert any(block.channel == "self" for block in presenter.snapshot().blocks)
        stale = await presenter.emit(caption(2, unit, "late fitting text", revision=2))
        assert (stale.outcome, stale.cause) == ("stale", "stale")
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_overlay_reattachment_preserves_capture_generation_and_revokes_old_pending_content():
    timeline = Timeline()
    old_surface, new_surface = Surface(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        bridge=old_surface,
        native_retry_enabled=True,
    )
    output = OutputRuntime(chatbox=RecordingChatbox(), clock=timeline.clock, overlay_sink=presenter)
    output.activate_peer_generation(1)
    visible, pending, successor = identity(4501), identity(4502), identity(4503)
    try:
        await presenter.emit(own_speech())
        for order, unit, text in (
            (1, visible, "가" * 64),
            (2, pending, "old pending final"),
        ):
            await output.publish_overlay_event(
                caption(order, unit, text), publication_generation=1, source_order=order
            )
            await output.wait_for_peer_output_idle()
        assert presenter.peer_fragment_state(visible, 1).pending
        assert presenter.peer_fragment_state(pending, 0).pending
        assert await output.replace_overlay_sink(None)
        assert presenter.peer_fragment_state(visible, 1) is None
        assert presenter.peer_fragment_states(pending) == ()
        presenter.detach_bridge()
        await presenter.begin_native_retry_epoch(enabled=False)
        presenter.attach_bridge(new_surface)
        assert await output.replace_overlay_sink(presenter)
        await presenter.begin_native_retry_epoch(enabled=True)
        correction = caption(3, visible, "current visible correction", revision=2)
        await output.publish_overlay_event(correction, publication_generation=1, source_order=1)
        await output.wait_for_peer_output_idle()
        assert presenter.snapshot().blocks[-1].primary_text == "current visible correction"
        assert presenter.application_receipt(correction.event_id).outcome == "applied"
        await output.publish_overlay_event(
            caption(4, successor, "new same-capture final"),
            publication_generation=1,
            source_order=3,
        )
        await output.wait_for_peer_output_idle()
        await timeline.advance(0)
        await timeline.advance(1)
        assert [block.channel for block in presenter.snapshot().blocks] == ["self", "peer"]
        assert presenter.snapshot().blocks[-1].primary_text == "new same-capture final"
        current_scene = presenter.snapshot()
        stale = caption(5, pending, "late old pending final", revision=2)
        await output.publish_overlay_event(stale, publication_generation=1, source_order=2)
        await output.wait_for_peer_output_idle()
        assert presenter.snapshot() == current_scene
        assert all(
            block.id not in {fragment_id(visible, 1), fragment_id(pending, 0)}
            for snapshot in new_surface.snapshots
            for block in snapshot.blocks
        )
    finally:
        await output.close()
        await presenter.close()
