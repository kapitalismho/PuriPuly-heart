from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.overlay.peer_caption_layout import peer_translation_fragments
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.presenter_acceptance import PRESENTER_ENTRY_LIMIT
from puripuly_heart.core.overlay.sink import OverlayEventAdapter, OverlayPublicationScope
from puripuly_heart.core.overlay.state import (
    PEER_FRAGMENT_DEPARTURE_BYTES,
    PEER_FRAGMENT_REGISTRATION_BYTES,
)
from puripuly_heart.core.runtime.output import OutputRuntime
from puripuly_heart.core.runtime.output_batch import DestinationBatchAdmission
from tests.core.runtime.test_output_runtime import RecordingChatbox, _publish_peer_overlay
from tests.core.test_peer_caption_fragments import fragment_id, identity, own_speech
from tests.core.test_peer_overlay_policy import Surface, Timeline, caption


@pytest.mark.asyncio
async def test_f1_visible_correction_bypasses_incrementally_saturated_new_item_admission():
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep, bridge=surface
    )
    output = OutputRuntime(chatbox=RecordingChatbox(), clock=timeline.clock, overlay_sink=presenter)
    try:
        for number in (1, 2):
            await _publish_peer_overlay(
                output, caption(number, identity(number), f"visible {number}"), source_order=number
            )
            await output.wait_for_peer_output_idle()
        for number in range(3, 12):
            await _publish_peer_overlay(
                output, caption(number, identity(number), f"waiting {number}"), source_order=number
            )
            await timeline.advance(0)
        waiting = [presenter.peer_fragment_state(identity(number), 0) for number in range(3, 12)]
        assert all(item.pending and item.registered_at == 0 for item in waiting)
        assert [item.registration_order for item in waiting] == sorted(
            item.registration_order for item in waiting
        )
        await timeline.advance(0.1)
        await _publish_peer_overlay(
            output, caption(12, identity(2), "corrected B", revision=2), source_order=2
        )
        await timeline.advance(0)
        assert [block.primary_text for block in presenter.snapshot().blocks] == [
            "visible 1",
            "corrected B",
        ]
        assert presenter.peer_fragment_state(identity(2), 0).last_meaningful_visible_at == 0.1
        assert [
            (
                presenter.peer_fragment_state(identity(number), 0).registered_at,
                presenter.peer_fragment_state(identity(number), 0).registration_order,
            )
            for number in range(3, 12)
        ] == [(item.registered_at, item.registration_order) for item in waiting]
        await timeline.advance(0.9)
        await output.wait_for_peer_output_idle()
        assert [block.primary_text for block in presenter.snapshot().blocks] == [
            "waiting 10",
            "waiting 11",
        ]
        assert any(
            [block.primary_text for block in snapshot.blocks] == ["visible 1", "corrected B"]
            for snapshot in surface.snapshots
        )
    finally:
        await output.close()
        await presenter.close()


async def admit(output, number):
    return await output.admit_translation_parent(
        parent_id=str(identity(1000 + number)),
        channel="peer",
        origin="peer",
        turn_generation=1,
        turn_order=number,
        retained_payloads=(f"source {number}",),
        destination_targets={"overlay": frozenset({0})},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reservation", ["pending", "applied", "revoked"])
@pytest.mark.parametrize("preview", [False, True])
async def test_f2_early_delivery_bypasses_activation_but_never_terminal_reservation(
    reservation, preview
):
    revoke = reservation == "revoked"
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep
    )
    output = OutputRuntime(chatbox=RecordingChatbox(), clock=timeline.clock, overlay_sink=presenter)
    adapter = OverlayEventAdapter(clock=timeline.clock)
    output.activate_peer_generation(1)
    try:
        assert await admit(output, 1) == frozenset({"overlay"})
        assert await admit(output, 2) == frozenset({"overlay"})
        scope = OverlayPublicationScope(
            turn_kind="peer", parent_utterance_id=identity(1002), turn_generation=1, turn_order=2
        )
        build = adapter.translation_stream_update if preview else adapter.translation_final
        event = build(
            utterance_id=identity(2002),
            channel="peer",
            text="revoked pending final" if revoke else "valid early final",
            source_text="source 2",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            peer_caption_revision=1 if preview else 0,
            output_scope=scope,
        )
        result = await output.publish_overlay_event(event, publication_generation=1, source_order=2)
        assert result.decision.reason == "accepted_handoff"
        if revoke:
            for number in range(3, 12):
                assert await admit(output, number) == frozenset({"overlay"})
        elif reservation == "applied":
            await output.complete_translation_parent_target(
                parent_id=str(identity(1002)),
                origin="peer",
                destination_indexes={"overlay": 0},
                destinations={"overlay"},
            )
        await asyncio.wait_for(output.wait_for_peer_output_idle(), 1)
        await timeline.advance(0)
        decisions = [
            decision
            for decision in output.routing_decisions
            if decision.publication_id == event.event_id
        ]
        if revoke:
            assert presenter.snapshot().blocks == []
            assert decisions[-1].reason == "output_overload"
            await timeline.advance(2)
            assert presenter.snapshot().blocks == []
        else:
            assert [block.primary_text for block in presenter.snapshot().blocks] == [
                "valid early final"
            ]
            assert decisions[-1].reason == "application_applied"
    finally:
        await output.close()
        await presenter.close()


@pytest.mark.asyncio
async def test_f3_large_valid_partition_keeps_logical_capacity_and_self_progress():
    timeline, surface = Timeline(), Surface()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(), clock=timeline.clock, sleep=timeline.sleep, bridge=surface
    )
    unit, text = identity(8001), "가" * 3600
    fragments = peer_translation_fragments(text)
    try:
        assert len(fragments) == 65
        event = caption(1, unit, text)
        assert (await presenter.emit(event)).outcome == "applied"
        assert len(presenter._entries) <= PRESENTER_ENTRY_LIMIT
        assert presenter._entries[("peer", unit)].peer_fragments == fragments
        parent = presenter._entries[("peer", unit)]
        base_bytes = presenter._acceptance.entry_payload_bytes(
            replace(parent, peer_fragments=(), peer_departed_indices=set())
        )
        expected_bytes = (
            base_bytes
            + sum(len(fragment.encode("utf-8")) for fragment in fragments)
            + len(fragments) * PEER_FRAGMENT_REGISTRATION_BYTES
        )
        assert presenter._acceptance.entry_payload_bytes(parent) == expected_bytes
        assert (
            presenter._acceptance.projected_entry_payload_bytes(
                None, event, peer_fragments=fragments
            )
            >= expected_bytes
        )
        assert (await presenter.emit(own_speech())).outcome == "applied"
        await timeline.advance(0)
        assert any(block.channel == "self" for block in presenter.snapshot().blocks)
        states = presenter.peer_fragment_states(unit)
        assert len(states) == len(fragments)
        assert sum(state.pending for state in states) == len(fragments) - 1
        assert all(
            state.visible_since is None and state.expires_at is None
            for state in states
            if state.pending
        )
        assert "".join("".join(fragment.split()) for fragment in fragments) == text
        await timeline.advance(1)
        observed = []
        for snapshot in surface.snapshots:
            for block in snapshot.blocks:
                if block.channel == "peer" and block.id not in observed:
                    observed.append(block.id)
        assert observed == [fragment_id(unit, index) for index in range(len(fragments))]
        assert len(presenter._entries) <= PRESENTER_ENTRY_LIMIT
        assert any(block.channel == "self" for block in presenter.snapshot().blocks)
        assert presenter._acceptance.entry_payload_bytes(parent) == (
            base_bytes
            + sum(len(fragment.encode("utf-8")) for fragment in fragments)
            + PEER_FRAGMENT_REGISTRATION_BYTES
            + (len(fragments) - 1) * PEER_FRAGMENT_DEPARTURE_BYTES
        )
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["retire", "shutdown"])
@pytest.mark.parametrize("root_waiting", [False, True])
async def test_f4_retirement_settles_pending_descendants_after_output_transfer(
    ending, root_waiting
):
    timeline = Timeline()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=timeline.clock,
        sleep=timeline.sleep,
        native_retry_enabled=True,
    )
    output = OutputRuntime(chatbox=RecordingChatbox(), clock=timeline.clock, overlay_sink=presenter)
    unit = identity(70)
    try:
        await presenter.emit(own_speech())
        if root_waiting:
            await _publish_peer_overlay(
                output, caption(1, identity(69), "protected neighbor"), source_order=1
            )
            await output.wait_for_peer_output_idle()
        event = caption(2, unit, "가" * 64)
        await _publish_peer_overlay(output, event, source_order=2)
        await output.wait_for_peer_output_idle()
        await timeline.advance(0)
        root = presenter.peer_fragment_state(unit, 0)
        tail = presenter.peer_fragment_state(unit, 1)
        assert root.pending is root_waiting
        assert tail.pending and tail.visible_since is None
        before = presenter.snapshot()
        await timeline.advance(0.2)
        if ending == "retire":
            output.retire_peer_generation(1)
            await output.wait_for_peer_output_idle()
        else:
            await output.close()
        await timeline.advance(0.8)
        assert presenter.peer_fragment_state(unit, 1) is None
        assert all(block.id != fragment_id(unit, 1) for block in presenter.snapshot().blocks)
        assert presenter.snapshot().native_quiet_tail_episodes == before.native_quiet_tail_episodes
        if root_waiting:
            assert all(block.id != fragment_id(unit, 0) for block in presenter.snapshot().blocks)
        else:
            assert any(block.id == fragment_id(unit, 0) for block in presenter.snapshot().blocks)
        if ending == "retire":
            output.activate_peer_generation(2)
            fresh = caption(3, identity(71), "current generation")
            await output.publish_overlay_event(fresh, publication_generation=2, source_order=1)
            await timeline.advance(0)
            await timeline.advance(1)
            await output.wait_for_peer_output_idle()
            assert any(
                block.primary_text == "current generation" for block in presenter.snapshot().blocks
            )
            stale = await output.publish_overlay_event(
                replace(event, event_id="late-retired", seq=100),
                publication_generation=1,
                source_order=2,
            )
            assert stale.decision.reason == "publication_generation_retired"
    finally:
        await output.close()
        await presenter.close()


def test_release_drops_payload_ownership_after_accounting_and_preserves_terminal_identity():
    admission = DestinationBatchAdmission()
    for order, text in ((1, "first payload"), (2, "second payload")):
        assert admission.admit_parent(
            parent_id=str(order),
            channel="peer",
            origin="peer",
            turn_generation=1,
            turn_order=order,
            payloads={id(text): text},
            destination_targets={"overlay": frozenset({0})},
        ) == frozenset({"overlay"})
    first = admission.batches[("peer", "1")]
    second = admission.batches[("peer", "2")]
    assert admission.snapshot()["reserved_bytes"] == len("first payloadsecond payload")
    admission.release(first, "applied")
    assert first.retained_payloads == {} and first.reserved_bytes == 0
    assert first.disposition == "applied" and (first.turn_generation, first.turn_order) == (1, 1)
    assert second.active and admission.snapshot()["reserved_bytes"] == len("second payload")
    admission.release(first, "cancelled_local")
    assert first.disposition == "applied"
    assert admission.snapshot()["reserved_bytes"] == len("second payload")
    admission.release(second, "output_overload")
    assert second.retained_payloads == {} and second.reserved_bytes == 0
    assert admission.snapshot()["reserved_bytes"] == 0
    assert admission.snapshot()["batches"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("settlement", ["applied", "overload"])
async def test_pending_sibling_authority_outlives_success_but_not_parent_revocation(settlement):
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
    unit = identity(2070)
    try:
        await presenter.emit(own_speech())
        assert await admit(output, 1) == frozenset({"overlay"})
        assert await admit(output, 2) == frozenset({"overlay"})
        reservation = output._batch_admission.batches[("peer", str(identity(1002)))]
        scope = OverlayPublicationScope(
            turn_kind="peer", parent_utterance_id=identity(1002), turn_generation=1, turn_order=2
        )
        event = adapter.translation_final(
            utterance_id=unit,
            channel="peer",
            text="가" * 64,
            source_text="source 2",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            peer_caption_revision=0,
            output_scope=scope,
        )
        await output.publish_overlay_event(event, publication_generation=1, source_order=2)
        await output.wait_for_peer_output_idle()
        assert presenter.peer_fragment_state(unit, 0).visible_since == 0
        assert presenter.peer_fragment_state(unit, 1).pending
        if settlement == "applied":
            await output.publish_overlay_event(
                adapter.utterance_closed(
                    utterance_id=unit, channel="peer", is_final=True, output_scope=scope
                ),
                publication_generation=1,
                source_order=2,
            )
            await output.wait_for_peer_output_idle()
            assert reservation.disposition == "applied"
            assert reservation.retained_payloads == {} and reservation.reserved_bytes == 0
            output.retire_peer_generation(0)
        else:
            for number in range(3, 12):
                assert await admit(output, number) == frozenset({"overlay"})
            assert reservation.disposition == "output_overload"
        await timeline.advance(1)
        tail = presenter.peer_fragment_state(unit, 1)
        if settlement == "applied":
            assert tail is not None and not tail.pending
            assert tail.visible_since == 1 and tail.expires_at == 9
            assert any(block.id == tail.id for block in presenter.snapshot().blocks)
        else:
            assert tail is None
            assert all(block.id != fragment_id(unit, 1) for block in presenter.snapshot().blocks)
    finally:
        await output.close()
        await presenter.close()
