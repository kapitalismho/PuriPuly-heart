from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter
from puripuly_heart.domain.models import Transcript
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.helpers.overlay_source_cards import assignment, source_child, translated_event


@dataclass
class RecordingBridge:
    scenes: list = field(default_factory=list)
    deadlines: dict = field(default_factory=dict)

    async def replace_snapshot(self, snapshot, *, block_expirations=None):
        self.scenes.append(snapshot)
        self.deadlines = dict(block_expirations or {})


@pytest.mark.asyncio
async def test_internal_request_a_b_a_accumulates_atomically_and_renews_only_a():
    clock = FakeClock(_now=10.0)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(
        OverlayCalibration(), bridge=bridge, clock=clock, native_retry_enabled=True
    )
    adapter = OverlayEventAdapter(clock=clock)
    group = UUID(int=217)
    sources = [
        source_child(i, text, group=group) for i, text in enumerate(("first", "B", "again"), 1)
    ]
    try:
        await presenter.emit(translated_event(adapter, sources[0], "A one", assignment(1, "A")))
        first = presenter.snapshot().blocks[0]
        first_deadline = bridge.deadlines[first.id]
        clock.advance(1)
        await presenter.emit(
            translated_event(adapter, sources[1], "B one", assignment(2, "B", index=1))
        )
        other = presenter.snapshot().blocks[-1]
        other_deadline = bridge.deadlines[other.id]
        clock.advance(1)
        await presenter.emit(translated_event(adapter, sources[2], "A two", assignment(3, "A")))
        blocks = presenter.snapshot().blocks
        assert [block.id for block in blocks] == [other.id, first.id]
        assert blocks[0] == other
        assert blocks[-1].occupant_key == first.occupant_key
        assert blocks[-1].primary_text == "A one A two"
        assert blocks[-1].secondary_text == "first again"
        assert blocks[-1].appearance_seq > other.appearance_seq
        assert blocks[-1].card_revision > first.card_revision
        assert blocks[-1].publication_order == 3
        assert presenter.snapshot().native_fresh_render_targets.peer == first.id
        assert bridge.deadlines[first.id] == 20.0 > first_deadline
        assert bridge.deadlines[other.id] == other_deadline
        assert [source.order for source in sources] == [(1, 1, 0), (1, 2, 0), (1, 3, 0)]
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("groups", [(None, None, None), (UUID(int=1), UUID(int=1), UUID(int=2))])
async def test_unknown_and_different_groups_never_append(groups):
    clock = FakeClock(_now=10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    try:
        ids = []
        for order, speaker in enumerate(("A", "B", "A"), 1):
            source = source_child(order, f"source {order}", group=groups[order - 1])
            await presenter.emit(
                translated_event(
                    adapter, source, f"translation {order}", assignment(order, speaker)
                )
            )
            ids.append(presenter.snapshot().blocks[-1].id)
        assert len(set(ids)) == 3
        assert presenter.snapshot().blocks[-1].primary_text == "translation 3"
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("known", [False, True])
async def test_gray_requires_genuine_identity_and_overflow_divider(known):
    clock = FakeClock(_now=10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    try:
        ids = []
        for order, speaker in enumerate(("E", "F", "E"), 1):
            source = source_child(order, f"source {order}", group=UUID(int=217))
            evidence = assignment(order, speaker if known else None, index=None, overflow=known)
            await presenter.emit(translated_event(adapter, source, f"words {order}", evidence))
            ids.append(presenter.snapshot().blocks[-1].id)
        assert presenter.snapshot().blocks[-1].primary_text == (
            "words 1 words 3" if known else "words 3"
        )
        assert (ids[0] == ids[2]) is known
        assert presenter.snapshot().speaker_divider is known
        assert all(block.speaker_style == "gray" for block in presenter.snapshot().blocks)
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_source_completion_targets_retry_metadata_markers_and_reconnect_do_not_renew():
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(
        OverlayCalibration(), bridge=bridge, clock=clock, native_retry_enabled=True
    )
    adapter = OverlayEventAdapter(clock=clock)
    source = source_child(1, "selected source", group=UUID(int=217))
    attempt = uuid4()
    try:
        transcript = Transcript(
            attempt, "ignored downstream source", True, channel="peer", source_child=source
        )
        source_event = adapter.transcript_final(
            transcript,
            source_language="en",
            target_language="ko",
            speaker_assignment=assignment(1, "A"),
        )
        await presenter.emit(source_event)
        first = presenter.snapshot().blocks[0]
        deadline = bridge.deadlines[first.id]
        clock.advance(2)
        completed = translated_event(
            adapter, source, "translation", assignment(1, "A"), attempt=attempt
        )
        await presenter.emit(completed)
        assert presenter.snapshot().blocks[0].secondary_text == source.effective_text
        assert bridge.deadlines[first.id] == deadline
        translated = translated_event(
            adapter, source, "another target", assignment(1, "A"), target="ja", attempt=uuid4()
        )
        await presenter.emit(translated)
        current = presenter.snapshot().blocks[0]
        assert current.id == first.id
        assert current.primary_text == "another target"
        assert current.primary_language == "ja"
        assert current.appearance_seq == first.appearance_seq
        assert bridge.deadlines[first.id] == deadline
        await presenter.emit(translated)
        retry = translated_event(
            adapter,
            source,
            "retry result",
            assignment(1, "A", index=2),
            target="ja",
            attempt=uuid4(),
        )
        await presenter.emit(retry)
        metadata = replace(retry, event_id="metadata", seq=retry.seq + 1, update_id="metadata-only")
        await presenter.emit(metadata)
        await presenter.emit(adapter.utterance_closed(utterance_id=attempt, channel="peer"))
        marker = replace(
            adapter.utterance_closed(utterance_id=uuid4(), channel="peer"), source_child=source
        )
        await presenter.emit(marker)
        await presenter.begin_native_retry_epoch(enabled=True)
        current = presenter.snapshot().blocks[0]
        assert current.id == first.id
        assert current.speaker_style == first.speaker_style
        assert current.appearance_seq == first.appearance_seq
        assert current.secondary_text == source.effective_text
        assert bridge.deadlines[first.id] == deadline
        assert len(presenter.snapshot().blocks) == 1
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_distinct_equal_text_source_units_are_not_deduplicated():
    clock = FakeClock(_now=10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    try:
        for order in (1, 2):
            await presenter.emit(
                translated_event(
                    adapter,
                    source_child(order, "yes", group=UUID(int=217)),
                    "yes",
                    assignment(order, "A"),
                )
            )
        block = presenter.snapshot().blocks[0]
        assert block.primary_text == "yes yes"
        assert block.secondary_text == "yes yes"
        assert block.card_revision == 2
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("removal", ["expiry", "eviction", "self"])
async def test_removed_card_history_never_resurrects(removal):
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(OverlayCalibration(), bridge=bridge, clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    group = UUID(int=217)
    source = source_child(1, "old A", group=group)
    first_event = translated_event(adapter, source, "old A translation", assignment(1, "A"))
    try:
        await presenter.emit(first_event)
        first_id = presenter.snapshot().blocks[0].id
        await presenter.emit(
            translated_event(
                adapter, source_child(2, "B", group=group), "B", assignment(2, "B", index=1)
            )
        )
        if removal == "expiry":
            clock.advance(8.1)
        elif removal == "eviction":
            await presenter.emit(
                translated_event(
                    adapter, source_child(3, "C", group=group), "C", assignment(3, "C", index=2)
                )
            )
        else:
            self_turn = uuid4()
            await presenter.emit(
                adapter.self_active_update(
                    text="protected SELF", utterance_id=self_turn, occupant_key=f"self:{self_turn}"
                )
            )
            assert any(block.channel == "self" for block in presenter.snapshot().blocks)
        replay = translated_event(adapter, source, "old retry", assignment(1, "A"), attempt=uuid4())
        assert (await presenter.emit(replay)).outcome == "stale"
        await presenter.emit(
            translated_event(
                adapter,
                source_child(4, "new A", group=group),
                "new A translation",
                assignment(4, "A"),
            )
        )
        newest = next(
            block
            for block in presenter.snapshot().blocks
            if block.channel == "peer" and block.primary_text == "new A translation"
        )
        assert newest.id != first_id
        assert newest.secondary_text == "new A"
        if removal == "self":
            assert any(block.channel == "self" for block in presenter.snapshot().blocks)
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("incompatibility", ["target", "source", "style", "epoch", "layout"])
async def test_new_source_requires_compatible_language_style_epoch_and_layout(incompatibility):
    clock = FakeClock(_now=10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    group = UUID(int=217)
    try:
        await presenter.emit(
            translated_event(
                adapter, source_child(1, "first", group=group), "first", assignment(1, "A")
            )
        )
        old = presenter.snapshot().blocks[0]
        if incompatibility == "layout":
            await presenter.update_display_preferences(
                show_translation=True, show_peer_original=False
            )
        evidence = assignment(
            2,
            "A",
            scope="next" if incompatibility == "epoch" else "epoch",
            index=1 if incompatibility == "style" else 0,
        )
        source = source_child(
            2, "second", group=group, language="es" if incompatibility == "source" else "en"
        )
        await presenter.emit(
            translated_event(
                adapter,
                source,
                "second",
                evidence,
                target="ja" if incompatibility == "target" else "ko",
            )
        )
        assert len(presenter.snapshot().blocks) == 2
        assert presenter.snapshot().blocks[-1].id != old.id
        assert presenter.snapshot().blocks[-1].primary_text == "second"
        assert presenter.snapshot().blocks[0].speaker_style == old.speaker_style
        if incompatibility == "epoch":
            assert presenter.snapshot().blocks[-1].speaker_style == "gray"
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_source_application_waits_only_for_protection_and_ttl_starts_at_application():
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(
        OverlayCalibration(), bridge=bridge, clock=clock, visible_window_target_blocks=1
    )
    adapter = OverlayEventAdapter(clock=clock)
    self_turn = uuid4()
    await presenter.emit(
        adapter.self_active_update(
            text="protected", utterance_id=self_turn, occupant_key=f"self:{self_turn}"
        )
    )
    event = translated_event(
        adapter, source_child(1, "A", group=UUID(int=217)), "A", assignment(1, "A")
    )
    waiting = asyncio.create_task(presenter.emit_peer_when_admissible(event))
    try:
        await asyncio.sleep(0)
        assert not waiting.done()
        clock.advance(3)
        await presenter.emit(adapter.self_active_clear())
        assert (await asyncio.wait_for(waiting, 1)).outcome == "applied"
        first = presenter.snapshot().blocks[0]
        assert bridge.deadlines[first.id] == 21
        replacement = translated_event(
            adapter, source_child(2, "B", group=UUID(int=218)), "B", assignment(2, "B", index=1)
        )
        assert (
            await asyncio.wait_for(presenter.emit_peer_when_admissible(replacement), 1)
        ).outcome == "applied"
        assert clock.now() == 13
        assert presenter.snapshot().blocks[0].primary_text == "B"
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await presenter.close()


@pytest.mark.asyncio
async def test_stale_expiry_callback_cannot_remove_appended_card():
    clock = FakeClock(_now=10)
    sleepers = []

    async def sleep(delay):
        release = asyncio.Event()
        sleepers.append(release)
        try:
            await release.wait()
        except asyncio.CancelledError:
            if len(sleepers) == 1:
                await release.wait()
            else:
                raise

    presenter = OverlayPresenter(OverlayCalibration(), clock=clock, sleep=sleep)
    adapter = OverlayEventAdapter(clock=clock)
    try:
        await presenter.emit(
            translated_event(
                adapter, source_child(1, "first", group=UUID(int=217)), "first", assignment(1, "A")
            )
        )
        await asyncio.sleep(0)
        old_timer = sleepers[0]
        clock.advance(2)
        await presenter.emit(
            translated_event(
                adapter,
                source_child(2, "second", group=UUID(int=217)),
                "second",
                assignment(2, "A"),
            )
        )
        await asyncio.sleep(0)
        clock.advance(6)
        old_timer.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert presenter.snapshot().blocks[0].primary_text == "first second"
        clock.advance(2)
        sleepers[-1].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert presenter.snapshot().blocks == []
    finally:
        for release in sleepers:
            release.set()
        await presenter.close()


@pytest.mark.asyncio
async def test_many_enclosing_groups_retire_history_without_resurrecting_old_sources():
    clock = FakeClock(_now=10)
    presenter = OverlayPresenter(OverlayCalibration(), clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    first = source_child(1, "again", group=UUID(int=1))
    try:
        for order in range(1, 140):
            source = source_child(order, "again", group=UUID(int=order))
            result = await presenter.emit(
                translated_event(adapter, source, "again", assignment(order, "A"))
            )
            assert result.outcome == "applied"
            assert len(presenter.snapshot().blocks) <= 2
            assert all(block.primary_text == "again" for block in presenter.snapshot().blocks)
        assert (
            await presenter.emit(
                translated_event(adapter, first, "old retry", assignment(1, "A"), attempt=uuid4())
            )
        ).outcome == "stale"
        assert all(block.primary_text == "again" for block in presenter.snapshot().blocks)
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_hidden_source_binds_and_starts_lifetime_at_first_readable_translation():
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(
        OverlayCalibration(), bridge=bridge, clock=clock, show_peer_original=False
    )
    adapter = OverlayEventAdapter(clock=clock)
    group = UUID(int=217)
    first_source = source_child(1, "first source", group=group)
    pending_source = source_child(2, "next source", group=group)
    attempt = uuid4()
    try:
        await presenter.emit(translated_event(adapter, first_source, "first", assignment(1, "A")))
        first = presenter.snapshot().blocks[0]
        deadline = bridge.deadlines[first.id]
        clock.advance(1)
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    attempt, "next source", True, channel="peer", source_child=pending_source
                ),
                source_language="en",
                target_language="ko",
                speaker_assignment=assignment(2, "A"),
            )
        )
        assert presenter.snapshot().blocks == [first]
        assert bridge.deadlines[first.id] == deadline
        clock.advance(2)
        await presenter.emit(
            translated_event(adapter, pending_source, "next", assignment(2, "A"), attempt=attempt)
        )
        assert len(presenter.snapshot().blocks) == 1
        block = presenter.snapshot().blocks[0]
        assert block.id == first.id
        assert block.primary_text == "first next"
        assert block.secondary_enabled is False
        assert bridge.deadlines[first.id] == 21
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_mailbox_reconnect_preserves_card_revision_and_original_deadlines():
    import json

    from puripuly_heart.core.overlay.bridge_mailbox import OverlayBridgeMailbox

    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(OverlayCalibration(), bridge=bridge, clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    group = UUID(int=217)
    try:
        for order, speaker in enumerate(("A", "B", "A"), 1):
            await presenter.emit(
                translated_event(
                    adapter,
                    source_child(order, f"source {order}", group=group),
                    f"words {order}",
                    assignment(order, speaker, index=1 if speaker == "B" else 0),
                )
            )
            clock.advance(1)
        snapshot = presenter.snapshot()
        mailbox = OverlayBridgeMailbox(
            initial_snapshot=snapshot,
            clock=clock,
            scene_byte_limit=1024 * 1024,
            control_byte_limit=4096,
            control_slot_limit=8,
            delivery_receipt_limit=64,
        )
        scene = mailbox.make_scene(snapshot, bridge.deadlines)
        mailbox.current_scene = scene
        mailbox.rebuild_current()
        assert mailbox.current_scene.block_expirations == scene.block_expirations
        assert (
            json.loads(mailbox.revalidated_scene_message(mailbox.current_scene))["payload"]
            == snapshot.to_dict()
        )
        clock.advance(6.5)
        replay = json.loads(mailbox.revalidated_scene_message(mailbox.current_scene))["payload"]
        assert len(replay["blocks"]) == 1
        assert replay["blocks"][0] == snapshot.blocks[-1].to_dict()
        assert replay["blocks"][0]["card_revision"] == snapshot.blocks[-1].card_revision
        assert scene.block_expirations[snapshot.blocks[-1].id] == 20
        clock.advance(0.5)
        assert (
            json.loads(mailbox.revalidated_scene_message(mailbox.current_scene))["payload"][
                "blocks"
            ]
            == []
        )
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_expired", [False, True])
async def test_hidden_pending_late_arrival_fence_never_resurrects_former_candidate(pending_expired):
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(
        OverlayCalibration(), bridge=bridge, clock=clock, show_peer_original=False
    )
    adapter = OverlayEventAdapter(clock=clock)
    group = UUID(int=217)
    pending = source_child(2, "next source", group=group)
    attempt = uuid4()
    try:
        await presenter.emit(
            translated_event(
                adapter, source_child(1, "old source", group=group), "old", assignment(1, "A")
            )
        )
        former_id = presenter.snapshot().blocks[0].id
        clock.advance(7)
        await presenter.emit(
            adapter.transcript_final(
                Transcript(attempt, "next source", True, channel="peer", source_child=pending),
                source_language="en",
                target_language="ko",
                speaker_assignment=assignment(2, "A"),
            )
        )
        await presenter.emit(adapter.utterance_closed(utterance_id=attempt, channel="peer"))
        clock.advance(5.1 if pending_expired else 2)
        receipt = await presenter.emit(
            translated_event(adapter, pending, "next", assignment(2, "A"), attempt=attempt)
        )
        if pending_expired:
            assert receipt.outcome == "stale"
            assert presenter.snapshot().blocks == []
        else:
            assert receipt.outcome == "applied"
            current = presenter.snapshot().blocks[0]
            assert current.id != former_id
            assert current.primary_text == "next"
            assert bridge.deadlines[current.id] == 27
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_visible_a_append_preserves_active_self_identity_and_deadline():
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(OverlayCalibration(), bridge=bridge, clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    group = UUID(int=217)
    try:
        await presenter.emit(
            translated_event(
                adapter, source_child(1, "first", group=group), "first", assignment(1, "A")
            )
        )
        first = presenter.snapshot().blocks[0]
        clock.advance(1)
        self_turn = uuid4()
        await presenter.emit(
            adapter.self_active_update(
                text="active SELF", utterance_id=self_turn, occupant_key=f"self:{self_turn}"
            )
        )
        self_block = next(block for block in presenter.snapshot().blocks if block.channel == "self")
        self_deadline = bridge.deadlines[self_block.id]
        clock.advance(1)
        await presenter.emit_peer_when_admissible(
            translated_event(
                adapter, source_child(2, "next", group=group), "next", assignment(2, "A")
            )
        )
        blocks = presenter.snapshot().blocks
        assert blocks[0] == self_block
        assert blocks[-1].id == first.id
        assert blocks[-1].primary_text == "first next"
        assert bridge.deadlines[self_block.id] == self_deadline
        assert bridge.deadlines[first.id] == 20
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_legacy_finalized_completion_never_renews_or_extends_lifetime(channel):
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(OverlayCalibration(), bridge=bridge, clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    turn = uuid4()
    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(turn, "readable source", True, channel=channel),
                source_language="en",
                target_language="ko",
            )
        )
        initial = presenter.snapshot().blocks[0]
        deadline = bridge.deadlines[initial.id]
        clock.advance(7)
        await presenter.emit(
            adapter.translation_final(
                utterance_id=turn,
                channel=channel,
                text="late translation",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
            )
        )
        assert bridge.deadlines[initial.id] == deadline == 18
        assert presenter.snapshot().blocks[0].card_revision > initial.card_revision
        clock.advance(1.1)
        replay = adapter.translation_final(
            utterance_id=turn,
            channel=channel,
            text="later translation",
            source_language="en",
            target_language="ja",
            applied_context_mode=None,
        )
        assert (await presenter.emit(replay)).outcome == "stale"
        assert presenter.snapshot().blocks == []
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("canonical", [False, True])
async def test_self_preview_progress_and_finalization_keep_existing_lifetime(canonical):
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(OverlayCalibration(), bridge=bridge, clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    turn = uuid4()
    try:
        await presenter.emit(
            adapter.self_active_update(
                text="preview", utterance_id=turn, occupant_key=f"self:{turn}"
            )
        )
        initial = presenter.snapshot().blocks[0]
        assert bridge.deadlines[initial.id] == 18
        clock.advance(1)
        await presenter.emit(
            adapter.self_active_update(
                text="preview progress", utterance_id=turn, occupant_key=f"self:{turn}"
            )
        )
        assert bridge.deadlines[initial.id] == 19
        clock.advance(1)
        source = (
            replace(source_child(1, "final source", group=None), channel="self")
            if canonical
            else None
        )
        await presenter.emit(
            adapter.transcript_final(
                Transcript(turn, "final source", True, channel="self", source_child=source),
                source_language="en",
                target_language="ko",
            )
        )
        finalized = presenter.snapshot().blocks[0]
        assert finalized.id == initial.id
        assert finalized.occupant_key == initial.occupant_key
        assert finalized.block_variant == "finalized"
        assert bridge.deadlines[initial.id] == 19
        clock.advance(5)
        completed = adapter.translation_final(
            utterance_id=turn,
            channel="self",
            text="completion",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
        )
        if canonical:
            completed = replace(completed, source_child=source)
        await presenter.emit(completed)
        assert bridge.deadlines[initial.id] == 19
    finally:
        await presenter.close()


@pytest.mark.asyncio
async def test_first_readable_preference_reveal_is_not_later_renewed_by_completion():
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(
        OverlayCalibration(), bridge=bridge, clock=clock, show_peer_original=False
    )
    adapter = OverlayEventAdapter(clock=clock)
    source = source_child(1, "hidden source", group=UUID(int=217))
    attempt = uuid4()
    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(attempt, "hidden source", True, channel="peer", source_child=source),
                source_language="en",
                target_language="ko",
                speaker_assignment=assignment(1, "A"),
            )
        )
        assert presenter.snapshot().blocks == []
        clock.advance(2)
        await presenter.update_display_preferences(show_translation=True, show_peer_original=True)
        first = presenter.snapshot().blocks[0]
        assert bridge.deadlines[first.id] == 20
        clock.advance(1)
        await presenter.emit(
            translated_event(adapter, source, "translation", assignment(1, "A"), attempt=attempt)
        )
        assert presenter.snapshot().blocks[0].id == first.id
        assert bridge.deadlines[first.id] == 20
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("detected_language", "resolved_language", "expected_language"),
    [(None, "ja", "ja"), ("en", "ja", "en")],
)
async def test_canonical_text_keeps_resolved_language_fallback_without_overriding_detection(
    detected_language, resolved_language, expected_language
):
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(OverlayCalibration(), bridge=bridge, clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    source = replace(
        source_child(1, "canonical source", group=UUID(int=217)),
        language=detected_language,
    )
    attempt = uuid4()
    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(attempt, "ignored source", True, channel="peer", source_child=source),
                source_language="",
                target_language="zh-TW",
                speaker_assignment=assignment(1, "A"),
            )
        )
        first = presenter.snapshot().blocks[0]
        deadline = bridge.deadlines[first.id]
        clock.advance(1)
        await presenter.emit(
            replace(
                adapter.translation_final(
                    utterance_id=attempt,
                    channel="peer",
                    text="translation",
                    source_text="ignored downstream source",
                    source_language=resolved_language,
                    target_language="zh-TW",
                    applied_context_mode=None,
                ),
                source_child=source,
            )
        )
        current = presenter.snapshot().blocks[0]
        assert current.id == first.id
        assert current.secondary_text == source.effective_text
        assert current.secondary_language == expected_language
        assert current.primary_language == "zh-TW"
        assert source.language == detected_language
        assert bridge.deadlines[first.id] == deadline
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("removal", ["expiry", "eviction"])
@pytest.mark.parametrize("caption_churn", [0, 100])
async def test_first_canonical_self_completion_carries_exact_legacy_attempt_retirement(
    removal, caption_churn
):
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(OverlayCalibration(), bridge=bridge, clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    attempt = uuid4()
    source = replace(source_child(1, "old SELF", group=None), channel="self")
    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(attempt, source.effective_text, True, channel="self"),
                source_language="en",
                target_language="ko",
            )
        )
        original = presenter.snapshot().blocks[0]
        assert bridge.deadlines[original.id] == 18
        if removal == "expiry":
            clock.advance(9)
            await presenter.begin_native_retry_epoch(enabled=True)
        else:
            clock.advance(1)
            for text in ("newer one", "newer two"):
                await presenter.emit(
                    adapter.transcript_final(
                        Transcript(uuid4(), text, True, channel="self"),
                        source_language="en",
                        target_language="ko",
                    )
                )
        for index in range(caption_churn):
            await presenter.emit(
                adapter.transcript_final(
                    Transcript(uuid4(), f"newer {index}", True, channel="self"),
                    source_language="en",
                    target_language="ko",
                )
            )
        before = presenter.snapshot()
        deadlines = dict(bridge.deadlines)
        assert original.id not in {block.id for block in before.blocks}
        completion = replace(
            adapter.translation_final(
                utterance_id=attempt,
                channel="self",
                text="late translation",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
            ),
            source_child=source,
        )
        assert (await presenter.emit(completion)).outcome == "stale"
        assert presenter.snapshot() == before
        assert bridge.deadlines == deadlines
        retry = replace(
            completion, utterance_id=uuid4(), event_id=uuid4().hex, seq=completion.seq + 1
        )
        assert (await presenter.emit(retry)).outcome == "stale"
        assert presenter.snapshot() == before
        new_source = replace(source_child(2, "old SELF", group=None), channel="self")
        assert (
            await presenter.emit(
                adapter.transcript_final(
                    Transcript(
                        uuid4(),
                        new_source.effective_text,
                        True,
                        channel="self",
                        source_child=new_source,
                    ),
                    source_language="en",
                    target_language="ko",
                )
            )
        ).outcome == "applied"
        newest = presenter.snapshot().blocks[-1]
        assert newest.primary_text == source.effective_text
        assert newest.id != original.id
        assert bridge.deadlines[newest.id] == clock.now() + 8
        assert len(presenter.snapshot().blocks) <= 2
    finally:
        await presenter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("active_preview", [False, True])
async def test_first_canonical_self_completion_upgrades_live_legacy_caption_without_new_lifetime(
    active_preview,
):
    clock = FakeClock(_now=10)
    bridge = RecordingBridge()
    presenter = OverlayPresenter(OverlayCalibration(), bridge=bridge, clock=clock)
    adapter = OverlayEventAdapter(clock=clock)
    attempt = uuid4()
    source = replace(source_child(1, "SELF source", group=None), channel="self")
    try:
        if active_preview:
            await presenter.emit(
                adapter.self_active_update(
                    text="preview", utterance_id=attempt, occupant_key=f"self:{attempt}"
                )
            )
            clock.advance(1)
            await presenter.emit(
                adapter.self_active_update(
                    text="preview progress", utterance_id=attempt, occupant_key=f"self:{attempt}"
                )
            )
            clock.advance(1)
        await presenter.emit(
            adapter.transcript_final(
                Transcript(attempt, source.effective_text, True, channel="self"),
                source_language="en",
                target_language="ko",
            )
        )
        original = presenter.snapshot().blocks[0]
        deadline = bridge.deadlines[original.id]
        clock.advance(5)
        completed = replace(
            adapter.translation_final(
                utterance_id=attempt,
                channel="self",
                text="completion",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
            ),
            source_child=source,
        )
        assert (await presenter.emit(completed)).outcome == "applied"
        current = presenter.snapshot().blocks[0]
        assert current.id == original.id
        assert current.occupant_key == original.occupant_key
        assert current.secondary_text == "completion"
        assert bridge.deadlines[current.id] == deadline
        assert deadline == (19 if active_preview else 18)
    finally:
        await presenter.close()
