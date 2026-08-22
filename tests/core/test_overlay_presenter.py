from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.diagnostics import OverlayDiagnosticsRecorder
from puripuly_heart.core.overlay.presenter import (
    SELF_TRANSLATION_MIN_VISIBLE_SECONDS,
    OverlayPresenter,
)
from puripuly_heart.core.overlay.protocol import (
    U64_MAX,
    NativeFreshRenderGenerations,
    OverlayPresentationBlock,
    OverlayPresentationCalibration,
)
from puripuly_heart.core.overlay.sink import (
    OverlayEventAdapter,
    PeerActiveUpdate,
    SelfActiveClear,
    SelfActiveUpdate,
    SelfTranscriptFinal,
    TranslationFinal,
    TranslationStreamUpdate,
)
from puripuly_heart.core.overlay.state import (
    ActiveSelfOverlayMetadata,
    OverlayPresentationState,
)
from puripuly_heart.core.runtime_logging import SessionLoggingMode
from puripuly_heart.domain.models import Transcript
from puripuly_heart.ui.overlay_calibration import OverlayCalibration
from tests.core.test_translation_owner_branch_coverage import (
    _make_runtime_logging_capture,
    _runtime_log_messages,
)


@dataclass(slots=True)
class RecordingPresentationBridge:
    snapshots: list[object] = field(default_factory=list)
    shutdown_calls: int = 0

    async def replace_snapshot(self, snapshot: object) -> None:
        self.snapshots.append(snapshot)

    async def broadcast_shutdown(self) -> None:
        self.shutdown_calls += 1


def _overlay_presenter_decisions(stream: object) -> list[str]:
    decisions: list[str] = []
    for message in _runtime_log_messages(stream):
        if "[OverlayPresenter][Decision]" not in message or "decision=" not in message:
            continue
        decisions.append(message.split("decision=", 1)[1].split()[0])
    return decisions


def _overlay_presenter_pair_messages(stream: object) -> list[str]:
    return [
        message
        for message in _runtime_log_messages(stream)
        if "[OverlayPresenter][PairState]" in message
    ]


def _overlay_presenter_disposition_messages(stream: object) -> list[str]:
    return [
        message
        for message in _runtime_log_messages(stream)
        if "[OverlayPresenter][Decision]" in message and "disposition=" in message
    ]


@dataclass(slots=True)
class RecordingPresenterRemovalDiagnostics:
    removal_events: list[dict[str, object]] = field(default_factory=list)

    def record_presenter(self, event: str, **fields: object) -> dict[str, object]:
        _ = (event, fields)
        return {}

    def record_presenter_removal(
        self, event: str = "entry_removed", **fields: object
    ) -> dict[str, object]:
        payload = {"event": event, **fields}
        self.removal_events.append(payload)
        return payload


@dataclass(slots=True)
class RecordingPresenterDiagnostics:
    events: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def record_presenter(self, event: str, **fields: object) -> dict[str, object]:
        payload = dict(fields)
        self.events.append((event, payload))
        return payload

    def record_presenter_removal(
        self, event: str = "entry_removed", **fields: object
    ) -> dict[str, object]:
        payload = dict(fields)
        self.events.append((event, payload))
        return payload


class _ExplodingValue:
    def __str__(self) -> str:
        raise AssertionError("formatted eagerly")


def _generate_overlay_state_snapshot(
    state: OverlayPresentationState,
    *,
    revision: int,
    show_translation: bool = True,
    peer_presentation_refresh_burst: bool = False,
    self_presentation_refresh_burst: bool = True,
):
    next_appearance_seq = 0

    def next_appearance() -> int:
        nonlocal next_appearance_seq
        next_appearance_seq += 1
        return next_appearance_seq

    selection = state.visible_block_selection(
        entries=state.entries,
        live_self_entry=state.live_entry_for_channel("self"),
        live_peer_entry=state.live_entry_for_channel("peer"),
        visible_window_target_blocks=2,
        show_translation=show_translation,
        show_peer_original=True,
        peer_presentation_refresh_burst=peer_presentation_refresh_burst,
        self_presentation_refresh_burst=self_presentation_refresh_burst,
        next_appearance_seq=next_appearance,
    )
    return state.generate_snapshot(
        revision=revision,
        calibration=OverlayPresentationCalibration(),
        rendered_entries=selection.rendered_entries,
    )


def test_overlay_presentation_state_peer_refresh_methods_own_target_and_nonce() -> None:
    state = OverlayPresentationState()
    key = ("peer", uuid4())
    other_key = ("peer", uuid4())

    assert callable(getattr(state, "begin_peer_presentation_refresh", None))
    assert callable(getattr(state, "tick_peer_presentation_refresh", None))
    assert callable(getattr(state, "end_peer_presentation_refresh", None))

    assert state.begin_peer_presentation_refresh(key) is False
    assert state.peer_presentation_refresh_target_key == key
    assert state.peer_presentation_refresh_nonce == 0

    assert state.tick_peer_presentation_refresh(other_key) is False
    assert state.peer_presentation_refresh_nonce == 0

    assert state.tick_peer_presentation_refresh(key) is True
    assert state.peer_presentation_refresh_nonce == 1

    assert state.begin_peer_presentation_refresh(other_key) is False
    assert state.peer_presentation_refresh_target_key == other_key
    assert state.peer_presentation_refresh_nonce == 0

    assert state.tick_peer_presentation_refresh(key) is False
    assert state.peer_presentation_refresh_nonce == 0

    assert state.tick_peer_presentation_refresh(other_key) is True
    assert state.peer_presentation_refresh_nonce == 1

    assert state.end_peer_presentation_refresh(key) is False
    assert state.peer_presentation_refresh_target_key == other_key
    assert state.peer_presentation_refresh_nonce == 1

    assert state.end_peer_presentation_refresh(other_key) is False
    assert state.peer_presentation_refresh_target_key is None
    assert state.peer_presentation_refresh_nonce == 0
    assert state.end_peer_presentation_refresh(other_key) is False


def test_overlay_presentation_state_self_refresh_methods_own_target_and_nonce_without_peer_state() -> (
    None
):
    state = OverlayPresentationState()
    self_key = ("self", uuid4())
    other_self_key = ("self", uuid4())
    peer_key = ("peer", uuid4())

    state.begin_peer_presentation_refresh(peer_key)
    assert state.tick_peer_presentation_refresh(peer_key) is True

    assert callable(getattr(state, "begin_self_presentation_refresh", None))
    assert callable(getattr(state, "tick_self_presentation_refresh", None))
    assert callable(getattr(state, "end_self_presentation_refresh", None))

    assert state.begin_self_presentation_refresh(self_key) is False
    assert state.self_presentation_refresh_target_key == self_key
    assert state.self_presentation_refresh_nonce == 0
    assert state.peer_presentation_refresh_target_key == peer_key
    assert state.peer_presentation_refresh_nonce == 1

    assert state.tick_self_presentation_refresh(other_self_key) is False
    assert state.self_presentation_refresh_nonce == 0

    assert state.tick_self_presentation_refresh(self_key) is True
    assert state.self_presentation_refresh_nonce == 1
    assert state.peer_presentation_refresh_target_key == peer_key
    assert state.peer_presentation_refresh_nonce == 1

    assert state.begin_self_presentation_refresh(other_self_key) is False
    assert state.self_presentation_refresh_target_key == other_self_key
    assert state.self_presentation_refresh_nonce == 0
    assert state.peer_presentation_refresh_target_key == peer_key
    assert state.peer_presentation_refresh_nonce == 1

    assert state.end_self_presentation_refresh(self_key) is False
    assert state.self_presentation_refresh_target_key == other_self_key
    assert state.self_presentation_refresh_nonce == 0

    assert state.tick_self_presentation_refresh(other_self_key) is True
    assert state.self_presentation_refresh_nonce == 1

    assert state.end_self_presentation_refresh(other_self_key) is False
    assert state.self_presentation_refresh_target_key is None
    assert state.self_presentation_refresh_nonce == 0
    assert state.peer_presentation_refresh_target_key == peer_key
    assert state.peer_presentation_refresh_nonce == 1


def test_overlay_presentation_state_self_refresh_marker_revises_source_only_finalized_self() -> (
    None
):
    state = OverlayPresentationState()
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()
    key = ("self", turn_id)

    result = state.apply_self_finalized_update(
        adapter.transcript_final(
            Transcript(
                utterance_id=turn_id,
                channel="self",
                text="hello source only",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        ),
        now=10.0,
        show_translation=True,
        next_appearance_seq=lambda: 1,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )
    assert result.changed is True

    initial_snapshot = _generate_overlay_state_snapshot(state, revision=1)
    initial_block = initial_snapshot.blocks[0]
    initial_signature = state.rendered_block_signature(initial_block)
    assert initial_block.channel == "self"
    assert initial_block.block_variant == "finalized"
    assert initial_block.primary_text == "hello source only"
    assert initial_block.secondary_text == ""
    assert initial_block.session_scope is None

    assert state.begin_self_presentation_refresh(key) is False
    assert state.tick_self_presentation_refresh(key) is True
    first_refresh = _generate_overlay_state_snapshot(state, revision=2)
    first_refresh_block = first_refresh.blocks[0]

    assert first_refresh_block.session_scope == "self_presentation_refresh=1"
    assert state._snapshot_has_self_presentation_refresh_marker() is True
    first_refresh_signature = state.rendered_block_signature(first_refresh_block)
    assert first_refresh_signature != initial_signature

    assert state.tick_self_presentation_refresh(key) is True
    second_refresh = _generate_overlay_state_snapshot(state, revision=3)
    second_refresh_block = second_refresh.blocks[0]

    assert second_refresh_block.session_scope == "self_presentation_refresh=2"
    assert state.rendered_block_signature(second_refresh_block) != first_refresh_signature

    assert state.end_self_presentation_refresh(key) is True
    clean_snapshot = _generate_overlay_state_snapshot(state, revision=4)

    assert clean_snapshot.blocks[0].session_scope is None
    assert state._snapshot_has_self_presentation_refresh_marker() is False
    assert state.rendered_block_signature(clean_snapshot.blocks[0]) == initial_signature


def test_overlay_presentation_state_peer_refresh_marker_revises_translated_peer() -> None:
    state = OverlayPresentationState()
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()
    key = ("peer", turn_id)

    state.apply_peer_finalized_update(
        adapter.transcript_final(
            Transcript(
                utterance_id=turn_id,
                channel="peer",
                text="peer source",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        ),
        now=10.0,
        show_peer_original=True,
        next_appearance_seq=lambda: 1,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )
    state.apply_peer_translation_update(
        adapter.translation_final(
            utterance_id=turn_id,
            channel="peer",
            text="피어 번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.1,
            update_id="upd-peer-final",
            origin_wall_clock_ms=1712345678901,
            session_scope="session:peer",
            source_text_hash="peerfinalhash123",
            source_text_len=len("peer source"),
            logical_turn_key=f"peer:{turn_id}",
        ),
        now=10.1,
        show_peer_original=True,
        next_appearance_seq=lambda: 1,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )

    initial_snapshot = _generate_overlay_state_snapshot(
        state,
        revision=1,
        peer_presentation_refresh_burst=True,
    )
    initial_block = initial_snapshot.blocks[0]
    initial_signature = state.rendered_block_signature(initial_block)
    assert initial_block.channel == "peer"
    assert initial_block.block_variant == "finalized"
    assert initial_block.primary_text == "피어 번역"
    assert initial_block.secondary_text == "peer source"
    assert initial_block.session_scope == "session:peer"

    assert state.begin_peer_presentation_refresh(key) is False
    assert state.tick_peer_presentation_refresh(key) is True
    first_refresh = _generate_overlay_state_snapshot(
        state,
        revision=2,
        peer_presentation_refresh_burst=True,
    )
    first_refresh_block = first_refresh.blocks[0]

    assert first_refresh_block.session_scope == "session:peer|peer_presentation_refresh=1"
    assert state._snapshot_has_peer_presentation_refresh_marker() is True
    first_refresh_signature = state.rendered_block_signature(first_refresh_block)
    assert first_refresh_signature != initial_signature

    assert state.tick_peer_presentation_refresh(key) is True
    second_refresh = _generate_overlay_state_snapshot(
        state,
        revision=3,
        peer_presentation_refresh_burst=True,
    )
    second_refresh_block = second_refresh.blocks[0]

    assert second_refresh_block.session_scope == "session:peer|peer_presentation_refresh=2"
    assert state.rendered_block_signature(second_refresh_block) != first_refresh_signature

    assert state.end_peer_presentation_refresh(key) is True
    clean_snapshot = _generate_overlay_state_snapshot(
        state,
        revision=4,
        peer_presentation_refresh_burst=True,
    )

    assert clean_snapshot.blocks[0].session_scope == "session:peer"
    assert state._snapshot_has_peer_presentation_refresh_marker() is False
    assert state.rendered_block_signature(clean_snapshot.blocks[0]) == initial_signature


def test_overlay_presentation_state_self_refresh_marker_appends_to_existing_self_session_scope() -> (
    None
):
    state = OverlayPresentationState()
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()
    key = ("self", turn_id)

    state.apply_self_finalized_update(
        adapter.transcript_final(
            Transcript(
                utterance_id=turn_id,
                channel="self",
                text="hello source",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        ),
        now=10.0,
        show_translation=True,
        next_appearance_seq=lambda: 1,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )
    state.apply_self_translation_update(
        adapter.translation_final(
            utterance_id=turn_id,
            channel="self",
            text="translated self",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=10.1,
            update_id="upd-self-final",
            origin_wall_clock_ms=1712345678901,
            session_scope="session:self",
            source_text_hash="selffinalhash123",
            source_text_len=len("hello source"),
            logical_turn_key=f"self:{turn_id}",
        ),
        now=10.1,
        show_translation=True,
        next_appearance_seq=lambda: 1,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )

    assert state.begin_self_presentation_refresh(key) is False
    assert state.tick_self_presentation_refresh(key) is True
    snapshot = _generate_overlay_state_snapshot(state, revision=1)
    block = snapshot.blocks[0]

    assert block.session_scope == "session:self|self_presentation_refresh=1"
    assert block.update_id == "upd-self-final"
    assert block.origin_wall_clock_ms == 1712345678901
    assert block.source_text_hash == "selffinalhash123"
    assert block.source_text_len == len("hello source")
    assert block.logical_turn_key == f"self:{turn_id}"


def test_overlay_presentation_state_self_refresh_marker_respects_disabled_flag() -> None:
    state = OverlayPresentationState()
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()
    key = ("self", turn_id)

    state.apply_self_finalized_update(
        adapter.transcript_final(
            Transcript(
                utterance_id=turn_id,
                channel="self",
                text="hello disabled refresh",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        ),
        now=10.0,
        show_translation=True,
        next_appearance_seq=lambda: 1,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )

    assert state.begin_self_presentation_refresh(key) is False
    assert state.tick_self_presentation_refresh(key) is True
    snapshot = _generate_overlay_state_snapshot(
        state,
        revision=1,
        self_presentation_refresh_burst=False,
    )

    assert snapshot.blocks[0].session_scope is None
    assert state._snapshot_has_self_presentation_refresh_marker() is False


@pytest.mark.asyncio
async def test_presenter_self_transcript_final_refresh_request_requires_changed_visible_source_row() -> (
    None
):
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()

    initial_event = adapter.transcript_final(
        Transcript(
            utterance_id=turn_id,
            channel="self",
            text="self source final one",
            is_final=True,
            created_at=10.0,
        ),
        source_language="ko",
        target_language="en",
    )
    try:
        previous_snapshot = presenter.snapshot()

        await presenter.emit(initial_event)

        assert presenter._self_presentation_refresh_request_key_for_event(
            initial_event,
            previous_snapshot=previous_snapshot,
        ) == ("self", turn_id)

        previous_snapshot = presenter.snapshot()
        await presenter.emit(initial_event)

        assert (
            presenter._self_presentation_refresh_request_key_for_event(
                initial_event,
                previous_snapshot=previous_snapshot,
            )
            is None
        )

        changed_event = SelfTranscriptFinal(
            event_id="self-source-final-changed",
            seq=initial_event.seq + 1,
            utterance_id=turn_id,
            channel="self",
            created_at=10.2,
            text="self source final two",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
        previous_snapshot = presenter.snapshot()

        await presenter.emit(changed_event)

        assert presenter._self_presentation_refresh_request_key_for_event(
            changed_event,
            previous_snapshot=previous_snapshot,
        ) == ("self", turn_id)
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_self_refresh_request_requires_feature_enabled() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()
    event = adapter.transcript_final(
        Transcript(
            utterance_id=turn_id,
            channel="self",
            text="self disabled source final",
            is_final=True,
            created_at=10.0,
        ),
        source_language="ko",
        target_language="en",
    )
    previous_snapshot = presenter.snapshot()

    await presenter.emit(event)

    assert (
        presenter._self_presentation_refresh_request_key_for_event(
            event,
            previous_snapshot=previous_snapshot,
        )
        is None
    )


@pytest.mark.asyncio
async def test_presenter_translation_final_refresh_request_requires_visible_content_change() -> (
    None
):
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()

    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=turn_id,
                    channel="self",
                    text="self source before translation",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        translation_event = adapter.translation_final(
            utterance_id=turn_id,
            channel="self",
            text="visible self translation",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=10.1,
        )
        previous_snapshot = presenter.snapshot()

        await presenter.emit(translation_event)

        assert presenter._self_presentation_refresh_request_key_for_event(
            translation_event,
            previous_snapshot=previous_snapshot,
        ) == ("self", turn_id)

        duplicate_event = TranslationFinal(
            event_id="duplicate-self-translation-final",
            seq=translation_event.seq,
            utterance_id=turn_id,
            channel="self",
            created_at=10.2,
            text="visible self translation",
            source_language="ko",
            target_language="en",
            is_final=True,
            applied_context_mode=None,
        )
        previous_snapshot = presenter.snapshot()

        await presenter.emit(duplicate_event)

        assert (
            presenter._self_presentation_refresh_request_key_for_event(
                duplicate_event,
                previous_snapshot=previous_snapshot,
            )
            is None
        )
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_hidden_translation_final_without_visible_change_is_not_self_refresh_request() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
        show_translation=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()
    source_event = adapter.transcript_final(
        Transcript(
            utterance_id=turn_id,
            channel="self",
            text="visible source only while translation hidden",
            is_final=True,
            created_at=10.0,
        ),
        source_language="ko",
        target_language="en",
    )
    try:
        previous_snapshot = presenter.snapshot()

        await presenter.emit(source_event)

        assert presenter._self_presentation_refresh_request_key_for_event(
            source_event,
            previous_snapshot=previous_snapshot,
        ) == ("self", turn_id)

        translation_event = adapter.translation_final(
            utterance_id=turn_id,
            channel="self",
            text="hidden translation final",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=10.1,
        )
        previous_snapshot = presenter.snapshot()
        snapshot_count_before_translation = len(bridge.snapshots)

        await presenter.emit(translation_event)

        assert presenter.snapshot() == previous_snapshot
        assert len(bridge.snapshots) == snapshot_count_before_translation
        assert presenter.snapshot().native_fresh_render_generations is None
        assert presenter.snapshot().native_fresh_render_targets is None
        assert (
            presenter._self_presentation_refresh_request_key_for_event(
                translation_event,
                previous_snapshot=previous_snapshot,
            )
            is None
        )
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_self_refresh_request_rejects_named_non_trigger_events() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    live_turn_id = uuid4()

    try:
        source_only_live_event = adapter.self_active_update(
            text="source-only live self",
            utterance_id=live_turn_id,
            occupant_key=f"self:{live_turn_id}",
            created_at=10.0,
        )
        previous_snapshot = presenter.snapshot()

        await presenter.emit(source_only_live_event)

        assert (
            presenter._self_presentation_refresh_request_key_for_event(
                source_only_live_event,
                previous_snapshot=previous_snapshot,
            )
            is None
        )

        preview_turn_id = uuid4()
        active_preview_event = adapter.self_active_update(
            text="active self source",
            secondary_text="active preview translation",
            utterance_id=preview_turn_id,
            occupant_key=f"self:{preview_turn_id}",
            created_at=10.1,
        )
        previous_snapshot = presenter.snapshot()

        await presenter.emit(active_preview_event)

        assert (
            presenter._self_presentation_refresh_request_key_for_event(
                active_preview_event,
                previous_snapshot=previous_snapshot,
            )
            is None
        )

        finalized_turn_id = uuid4()
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=finalized_turn_id,
                    channel="self",
                    text="self source before stream",
                    is_final=True,
                    created_at=10.2,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        stream_event = TranslationStreamUpdate(
            event_id="self-translation-stream",
            seq=50,
            utterance_id=finalized_turn_id,
            channel="self",
            created_at=10.3,
            text="streaming translation preview",
            source_language="ko",
            target_language="en",
            is_final=False,
            applied_context_mode=None,
        )
        previous_snapshot = presenter.snapshot()

        await presenter.emit(stream_event)

        assert (
            presenter._self_presentation_refresh_request_key_for_event(
                stream_event,
                previous_snapshot=previous_snapshot,
            )
            is None
        )
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_self_refresh_request_requires_utterance_id_and_current_matching_block() -> (
    None
):
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    visible_turn_id = uuid4()

    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=visible_turn_id,
                    channel="self",
                    text="visible self source",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="ko",
                target_language="en",
            )
        )

        previous_snapshot = presenter.snapshot()
        missing_utterance_event = SelfTranscriptFinal(
            event_id="missing-utterance-self-final",
            seq=100,
            utterance_id=None,
            channel="self",
            created_at=10.1,
            text="missing utterance source",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
        other_turn_event = TranslationFinal(
            event_id="mismatched-self-translation-final",
            seq=101,
            utterance_id=uuid4(),
            channel="self",
            created_at=10.2,
            text="other translation",
            source_language="ko",
            target_language="en",
            is_final=True,
            applied_context_mode=None,
        )

        assert (
            presenter._self_presentation_refresh_request_key_for_event(
                missing_utterance_event,
                previous_snapshot=previous_snapshot,
            )
            is None
        )
        assert (
            presenter._self_presentation_refresh_request_key_for_event(
                other_turn_event,
                previous_snapshot=previous_snapshot,
            )
            is None
        )
    finally:
        await presenter.clear_for_runtime_detach()


def test_overlay_presentation_state_exposes_active_self_metadata() -> None:
    state = OverlayPresentationState()
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    utterance_id = uuid4()
    logical_turn_key = f"self:{utterance_id}"

    result = state.apply_self_active_update(
        adapter.self_active_update(
            text="hello live",
            utterance_id=utterance_id,
            secondary_text="translated live",
            occupant_key=f"self:{utterance_id}",
            update_id="self-active-update-1",
            origin_wall_clock_ms=123456789,
            session_scope="self-active-session",
            source_text_hash="0123456789abcdef",
            source_text_len=len("hello live"),
            logical_turn_key=logical_turn_key,
        ),
        now=10.0,
        show_translation=True,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )

    assert result.changed is True
    assert state.active_self_overlay_metadata() == ActiveSelfOverlayMetadata(
        text="hello live",
        secondary_text="translated live",
        utterance_id=utterance_id,
        occupant_key=f"self:{utterance_id}",
        update_id="self-active-update-1",
        origin_wall_clock_ms=123456789,
        session_scope="self-active-session",
        source_text_hash="0123456789abcdef",
        source_text_len=len("hello live"),
        logical_turn_key=logical_turn_key,
    )

    state.apply_self_active_clear(
        adapter.self_active_clear(created_at=10.1),
        now=10.1,
        show_translation=True,
    )
    assert state.active_self_overlay_metadata() is None


@pytest.mark.asyncio
async def test_presenter_shows_first_self_transcript_without_waiting_for_next_utterance() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    transcript = Transcript(
        utterance_id=uuid4(),
        channel="self",
        text="hello now",
        is_final=True,
        created_at=11.0,
    )

    await presenter.emit(
        adapter.transcript_final(
            transcript,
            source_language="ko",
            target_language="en",
        )
    )

    assert bridge.snapshots[-1].blocks[-1].channel == "self"
    assert bridge.snapshots[-1].blocks[-1].block_variant == "finalized"
    assert bridge.snapshots[-1].blocks[-1].primary_text == "hello now"
    assert bridge.snapshots[-1].blocks[-1].secondary_text == ""
    assert bridge.snapshots[-1].blocks[-1].secondary_enabled is True


@pytest.mark.asyncio
async def test_presenter_self_rows_use_source_and_target_content_languages() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    self_turn_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=self_turn_id,
                channel="self",
                text="안녕하세요",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )

    block = presenter.snapshot().blocks[0]
    assert block.primary_language == "ko"
    assert block.secondary_language is None

    await presenter.emit(
        adapter.translation_final(
            utterance_id=self_turn_id,
            channel="self",
            text="hello",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=10.1,
        )
    )

    block = presenter.snapshot().blocks[0]
    assert block.primary_language == "ko"
    assert block.secondary_language == "en"


@pytest.mark.asyncio
async def test_presenter_self_active_row_uses_source_and_target_content_languages() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    self_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="안녕 live",
            secondary_text="hello live",
            utterance_id=self_turn_id,
            occupant_key=f"self:{self_turn_id}",
            source_language="ko",
            target_language="en",
            created_at=10.0,
        )
    )

    block = presenter.snapshot().blocks[0]
    assert block.primary_language == "ko"
    assert block.secondary_language == "en"


@pytest.mark.asyncio
async def test_presenter_language_only_self_active_update_publishes_without_resetting_visible_identity() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="こんにちは",
            utterance_id=self_turn_id,
            occupant_key=f"self:{self_turn_id}",
            source_language="ko",
            target_language="en",
            created_at=10.0,
        )
    )

    initial_snapshot = presenter.snapshot()
    initial_block = initial_snapshot.blocks[0]
    entry = presenter._entries[("self", self_turn_id)]
    initial_visible_since = entry.visible_since
    initial_last_meaningful_visible_at = entry.last_meaningful_visible_at
    initial_occupant_key = entry.occupant_key
    initial_entry_appearance_seq = entry.appearance_seq

    clock.advance(2.0)
    await presenter.emit(
        adapter.self_active_update(
            text="こんにちは",
            utterance_id=self_turn_id,
            occupant_key=f"self:{self_turn_id}",
            source_language="ja",
            target_language="en",
            created_at=12.0,
        )
    )

    updated_snapshot = presenter.snapshot()
    updated_block = updated_snapshot.blocks[0]

    assert updated_snapshot.revision == initial_snapshot.revision + 1
    assert len(bridge.snapshots) == 2
    assert updated_block.primary_language == "ja"
    assert updated_block.primary_text == initial_block.primary_text
    assert updated_block.occupant_key == initial_block.occupant_key == initial_occupant_key
    assert updated_block.appearance_seq == initial_block.appearance_seq
    assert entry.appearance_seq == initial_entry_appearance_seq
    assert entry.visible_since == initial_visible_since == 10.0
    assert entry.last_meaningful_visible_at == initial_last_meaningful_visible_at == 10.0


@pytest.mark.asyncio
async def test_presenter_does_not_reorder_existing_turn_when_translation_updates() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    first_peer = Transcript(
        utterance_id=uuid4(), channel="peer", text="peer one", is_final=True, created_at=11.0
    )
    second_self = Transcript(
        utterance_id=uuid4(), channel="self", text="self two", is_final=True, created_at=12.0
    )

    await presenter.emit(
        adapter.transcript_final(first_peer, source_language="en", target_language="ko")
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=first_peer.utterance_id,
            channel="peer",
            text="피어 하나",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=12.5,
        )
    )
    await presenter.emit(
        adapter.transcript_final(second_self, source_language="ko", target_language="en")
    )
    first_order = presenter.snapshot().blocks[0].appearance_seq

    await presenter.emit(
        adapter.translation_final(
            utterance_id=first_peer.utterance_id,
            channel="peer",
            text="피어 하나 수정",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=13.0,
        )
    )

    assert presenter.snapshot().blocks[0].appearance_seq == first_order
    assert [block.occupant_key for block in presenter.snapshot().blocks] == [
        f"peer:{first_peer.utterance_id}",
        f"self:{second_self.utterance_id}",
    ]


@pytest.mark.asyncio
async def test_presenter_reserved_peer_active_update_can_emit_compatibility_row() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="peer original",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=11.0,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"peer:{peer_turn_id}"]
    assert presenter.snapshot().blocks[0].block_variant == "active_peer"
    assert presenter.snapshot().blocks[0].primary_text == ""
    assert presenter.snapshot().blocks[0].secondary_text == "peer original"
    assert presenter.snapshot().blocks[0].secondary_enabled is True

    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="상대 번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=12.0,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"peer:{peer_turn_id}"]
    assert presenter.snapshot().blocks[0].primary_text == "상대 번역"
    assert presenter.snapshot().blocks[0].secondary_text == "peer original"
    assert presenter.snapshot().blocks[0].secondary_enabled is True


@pytest.mark.asyncio
async def test_presenter_peer_rows_use_translation_primary_and_source_secondary_languages() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer source",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )

    source_only_block = presenter.snapshot().blocks[0]
    assert source_only_block.primary_text == ""
    assert source_only_block.primary_language is None
    assert source_only_block.secondary_text == "peer source"
    assert source_only_block.secondary_language == "en"

    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="상대 번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.1,
        )
    )

    translated_block = presenter.snapshot().blocks[0]
    assert translated_block.primary_text == "상대 번역"
    assert translated_block.primary_language == "ko"
    assert translated_block.secondary_text == "peer source"
    assert translated_block.secondary_language == "en"


@pytest.mark.asyncio
async def test_presenter_reserved_peer_active_source_renders_secondary_only_before_translation() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="What about now?",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=10.0,
        )
    )

    block = presenter.snapshot().blocks[0]
    assert block.id == f"peer:{peer_turn_id}"
    assert block.occupant_key == f"peer:{peer_turn_id}"
    assert block.channel == "peer"
    assert block.block_variant == "active_peer"
    assert block.primary_text == ""
    assert block.secondary_text == "What about now?"
    assert block.secondary_enabled is True


@pytest.mark.asyncio
async def test_presenter_reserved_peer_active_source_only_block_uses_secondary_language() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="何ですか",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            source_language="ja",
            target_language="ko",
            created_at=10.0,
        )
    )

    block = presenter.snapshot().blocks[0]
    assert block.primary_text == ""
    assert block.primary_language is None
    assert block.secondary_text == "何ですか"
    assert block.secondary_language == "ja"


@pytest.mark.asyncio
async def test_presenter_presentation_state_shell_tracks_self_and_peer_snapshots() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="self live shell parity",
            utterance_id=self_turn_id,
            occupant_key=f"self:{self_turn_id}",
            created_at=10.0,
        )
    )
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer source shell parity",
                is_final=True,
                created_at=10.1,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="피어 셸 동기화",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.2,
        )
    )

    presentation_state = getattr(presenter, "_presentation_state", None)
    assert presentation_state is not None
    assert presenter.snapshot() == presenter._presentation_state.snapshot()


class _RecordingSnapshotGenerationState(OverlayPresentationState):
    def __init__(self) -> None:
        super().__init__()
        self.generated_snapshot_count = 0

    def generate_snapshot(self, **kwargs: object) -> object:
        self.generated_snapshot_count += 1
        return super().generate_snapshot(**kwargs)


class _RecordingSelfReductionState(OverlayPresentationState):
    def __init__(self) -> None:
        super().__init__()
        self.self_active_updates = 0
        self.self_finalized_updates = 0
        self.self_translation_updates = 0

    def apply_self_active_update(self, *args: object, **kwargs: object) -> bool:
        self.self_active_updates += 1
        return super().apply_self_active_update(*args, **kwargs)

    def apply_self_finalized_update(self, *args: object, **kwargs: object) -> bool:
        self.self_finalized_updates += 1
        return super().apply_self_finalized_update(*args, **kwargs)

    def apply_self_translation_update(self, *args: object, **kwargs: object) -> bool:
        self.self_translation_updates += 1
        return super().apply_self_translation_update(*args, **kwargs)


class _RecordingPeerReductionState(OverlayPresentationState):
    def __init__(self) -> None:
        super().__init__()
        self.peer_active_updates = 0
        self.peer_finalized_updates = 0
        self.peer_translation_updates = 0
        self.peer_utterance_closed_updates = 0

    def apply_peer_active_update(self, *args: object, **kwargs: object) -> bool:
        self.peer_active_updates += 1
        return super().apply_peer_active_update(*args, **kwargs)

    def apply_peer_finalized_update(self, *args: object, **kwargs: object) -> bool:
        self.peer_finalized_updates += 1
        return super().apply_peer_finalized_update(*args, **kwargs)

    def apply_peer_translation_update(self, *args: object, **kwargs: object) -> bool:
        self.peer_translation_updates += 1
        return super().apply_peer_translation_update(*args, **kwargs)

    def apply_peer_utterance_closed(self, *args: object, **kwargs: object) -> bool:
        self.peer_utterance_closed_updates += 1
        return super().apply_peer_utterance_closed(*args, **kwargs)


class _NonSelectableState(OverlayPresentationState):
    def __init__(self) -> None:
        super().__init__()
        self.selectable_calls = 0

    def entry_is_selectable(self, *args: object, **kwargs: object) -> bool:
        self.selectable_calls += 1
        return False


def test_presentation_state_self_reducers_return_diagnostics_without_emit_callbacks() -> None:
    reducer_methods = [
        OverlayPresentationState.apply_self_active_update,
        OverlayPresentationState.apply_self_active_clear,
        OverlayPresentationState.apply_self_finalized_update,
        OverlayPresentationState.apply_self_translation_update,
        OverlayPresentationState.apply_self_utterance_closed,
    ]
    for reducer_method in reducer_methods:
        assert not {
            "emit_skip_disposition",
            "emit_turn_decision",
        }.intersection(inspect.signature(reducer_method).parameters)

    state = OverlayPresentationState()
    utterance_id = uuid4()
    initial_result = state.apply_self_active_update(
        SelfActiveUpdate(
            event_id="self-active-newer",
            seq=2,
            utterance_id=utterance_id,
            channel="self",
            created_at=10.0,
            text="current preview",
            occupant_key=f"self:{utterance_id}",
        ),
        now=10.0,
        show_translation=True,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )

    stale_result = state.apply_self_active_update(
        SelfActiveUpdate(
            event_id="self-active-stale",
            seq=1,
            utterance_id=utterance_id,
            channel="self",
            created_at=9.9,
            text="stale preview",
            occupant_key=f"self:{utterance_id}",
        ),
        now=10.1,
        show_translation=True,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )

    assert initial_result.changed is True
    assert stale_result.changed is False
    assert [decision.decision for decision in stale_result.decisions] == ["overlay_turn_superseded"]


def test_presentation_state_coalesced_self_active_update_refreshes_language_metadata() -> None:
    state = OverlayPresentationState()
    utterance_id = uuid4()

    initial_result = state.apply_self_active_update(
        SelfActiveUpdate(
            event_id="self-active-initial-language",
            seq=1,
            utterance_id=utterance_id,
            channel="self",
            created_at=10.0,
            text="こんにちは",
            secondary_text="",
            occupant_key=f"self:{utterance_id}",
            source_language="ko",
            target_language="en",
        ),
        now=10.0,
        show_translation=True,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )

    metadata = state.active_self_overlay_metadata()
    assert initial_result.changed is True
    assert metadata is not None
    assert metadata.primary_language == "ko"

    language_only_result = state.apply_self_active_update(
        SelfActiveUpdate(
            event_id="self-active-provider-language",
            seq=2,
            utterance_id=utterance_id,
            channel="self",
            created_at=10.1,
            text="こんにちは",
            secondary_text="",
            occupant_key=f"self:{utterance_id}",
            source_language="ja",
            target_language="zh-TW",
        ),
        now=10.1,
        show_translation=True,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )

    metadata = state.active_self_overlay_metadata()
    assert language_only_result.changed is True
    assert [decision.disposition for decision in language_only_result.decisions] == ["coalesced"]
    assert metadata is not None
    assert metadata.primary_language == "ja"
    assert metadata.secondary_language is None


def test_presentation_state_coalesced_peer_active_update_refreshes_language_metadata() -> None:
    state = OverlayPresentationState()
    utterance_id = uuid4()
    next_appearance_seq = 0

    def next_seq() -> int:
        nonlocal next_appearance_seq
        next_appearance_seq += 1
        return next_appearance_seq

    initial_result = state.apply_peer_active_update(
        PeerActiveUpdate(
            event_id="peer-active-initial-language",
            seq=1,
            utterance_id=utterance_id,
            channel="peer",
            created_at=10.0,
            text="こんにちは",
            occupant_key=f"peer:{utterance_id}",
            source_language="ko",
            target_language="en",
        ),
        now=10.0,
        show_peer_original=True,
        next_appearance_seq=next_seq,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )
    entry = state.entries[("peer", utterance_id)]
    assert initial_result.changed is True
    assert entry.original_language == "ko"

    language_only_result = state.apply_peer_active_update(
        PeerActiveUpdate(
            event_id="peer-active-provider-language",
            seq=2,
            utterance_id=utterance_id,
            channel="peer",
            created_at=10.1,
            text="こんにちは",
            occupant_key=f"peer:{utterance_id}",
            source_language="ja",
            target_language="zh-TW",
        ),
        now=10.1,
        show_peer_original=True,
        next_appearance_seq=next_seq,
        terminal_update_reason=lambda _channel, _utterance_id: None,
    )

    assert language_only_result.changed is True
    assert [decision.disposition for decision in language_only_result.decisions] == ["coalesced"]
    assert entry.original_language == "ja"


def test_presentation_state_peer_reducers_return_diagnostics_without_emit_callbacks() -> None:
    reducer_methods = [
        "apply_peer_active_update",
        "apply_peer_finalized_update",
        "apply_peer_translation_update",
        "apply_peer_utterance_closed",
    ]
    for reducer_method_name in reducer_methods:
        reducer_method = getattr(OverlayPresentationState, reducer_method_name, None)
        assert reducer_method is not None
        assert not {
            "emit_skip_disposition",
            "emit_turn_decision",
        }.intersection(inspect.signature(reducer_method).parameters)


@pytest.mark.asyncio
async def test_presenter_delegates_snapshot_generation_to_presentation_state() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        peer_presentation_refresh_burst=False,
    )
    state = _RecordingSnapshotGenerationState()
    presenter._presentation_state = state
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="snapshot generation is reducer-owned",
            utterance_id=self_turn_id,
            occupant_key=f"self:{self_turn_id}",
            created_at=10.0,
        )
    )

    assert state.generated_snapshot_count == 1
    assert presenter.snapshot() == state.snapshot()


@pytest.mark.asyncio
async def test_presenter_delegates_self_event_reduction_to_presentation_state() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    state = _RecordingSelfReductionState()
    presenter._presentation_state = state
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="self reducer-owned preview",
            secondary_text="preview translation",
            utterance_id=self_turn_id,
            occupant_key=f"self:{self_turn_id}",
            created_at=10.0,
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=self_turn_id,
            channel="self",
            text="final reducer-owned translation",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=10.1,
        )
    )
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=self_turn_id,
                channel="self",
                text="final reducer-owned source",
                is_final=True,
                created_at=10.2,
            ),
            source_language="ko",
            target_language="en",
        )
    )

    assert state.self_active_updates == 1
    assert state.self_translation_updates == 1
    assert state.self_finalized_updates == 1
    assert presenter.snapshot() == state.snapshot()


@pytest.mark.asyncio
async def test_presenter_delegates_peer_event_reduction_to_presentation_state() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        peer_presentation_refresh_burst=False,
    )
    state = _RecordingPeerReductionState()
    presenter._presentation_state = state
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="peer reducer-owned fallback source",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=10.0,
        )
    )
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer reducer-owned source final",
                is_final=True,
                created_at=10.1,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer reducer-owned translation",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.2,
        )
    )
    await presenter.emit(
        adapter.utterance_closed(
            utterance_id=peer_turn_id,
            channel="peer",
            created_at=10.3,
        )
    )

    assert state.peer_active_updates == 1
    assert state.peer_finalized_updates == 1
    assert state.peer_translation_updates == 1
    assert state.peer_utterance_closed_updates == 1
    assert presenter.snapshot().blocks[0].primary_text == "peer reducer-owned translation"
    assert presenter.snapshot() == state.snapshot()


@pytest.mark.asyncio
async def test_presenter_visible_window_diagnostics_include_reducer_retained_hidden_entries() -> (
    None
):
    diagnostics = RecordingPresenterDiagnostics()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        diagnostics=diagnostics,
        visible_window_target_blocks=2,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    retained_turn_id = uuid4()
    visible_turn_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=retained_turn_id,
                channel="self",
                text="retained diagnostic source",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    retained_entry = presenter._entries[("self", retained_turn_id)]
    retained_entry.retained_hidden = True
    retained_entry.window_evicted_at = clock.now()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=visible_turn_id,
                channel="self",
                text="visible diagnostic source",
                is_final=True,
                created_at=10.1,
            ),
            source_language="ko",
            target_language="en",
        )
    )

    visible_window_events = [
        fields for event, fields in diagnostics.events if event == "visible_window"
    ]
    assert any(
        f"self:{retained_turn_id}" in fields.get("retained_hidden", [])
        for fields in visible_window_events
    )


@pytest.mark.asyncio
async def test_presenter_visible_selection_uses_reducer_selectability_source() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    state = _NonSelectableState()
    presenter._presentation_state = state
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="should stay hidden by reducer selectability",
            source_text="peer source",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.0,
        )
    )

    assert state.selectable_calls >= 1
    assert presenter.snapshot().blocks == []


@pytest.mark.asyncio
async def test_presenter_hides_peer_source_only_when_peer_original_disabled() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        show_peer_original=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="peer original",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=10.0,
        )
    )

    assert presenter.snapshot().blocks == []

    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="상대 번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.1,
        )
    )

    block = presenter.snapshot().blocks[0]
    assert block.primary_text == "상대 번역"
    assert block.secondary_text == "peer original"
    assert block.secondary_enabled is False


@pytest.mark.asyncio
async def test_presenter_hidden_peer_source_waits_past_ttl_for_translation() -> None:
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        show_peer_original=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="peer original",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=clock.now(),
        )
    )

    assert presenter.snapshot().blocks == []

    clock.advance(9.0)
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="상대 번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=clock.now(),
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"peer:{peer_turn_id}"]
    assert presenter.snapshot().blocks[0].primary_text == "상대 번역"
    assert presenter.snapshot().blocks[0].secondary_enabled is False


@pytest.mark.asyncio
async def test_presenter_peer_transcript_uses_translation_as_primary_text_in_normal_flow() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()
    transcript = Transcript(
        utterance_id=peer_turn_id,
        channel="peer",
        text="peer source final",
        is_final=True,
        created_at=10.0,
    )

    await presenter.emit(
        adapter.transcript_final(
            transcript,
            source_language="en",
            target_language="ko",
        )
    )

    source_only_blocks = presenter.snapshot().blocks
    assert all(block.block_variant != "active_peer" for block in source_only_blocks)
    assert all(block.primary_text != "peer source final" for block in source_only_blocks)
    assert not any(block.primary_text.strip() for block in source_only_blocks)
    assert source_only_blocks[0].primary_text == ""
    assert source_only_blocks[0].secondary_text == "peer source final"
    assert source_only_blocks[0].secondary_enabled is True

    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="피어 최종 번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=11.0,
        )
    )

    translated_blocks = presenter.snapshot().blocks
    assert all(block.block_variant != "active_peer" for block in translated_blocks)
    translated_block = translated_blocks[0]
    assert translated_block.block_variant == "finalized"
    assert translated_block.primary_text == "피어 최종 번역"
    assert translated_block.secondary_text == "peer source final"
    assert translated_block.secondary_enabled is True


@pytest.mark.asyncio
async def test_presenter_reserved_peer_active_update_can_be_finalized_by_translation_without_reordering() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="Can you hear me?",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=10.0,
        )
    )
    active_block = presenter.snapshot().blocks[0]

    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="내 말 들려?",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=11.0,
        )
    )

    finalized_block = presenter.snapshot().blocks[0]
    assert finalized_block.id == active_block.id
    assert finalized_block.occupant_key == active_block.occupant_key
    assert finalized_block.appearance_seq == active_block.appearance_seq
    assert finalized_block.block_variant == "finalized"
    assert finalized_block.primary_text == "내 말 들려?"
    assert finalized_block.secondary_text == "Can you hear me?"
    assert finalized_block.secondary_enabled is True


@pytest.mark.asyncio
async def test_presenter_self_active_source_row_remains_allowed_before_translation() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    self_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="self live source",
            utterance_id=self_turn_id,
            occupant_key=f"self:{self_turn_id}",
            created_at=10.0,
        )
    )

    block = presenter.snapshot().blocks[0]
    assert block.channel == "self"
    assert block.block_variant == "active_self"
    assert block.primary_text == "self live source"
    assert block.secondary_text == ""


@pytest.mark.asyncio
async def test_presenter_translation_min_visible_deadline_remains_self_only() -> None:
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=self_turn_id,
                channel="self",
                text="self original",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=self_turn_id,
            channel="self",
            text="self translation",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=10.1,
        )
    )
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer original",
                is_final=True,
                created_at=10.2,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer translation",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.3,
        )
    )

    self_entry = presenter._entries[("self", self_turn_id)]
    peer_entry = presenter._entries[("peer", peer_turn_id)]
    _, _, self_translation_deadline = presenter._entry_expiration_components(self_entry)
    _, _, peer_translation_deadline = presenter._entry_expiration_components(peer_entry)

    assert self_entry.translation_visible_since is not None
    assert self_translation_deadline == (
        self_entry.translation_visible_since + SELF_TRANSLATION_MIN_VISIBLE_SECONDS
    )
    assert peer_entry.translation_visible_since is not None
    assert peer_translation_deadline is None


@pytest.mark.asyncio
async def test_presenter_retained_hidden_self_entry_accepts_late_translation() -> None:
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        visible_window_target_blocks=2,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    retained_turn_id = uuid4()
    visible_turn_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=retained_turn_id,
                channel="self",
                text="retained self source",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    retained_entry = presenter._entries[("self", retained_turn_id)]
    retained_entry.retained_hidden = True
    retained_entry.window_evicted_at = clock.now()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=visible_turn_id,
                channel="self",
                text="visible self source",
                is_final=True,
                created_at=10.1,
            ),
            source_language="ko",
            target_language="en",
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{visible_turn_id}"]

    await presenter.emit(
        adapter.translation_final(
            utterance_id=retained_turn_id,
            channel="self",
            text="late retained translation",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=10.2,
        )
    )

    retained_entry = presenter._entries[("self", retained_turn_id)]
    blocks_by_id = {block.id: block for block in presenter.snapshot().blocks}
    assert retained_entry.retained_hidden is False
    assert retained_entry.window_evicted_at is None
    assert blocks_by_id[f"self:{retained_turn_id}"].secondary_text == ("late retained translation")


@pytest.mark.asyncio
async def test_presenter_retired_preview_self_seqs_remain_self_only() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    shared_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="self preview only",
            utterance_id=shared_turn_id,
            occupant_key=f"self:{shared_turn_id}",
            created_at=10.0,
        )
    )
    await presenter.emit(adapter.self_active_clear(created_at=10.1))

    assert ("self", shared_turn_id) in presenter._retired_preview_self_seqs

    await presenter.emit(
        adapter.peer_active_update(
            text="reserved peer active fallback",
            utterance_id=shared_turn_id,
            occupant_key=f"peer:{shared_turn_id}",
            created_at=10.2,
        )
    )

    peer_block = presenter.snapshot().blocks[0]
    assert ("peer", shared_turn_id) not in presenter._retired_preview_self_seqs
    assert peer_block.id == f"peer:{shared_turn_id}"
    assert peer_block.block_variant == "active_peer"
    assert peer_block.secondary_text == "reserved peer active fallback"


@pytest.mark.asyncio
async def test_presenter_late_peer_translation_does_not_reinsert_evicted_old_turn() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        visible_window_target_blocks=2,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_a = uuid4()
    turn_b = uuid4()
    turn_c = uuid4()

    for text, turn_id in [
        ("old source", turn_a),
        ("middle source", turn_b),
        ("latest source", turn_c),
    ]:
        await presenter.emit(
            adapter.peer_active_update(
                text=text,
                utterance_id=turn_id,
                occupant_key=f"peer:{turn_id}",
                created_at=10.0,
            )
        )

    assert [block.id for block in presenter.snapshot().blocks] == [
        f"peer:{turn_b}",
        f"peer:{turn_c}",
    ]

    await presenter.emit(
        adapter.translation_final(
            utterance_id=turn_a,
            channel="peer",
            text="오래된 번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=12.0,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [
        f"peer:{turn_b}",
        f"peer:{turn_c}",
    ]


@pytest.mark.asyncio
async def test_presenter_peer_source_only_terminalization_finalizes_before_close() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="translation unavailable",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=10.0,
        )
    )
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="translation unavailable",
                is_final=True,
                created_at=10.5,
            ),
            source_language="en",
            target_language="ko",
        )
    )

    block_before_close = presenter.snapshot().blocks[0]
    assert block_before_close.block_variant == "finalized"
    assert block_before_close.primary_text == ""
    assert block_before_close.secondary_text == "translation unavailable"
    assert block_before_close.secondary_enabled is True

    await presenter.emit(
        adapter.utterance_closed(
            utterance_id=peer_turn_id,
            channel="peer",
            is_final=False,
            created_at=11.0,
        )
    )

    assert presenter.snapshot().blocks[0].id == f"peer:{peer_turn_id}"


@pytest.mark.asyncio
async def test_presenter_protects_current_peer_live_row_from_generic_window() -> None:
    diagnostics = RecordingPresenterDiagnostics()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        diagnostics=diagnostics,
        visible_window_target_blocks=2,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    old_turn = uuid4()
    live_turn = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="old source",
            utterance_id=old_turn,
            occupant_key=f"peer:{old_turn}",
            created_at=10.0,
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=old_turn,
            channel="peer",
            text="이전 번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.1,
        )
    )
    await presenter.emit(
        adapter.peer_active_update(
            text="current source",
            utterance_id=live_turn,
            occupant_key=f"peer:{live_turn}",
            created_at=10.2,
        )
    )

    visible_window_events = [
        fields for event, fields in diagnostics.events if event == "visible_window"
    ]

    assert any(
        f"peer:{live_turn}" in fields.get("protected_selected", [])
        for fields in visible_window_events
    )
    assert presenter.snapshot().blocks[-1].id == f"peer:{live_turn}"
    assert presenter.snapshot().blocks[-1].block_variant == "active_peer"
    assert presenter.snapshot().blocks[-1].primary_text == ""
    assert presenter.snapshot().blocks[-1].secondary_text == "current source"


@pytest.mark.asyncio
async def test_presenter_peer_translation_final_with_source_text_publishes_paired_row() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer translation",
            source_text="peer source",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.0,
        )
    )

    assert len(presenter.snapshot().blocks) == 1
    block = presenter.snapshot().blocks[0]
    assert block.channel == "peer"
    assert block.block_variant == "finalized"
    assert block.primary_text == "peer translation"
    assert block.secondary_text == "peer source"
    assert block.secondary_enabled is True


@pytest.mark.asyncio
async def test_presenter_peer_active_duplicate_uses_shared_coalesced_disposition() -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        runtime_log_detailed=runtime_logging.emit_detailed,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    try:
        await presenter.emit(
            adapter.peer_active_update(
                text="peer source duplicate",
                utterance_id=peer_turn_id,
                occupant_key=f"peer:{peer_turn_id}",
                created_at=10.0,
            )
        )
        revision_before_duplicate = presenter.snapshot().revision
        snapshot_count_before_duplicate = len(bridge.snapshots)

        await presenter.emit(
            adapter.peer_active_update(
                text="peer source duplicate",
                utterance_id=peer_turn_id,
                occupant_key=f"peer:{peer_turn_id}",
                created_at=10.1,
            )
        )

        assert presenter.snapshot().revision == revision_before_duplicate
        assert len(bridge.snapshots) == snapshot_count_before_duplicate
        disposition_messages = _overlay_presenter_disposition_messages(log_stream)
        assert any(
            f"entry=peer:{peer_turn_id}" in message
            and "decision=overlay_turn_coalesced" in message
            and "disposition=coalesced" in message
            for message in disposition_messages
        )
        assert not any(
            "decision=overlay_turn_no_visible_change" in message
            and "disposition=rendered_signature_unchanged" in message
            for message in disposition_messages
        )
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_presenter_peer_presentation_refresh_burst_waits_for_translated_peer_primary_text() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()
    transcript = Transcript(
        utterance_id=peer_turn_id,
        channel="peer",
        text="peer source before translation",
        is_final=True,
        created_at=10.0,
    )

    try:
        await presenter.emit(
            adapter.transcript_final(
                transcript,
                source_language="en",
                target_language="ko",
            )
        )
        await asyncio.sleep(0)

        source_only_block = presenter.snapshot().blocks[0]
        assert source_only_block.primary_text == ""
        assert source_only_block.secondary_text == "peer source before translation"
        assert source_only_block.block_variant == "finalized"
        assert presenter._peer_presentation_refresh_burst_task is None
        assert [delay for delay in sleep_calls if delay == 0.1] == []

        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="번역 후 표시",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=11.0,
            )
        )
        await asyncio.sleep(0)

        translated_block = presenter.snapshot().blocks[0]
        assert translated_block.primary_text == "번역 후 표시"
        assert translated_block.secondary_text == "peer source before translation"
        assert presenter._peer_presentation_refresh_burst_task is not None
        assert [delay for delay in sleep_calls if delay == 0.1] == [0.1]
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_reserved_peer_active_update_does_not_start_refresh_burst_without_translation() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()

    try:
        await presenter.emit(
            adapter.peer_active_update(
                text="reserved peer active source",
                utterance_id=peer_turn_id,
                occupant_key=f"peer:{peer_turn_id}",
                created_at=10.0,
            )
        )
        await asyncio.sleep(0)

        block = presenter.snapshot().blocks[0]
        assert block.block_variant == "active_peer"
        assert block.primary_text == ""
        assert block.secondary_text == "reserved peer active source"
        assert presenter._peer_presentation_refresh_burst_task is None
        assert [delay for delay in sleep_calls if delay == 0.1] == []
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_peer_presentation_refresh_burst_defaults_on_and_rerenders_peer_snapshot_without_visible_text_change() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
    )
    assert presenter.peer_presentation_refresh_burst is True
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()

    def refresh_sleep_indices() -> list[int]:
        return [index for index, delay in enumerate(sleep_calls) if delay == 0.1]

    try:
        transcript = Transcript(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer source unchanged during refresh",
            is_final=True,
            created_at=10.0,
        )
        await presenter.emit(
            adapter.transcript_final(
                transcript,
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="번역 refresh 유지",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.1,
            )
        )

        initial_snapshot = presenter.snapshot()
        initial_snapshot_count = len(bridge.snapshots)
        initial_visible_text = [
            (block.primary_text, block.secondary_text, block.secondary_enabled)
            for block in initial_snapshot.blocks
        ]
        assert initial_visible_text == [
            ("번역 refresh 유지", "peer source unchanged during refresh", True)
        ]

        await asyncio.sleep(0)
        assert len(refresh_sleep_indices()) == 1

        sleep_events[refresh_sleep_indices()[-1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        first_refresh = presenter.snapshot()

        assert first_refresh.revision == initial_snapshot.revision + 1
        assert len(bridge.snapshots) == initial_snapshot_count + 1
        assert [
            (block.primary_text, block.secondary_text, block.secondary_enabled)
            for block in first_refresh.blocks
        ] == initial_visible_text
        assert first_refresh.blocks[0].session_scope == "peer_presentation_refresh=1"
        assert (
            first_refresh.native_fresh_render_generations
            == initial_snapshot.native_fresh_render_generations
        )

        assert len(refresh_sleep_indices()) == 2
        sleep_events[refresh_sleep_indices()[-1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second_refresh = presenter.snapshot()

        assert second_refresh.revision == initial_snapshot.revision + 2
        assert len(bridge.snapshots) == initial_snapshot_count + 2
        assert [
            (block.primary_text, block.secondary_text, block.secondary_enabled)
            for block in second_refresh.blocks
        ] == initial_visible_text
        assert second_refresh.blocks[0].session_scope == "peer_presentation_refresh=2"
        assert (
            second_refresh.native_fresh_render_generations
            == initial_snapshot.native_fresh_render_generations
        )

        for _ in range(25):
            if presenter._peer_presentation_refresh_burst_task is None:
                break
            assert sleep_events, "refresh burst should be waiting for its next tick"
            sleep_events[-1].set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        clean_snapshot = presenter.snapshot()
        assert presenter._peer_presentation_refresh_burst_task is None
        assert clean_snapshot.revision > second_refresh.revision
        assert clean_snapshot.blocks[0].primary_text == "번역 refresh 유지"
        assert clean_snapshot.blocks[0].secondary_text == "peer source unchanged during refresh"
        assert clean_snapshot.blocks[0].session_scope is None
        assert bridge.snapshots[-1].blocks[0].session_scope is None
        assert (
            clean_snapshot.native_fresh_render_generations
            == initial_snapshot.native_fresh_render_generations
        )
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_bridge_restart_during_peer_refresh_receives_marker_and_clean_end() -> None:
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()

    try:
        transcript = Transcript(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer source bridge restart",
            is_final=True,
            created_at=10.0,
        )
        await presenter.emit(
            adapter.transcript_final(
                transcript,
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="브리지 재시작 번역",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.1,
            )
        )
        await asyncio.sleep(0)

        assert [delay for delay in sleep_calls if delay == 0.1] == [0.1]
        sleep_events[-1].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        marker_snapshot = presenter.snapshot()
        assert marker_snapshot.blocks[0].session_scope == "peer_presentation_refresh=1"

        restarted_bridge = OverlayBridge(
            session_token="test-token",
            initial_snapshot=marker_snapshot,
        )
        presenter.attach_bridge(restarted_bridge)

        assert restarted_bridge.snapshot().revision == marker_snapshot.revision
        assert restarted_bridge.snapshot().blocks[0].primary_text == "브리지 재시작 번역"
        assert restarted_bridge.snapshot().blocks[0].session_scope == (
            "peer_presentation_refresh=1"
        )

        for _ in range(25):
            if presenter._peer_presentation_refresh_burst_task is None:
                break
            assert sleep_events, "refresh burst should be waiting for its next tick"
            sleep_events[-1].set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        clean_snapshot = presenter.snapshot()
        assert presenter._peer_presentation_refresh_burst_task is None
        assert clean_snapshot.blocks[0].primary_text == "브리지 재시작 번역"
        assert clean_snapshot.blocks[0].session_scope is None
        assert restarted_bridge.snapshot().revision == clean_snapshot.revision
        assert restarted_bridge.snapshot().blocks[0].session_scope is None
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_peer_presentation_refresh_burst_naturally_ends_with_clean_peer_snapshot() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()

    try:
        transcript = Transcript(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer source clean after natural burst",
            is_final=True,
            created_at=10.0,
        )
        await presenter.emit(
            adapter.transcript_final(
                transcript,
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="자연 종료 번역",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.1,
            )
        )
        await asyncio.sleep(0)

        for _ in range(25):
            if presenter._peer_presentation_refresh_burst_task is None:
                break
            assert sleep_events, "refresh burst should be waiting for its next tick"
            sleep_events[-1].set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        assert presenter._peer_presentation_refresh_burst_task is None
        assert len([delay for delay in sleep_calls if delay == 0.1]) >= 1

        clean_snapshot = presenter.snapshot()
        assert clean_snapshot.blocks[0].primary_text == "자연 종료 번역"
        assert clean_snapshot.blocks[0].secondary_text == "peer source clean after natural burst"
        assert clean_snapshot.blocks[0].session_scope is None
        assert bridge.snapshots[-1].blocks[0].session_scope is None
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_peer_presentation_refresh_burst_restarts_after_coalesced_peer_translation_update() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    cancelled_delays: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_delays.append(delay)
            raise
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()

    def refresh_sleep_indices() -> list[int]:
        return [index for index, delay in enumerate(sleep_calls) if delay == 0.1]

    try:
        transcript = Transcript(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer source coalesced refresh",
            is_final=True,
            created_at=10.0,
        )
        await presenter.emit(
            adapter.transcript_final(
                transcript,
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="번역 coalesced refresh",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.05,
                update_id="same-peer-translation-update",
            )
        )
        await asyncio.sleep(0)
        assert len(refresh_sleep_indices()) == 1
        first_task = presenter._peer_presentation_refresh_burst_task
        assert first_task is not None

        revision_before_duplicate = presenter.snapshot().revision
        snapshot_count_before_duplicate = len(bridge.snapshots)

        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="번역 coalesced refresh",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.1,
                update_id="same-peer-translation-update",
            )
        )
        await asyncio.sleep(0)

        restarted_task = presenter._peer_presentation_refresh_burst_task
        assert restarted_task is not None
        assert restarted_task is not first_task
        assert first_task.cancelled()
        assert cancelled_delays == [0.1]
        assert len(refresh_sleep_indices()) == 2
        sleep_events[refresh_sleep_indices()[-1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert presenter.snapshot().revision == revision_before_duplicate + 2
        assert len(bridge.snapshots) == snapshot_count_before_duplicate + 2
        assert presenter.snapshot().blocks[0].primary_text == "번역 coalesced refresh"
        assert presenter.snapshot().blocks[0].secondary_text == "peer source coalesced refresh"
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_peer_presentation_refresh_restart_after_visible_marker_resets_nonce_and_cleans_old_target() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    cancelled_delays: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_delays.append(delay)
            raise
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    first_peer_turn_id = uuid4()
    second_peer_turn_id = uuid4()

    def refresh_sleep_indices() -> list[int]:
        return [index for index, delay in enumerate(sleep_calls) if delay == 0.1]

    def blocks_by_id() -> dict[str, OverlayPresentationBlock]:
        return {block.id: block for block in presenter.snapshot().blocks}

    try:
        second_transcript = Transcript(
            utterance_id=second_peer_turn_id,
            channel="peer",
            text="second peer source",
            is_final=True,
            created_at=10.0,
        )
        await presenter.emit(
            adapter.transcript_final(
                second_transcript,
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=second_peer_turn_id,
                channel="peer",
                text="두 번째 번역",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.05,
                update_id="same-second-peer-update",
            )
        )
        await asyncio.sleep(0)
        initial_second_task = presenter._peer_presentation_refresh_burst_task
        assert initial_second_task is not None

        first_transcript = Transcript(
            utterance_id=first_peer_turn_id,
            channel="peer",
            text="first peer source",
            is_final=True,
            created_at=10.1,
        )
        await presenter.emit(
            adapter.transcript_final(
                first_transcript,
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=first_peer_turn_id,
                channel="peer",
                text="첫 번째 번역",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.15,
            )
        )
        await asyncio.sleep(0)
        first_peer_task = presenter._peer_presentation_refresh_burst_task
        assert first_peer_task is not None
        assert first_peer_task is not initial_second_task

        sleep_events[refresh_sleep_indices()[-1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        first_marker_snapshot = presenter.snapshot()
        first_marker_blocks = blocks_by_id()
        assert first_marker_blocks[f"peer:{first_peer_turn_id}"].session_scope == (
            "peer_presentation_refresh=1"
        )
        assert first_marker_blocks[f"peer:{second_peer_turn_id}"].session_scope is None

        snapshot_count_before_restart = len(bridge.snapshots)

        await presenter.emit(
            adapter.translation_final(
                utterance_id=second_peer_turn_id,
                channel="peer",
                text="두 번째 번역",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.2,
                update_id="same-second-peer-update",
            )
        )
        await asyncio.sleep(0)

        restarted_task = presenter._peer_presentation_refresh_burst_task
        assert restarted_task is not None
        assert restarted_task is not first_peer_task
        restart_clean_blocks = blocks_by_id()
        assert presenter.snapshot().revision == first_marker_snapshot.revision + 2
        assert len(bridge.snapshots) == snapshot_count_before_restart + 2
        assert restart_clean_blocks[f"peer:{first_peer_turn_id}"].session_scope is None
        assert restart_clean_blocks[f"peer:{second_peer_turn_id}"].session_scope is None

        sleep_events[refresh_sleep_indices()[-1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        restarted_marker_blocks = blocks_by_id()
        assert restarted_marker_blocks[f"peer:{first_peer_turn_id}"].session_scope is None
        assert restarted_marker_blocks[f"peer:{second_peer_turn_id}"].session_scope == (
            "peer_presentation_refresh=1"
        )

        for _ in range(25):
            if presenter._peer_presentation_refresh_burst_task is None:
                break
            assert sleep_events, "refresh burst should be waiting for its next tick"
            sleep_events[-1].set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        clean_blocks = blocks_by_id()
        assert presenter._peer_presentation_refresh_burst_task is None
        assert clean_blocks[f"peer:{first_peer_turn_id}"].session_scope is None
        assert clean_blocks[f"peer:{second_peer_turn_id}"].session_scope is None
        assert bridge.snapshots[-1].blocks == presenter.snapshot().blocks
        assert cancelled_delays
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_disabling_peer_presentation_refresh_burst_publishes_clean_peer_snapshot() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()

    def refresh_sleep_indices() -> list[int]:
        return [index for index, delay in enumerate(sleep_calls) if delay == 0.1]

    try:
        transcript = Transcript(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer source clean disable",
            is_final=True,
            created_at=10.0,
        )
        await presenter.emit(
            adapter.transcript_final(
                transcript,
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="비활성화 전 번역",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.1,
            )
        )
        await asyncio.sleep(0)
        sleep_events[refresh_sleep_indices()[-1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        refresh_snapshot = presenter.snapshot()
        revision_before_disable = refresh_snapshot.revision
        snapshot_count_before_disable = len(bridge.snapshots)

        assert refresh_snapshot.blocks[0].primary_text == "비활성화 전 번역"
        assert refresh_snapshot.blocks[0].secondary_text == "peer source clean disable"
        assert refresh_snapshot.blocks[0].session_scope == "peer_presentation_refresh=1"

        await presenter.update_peer_presentation_refresh_burst(False)

        clean_snapshot = presenter.snapshot()
        assert clean_snapshot.revision == revision_before_disable + 1
        assert len(bridge.snapshots) == snapshot_count_before_disable + 1
        assert clean_snapshot.blocks[0].primary_text == "비활성화 전 번역"
        assert clean_snapshot.blocks[0].secondary_text == "peer source clean disable"
        assert clean_snapshot.blocks[0].session_scope is None
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_peer_presentation_refresh_burst_disabled_keeps_peer_active_duplicates_coalesced() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="peer source duplicate",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=10.0,
        )
    )
    revision_before_duplicate = presenter.snapshot().revision
    snapshot_count_before_duplicate = len(bridge.snapshots)

    await presenter.emit(
        adapter.peer_active_update(
            text="peer source duplicate",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=10.1,
        )
    )
    await asyncio.sleep(0)

    assert presenter.snapshot().revision == revision_before_duplicate
    assert len(bridge.snapshots) == snapshot_count_before_duplicate
    assert presenter._peer_presentation_refresh_burst_task is None


@pytest.mark.asyncio
async def test_presenter_self_presentation_refresh_burst_defaults_on_and_rerenders_source_only_self_snapshot_without_visible_text_change() -> (
    None
):
    bridge = RecordingPresentationBridge()
    diagnostics = RecordingPresenterDiagnostics()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        diagnostics=diagnostics,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=False,
    )
    assert presenter.self_presentation_refresh_burst is True
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()

    def refresh_sleep_indices() -> list[int]:
        return [index for index, delay in enumerate(sleep_calls) if delay == 0.1]

    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=self_turn_id,
                    channel="self",
                    text="self source unchanged during refresh",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="ko",
                target_language="en",
            )
        )

        initial_snapshot = presenter.snapshot()
        initial_snapshot_count = len(bridge.snapshots)
        initial_visible_text = [
            (block.primary_text, block.secondary_text, block.secondary_enabled)
            for block in initial_snapshot.blocks
        ]
        assert initial_visible_text == [("self source unchanged during refresh", "", True)]
        assert initial_snapshot.blocks[0].session_scope is None

        await asyncio.sleep(0)
        assert presenter._self_presentation_refresh_burst_task is not None
        assert len(refresh_sleep_indices()) == 1

        sleep_events[refresh_sleep_indices()[-1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        first_refresh = presenter.snapshot()

        assert first_refresh.revision == initial_snapshot.revision + 1
        assert len(bridge.snapshots) == initial_snapshot_count + 1
        assert [
            (block.primary_text, block.secondary_text, block.secondary_enabled)
            for block in first_refresh.blocks
        ] == initial_visible_text
        assert first_refresh.blocks[0].session_scope == "self_presentation_refresh=1"
        assert (
            first_refresh.native_fresh_render_generations
            == initial_snapshot.native_fresh_render_generations
        )

        assert len(refresh_sleep_indices()) == 2
        sleep_events[refresh_sleep_indices()[-1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second_refresh = presenter.snapshot()

        assert second_refresh.revision == initial_snapshot.revision + 2
        assert len(bridge.snapshots) == initial_snapshot_count + 2
        assert [
            (block.primary_text, block.secondary_text, block.secondary_enabled)
            for block in second_refresh.blocks
        ] == initial_visible_text
        assert second_refresh.blocks[0].session_scope == "self_presentation_refresh=2"
        assert (
            second_refresh.native_fresh_render_generations
            == initial_snapshot.native_fresh_render_generations
        )

        for _ in range(25):
            if presenter._self_presentation_refresh_burst_task is None:
                break
            assert sleep_events, "self refresh burst should be waiting for its next tick"
            sleep_events[refresh_sleep_indices()[-1]].set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        clean_snapshot = presenter.snapshot()
        assert presenter._self_presentation_refresh_burst_task is None
        assert clean_snapshot.revision > second_refresh.revision
        assert clean_snapshot.blocks[0].primary_text == "self source unchanged during refresh"
        assert clean_snapshot.blocks[0].secondary_text == ""
        assert clean_snapshot.blocks[0].session_scope is None
        assert bridge.snapshots[-1].blocks[0].session_scope is None
        assert (
            clean_snapshot.native_fresh_render_generations
            == initial_snapshot.native_fresh_render_generations
        )

        start_events = [
            fields
            for event, fields in diagnostics.events
            if event == "self_presentation_refresh_burst_start"
        ]
        end_events = [
            fields
            for event, fields in diagnostics.events
            if event == "self_presentation_refresh_burst_end"
        ]
        assert start_events == [
            {
                "reason": "eligible_finalized_self_update",
                "target_key": f"self:{self_turn_id}",
            }
        ]
        assert end_events[-1]["reason"] == "deadline_expired"
        assert end_events[-1]["target_key"] == f"self:{self_turn_id}"
        assert end_events[-1]["cleanup_publish_count"] == 1
        assert 20 <= end_events[-1]["tick_count"] <= 21
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_unchanged_self_finalized_duplicate_does_not_restart_or_extend_refresh_burst() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    cancelled_delays: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_delays.append(delay)
            raise
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()
    source_event = adapter.transcript_final(
        Transcript(
            utterance_id=self_turn_id,
            channel="self",
            text="duplicate self source final",
            is_final=True,
            created_at=10.0,
        ),
        source_language="ko",
        target_language="en",
    )

    def refresh_sleep_indices() -> list[int]:
        return [index for index, delay in enumerate(sleep_calls) if delay == 0.1]

    try:
        await presenter.emit(source_event)
        await asyncio.sleep(0)

        first_task = presenter._self_presentation_refresh_burst_task
        assert first_task is not None
        refresh_sleep_count_before_duplicate = len(refresh_sleep_indices())
        snapshot_count_before_duplicate = len(bridge.snapshots)
        revision_before_duplicate = presenter.snapshot().revision

        await presenter.emit(source_event)
        await asyncio.sleep(0)

        assert presenter._self_presentation_refresh_burst_task is first_task
        assert len(refresh_sleep_indices()) == refresh_sleep_count_before_duplicate
        assert len(bridge.snapshots) == snapshot_count_before_duplicate
        assert presenter.snapshot().revision == revision_before_duplicate
        assert cancelled_delays == []
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_self_presentation_refresh_zero_tick_replacement_records_end_and_cleans_cancel_metadata() -> (
    None
):
    diagnostics = RecordingPresenterDiagnostics()
    clock = FakeClock(_now=10.0)
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        _ = delay
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()

    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        diagnostics=diagnostics,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    first_self_turn_id = uuid4()
    second_self_turn_id = uuid4()

    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=first_self_turn_id,
                    channel="self",
                    text="first self source cancelled before start",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        first_task = presenter._self_presentation_refresh_burst_task
        assert first_task is not None

        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=second_self_turn_id,
                    channel="self",
                    text="second self source replaces zero tick task",
                    is_final=True,
                    created_at=10.1,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        assert presenter._self_presentation_refresh_burst_task is not first_task

        for _ in range(3):
            await asyncio.sleep(0)

        assert first_task.cancelled()
        assert first_task not in presenter._self_presentation_refresh_burst_cancel_reasons
        assert first_task not in presenter._self_presentation_refresh_burst_cancel_cleanup_counts

        first_target_key = f"self:{first_self_turn_id}"
        first_start_events = [
            fields
            for event, fields in diagnostics.events
            if event == "self_presentation_refresh_burst_start"
            and fields.get("target_key") == first_target_key
        ]
        first_end_events = [
            fields
            for event, fields in diagnostics.events
            if event == "self_presentation_refresh_burst_end"
            and fields.get("target_key") == first_target_key
        ]
        assert first_start_events == [
            {
                "reason": "eligible_finalized_self_update",
                "target_key": first_target_key,
            }
        ]
        assert first_end_events == [
            {
                "reason": "target_replaced",
                "target_key": first_target_key,
                "tick_count": 0,
                "cleanup_publish_count": 0,
            }
        ]
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_self_presentation_refresh_restart_after_visible_marker_resets_nonce_and_cleans_old_target() -> (
    None
):
    bridge = RecordingPresentationBridge()
    diagnostics = RecordingPresenterDiagnostics()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    cancelled_delays: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_delays.append(delay)
            raise
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        diagnostics=diagnostics,
        sleep=fake_sleep,
        visible_window_target_blocks=2,
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    first_self_turn_id = uuid4()
    second_self_turn_id = uuid4()

    def refresh_sleep_indices() -> list[int]:
        return [index for index, delay in enumerate(sleep_calls) if delay == 0.1]

    def blocks_by_id() -> dict[str, OverlayPresentationBlock]:
        return {block.id: block for block in presenter.snapshot().blocks}

    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=first_self_turn_id,
                    channel="self",
                    text="first self source",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await asyncio.sleep(0)
        first_task = presenter._self_presentation_refresh_burst_task
        assert first_task is not None

        sleep_events[refresh_sleep_indices()[-1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        first_marker_snapshot = presenter.snapshot()
        first_marker_blocks = blocks_by_id()
        assert first_marker_blocks[f"self:{first_self_turn_id}"].session_scope == (
            "self_presentation_refresh=1"
        )

        snapshot_count_before_restart = len(bridge.snapshots)

        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=second_self_turn_id,
                    channel="self",
                    text="second self source",
                    is_final=True,
                    created_at=10.2,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await asyncio.sleep(0)

        restarted_task = presenter._self_presentation_refresh_burst_task
        assert restarted_task is not None
        assert restarted_task is not first_task
        assert len(bridge.snapshots) == snapshot_count_before_restart + 2
        changed_snapshot_blocks = {block.id: block for block in bridge.snapshots[-2].blocks}
        assert f"self:{second_self_turn_id}" in changed_snapshot_blocks
        assert changed_snapshot_blocks[f"self:{first_self_turn_id}"].session_scope == (
            "self_presentation_refresh=1"
        )

        restart_clean_blocks = blocks_by_id()
        assert presenter.snapshot().revision == first_marker_snapshot.revision + 2
        assert restart_clean_blocks[f"self:{first_self_turn_id}"].session_scope is None
        assert restart_clean_blocks[f"self:{second_self_turn_id}"].session_scope is None
        target_replaced_end_events = [
            fields
            for event, fields in diagnostics.events
            if event == "self_presentation_refresh_burst_end"
            and fields.get("reason") == "target_replaced"
            and fields.get("target_key") == f"self:{first_self_turn_id}"
        ]
        assert target_replaced_end_events[-1]["cleanup_publish_count"] == 1

        sleep_events[refresh_sleep_indices()[-1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        restarted_marker_blocks = blocks_by_id()
        assert restarted_marker_blocks[f"self:{first_self_turn_id}"].session_scope is None
        assert restarted_marker_blocks[f"self:{second_self_turn_id}"].session_scope == (
            "self_presentation_refresh=1"
        )
        assert cancelled_delays
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_detach_bridge_preserves_self_refresh_but_runtime_detach_cleans_marker() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()

    def refresh_sleep_indices() -> list[int]:
        return [index for index, delay in enumerate(sleep_calls) if delay == 0.1]

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=self_turn_id,
                channel="self",
                text="self source across bridge detach",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    await asyncio.sleep(0)
    sleep_events[refresh_sleep_indices()[-1]].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    marker_snapshot = presenter.snapshot()
    assert marker_snapshot.blocks[0].session_scope == "self_presentation_refresh=1"
    active_task = presenter._self_presentation_refresh_burst_task
    assert active_task is not None

    presenter.detach_bridge()

    assert presenter.bridge is None
    assert presenter._self_presentation_refresh_burst_task is active_task
    assert presenter._presentation_state.self_presentation_refresh_target_key == (
        "self",
        self_turn_id,
    )
    assert presenter.snapshot().blocks[0].session_scope == "self_presentation_refresh=1"

    replacement_bridge = RecordingPresentationBridge()
    presenter.attach_bridge(replacement_bridge)
    await presenter.clear_for_runtime_detach()

    assert presenter._self_presentation_refresh_burst_task is None
    assert presenter._presentation_state.self_presentation_refresh_target_key is None
    assert presenter._presentation_state.self_presentation_refresh_nonce == 0
    assert presenter.snapshot().blocks == []
    assert replacement_bridge.snapshots[-1].blocks == []


@pytest.mark.asyncio
async def test_presenter_disabling_self_presentation_refresh_burst_preserves_active_peer_refresh_marker() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()
    peer_turn_id = uuid4()

    def refresh_sleep_indices() -> list[int]:
        return [index for index, delay in enumerate(sleep_calls) if delay == 0.1]

    def blocks_by_id() -> dict[str, OverlayPresentationBlock]:
        return {block.id: block for block in presenter.snapshot().blocks}

    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=self_turn_id,
                    channel="self",
                    text="self source during peer refresh",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=peer_turn_id,
                    channel="peer",
                    text="peer source during self refresh",
                    is_final=True,
                    created_at=10.1,
                ),
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer translation during self refresh",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.2,
            )
        )
        await asyncio.sleep(0)
        assert len(refresh_sleep_indices()) >= 2

        sleep_events[refresh_sleep_indices()[0]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        sleep_events[refresh_sleep_indices()[1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        marker_blocks = blocks_by_id()
        assert marker_blocks[f"self:{self_turn_id}"].session_scope == (
            "self_presentation_refresh=1"
        )
        assert marker_blocks[f"peer:{peer_turn_id}"].session_scope == (
            "peer_presentation_refresh=1"
        )
        peer_task = presenter._peer_presentation_refresh_burst_task
        assert peer_task is not None

        await presenter.update_self_presentation_refresh_burst(False)

        clean_self_blocks = blocks_by_id()
        assert presenter._self_presentation_refresh_burst_task is None
        assert presenter._peer_presentation_refresh_burst_task is peer_task
        assert clean_self_blocks[f"self:{self_turn_id}"].session_scope is None
        assert clean_self_blocks[f"peer:{peer_turn_id}"].session_scope == (
            "peer_presentation_refresh=1"
        )
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_disabling_peer_presentation_refresh_burst_preserves_active_self_refresh_marker() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()
    peer_turn_id = uuid4()

    def refresh_sleep_indices() -> list[int]:
        return [index for index, delay in enumerate(sleep_calls) if delay == 0.1]

    def blocks_by_id() -> dict[str, OverlayPresentationBlock]:
        return {block.id: block for block in presenter.snapshot().blocks}

    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=self_turn_id,
                    channel="self",
                    text="self source while peer disable",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=peer_turn_id,
                    channel="peer",
                    text="peer source while self refresh stays",
                    is_final=True,
                    created_at=10.1,
                ),
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer translation while self refresh stays",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.2,
            )
        )
        await asyncio.sleep(0)
        assert len(refresh_sleep_indices()) >= 2

        sleep_events[refresh_sleep_indices()[0]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        sleep_events[refresh_sleep_indices()[1]].set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        marker_blocks = blocks_by_id()
        assert marker_blocks[f"self:{self_turn_id}"].session_scope == (
            "self_presentation_refresh=1"
        )
        assert marker_blocks[f"peer:{peer_turn_id}"].session_scope == (
            "peer_presentation_refresh=1"
        )
        self_task = presenter._self_presentation_refresh_burst_task
        assert self_task is not None

        await presenter.update_peer_presentation_refresh_burst(False)

        clean_peer_blocks = blocks_by_id()
        assert presenter._self_presentation_refresh_burst_task is self_task
        assert presenter._peer_presentation_refresh_burst_task is None
        assert clean_peer_blocks[f"self:{self_turn_id}"].session_scope == (
            "self_presentation_refresh=1"
        )
        assert clean_peer_blocks[f"peer:{peer_turn_id}"].session_scope is None
    finally:
        await presenter.clear_for_runtime_detach()


@pytest.mark.asyncio
async def test_presenter_reset_scene_clears_active_self_refresh_marker() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    self_turn_id = uuid4()

    def refresh_sleep_indices() -> list[int]:
        return [index for index, delay in enumerate(sleep_calls) if delay == 0.1]

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=self_turn_id,
                channel="self",
                text="self source before scene reset",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    await asyncio.sleep(0)
    sleep_events[refresh_sleep_indices()[-1]].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks[0].session_scope == "self_presentation_refresh=1"

    presenter.reset_scene()
    await asyncio.sleep(0)

    assert presenter._self_presentation_refresh_burst_task is None
    assert presenter._presentation_state.self_presentation_refresh_target_key is None
    assert presenter._presentation_state.self_presentation_refresh_nonce == 0
    assert presenter.snapshot().blocks == []


@pytest.mark.asyncio
async def test_presenter_active_turn_replacement_uses_shared_lifecycle_with_channel_publishability() -> (
    None
):
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        visible_window_target_blocks=3,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    first_self = uuid4()
    second_self = uuid4()
    first_peer = uuid4()
    second_peer = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="first self live only",
            utterance_id=first_self,
            occupant_key=f"self:{first_self}",
            created_at=10.0,
        )
    )
    await presenter.emit(
        adapter.self_active_update(
            text="second self live",
            utterance_id=second_self,
            occupant_key=f"self:{second_self}",
            created_at=10.1,
        )
    )

    self_blocks = [block for block in presenter.snapshot().blocks if block.channel == "self"]
    assert [block.id for block in self_blocks] == [f"self:{second_self}"]
    assert self_blocks[0].block_variant == "active_self"
    assert self_blocks[0].primary_text == "second self live"

    await presenter.emit(
        adapter.peer_active_update(
            text="first peer source",
            utterance_id=first_peer,
            occupant_key=f"peer:{first_peer}",
            created_at=11.0,
        )
    )
    await presenter.emit(
        adapter.peer_active_update(
            text="second peer source",
            utterance_id=second_peer,
            occupant_key=f"peer:{second_peer}",
            created_at=11.1,
        )
    )

    peer_blocks = [block for block in presenter.snapshot().blocks if block.channel == "peer"]
    assert [block.id for block in peer_blocks] == [f"peer:{first_peer}", f"peer:{second_peer}"]
    assert peer_blocks[0].block_variant == "finalized"
    assert peer_blocks[0].primary_text == ""
    assert peer_blocks[0].secondary_text == "first peer source"
    assert peer_blocks[1].block_variant == "active_peer"
    assert peer_blocks[1].primary_text == ""
    assert peer_blocks[1].secondary_text == "second peer source"


@pytest.mark.asyncio
async def test_presenter_new_peer_turn_demotes_previous_live_peer_and_protects_new_peer() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        visible_window_target_blocks=2,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    first = uuid4()
    second = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="first source",
            utterance_id=first,
            occupant_key=f"peer:{first}",
            created_at=10.0,
        )
    )
    await presenter.emit(
        adapter.peer_active_update(
            text="second source",
            utterance_id=second,
            occupant_key=f"peer:{second}",
            created_at=10.1,
        )
    )

    blocks = presenter.snapshot().blocks

    assert [block.id for block in blocks] == [f"peer:{first}", f"peer:{second}"]

    first_block, second_block = blocks
    assert first_block.block_variant == "finalized"
    assert first_block.primary_text == ""
    assert first_block.secondary_text == "first source"
    assert first_block.secondary_enabled is True

    assert second_block.block_variant == "active_peer"
    assert second_block.primary_text == ""
    assert second_block.secondary_text == "second source"
    assert second_block.secondary_enabled is True


@pytest.mark.asyncio
async def test_presenter_latest_peer_translation_not_displaced_by_older_translated_self_row() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        visible_window_target_blocks=1,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    self_turn_id = uuid4()
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=self_turn_id,
                channel="self",
                text="older self",
                is_final=True,
                created_at=10.1,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=self_turn_id,
            channel="self",
            text="older self translation",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=10.2,
        )
    )
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn_id,
                channel="peer",
                text="newer peer",
                is_final=True,
                created_at=10.3,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="newer peer translation",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.4,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"peer:{peer_turn_id}"]
    assert presenter.snapshot().blocks[0].primary_text == "newer peer translation"


@pytest.mark.asyncio
async def test_presenter_reschedules_closed_peer_expiration_with_translation_min_visibility() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    cancelled_delays: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_delays.append(delay)
            raise
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="peer original",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=10.0,
        )
    )
    await presenter.emit(
        adapter.utterance_closed(
            utterance_id=peer_turn_id,
            channel="peer",
            created_at=10.1,
        )
    )

    await asyncio.sleep(0)
    assert sleep_calls == [8.0]

    clock.advance(7.0)
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer translation",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=17.0,
        )
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert cancelled_delays == [8.0]
    assert sleep_calls == [8.0, 8.0]
    assert presenter.snapshot().blocks[0].primary_text == "peer translation"
    assert presenter.snapshot().blocks[0].secondary_text == "peer original"

    sleep_events[0].set()
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks[0].primary_text == "peer translation"

    sleep_events[1].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks == []


@pytest.mark.asyncio
async def test_presenter_reschedules_closed_self_expiration_with_translation_min_visibility() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    cancelled_delays: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_delays.append(delay)
            raise
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    transcript = Transcript(
        utterance_id=uuid4(),
        channel="self",
        text="self original",
        is_final=True,
        created_at=10.0,
    )

    await presenter.emit(
        adapter.transcript_final(
            transcript,
            source_language="ko",
            target_language="en",
        )
    )
    await presenter.emit(
        adapter.utterance_closed(
            utterance_id=transcript.utterance_id,
            channel="self",
            created_at=10.1,
        )
    )

    await asyncio.sleep(0)

    assert sleep_calls == [8.0]

    clock.advance(7.0)
    await presenter.emit(
        adapter.translation_final(
            utterance_id=transcript.utterance_id,
            channel="self",
            text="self translation",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=17.0,
        )
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert cancelled_delays == [8.0]
    assert sleep_calls == [8.0, 8.0]
    assert presenter.snapshot().blocks[0].secondary_text == "self translation"

    sleep_events[0].set()
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks[0].secondary_text == "self translation"

    sleep_events[1].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks == []


@pytest.mark.asyncio
async def test_presenter_restarts_self_translation_min_visibility_when_translation_changes() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    cancelled_delays: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_delays.append(delay)
            raise
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    transcript = Transcript(
        utterance_id=uuid4(),
        channel="self",
        text="self original",
        is_final=True,
        created_at=10.0,
    )

    await presenter.emit(
        adapter.transcript_final(
            transcript,
            source_language="ko",
            target_language="en",
        )
    )
    await presenter.emit(
        adapter.utterance_closed(
            utterance_id=transcript.utterance_id,
            channel="self",
            created_at=10.1,
        )
    )

    await asyncio.sleep(0)
    assert sleep_calls == [8.0]

    clock.advance(7.5)
    await presenter.emit(
        adapter.translation_final(
            utterance_id=transcript.utterance_id,
            channel="self",
            text="self translation one",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=17.5,
        )
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert cancelled_delays == [8.0]
    assert sleep_calls == [8.0, 8.0]
    assert presenter.snapshot().blocks[0].secondary_text == "self translation one"

    clock.advance(2.0)
    await presenter.emit(
        adapter.translation_final(
            utterance_id=transcript.utterance_id,
            channel="self",
            text="self translation two",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=19.5,
        )
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert cancelled_delays == [8.0, 8.0]
    assert sleep_calls == [8.0, 8.0, 8.0]
    assert presenter.snapshot().blocks[0].secondary_text == "self translation two"

    sleep_events[0].set()
    await asyncio.sleep(0)
    assert presenter.snapshot().blocks[0].secondary_text == "self translation two"

    sleep_events[1].set()
    await asyncio.sleep(0)
    assert presenter.snapshot().blocks[0].secondary_text == "self translation two"

    sleep_events[2].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks == []


@pytest.mark.asyncio
async def test_presenter_hidden_self_translation_update_does_not_extend_visible_ttl() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    utterance_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=utterance_id,
                channel="self",
                text="self original",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=utterance_id,
            channel="self",
            text="visible translation",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=10.0,
        )
    )
    await presenter.emit(
        adapter.utterance_closed(
            utterance_id=utterance_id,
            channel="self",
            created_at=10.0,
        )
    )

    await asyncio.sleep(0)

    clock.advance(2.0)
    await presenter.update_display_preferences(
        show_translation=False,
        show_peer_original=True,
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks[0].secondary_enabled is False
    assert sleep_calls[-1] == 8.0

    clock.advance(7.0)
    await presenter.emit(
        adapter.translation_final(
            utterance_id=utterance_id,
            channel="self",
            text="hidden translation update",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=19.0,
        )
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks[0].secondary_enabled is False
    assert sleep_calls[-1] == 1.0

    sleep_events[-1].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks == []


@pytest.mark.asyncio
async def test_presenter_records_expired_entry_diagnostic_with_deadlines(
    tmp_path,
) -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)

    async def fake_sleep(delay: float) -> None:
        clock.advance(delay)
        await asyncio.sleep(0)

    diagnostics = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-test",
        diagnostics_dir=tmp_path,
    )
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        diagnostics=diagnostics,
        sleep=fake_sleep,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    utterance_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=utterance_id,
                channel="self",
                text="self original",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    await presenter.emit(
        adapter.utterance_closed(
            utterance_id=utterance_id,
            channel="self",
            created_at=10.1,
        )
    )
    clock.advance(7.0)
    await presenter.emit(
        adapter.translation_final(
            utterance_id=utterance_id,
            channel="self",
            text="self translation",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=17.0,
        )
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert list(diagnostics.presenter_events) == []
    assert list(diagnostics.presenter_removal_events) == []


@pytest.mark.asyncio
async def test_presenter_records_untranslated_self_visibility_duration(
    tmp_path,
) -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)

    async def fake_sleep(delay: float) -> None:
        clock.advance(delay)
        await asyncio.sleep(0)

    diagnostics = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-test",
        diagnostics_dir=tmp_path,
    )
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        diagnostics=diagnostics,
        sleep=fake_sleep,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    utterance_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=utterance_id,
                channel="self",
                text="self original",
                is_final=True,
                created_at=10.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    await presenter.emit(
        adapter.utterance_closed(
            utterance_id=utterance_id,
            channel="self",
            created_at=10.1,
        )
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert list(diagnostics.presenter_events) == []
    assert list(diagnostics.presenter_removal_events) == []


@pytest.mark.asyncio
async def test_presenter_records_peer_displacement_as_removal_diagnostic(tmp_path) -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    diagnostics = OverlayDiagnosticsRecorder(
        overlay_instance_id="overlay-test",
        diagnostics_dir=tmp_path,
    )
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        diagnostics=diagnostics,
    )
    adapter = OverlayEventAdapter(clock=clock)
    utterance_ids = [uuid4(), uuid4(), uuid4()]

    for index, utterance_id in enumerate(utterance_ids, start=1):
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=utterance_id,
                    channel="peer",
                    text=f"peer original {index}",
                    is_final=True,
                    created_at=float(index),
                ),
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=utterance_id,
                channel="peer",
                text=f"peer translation {index}",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=float(index) + 0.1,
            )
        )
        await presenter.emit(
            adapter.utterance_closed(
                utterance_id=utterance_id,
                channel="peer",
                is_final=True,
                created_at=float(index) + 0.2,
            )
        )

    assert list(diagnostics.presenter_events) == []
    assert list(diagnostics.presenter_removal_events) == []


@pytest.mark.asyncio
async def test_presenter_includes_calibration_inside_snapshot_updates() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        self_presentation_refresh_burst=False,
    )

    await presenter.update_calibration(
        OverlayCalibration(
            anchor="head_locked",
            offset_x=0.2,
            offset_y=-0.1,
            distance=1.5,
            text_scale=1.1,
            background_alpha=0.33,
        )
    )

    latest = bridge.snapshots[-1]

    assert latest.calibration == OverlayPresentationCalibration(
        anchor="head_locked",
        offset_x=0.2,
        offset_y=-0.1,
        distance=1.5,
        text_scale=1.1,
        background_alpha=0.33,
    )


@pytest.mark.asyncio
async def test_presenter_shutdown_is_control_plane_only() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        self_presentation_refresh_burst=False,
    )

    await presenter.broadcast_shutdown()

    assert bridge.shutdown_calls == 1
    assert bridge.snapshots == []


@pytest.mark.asyncio
async def test_presenter_ignores_stale_self_active_clear() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        self_presentation_refresh_burst=False,
    )
    active_utterance_id = uuid4()

    await presenter.emit(
        SelfActiveUpdate(
            event_id="active-new",
            seq=2,
            utterance_id=active_utterance_id,
            channel="self",
            created_at=10.0,
            text="live self",
            occupant_key=f"self:{active_utterance_id}",
        )
    )
    revision_before_clear = presenter.snapshot().revision

    await presenter.emit(
        SelfActiveClear(
            event_id="clear-old",
            seq=1,
            utterance_id=None,
            channel="self",
            created_at=9.0,
        )
    )

    assert presenter.snapshot().revision == revision_before_clear
    assert presenter.snapshot().blocks[-1].id == f"self:{active_utterance_id}"
    assert presenter.snapshot().blocks[-1].occupant_key == f"self:{active_utterance_id}"
    assert presenter.snapshot().blocks[-1].block_variant == "active_self"
    assert presenter.snapshot().blocks[-1].primary_text == "live self"
    assert presenter.snapshot().blocks[-1].secondary_text == ""
    assert presenter.snapshot().blocks[-1].secondary_enabled is True


@pytest.mark.asyncio
async def test_presenter_ignores_stale_cross_turn_self_active_update_without_disturbing_current_live_row() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
    )
    current_turn_id = uuid4()
    stale_turn_id = uuid4()

    await presenter.emit(
        SelfActiveUpdate(
            event_id="active-current",
            seq=5,
            utterance_id=current_turn_id,
            channel="self",
            created_at=10.0,
            text="current live",
            occupant_key=f"self:{current_turn_id}",
        )
    )
    revision_before_stale = presenter.snapshot().revision

    await presenter.emit(
        SelfActiveUpdate(
            event_id="active-stale-other",
            seq=4,
            utterance_id=stale_turn_id,
            channel="self",
            created_at=9.0,
            text="stale live",
            occupant_key=f"self:{stale_turn_id}",
        )
    )

    assert presenter.snapshot().revision == revision_before_stale
    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{current_turn_id}"]
    assert presenter.snapshot().blocks[0].primary_text == "current live"
    assert len(bridge.snapshots) == 1
    assert [block.id for block in bridge.snapshots[-1].blocks] == [f"self:{current_turn_id}"]


@pytest.mark.asyncio
async def test_presenter_ignores_stale_history_updates() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        self_presentation_refresh_burst=False,
    )
    utterance_id = uuid4()

    await presenter.emit(
        SelfTranscriptFinal(
            event_id="self-new",
            seq=10,
            utterance_id=utterance_id,
            channel="self",
            created_at=10.0,
            text="latest original",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )
    await presenter.emit(
        TranslationFinal(
            event_id="translation-new",
            seq=12,
            utterance_id=utterance_id,
            channel="self",
            created_at=12.0,
            text="latest translation",
            source_language="ko",
            target_language="en",
            is_final=True,
            applied_context_mode=None,
        )
    )
    revision_before_stale = presenter.snapshot().revision

    await presenter.emit(
        SelfTranscriptFinal(
            event_id="self-old",
            seq=9,
            utterance_id=utterance_id,
            channel="self",
            created_at=9.0,
            text="stale original",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )
    await presenter.emit(
        TranslationFinal(
            event_id="translation-old",
            seq=11,
            utterance_id=utterance_id,
            channel="self",
            created_at=11.0,
            text="stale translation",
            source_language="ko",
            target_language="en",
            is_final=True,
            applied_context_mode=None,
        )
    )

    assert presenter.snapshot().revision == revision_before_stale
    assert presenter.snapshot().blocks[-1].primary_text == "latest original"
    assert presenter.snapshot().blocks[-1].secondary_text == "latest translation"


@pytest.mark.asyncio
async def test_presenter_allows_two_self_rows_and_evicts_oldest_on_third_self_turn() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        self_presentation_refresh_burst=False,
    )

    turn_ids = [uuid4(), uuid4(), uuid4()]

    for index, turn_id in enumerate(turn_ids, start=1):
        await presenter.emit(
            SelfTranscriptFinal(
                event_id=f"self-{index}",
                seq=index,
                utterance_id=turn_id,
                channel="self",
                created_at=float(index),
                text=f"original {index}",
                source_language="ko",
                target_language="en",
                is_final=True,
            )
        )

    assert [block.id for block in presenter.snapshot().blocks] == [
        f"self:{turn_ids[1]}",
        f"self:{turn_ids[2]}",
    ]
    assert [block.primary_text for block in presenter.snapshot().blocks] == [
        "original 2",
        "original 3",
    ]


@pytest.mark.asyncio
async def test_presenter_preserves_input_order_when_two_turns_first_become_visible_in_same_window() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
    )
    adapter = OverlayEventAdapter(clock=clock)
    older = Transcript(
        utterance_id=uuid4(),
        channel="peer",
        text="peer older",
        is_final=True,
        created_at=11.0,
    )
    newer = Transcript(
        utterance_id=uuid4(),
        channel="peer",
        text="peer newer",
        is_final=True,
        created_at=12.0,
    )

    await presenter.emit(
        adapter.transcript_final(older, source_language="en", target_language="ko")
    )
    await presenter.emit(
        adapter.transcript_final(newer, source_language="en", target_language="ko")
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=newer.utterance_id,
            channel="peer",
            text="피어 둘",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=13.0,
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=older.utterance_id,
            channel="peer",
            text="피어 하나",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=13.1,
        )
    )

    assert [block.id for block in bridge.snapshots[-1].blocks] == [
        f"peer:{older.utterance_id}",
        f"peer:{newer.utterance_id}",
    ]
    assert [block.primary_text for block in bridge.snapshots[-1].blocks] == [
        "피어 하나",
        "피어 둘",
    ]


@pytest.mark.asyncio
async def test_presenter_evicted_turn_late_update_is_ignored() -> None:
    bridge = RecordingPresentationBridge()
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        self_presentation_refresh_burst=False,
    )

    displaced_turn_id = uuid4()
    newer_turn_ids = [uuid4(), uuid4()]

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=displaced_turn_id,
                channel="self",
                text="original 1",
                is_final=True,
                created_at=1.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )

    for index, turn_id in enumerate(newer_turn_ids, start=2):
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=turn_id,
                    channel="self",
                    text=f"original {index}",
                    is_final=True,
                    created_at=float(index),
                ),
                source_language="ko",
                target_language="en",
            )
        )

    assert [block.id for block in presenter.snapshot().blocks] == [
        f"self:{newer_turn_ids[0]}",
        f"self:{newer_turn_ids[1]}",
    ]

    revision_before_late_update = presenter.snapshot().revision
    blocks_before_late_update = presenter.snapshot().blocks
    snapshot_count_before_late_update = len(bridge.snapshots)

    await presenter.emit(
        adapter.translation_final(
            utterance_id=displaced_turn_id,
            channel="self",
            text="late translation",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=9.9,
        )
    )

    assert presenter.snapshot().revision == revision_before_late_update
    assert presenter.snapshot().blocks == blocks_before_late_update
    assert len(bridge.snapshots) == snapshot_count_before_late_update


@pytest.mark.asyncio
async def test_presenter_evicted_turn_remains_ignored_after_tombstone_cap_overflow() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        self_presentation_refresh_burst=False,
    )

    total_turns = 67
    turn_ids = [uuid4() for _ in range(total_turns)]
    for index, turn_id in enumerate(turn_ids, start=1):
        await presenter.emit(
            SelfTranscriptFinal(
                event_id=f"self-{index}",
                seq=index,
                utterance_id=turn_id,
                channel="self",
                created_at=float(index),
                text=f"original {index}",
                source_language="ko",
                target_language="en",
                is_final=True,
            )
        )

    assert [block.id for block in presenter.snapshot().blocks] == [
        f"self:{turn_ids[-2]}",
        f"self:{turn_ids[-1]}",
    ]

    revision_before_late_update = presenter.snapshot().revision
    blocks_before_late_update = presenter.snapshot().blocks
    snapshot_count_before_late_update = len(bridge.snapshots)

    await presenter.emit(
        SelfTranscriptFinal(
            event_id="late-self-1",
            seq=1000,
            utterance_id=turn_ids[0],
            channel="self",
            created_at=1000.0,
            text="late original",
            source_language="ko",
            target_language="en",
            is_final=True,
        )
    )

    assert presenter.snapshot().revision == revision_before_late_update
    assert presenter.snapshot().blocks == blocks_before_late_update
    assert len(bridge.snapshots) == snapshot_count_before_late_update
    assert bridge.snapshots[-1].blocks == blocks_before_late_update


@pytest.mark.asyncio
async def test_presenter_expires_visible_finalized_entry_after_eight_seconds() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    transcript = Transcript(
        utterance_id=uuid4(),
        channel="self",
        text="hello now",
        is_final=True,
        created_at=10.0,
    )

    await presenter.emit(
        adapter.transcript_final(
            transcript,
            source_language="ko",
            target_language="en",
        )
    )
    await presenter.emit(
        adapter.utterance_closed(
            utterance_id=transcript.utterance_id,
            channel="self",
            is_final=True,
            created_at=10.1,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [
        f"self:{transcript.utterance_id}"
    ]
    assert len(bridge.snapshots) == 1

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert sleep_calls == [8.0]
    assert presenter.snapshot().blocks == []
    assert len(bridge.snapshots) == 2
    assert bridge.snapshots[-1].blocks == []


@pytest.mark.asyncio
async def test_presenter_meaningful_update_refreshes_idle_deadline_but_duplicate_visible_content_does_not() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        release = asyncio.Event()
        sleep_events.append(release)
        await release.wait()
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
    )
    adapter = OverlayEventAdapter(clock=clock)
    turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="live one",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=10.0,
        )
    )
    await asyncio.sleep(0)

    clock.advance(4.0)
    await presenter.emit(
        adapter.self_active_update(
            text="live two",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=14.0,
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    clock.advance(7.5)

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{turn_id}"]
    assert presenter.snapshot().blocks[0].primary_text == "live two"

    revision_before_duplicate = presenter.snapshot().revision
    await presenter.emit(
        adapter.self_active_update(
            text="live two",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=21.5,
        )
    )

    assert presenter.snapshot().revision == revision_before_duplicate

    sleep_events[0].set()
    await asyncio.sleep(0)
    sleep_events[1].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks == []


@pytest.mark.asyncio
async def test_presenter_idle_hidden_turn_late_update_is_ignored() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)

    async def fake_sleep(delay: float) -> None:
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
    )
    adapter = OverlayEventAdapter(clock=clock)
    turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="live self",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=10.0,
        )
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks == []

    revision_before_late_update = presenter.snapshot().revision
    await presenter.emit(
        adapter.self_active_update(
            text="late self",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=18.1,
        )
    )

    assert presenter.snapshot().revision == revision_before_late_update
    assert presenter.snapshot().blocks == []


@pytest.mark.asyncio
async def test_presenter_channel_role_mapping_preserves_identity_with_required_text_roles() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    self_turn_id = uuid4()
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="self source live",
            secondary_text="self translation preview",
            utterance_id=self_turn_id,
            occupant_key=f"self:{self_turn_id}",
            created_at=10.0,
        )
    )
    self_active = presenter.snapshot().blocks[-1]

    await presenter.emit(
        SelfTranscriptFinal(
            event_id="self-final-role-mapping",
            seq=2,
            utterance_id=self_turn_id,
            channel="self",
            text="self source live",
            source_language="ko",
            target_language="en",
            created_at=10.1,
        )
    )
    self_final = next(
        block for block in presenter.snapshot().blocks if block.id == f"self:{self_turn_id}"
    )

    await presenter.emit(
        adapter.peer_active_update(
            text="peer source original",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=11.0,
        )
    )
    peer_active = presenter.snapshot().blocks[-1]

    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="peer translation final",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=11.1,
        )
    )
    peer_final = next(
        block for block in presenter.snapshot().blocks if block.id == f"peer:{peer_turn_id}"
    )

    assert self_active.id == self_final.id == f"self:{self_turn_id}"
    assert self_active.occupant_key == self_final.occupant_key == f"self:{self_turn_id}"
    assert self_active.primary_text == "self source live"
    assert self_active.secondary_text == "self translation preview"
    assert self_final.primary_text == "self source live"
    assert self_final.secondary_text == "self translation preview"

    assert peer_active.id == peer_final.id == f"peer:{peer_turn_id}"
    assert peer_active.occupant_key == peer_final.occupant_key == f"peer:{peer_turn_id}"
    assert peer_active.block_variant == "active_peer"
    assert peer_active.primary_text == ""
    assert peer_active.secondary_text == "peer source original"
    assert peer_final.block_variant == "finalized"
    assert peer_final.primary_text == "peer translation final"
    assert peer_final.secondary_text == "peer source original"


@pytest.mark.asyncio
async def test_presenter_self_active_self_final_and_self_translation_share_one_row_identity() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="hello live",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=10.0,
        )
    )
    await presenter.emit(
        SelfTranscriptFinal(
            event_id="final-1",
            seq=2,
            utterance_id=turn_id,
            channel="self",
            text="hello live",
            source_language="ko",
            target_language="en",
            created_at=11.0,
        )
    )
    await presenter.emit(
        TranslationFinal(
            event_id="translation-1",
            seq=3,
            utterance_id=turn_id,
            channel="self",
            created_at=12.0,
            text="translated live",
            source_language="ko",
            target_language="en",
            is_final=True,
            applied_context_mode=None,
        )
    )

    assert all(len(snapshot.blocks) == 1 for snapshot in bridge.snapshots)
    assert [snapshot.blocks[0].occupant_key for snapshot in bridge.snapshots] == [
        f"self:{turn_id}"
    ] * 3
    assert [snapshot.blocks[0].block_variant for snapshot in bridge.snapshots] == [
        "active_self",
        "finalized",
        "finalized",
    ]
    assert [snapshot.blocks[0].id for snapshot in bridge.snapshots[1:]] == [
        f"self:{turn_id}",
        f"self:{turn_id}",
    ]
    assert bridge.snapshots[-1].blocks[0].secondary_text == "translated live"


@pytest.mark.asyncio
async def test_presenter_promotes_same_turn_preview_secondary_into_finalized_self_row() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="hello live",
            secondary_text="translated live",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=10.0,
        )
    )
    await presenter.emit(
        SelfTranscriptFinal(
            event_id="final-promote-preview-secondary",
            seq=2,
            utterance_id=turn_id,
            channel="self",
            text="hello live",
            source_language="ko",
            target_language="en",
            created_at=11.0,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{turn_id}"]
    assert presenter.snapshot().blocks[0].block_variant == "finalized"
    assert presenter.snapshot().blocks[0].primary_text == "hello live"
    assert presenter.snapshot().blocks[0].secondary_text == "translated live"


@pytest.mark.asyncio
async def test_presenter_active_self_snapshot_round_trips_update_metadata() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="hello live",
            secondary_text="translated live",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=10.0,
            update_id="upd-active-self",
            origin_wall_clock_ms=1712345678901,
            session_scope="session:self",
            source_text_hash="activehash123456",
            source_text_len=10,
            logical_turn_key=f"self:{turn_id}",
        )
    )

    block = presenter.snapshot().blocks[0]

    assert block.block_variant == "active_self"
    assert block.update_id == "upd-active-self"
    assert block.origin_wall_clock_ms == 1712345678901
    assert block.session_scope == "session:self"
    assert block.source_text_hash == "activehash123456"
    assert block.source_text_len == 10
    assert block.logical_turn_key == f"self:{turn_id}"
    assert OverlayPresentationBlock.from_dict(block.to_dict()) == block


@pytest.mark.asyncio
async def test_presenter_promoted_finalized_self_snapshot_round_trips_update_metadata() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="hello live",
            secondary_text="translated live",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=10.0,
            update_id="upd-finalized-self",
            origin_wall_clock_ms=1712345678902,
            session_scope="session:self",
            source_text_hash="selffinalhash123",
            source_text_len=10,
            logical_turn_key=f"self:{turn_id}",
        )
    )
    await presenter.emit(
        SelfTranscriptFinal(
            event_id="final-promote-preview-secondary-metadata",
            seq=2,
            utterance_id=turn_id,
            channel="self",
            text="hello live",
            source_language="ko",
            target_language="en",
            created_at=11.0,
        )
    )

    block = presenter.snapshot().blocks[0]

    assert block.block_variant == "finalized"
    assert block.secondary_text == "translated live"
    assert block.update_id == "upd-finalized-self"
    assert block.origin_wall_clock_ms == 1712345678902
    assert block.session_scope == "session:self"
    assert block.source_text_hash == "selffinalhash123"
    assert block.source_text_len == 10
    assert block.logical_turn_key == f"self:{turn_id}"
    assert OverlayPresentationBlock.from_dict(block.to_dict()) == block


@pytest.mark.asyncio
async def test_presenter_hidden_self_translation_metadata_update_does_not_bump_revision_or_publish() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        self_presentation_refresh_burst=False,
        native_retry_trigger_emission=True,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=turn_id,
                channel="self",
                text="hello hidden",
                is_final=True,
                created_at=11.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=turn_id,
            channel="self",
            text="translated hidden",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=12.0,
            update_id="upd-hidden-self-before",
            origin_wall_clock_ms=1712345678910,
            session_scope="session:hidden-before",
            source_text_hash="hiddenbeforehash123",
            source_text_len=17,
            logical_turn_key=f"self:{turn_id}",
        )
    )
    await presenter.update_display_preferences(
        show_translation=False,
        show_peer_original=True,
    )

    snapshot_before_metadata = presenter.snapshot()
    snapshot_count_before_metadata = len(bridge.snapshots)

    await presenter.emit(
        adapter.translation_final(
            utterance_id=turn_id,
            channel="self",
            text="translated hidden",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=13.0,
            update_id="upd-hidden-self-after",
            origin_wall_clock_ms=1712345678920,
            session_scope="session:hidden-after",
            source_text_hash="hiddenafterhash456",
            source_text_len=17,
            logical_turn_key=f"self:{turn_id}",
        )
    )

    assert presenter.snapshot() == snapshot_before_metadata
    assert len(bridge.snapshots) == snapshot_count_before_metadata
    assert presenter.snapshot().native_fresh_render_generations.self == 2
    assert presenter.snapshot().blocks[0].secondary_enabled is False


@pytest.mark.asyncio
async def test_presenter_visible_peer_metadata_update_does_not_refresh_idle_ttl() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    sleep_calls: list[float] = []
    cancelled_delays: list[float] = []
    sleep_events: list[asyncio.Event] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)
        release = asyncio.Event()
        sleep_events.append(release)
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_delays.append(delay)
            raise
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        peer_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    turn_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=turn_id,
                channel="peer",
                text="peer original",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=turn_id,
            channel="peer",
            text="peer translation",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=10.0,
            update_id="peer-visible-before",
            origin_wall_clock_ms=1712345678901,
            session_scope="session:peer:before",
            source_text_hash="peerbeforehash123",
            source_text_len=16,
            logical_turn_key=f"peer:{turn_id}",
        )
    )
    await presenter.emit(
        adapter.utterance_closed(
            utterance_id=turn_id,
            channel="peer",
            created_at=10.1,
        )
    )

    await asyncio.sleep(0)

    entry = presenter._entries[("peer", turn_id)]
    visible_anchor_before_metadata = entry.last_meaningful_visible_at
    revision_before_metadata = presenter.snapshot().revision

    assert visible_anchor_before_metadata == 10.0
    assert sleep_calls == [8.0]
    assert presenter.snapshot().blocks[0].update_id == "peer-visible-before"

    clock.advance(7.0)
    await presenter.emit(
        adapter.translation_final(
            utterance_id=turn_id,
            channel="peer",
            text="peer translation",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=17.0,
            update_id="peer-visible-after",
            origin_wall_clock_ms=1712345678911,
            session_scope="session:peer:after",
            source_text_hash="peerafterhash456",
            source_text_len=16,
            logical_turn_key=f"peer:{turn_id}",
        )
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().revision == revision_before_metadata + 1
    assert presenter.snapshot().blocks[0].primary_text == "peer translation"
    assert presenter.snapshot().blocks[0].update_id == "peer-visible-after"
    assert entry.last_meaningful_visible_at == visible_anchor_before_metadata
    assert cancelled_delays == [8.0]
    assert sleep_calls == [8.0, 1.0]

    sleep_events[-1].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks == []


@pytest.mark.asyncio
async def test_presenter_finalized_peer_snapshot_round_trips_update_metadata() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()
    transcript = Transcript(
        utterance_id=peer_turn_id,
        channel="peer",
        text="peer original",
        is_final=True,
        created_at=11.0,
    )

    await presenter.emit(
        adapter.transcript_final(
            transcript,
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="상대 번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=12.0,
            update_id="upd-finalized-peer",
            origin_wall_clock_ms=1712345678903,
            session_scope="session:peer",
            source_text_hash="peerfinalhash123",
            source_text_len=13,
            logical_turn_key=f"peer:{peer_turn_id}",
        )
    )

    block = presenter.snapshot().blocks[0]

    assert block.block_variant == "finalized"
    assert block.primary_text == "상대 번역"
    assert block.update_id == "upd-finalized-peer"
    assert block.origin_wall_clock_ms == 1712345678903
    assert block.session_scope == "session:peer"
    assert block.source_text_hash == "peerfinalhash123"
    assert block.source_text_len == 13
    assert block.logical_turn_key == f"peer:{peer_turn_id}"
    assert OverlayPresentationBlock.from_dict(block.to_dict()) == block


@pytest.mark.asyncio
async def test_presenter_renders_active_self_secondary_text() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    active_utterance_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="hello live",
            secondary_text="translated live",
            utterance_id=active_utterance_id,
            occupant_key=f"self:{active_utterance_id}",
            created_at=10.0,
        )
    )

    blocks = presenter.snapshot().blocks
    assert len(blocks) == 1
    assert blocks[0].id == f"self:{active_utterance_id}"
    assert blocks[0].block_variant == "active_self"
    assert blocks[0].primary_text == "hello live"
    assert blocks[0].secondary_text == "translated live"
    assert blocks[0].secondary_enabled is True


@pytest.mark.asyncio
async def test_presenter_updates_active_self_when_secondary_changes_only() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    active_utterance_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="hello live",
            utterance_id=active_utterance_id,
            occupant_key=f"self:{active_utterance_id}",
            created_at=10.0,
        )
    )
    revision_before_secondary = presenter.snapshot().revision

    await presenter.emit(
        adapter.self_active_update(
            text="hello live",
            secondary_text="translated live",
            utterance_id=active_utterance_id,
            occupant_key=f"self:{active_utterance_id}",
            created_at=11.0,
        )
    )

    assert presenter.snapshot().revision == revision_before_secondary + 1
    assert presenter.snapshot().blocks[-1].id == f"self:{active_utterance_id}"
    assert presenter.snapshot().blocks[-1].secondary_text == "translated live"


@pytest.mark.asyncio
async def test_presenter_self_active_clear_removes_live_only_row_but_keeps_finalized_self_row() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    live_only_turn_id = uuid4()
    finalized_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="live only",
            utterance_id=live_only_turn_id,
            occupant_key=f"self:{live_only_turn_id}",
            created_at=10.0,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{live_only_turn_id}"]

    await presenter.emit(adapter.self_active_clear(created_at=10.1))

    assert presenter.snapshot().blocks == []

    await presenter.emit(
        adapter.self_active_update(
            text="live then final",
            utterance_id=finalized_turn_id,
            occupant_key=f"self:{finalized_turn_id}",
            created_at=11.0,
        )
    )
    await presenter.emit(
        SelfTranscriptFinal(
            event_id="final-2",
            seq=4,
            utterance_id=finalized_turn_id,
            channel="self",
            text="live then final",
            source_language="ko",
            target_language="en",
            created_at=11.1,
        )
    )

    revision_before_clear = presenter.snapshot().revision
    await presenter.emit(adapter.self_active_clear(created_at=11.2))

    assert presenter.snapshot().revision == revision_before_clear
    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{finalized_turn_id}"]
    assert presenter.snapshot().blocks[0].block_variant == "finalized"


@pytest.mark.asyncio
async def test_presenter_self_active_clear_retires_live_only_row_from_state() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    live_only_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="live only",
            utterance_id=live_only_turn_id,
            occupant_key=f"self:{live_only_turn_id}",
            created_at=10.0,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{live_only_turn_id}"]

    await presenter.emit(adapter.self_active_clear(created_at=10.1))

    assert presenter.snapshot().blocks == []
    assert bridge.snapshots[-1].blocks == []

    revision_before_stale = presenter.snapshot().revision
    snapshot_count_before_stale = len(bridge.snapshots)
    await presenter.emit(
        SelfActiveUpdate(
            event_id="stale-live-only",
            seq=1,
            utterance_id=live_only_turn_id,
            channel="self",
            created_at=10.0,
            text="late preview",
            occupant_key=f"self:{live_only_turn_id}",
        )
    )

    assert presenter.snapshot().revision == revision_before_stale
    assert presenter.snapshot().blocks == []
    assert len(bridge.snapshots) == snapshot_count_before_stale


@pytest.mark.asyncio
async def test_presenter_self_active_clear_retires_live_only_row_with_preview_secondary_from_state() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    live_only_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="live only",
            secondary_text="preview translation",
            utterance_id=live_only_turn_id,
            occupant_key=f"self:{live_only_turn_id}",
            created_at=10.0,
        )
    )

    assert presenter.snapshot().blocks[0].secondary_text == "preview translation"

    await presenter.emit(adapter.self_active_clear(created_at=10.1))

    assert presenter.snapshot().blocks == []
    assert bridge.snapshots[-1].blocks == []

    revision_before_stale = presenter.snapshot().revision
    snapshot_count_before_stale = len(bridge.snapshots)
    await presenter.emit(
        SelfActiveUpdate(
            event_id="stale-live-only-preview",
            seq=1,
            utterance_id=live_only_turn_id,
            channel="self",
            created_at=10.0,
            text="late preview",
            secondary_text="late preview translation",
            occupant_key=f"self:{live_only_turn_id}",
        )
    )

    assert presenter.snapshot().revision == revision_before_stale
    assert presenter.snapshot().blocks == []
    assert len(bridge.snapshots) == snapshot_count_before_stale


@pytest.mark.asyncio
async def test_presenter_ignores_stale_self_active_update_after_preview_only_retirement_but_allows_newer_final() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="preview only",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=10.0,
        )
    )
    await presenter.emit(adapter.self_active_clear(created_at=10.1))

    assert presenter.snapshot().blocks == []
    assert bridge.snapshots[-1].blocks == []

    revision_before_stale = presenter.snapshot().revision
    snapshot_count_before_stale = len(bridge.snapshots)
    await presenter.emit(
        SelfActiveUpdate(
            event_id="stale-preview",
            seq=1,
            utterance_id=turn_id,
            channel="self",
            created_at=10.0,
            text="late preview",
            occupant_key=f"self:{turn_id}",
        )
    )

    assert presenter.snapshot().revision == revision_before_stale
    assert presenter.snapshot().blocks == []
    assert len(bridge.snapshots) == snapshot_count_before_stale

    await presenter.emit(
        SelfTranscriptFinal(
            event_id="final-after-retire",
            seq=3,
            utterance_id=turn_id,
            channel="self",
            text="final text",
            source_language="ko",
            target_language="en",
            created_at=10.2,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{turn_id}"]
    assert presenter.snapshot().blocks[0].block_variant == "finalized"
    assert presenter.snapshot().blocks[0].primary_text == "final text"
    assert [block.id for block in bridge.snapshots[-1].blocks] == [f"self:{turn_id}"]


@pytest.mark.asyncio
async def test_presenter_replacing_preview_only_live_turn_cleans_up_old_row() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    first_turn_id = uuid4()
    second_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="first live",
            utterance_id=first_turn_id,
            occupant_key=f"self:{first_turn_id}",
            created_at=10.0,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{first_turn_id}"]

    await presenter.emit(
        adapter.self_active_update(
            text="second live",
            utterance_id=second_turn_id,
            occupant_key=f"self:{second_turn_id}",
            created_at=11.0,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{second_turn_id}"]
    assert presenter.snapshot().blocks[0].primary_text == "second live"

    revision_before_stale = presenter.snapshot().revision
    snapshot_count_before_stale = len(bridge.snapshots)
    await presenter.emit(
        SelfActiveUpdate(
            event_id="stale-first-live",
            seq=1,
            utterance_id=first_turn_id,
            channel="self",
            created_at=10.0,
            text="first live",
            occupant_key=f"self:{first_turn_id}",
        )
    )

    assert presenter.snapshot().revision == revision_before_stale
    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{second_turn_id}"]
    assert len(bridge.snapshots) == snapshot_count_before_stale


@pytest.mark.asyncio
async def test_presenter_ignores_stale_self_active_update_after_preview_secondary_replacement_retirement() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    first_turn_id = uuid4()
    second_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="first live",
            secondary_text="first preview translation",
            utterance_id=first_turn_id,
            occupant_key=f"self:{first_turn_id}",
            created_at=10.0,
        )
    )
    await presenter.emit(
        adapter.self_active_update(
            text="second live",
            utterance_id=second_turn_id,
            occupant_key=f"self:{second_turn_id}",
            created_at=11.0,
        )
    )
    await presenter.emit(adapter.self_active_clear(created_at=11.1))

    assert presenter.snapshot().blocks == []
    assert bridge.snapshots[-1].blocks == []

    revision_before_stale = presenter.snapshot().revision
    snapshot_count_before_stale = len(bridge.snapshots)
    await presenter.emit(
        SelfActiveUpdate(
            event_id="stale-first-preview",
            seq=1,
            utterance_id=first_turn_id,
            channel="self",
            created_at=10.0,
            text="first live",
            secondary_text="first preview translation",
            occupant_key=f"self:{first_turn_id}",
        )
    )

    assert presenter.snapshot().revision == revision_before_stale
    assert presenter.snapshot().blocks == []
    assert len(bridge.snapshots) == snapshot_count_before_stale


@pytest.mark.asyncio
async def test_presenter_replacing_preview_only_live_turn_with_preview_secondary_cleans_up_old_row() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    first_turn_id = uuid4()
    second_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="first live",
            secondary_text="first preview translation",
            utterance_id=first_turn_id,
            occupant_key=f"self:{first_turn_id}",
            created_at=10.0,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{first_turn_id}"]
    assert presenter.snapshot().blocks[0].secondary_text == "first preview translation"

    await presenter.emit(
        adapter.self_active_update(
            text="second live",
            secondary_text="second preview translation",
            utterance_id=second_turn_id,
            occupant_key=f"self:{second_turn_id}",
            created_at=11.0,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{second_turn_id}"]
    assert presenter.snapshot().blocks[0].secondary_text == "second preview translation"

    revision_before_stale = presenter.snapshot().revision
    snapshot_count_before_stale = len(bridge.snapshots)
    await presenter.emit(
        SelfActiveUpdate(
            event_id="stale-first-live-preview",
            seq=1,
            utterance_id=first_turn_id,
            channel="self",
            created_at=10.0,
            text="first live",
            secondary_text="first preview translation",
            occupant_key=f"self:{first_turn_id}",
        )
    )

    assert presenter.snapshot().revision == revision_before_stale
    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{second_turn_id}"]
    assert len(bridge.snapshots) == snapshot_count_before_stale


@pytest.mark.asyncio
async def test_presenter_clears_active_self_secondary_text_on_empty_update() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    active_utterance_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="hello live",
            secondary_text="translated live",
            utterance_id=active_utterance_id,
            occupant_key=f"self:{active_utterance_id}",
            created_at=10.0,
        )
    )
    revision_before_clear = presenter.snapshot().revision

    await presenter.emit(
        adapter.self_active_update(
            text="hello live",
            secondary_text="",
            utterance_id=active_utterance_id,
            occupant_key=f"self:{active_utterance_id}",
            created_at=11.0,
        )
    )

    assert presenter.snapshot().revision == revision_before_clear + 1
    assert presenter.snapshot().blocks[-1].id == f"self:{active_utterance_id}"
    assert presenter.snapshot().blocks[-1].secondary_text == ""


@pytest.mark.asyncio
async def test_presenter_reset_scene_clears_terminal_turn_memory() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)

    async def fake_sleep(delay: float) -> None:
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
    )
    adapter = OverlayEventAdapter(clock=clock)
    turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="before reset",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=10.0,
        )
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert presenter.snapshot().blocks == []

    revision_before_retry = presenter.snapshot().revision
    await presenter.emit(
        adapter.self_active_update(
            text="ignored before reset",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=18.1,
        )
    )

    assert presenter.snapshot().revision == revision_before_retry
    assert presenter.snapshot().blocks == []

    presenter.reset_scene()

    await presenter.emit(
        adapter.self_active_update(
            text="after reset",
            utterance_id=turn_id,
            occupant_key=f"self:{turn_id}",
            created_at=19.0,
        )
    )

    assert [block.id for block in presenter.snapshot().blocks] == [f"self:{turn_id}"]
    assert presenter.snapshot().blocks[0].primary_text == "after reset"


@pytest.mark.asyncio
async def test_presenter_assigns_peer_appearance_seq_on_active_source_arrival() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    peer_turn_id = uuid4()

    await presenter.emit(
        adapter.peer_active_update(
            text="peer one",
            utterance_id=peer_turn_id,
            occupant_key=f"peer:{peer_turn_id}",
            created_at=11.0,
        )
    )
    first_visible = presenter.snapshot().blocks[0]
    assert first_visible.block_variant == "active_peer"
    assert first_visible.occupant_key == f"peer:{peer_turn_id}"

    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="피어 하나",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=12.0,
        )
    )
    first_finalized = presenter.snapshot().blocks[0]

    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn_id,
            channel="peer",
            text="피어 하나 수정",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=13.0,
        )
    )

    assert first_finalized.appearance_seq == first_visible.appearance_seq
    assert presenter.snapshot().blocks[0].occupant_key == f"peer:{peer_turn_id}"
    assert presenter.snapshot().blocks[0].appearance_seq == first_visible.appearance_seq


@pytest.mark.asyncio
async def test_presenter_clear_for_runtime_detach_publishes_empty_snapshot_with_higher_revision() -> (
    None
):
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    utterance_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=utterance_id,
                channel="self",
                text="hello",
                is_final=True,
                created_at=11.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    revision_before_clear = presenter.snapshot().revision

    await presenter.clear_for_runtime_detach()

    assert presenter.snapshot().blocks == []
    assert presenter.snapshot().revision == revision_before_clear + 1
    assert bridge.snapshots[-1].blocks == []


@pytest.mark.asyncio
async def test_presenter_updates_secondary_visibility_preferences_without_changing_primary_semantics() -> (
    None
):
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=40.0)
    adapter = OverlayEventAdapter(clock=clock)
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        self_presentation_refresh_burst=False,
    )
    self_utterance_id = uuid4()
    peer_utterance_id = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=self_utterance_id,
                channel="self",
                text="self original",
                is_final=True,
                created_at=40.0,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=self_utterance_id,
            channel="self",
            text="self translation",
            source_language="ko",
            target_language="en",
            applied_context_mode=None,
            created_at=40.1,
        )
    )
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_utterance_id,
                channel="peer",
                text="peer original",
                is_final=True,
                created_at=41.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_utterance_id,
            channel="peer",
            text="peer translation",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            created_at=41.1,
        )
    )

    await presenter.update_display_preferences(
        show_translation=False,
        show_peer_original=False,
    )

    blocks_by_id = {block.id: block for block in presenter.snapshot().blocks}
    self_block = blocks_by_id[f"self:{self_utterance_id}"]
    peer_block = blocks_by_id[f"peer:{peer_utterance_id}"]

    assert self_block.primary_text == "self original"
    assert self_block.secondary_text == "self translation"
    assert self_block.secondary_enabled is False
    assert peer_block.primary_text == "peer translation"
    assert peer_block.secondary_text == "peer original"
    assert peer_block.secondary_enabled is False


@pytest.mark.asyncio
async def test_presenter_snapshot_publish_logs_only_to_detailed_runtime_logging() -> None:
    basic_runtime_logging, basic_stream = _make_runtime_logging_capture()
    detailed_runtime_logging, detailed_stream = _make_runtime_logging_capture()
    detailed_runtime_logging.set_mode(SessionLoggingMode.DETAILED)

    def basic_runtime_log_detailed(message: str, *, level: int = logging.INFO) -> bool:
        return basic_runtime_logging.emit_detailed(message, level=level)

    def detailed_runtime_log_detailed(message: str, *, level: int = logging.INFO) -> bool:
        return detailed_runtime_logging.emit_detailed(message, level=level)

    basic_presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        runtime_log_detailed=basic_runtime_log_detailed,
        self_presentation_refresh_burst=False,
    )
    detailed_presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        runtime_log_detailed=detailed_runtime_log_detailed,
        self_presentation_refresh_burst=False,
    )
    basic_adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    detailed_adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))

    try:
        await basic_presenter.emit(
            basic_adapter.transcript_final(
                Transcript(
                    utterance_id=uuid4(),
                    channel="self",
                    text="hello basic",
                    is_final=True,
                    created_at=11.0,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await detailed_presenter.emit(
            detailed_adapter.transcript_final(
                Transcript(
                    utterance_id=uuid4(),
                    channel="self",
                    text="hello detailed",
                    is_final=True,
                    created_at=11.0,
                ),
                source_language="ko",
                target_language="en",
            )
        )

        assert not any(
            "[OverlayPresenter] Snapshot publish" in message
            for message in _runtime_log_messages(basic_stream)
        )
        detailed_publish_messages = [
            message
            for message in _runtime_log_messages(detailed_stream)
            if "[OverlayPresenter] Snapshot publish" in message
        ]
        assert detailed_publish_messages
        assert any("update_id" in message for message in detailed_publish_messages)
    finally:
        basic_runtime_logging.close()
        detailed_runtime_logging.close()


@pytest.mark.asyncio
async def test_presenter_pair_state_same_text_different_turn_replacement_still_publishes_and_logs() -> (
    None
):
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        runtime_log_detailed=runtime_logging.emit_detailed,
        visible_window_target_blocks=1,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    first_turn_id = uuid4()
    second_turn_id = uuid4()

    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=first_turn_id,
                    channel="self",
                    text="same visible text",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=second_turn_id,
                    channel="self",
                    text="same visible text",
                    is_final=True,
                    created_at=10.1,
                ),
                source_language="ko",
                target_language="en",
            )
        )

        assert len(bridge.snapshots) == 2
        assert [block.id for block in bridge.snapshots[-1].blocks] == [f"self:{second_turn_id}"]
        assert any(
            f"entry=self:{second_turn_id}" in message and "publish_kind=first_visible" in message
            for message in _overlay_presenter_pair_messages(log_stream)
        )
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_presenter_turn_decision_logs_cover_latest_two_turn_decisions_in_detailed_mode() -> (
    None
):
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    clock = FakeClock(_now=10.0)

    async def fake_sleep(delay: float) -> None:
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        runtime_log_detailed=runtime_logging.emit_detailed,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    peer_turn_id = uuid4()
    first_self_turn_id = uuid4()
    second_self_turn_id = uuid4()
    third_self_turn_id = uuid4()
    idle_hidden_turn_id = uuid4()

    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=peer_turn_id,
                    channel="peer",
                    text="peer original",
                    is_final=True,
                    created_at=10.1,
                ),
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=first_self_turn_id,
                    channel="self",
                    text="self one",
                    is_final=True,
                    created_at=10.2,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=first_self_turn_id,
                channel="self",
                text="translated one",
                source_language="ko",
                target_language="en",
                applied_context_mode=None,
                created_at=10.3,
            )
        )
        clock.advance(5.0)
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=second_self_turn_id,
                    channel="self",
                    text="self two",
                    is_final=True,
                    created_at=15.4,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=third_self_turn_id,
                    channel="self",
                    text="self three",
                    is_final=True,
                    created_at=15.5,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=first_self_turn_id,
                channel="self",
                text="late after eviction",
                source_language="ko",
                target_language="en",
                applied_context_mode=None,
                created_at=15.6,
            )
        )
        await presenter.emit(
            adapter.self_active_update(
                text="idle hidden live",
                utterance_id=idle_hidden_turn_id,
                occupant_key=f"self:{idle_hidden_turn_id}",
                created_at=15.7,
            )
        )

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        await presenter.emit(
            adapter.self_active_update(
                text="late after idle hide",
                utterance_id=idle_hidden_turn_id,
                occupant_key=f"self:{idle_hidden_turn_id}",
                created_at=23.8,
            )
        )

        decisions = _overlay_presenter_decisions(log_stream)

        assert "overlay_turn_first_visible" in decisions
        assert "overlay_turn_updated" in decisions
        assert "overlay_turn_evicted_by_newer_turn" in decisions
        assert "overlay_turn_late_update_ignored_after_eviction" in decisions
        assert "overlay_turn_hidden_idle_ttl" in decisions
        assert "overlay_turn_late_update_ignored_after_idle_hide" in decisions
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_presenter_turn_decision_logs_do_not_emit_in_basic_mode() -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        runtime_log_detailed=runtime_logging.emit_detailed,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))

    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=uuid4(),
                    channel="peer",
                    text="peer original",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="en",
                target_language="ko",
            )
        )

        assert _overlay_presenter_decisions(log_stream) == []
    finally:
        runtime_logging.close()


def test_presenter_turn_decision_lazy_skips_formatting_when_detailed_mode_is_off() -> None:
    runtime_logging, _log_stream = _make_runtime_logging_capture()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        runtime_log_detailed=runtime_logging.emit_detailed,
    )

    try:
        assert (
            presenter._emit_turn_decision(
                "lazy_guard",
                key=("self", uuid4()),
                extras={"expensive": _ExplodingValue()},
            )
            is False
        )
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_presenter_pair_state_logs_publish_kind_and_sources_only_in_detailed_mode() -> None:
    basic_runtime_logging, basic_stream = _make_runtime_logging_capture()
    detailed_runtime_logging, detailed_stream = _make_runtime_logging_capture()
    detailed_runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    origin_wall_clock_ms = int(time.time() * 1000) - 50
    utterance_id = uuid4()

    basic_presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        runtime_log_detailed=basic_runtime_logging.emit_detailed,
    )
    detailed_presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        runtime_log_detailed=detailed_runtime_logging.emit_detailed,
    )
    basic_adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    detailed_adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))

    try:
        await basic_presenter.emit(
            basic_adapter.peer_active_update(
                text="peer original",
                utterance_id=utterance_id,
                occupant_key=f"peer:{utterance_id}",
                created_at=10.0,
            )
        )
        await detailed_presenter.emit(
            detailed_adapter.peer_active_update(
                text="peer original",
                utterance_id=utterance_id,
                occupant_key=f"peer:{utterance_id}",
                created_at=10.0,
            )
        )
        await basic_presenter.emit(
            basic_adapter.translation_final(
                utterance_id=utterance_id,
                channel="peer",
                text="peer translation",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.1,
                update_id="upd-peer-1",
                origin_wall_clock_ms=origin_wall_clock_ms,
                source_text_hash="hash-peer-1",
                source_text_len=13,
                logical_turn_key="peer-turn-1",
            )
        )
        await detailed_presenter.emit(
            detailed_adapter.translation_final(
                utterance_id=utterance_id,
                channel="peer",
                text="peer translation",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.1,
                update_id="upd-peer-1",
                origin_wall_clock_ms=origin_wall_clock_ms,
                source_text_hash="hash-peer-1",
                source_text_len=13,
                logical_turn_key="peer-turn-1",
            )
        )
        await detailed_presenter.emit(
            detailed_adapter.translation_final(
                utterance_id=utterance_id,
                channel="peer",
                text="peer translation v2",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.2,
                update_id="upd-peer-2",
                origin_wall_clock_ms=origin_wall_clock_ms,
                source_text_hash="hash-peer-2",
                source_text_len=16,
                logical_turn_key="peer-turn-1",
            )
        )

        assert _overlay_presenter_pair_messages(basic_stream) == []
        pair_messages = _overlay_presenter_pair_messages(detailed_stream)

        assert any(
            "publish_kind=first_visible" in message
            and "block_variant=active_peer" in message
            and "update_id=None" in message
            and "original_seq=1" in message
            and "translation_seq=None" in message
            and "rendered_pair_state=source_only" in message
            and "rendered_primary_source=blank" in message
            and "rendered_secondary_source=source" in message
            for message in pair_messages
        )
        assert any(
            "publish_kind=visible_update" in message
            and "update_id=upd-peer-1" in message
            and f"origin_wall_clock_ms={origin_wall_clock_ms}" in message
            and "source_text_hash=hash-peer-1" in message
            and "source_text_len=13" in message
            and "original_seq=1" in message
            and "translation_seq=2" in message
            and "rendered_pair_state=translation_with_original" in message
            and "rendered_primary_source=translation" in message
            and "rendered_secondary_source=source" in message
            and "elapsed_ms=" in message
            for message in pair_messages
        )
        assert any(
            "publish_kind=visible_update" in message
            and "update_id=upd-peer-2" in message
            and "source_text_hash=hash-peer-2" in message
            and "source_text_len=16" in message
            and "original_seq=1" in message
            and "translation_seq=3" in message
            and "rendered_pair_state=translation_with_original" in message
            and "rendered_primary_source=translation" in message
            and "rendered_secondary_source=source" in message
            for message in pair_messages
        )
    finally:
        basic_runtime_logging.close()
        detailed_runtime_logging.close()


@pytest.mark.asyncio
async def test_presenter_pair_state_logs_hidden_peer_original_as_blank() -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    utterance_id = uuid4()
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        runtime_log_detailed=runtime_logging.emit_detailed,
        show_peer_original=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))

    try:
        await presenter.emit(
            adapter.peer_active_update(
                text="peer original",
                utterance_id=utterance_id,
                occupant_key=f"peer:{utterance_id}",
                created_at=10.0,
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=utterance_id,
                channel="peer",
                text="peer translation",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.1,
                update_id="upd-peer-hidden-original",
            )
        )

        assert any(
            "publish_kind=first_visible" in message
            and "update_id=upd-peer-hidden-original" in message
            and "rendered_pair_state=translation_only" in message
            and "rendered_primary_source=translation" in message
            and "rendered_secondary_source=blank" in message
            for message in _overlay_presenter_pair_messages(log_stream)
        )
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_presenter_disposition_logs_skip_states_in_detailed_mode() -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=10.0),
        runtime_log_detailed=runtime_logging.emit_detailed,
        show_peer_original=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    self_turn_id = uuid4()
    peer_turn_id = uuid4()

    try:
        await presenter.emit(
            adapter.self_active_update(
                text="self preview",
                secondary_text="preview translation",
                utterance_id=self_turn_id,
                occupant_key=f"self:{self_turn_id}",
                created_at=10.0,
                update_id="live-preview-1",
            )
        )
        await presenter.emit(
            adapter.self_active_update(
                text="self preview",
                secondary_text="preview translation",
                utterance_id=self_turn_id,
                occupant_key=f"self:{self_turn_id}",
                created_at=10.1,
                update_id="live-preview-1",
            )
        )
        await presenter.emit(
            SelfActiveUpdate(
                event_id="evt-stale-preview",
                seq=1,
                utterance_id=self_turn_id,
                channel="self",
                created_at=9.9,
                text="stale preview",
                secondary_text="",
                occupant_key=f"self:{self_turn_id}",
            )
        )
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=peer_turn_id,
                    channel="peer",
                    text="peer original",
                    is_final=True,
                    created_at=10.2,
                ),
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=peer_turn_id,
                channel="peer",
                text="peer translation",
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
                created_at=10.3,
                update_id="peer-visible-1",
            )
        )
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=peer_turn_id,
                    channel="peer",
                    text="peer original changed",
                    is_final=True,
                    created_at=10.4,
                ),
                source_language="en",
                target_language="ko",
            )
        )

        disposition_messages = _overlay_presenter_disposition_messages(log_stream)

        assert any("disposition=coalesced" in message for message in disposition_messages)
        assert any("disposition=superseded" in message for message in disposition_messages)
        assert any(
            "disposition=rendered_signature_unchanged" in message
            for message in disposition_messages
        )
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_presenter_disposition_logs_terminal_states_in_detailed_mode() -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    clock = FakeClock(_now=10.0)

    async def fake_sleep(delay: float) -> None:
        clock.advance(delay)
        await asyncio.sleep(0)

    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        sleep=fake_sleep,
        runtime_log_detailed=runtime_logging.emit_detailed,
        visible_window_target_blocks=1,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    first_turn_id = uuid4()
    second_turn_id = uuid4()
    ttl_turn_id = uuid4()

    try:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=first_turn_id,
                    channel="self",
                    text="self one",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=second_turn_id,
                    channel="self",
                    text="self two",
                    is_final=True,
                    created_at=10.1,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=ttl_turn_id,
                    channel="self",
                    text="ttl turn",
                    is_final=True,
                    created_at=10.2,
                ),
                source_language="ko",
                target_language="en",
            )
        )
        await presenter.emit(
            adapter.utterance_closed(
                utterance_id=ttl_turn_id,
                channel="self",
                created_at=10.3,
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        disposition_messages = _overlay_presenter_disposition_messages(log_stream)

        assert any("disposition=evicted" in message for message in disposition_messages)
        assert any("disposition=hidden_idle_ttl" in message for message in disposition_messages)
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_presenter_preview_translation_visibility_is_counted_before_final_translation() -> (
    None
):
    diagnostics = RecordingPresenterRemovalDiagnostics()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        diagnostics=diagnostics,
    )
    adapter = OverlayEventAdapter(clock=clock)
    utterance_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="self preview",
            secondary_text="preview translation",
            utterance_id=utterance_id,
            occupant_key=f"self:{utterance_id}",
            created_at=10.0,
            update_id="preview-update",
        )
    )
    clock.advance(1.5)

    await presenter.emit(adapter.self_active_clear(created_at=11.5))

    assert diagnostics.removal_events
    removal_event = diagnostics.removal_events[-1]
    assert removal_event["reason"] == "live_self_cleared"
    assert removal_event["ever_visible_with_translation"] is True
    assert float(removal_event["translated_lifetime_ms"]) >= 1500.0


@pytest.mark.asyncio
async def test_presenter_preview_translation_visibility_stays_false_when_translation_hidden() -> (
    None
):
    diagnostics = RecordingPresenterRemovalDiagnostics()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        diagnostics=diagnostics,
        show_translation=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    utterance_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="self preview",
            secondary_text="hidden preview translation",
            utterance_id=utterance_id,
            occupant_key=f"self:{utterance_id}",
            created_at=10.0,
            update_id="hidden-preview-update",
        )
    )
    clock.advance(1.5)

    await presenter.emit(adapter.self_active_clear(created_at=11.5))

    assert diagnostics.removal_events
    removal_event = diagnostics.removal_events[-1]
    assert removal_event["reason"] == "live_self_cleared"
    assert removal_event["translation_visible_since"] is None
    assert removal_event["translation_observed_visible_since"] is None
    assert removal_event["ever_visible_with_translation"] is False
    assert float(removal_event["translated_lifetime_ms"]) == 0.0


@pytest.mark.asyncio
async def test_presenter_preview_translation_visibility_hidden_preview_secondary_promoted_to_final_does_not_count_as_visible() -> (
    None
):
    diagnostics = RecordingPresenterRemovalDiagnostics()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        clock=clock,
        diagnostics=diagnostics,
        show_translation=False,
        visible_window_target_blocks=1,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=clock)
    first_turn_id = uuid4()
    second_turn_id = uuid4()

    await presenter.emit(
        adapter.self_active_update(
            text="self preview",
            secondary_text="hidden preview translation",
            utterance_id=first_turn_id,
            occupant_key=f"self:{first_turn_id}",
            created_at=10.0,
            update_id="hidden-preview-promote",
        )
    )
    clock.advance(1.5)
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=first_turn_id,
                channel="self",
                text="self preview",
                is_final=True,
                created_at=11.5,
            ),
            source_language="ko",
            target_language="en",
        )
    )
    clock.advance(1.0)
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=second_turn_id,
                channel="self",
                text="new self turn",
                is_final=True,
                created_at=12.5,
            ),
            source_language="ko",
            target_language="en",
        )
    )

    assert diagnostics.removal_events
    removal_event = diagnostics.removal_events[-1]
    assert removal_event["reason"] == "evicted_by_newer_turn"
    assert removal_event["translation_visible_since"] is None
    assert removal_event["translation_observed_visible_since"] is None
    assert removal_event["ever_visible_with_translation"] is False
    assert float(removal_event["translated_lifetime_ms"]) == 0.0


@pytest.mark.asyncio
async def test_presenter_native_fresh_render_peer_event_matrix_uses_exact_visible_key() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
        native_retry_trigger_emission=True,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    visible = uuid4()
    source_only = uuid4()

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=visible,
                channel="peer",
                text="source",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_stream_update(
            utterance_id=visible,
            channel="peer",
            text="번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            update_id="peer-stream",
        )
    )
    assert presenter.snapshot().native_fresh_render_generations.peer == 1
    stream_episode = presenter.snapshot().native_quiet_tail_episodes.peer
    assert stream_episode is not None and stream_episode.phase == "stream"

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=source_only,
                channel="peer",
                text="hidden source",
                is_final=True,
                created_at=10.1,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.peer_active_update(
            text="still source only",
            utterance_id=source_only,
            occupant_key=f"peer:{source_only}",
        )
    )
    assert presenter.snapshot().native_fresh_render_generations.peer == 1

    await presenter.emit(
        adapter.peer_active_update(
            text="updated source",
            utterance_id=visible,
            occupant_key=f"peer:{visible}",
        )
    )
    assert presenter.snapshot().native_fresh_render_generations.peer == 2
    assert presenter.snapshot().native_quiet_tail_episodes.peer == stream_episode
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=visible,
                channel="peer",
                text="final source",
                is_final=True,
                created_at=10.2,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    assert presenter.snapshot().native_fresh_render_generations.peer == 3
    final_episode = presenter.snapshot().native_quiet_tail_episodes.peer
    assert final_episode is not None and final_episode.phase == "final"
    assert final_episode.generation != stream_episode.generation
    await presenter.emit(
        adapter.translation_final(
            utterance_id=visible,
            channel="peer",
            text="최종 번역",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
            update_id="peer-final",
        )
    )
    assert presenter.snapshot().native_fresh_render_generations.peer == 4
    assert presenter.snapshot().native_quiet_tail_episodes.peer == final_episode


@pytest.mark.asyncio
async def test_presenter_native_fresh_render_generations_preserve_channels_and_exact_keys() -> None:
    bridge = RecordingPresentationBridge()
    clock = FakeClock(_now=10.0)
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=clock,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
        native_retry_trigger_emission=True,
    )
    adapter = OverlayEventAdapter(clock=clock)
    self_turn = uuid4()
    peer_turn = uuid4()

    assert presenter.snapshot().native_fresh_render_generations is None
    await presenter.update_calibration(OverlayCalibration(distance=1.15))
    await presenter.update_display_preferences(
        show_translation=False,
        show_peer_original=True,
    )
    assert presenter.snapshot().native_fresh_render_generations is None
    await presenter.update_display_preferences(
        show_translation=True,
        show_peer_original=True,
    )

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=self_turn,
                channel="self",
                text="same visible text",
                is_final=True,
                created_at=10.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    self_generation = presenter.snapshot().native_fresh_render_generations.self
    self_episode = presenter.snapshot().native_quiet_tail_episodes.self
    assert self_episode is not None and self_episode.phase == "final"
    await presenter.emit(
        adapter.self_active_update(
            text="live update",
            utterance_id=self_turn,
            occupant_key=f"self:{self_turn}",
        )
    )
    await presenter.emit(
        adapter.translation_stream_update(
            utterance_id=self_turn,
            channel="self",
            text="stream update",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
        )
    )
    assert presenter.snapshot().native_fresh_render_generations.self == self_generation
    assert presenter.snapshot().native_quiet_tail_episodes.self == self_episode
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn,
                channel="peer",
                text="peer source",
                is_final=True,
                created_at=10.1,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    assert presenter.snapshot().native_fresh_render_generations.self == self_generation
    assert presenter.snapshot().native_fresh_render_generations.peer is None

    peer_translation = adapter.translation_final(
        utterance_id=peer_turn,
        channel="peer",
        text="peer translation",
        source_language="en",
        target_language="ko",
        applied_context_mode=None,
        created_at=10.2,
        update_id="stable-peer-update",
    )
    await presenter.emit(peer_translation)
    generations = presenter.snapshot().native_fresh_render_generations
    assert (generations.self, generations.peer) == (self_generation, 1)
    peer_episode = presenter.snapshot().native_quiet_tail_episodes.peer
    assert peer_episode is not None and peer_episode.phase == "final"

    await presenter.emit(peer_translation)
    generations = presenter.snapshot().native_fresh_render_generations
    assert (generations.self, generations.peer) == (self_generation, 1)
    assert presenter.snapshot().native_quiet_tail_episodes.peer == peer_episode
    await presenter.update_display_preferences(
        show_translation=False,
        show_peer_original=False,
    )
    assert presenter.snapshot().native_fresh_render_generations == generations
    await presenter.update_display_preferences(
        show_translation=True,
        show_peer_original=True,
    )
    assert presenter.snapshot().native_fresh_render_generations == generations
    await presenter.update_calibration(OverlayCalibration(distance=1.2))
    assert presenter.snapshot().native_fresh_render_generations == generations

    presenter._native_fresh_render_generations = NativeFreshRenderGenerations(
        self=generations.self,
        peer=U64_MAX,
    )
    await presenter.emit(peer_translation)
    rolled = presenter.snapshot().native_fresh_render_generations
    assert (rolled.self, rolled.peer) == (self_generation, 1)
    presenter.reset_scene()
    assert presenter.snapshot().native_fresh_render_generations is None


@pytest.mark.asyncio
async def test_presenter_episode_generation_is_monotonic_after_target_loss() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
        native_retry_trigger_emission=True,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    turn = uuid4()
    event = adapter.transcript_final(
        Transcript(
            utterance_id=turn,
            channel="self",
            text="first",
            is_final=True,
            created_at=10.0,
        ),
        source_language="en",
        target_language="ko",
    )
    await presenter.emit(event)
    first = presenter.snapshot().native_quiet_tail_episodes.self
    assert first is not None
    presenter.reset_scene()
    assert presenter.snapshot().native_quiet_tail_episodes is None
    await presenter.emit(event)
    second = presenter.snapshot().native_quiet_tail_episodes.self
    assert second is not None
    assert second.generation > first.generation


@pytest.mark.asyncio
async def test_presenter_native_fresh_render_same_text_new_turn_and_terminal_exclusions() -> None:
    presenter = OverlayPresenter(
        calibration=OverlayCalibration(),
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
        native_retry_trigger_emission=True,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    first_self = uuid4()
    second_self = uuid4()
    peer_turn = uuid4()

    for turn_id in (first_self, second_self):
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=turn_id,
                    channel="self",
                    text="identical text on a distinct turn",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="en",
                target_language="ko",
            )
        )
    assert presenter.snapshot().native_fresh_render_generations.self == 2

    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=peer_turn,
                channel="peer",
                text="peer source",
                is_final=True,
                created_at=10.1,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    await presenter.emit(
        adapter.translation_final(
            utterance_id=peer_turn,
            channel="peer",
            text="peer translation",
            source_language="en",
            target_language="ko",
            applied_context_mode=None,
        )
    )
    generations = presenter.snapshot().native_fresh_render_generations
    assert (generations.self, generations.peer) == (2, 1)

    await presenter.emit(adapter.self_active_clear())
    assert presenter.snapshot().native_fresh_render_generations == generations
    await presenter.emit(adapter.utterance_closed(utterance_id=second_self, channel="self"))
    assert presenter.snapshot().native_fresh_render_generations == generations
    await presenter.emit(adapter.utterance_closed(utterance_id=peer_turn, channel="peer"))
    assert presenter.snapshot().native_fresh_render_generations == generations


@pytest.mark.asyncio
async def test_native_ownership_transition_serializes_concurrent_target_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presenter = OverlayPresenter(calibration=OverlayCalibration())
    adapter = OverlayEventAdapter(clock=FakeClock(_now=10.0))
    first = uuid4()
    second = uuid4()

    async def publish_peer(turn_id, text: str) -> None:
        await presenter.emit(
            adapter.transcript_final(
                Transcript(
                    utterance_id=turn_id,
                    channel="peer",
                    text=f"source {text}",
                    is_final=True,
                    created_at=10.0,
                ),
                source_language="en",
                target_language="ko",
            )
        )
        await presenter.emit(
            adapter.translation_final(
                utterance_id=turn_id,
                channel="peer",
                text=text,
                source_language="en",
                target_language="ko",
                applied_context_mode=None,
            )
        )

    await publish_peer(first, "first")
    transition_blocked = asyncio.Event()
    release_transition = asyncio.Event()
    original = OverlayPresenter.update_peer_presentation_refresh_burst

    async def blocked_update(self, enabled: bool) -> None:
        if self is presenter and not enabled:
            transition_blocked.set()
            await release_transition.wait()
        await original(self, enabled)

    monkeypatch.setattr(OverlayPresenter, "update_peer_presentation_refresh_burst", blocked_update)
    transition = asyncio.create_task(presenter.update_native_retry_ownership(True))
    await transition_blocked.wait()
    replacement = asyncio.create_task(publish_peer(second, "second"))
    await asyncio.sleep(0)
    assert not replacement.done()
    release_transition.set()
    await transition
    await replacement

    snapshot = presenter.snapshot()
    assert snapshot.native_fresh_render_targets.peer == f"peer:{second}"
    assert snapshot.native_fresh_render_generations.peer is not None
    assert presenter._peer_presentation_refresh_burst_task is None
    presenter.peer_presentation_refresh_burst = True
    presenter.self_presentation_refresh_burst = True
    await presenter.update_native_retry_ownership(True)
    assert presenter.native_retry_trigger_emission is True
    assert presenter.peer_presentation_refresh_burst is False
    assert presenter.self_presentation_refresh_burst is False
    await presenter.update_native_retry_ownership(False)
    assert presenter.snapshot().native_fresh_render_generations is None
    assert presenter._presentation_state.peer_presentation_refresh_target_key == ("peer", second)
    assert presenter._peer_presentation_refresh_burst_task is not None
    await presenter.close()
    await presenter.update_native_retry_ownership(False)
    assert presenter._peer_presentation_refresh_burst_task is None
    assert presenter._self_presentation_refresh_burst_task is None


@pytest.mark.asyncio
async def test_discard_epoch_retry_intent_drops_old_epoch_generations_and_keeps_captions() -> None:
    bridge = RecordingPresentationBridge()
    presenter = OverlayPresenter(
        bridge=bridge,
        calibration=OverlayCalibration(),
        clock=FakeClock(_now=1.0),
        native_retry_trigger_emission=True,
        peer_presentation_refresh_burst=False,
        self_presentation_refresh_burst=False,
    )
    adapter = OverlayEventAdapter(clock=FakeClock(_now=1.0))
    turn_id = uuid4()
    await presenter.emit(
        adapter.transcript_final(
            Transcript(
                utterance_id=turn_id,
                channel="self",
                text="keep this caption",
                is_final=True,
                created_at=1.0,
            ),
            source_language="en",
            target_language="ko",
        )
    )
    presenter._native_fresh_render_generations = NativeFreshRenderGenerations(self=4)
    await presenter._publish_if_changed(force_protocol_publish=True)
    assert presenter.snapshot().native_fresh_render_generations is not None
    assert any(block.primary_text == "keep this caption" for block in presenter.snapshot().blocks)

    await presenter.discard_epoch_retry_intent()

    snapshot = presenter.snapshot()
    assert snapshot.native_fresh_render_generations is None
    assert snapshot.native_fresh_render_targets is None
    assert any(block.primary_text == "keep this caption" for block in snapshot.blocks)
    assert presenter.native_retry_trigger_emission is False
    assert presenter.peer_presentation_refresh_burst is False
    assert presenter.self_presentation_refresh_burst is False
