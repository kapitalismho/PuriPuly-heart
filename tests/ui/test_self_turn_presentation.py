from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks, SessionRuntimeLoggingService
from puripuly_heart.domain.events import (
    SelfTurnPresentation,
    SelfTurnTargetPresentation,
    UIEvent,
    UIEventType,
)
from puripuly_heart.domain.models import Transcript, Translation
from puripuly_heart.ui.event_projection import EventProjectionService
from puripuly_heart.ui.i18n import get_locale, set_locale, t
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from puripuly_heart.ui.views.dashboard import DashboardView
from puripuly_heart.ui.views.logs import ConversationViewModel, LogsView
from tests.helpers.self_turn_presentation import (
    ControlledSourceOverlay,
    OwnerPresentation,
    eventually,
    independent_recognition,
)


def draft(*, order=1, generation=1, targets=("en",)):
    return SelfTurnPresentation(
        utterance_id=uuid4(),
        turn_generation=generation,
        turn_order=order,
        revision=1,
        source_revision=1,
        source_text="最初",
        source_language="ja",
        source_closed=False,
        status="active",
        targets=tuple(SelfTurnTargetPresentation(language, 1, "pending") for language in targets),
        created_at=0.0,
    )


def ready(snapshot, text="first", *, target_index=0):
    targets = list(snapshot.targets)
    target = targets[target_index]
    targets[target_index] = replace(
        target,
        source_revision=snapshot.source_revision,
        status="ready",
        translation=Translation(
            snapshot.utterance_id,
            text,
            source_text=snapshot.source_text,
            source_language=snapshot.source_language,
            target_language=target.target_language,
        ),
    )
    return replace(snapshot, revision=snapshot.revision + 1, targets=tuple(targets))


@pytest.fixture
def presentation():
    locale = get_locale()
    set_locale("en")
    dashboard = DashboardView()
    logs = LogsView()
    root = logging.Logger(f"self-turn-root-{uuid4()}")
    session = logging.Logger(f"self-turn-session-{uuid4()}")
    session.propagate = False
    stream = io.StringIO()
    runtime = SessionRuntimeLoggingService(
        root_logger=root,
        session_logger=session,
        sinks=RuntimeLoggingSinks(
            stream_handler=logging.StreamHandler(stream),
            file_handler=logging.NullHandler(),
            log_file=Path("synthetic-self-turn.log"),
        ),
    )
    host = SimpleNamespace(view_dashboard=dashboard, view_logs=logs)
    adapter = FletUiPresentationAdapter(host)
    adapter.attach_runtime_log_sink(runtime)
    bridge = adapter.create_ui_event_bridge(
        event_queue=asyncio.Queue(maxsize=4), runtime_logging=runtime
    )
    logs._on_conversation_button_click(None)
    yield SimpleNamespace(dashboard=dashboard, logs=logs, runtime=runtime, bridge=bridge)
    bridge.close()
    runtime.close()
    logs.will_unmount()
    set_locale(locale)


async def publish(presentation, snapshot, *, authority=None):
    await presentation.bridge._handle_event(
        UIEvent(
            UIEventType.SELF_TURN_UPDATED,
            utterance_id=snapshot.utterance_id,
            payload=snapshot,
            source="Mic",
            delivery_authority=authority,
        )
    )


def accept(presentation, snapshot):
    target = snapshot.targets[0]
    presentation.runtime.record_conversation_observation(
        utterance_id=str(snapshot.utterance_id),
        speaker_channel="self",
        transcript_text=snapshot.source_text,
        translation_text=target.translation.text if target.translation else None,
        source_language=snapshot.source_language,
        target_language=target.target_language,
        metadata={"source": "Mic", "target_index": 0, "disposition": "translated"},
    )


@pytest.mark.asyncio
async def test_one_visible_turn_progresses_without_final_translation_flicker(
    presentation, monkeypatch
):
    card = presentation.dashboard.display_card
    observed = []
    sync = card._sync_display

    def capture():
        result = sync()
        observed.append(card._display_text.value)
        return result

    monkeypatch.setattr(card, "_sync_display", capture)
    source = draft()
    await publish(presentation, source)
    assert card._display_text.value == "最初"
    first = ready(source)
    await publish(presentation, first)
    assert card._display_text.value == "first"
    row = presentation.logs.conversation_records[0]
    timestamp = row.timestamp_label
    assert not row.accepted
    assert len(presentation.runtime._conversation_record_keys) == 0
    extended = replace(first, revision=3, source_revision=2, source_text="最初の続き")
    await publish(presentation, extended)
    assert card._display_text.value == "最初の続き"
    assert "earlier source r1 (current source r2)" in presentation.logs._log_text.value
    second = ready(extended, "extended translation")
    await publish(presentation, second)
    assert card._display_text.value == "extended translation"
    sealed = replace(second, revision=5, source_closed=True)
    await publish(presentation, sealed)
    assert card._display_text.value == "extended translation"
    assert card._turn_state_text.value == t("display.turn.translating")
    final = replace(sealed, revision=6, status="complete")
    accept(presentation, final)
    accept(presentation, final)
    before = len(observed)
    await publish(presentation, final)
    assert observed[before:] == ["extended translation"]
    assert card._turn_state_text.value == t("display.turn.complete")
    assert len(presentation.logs.conversation_records) == 1
    record = presentation.logs.conversation_records[0]
    assert record.timestamp_label == timestamp
    assert record.accepted and record.turn_status == "complete"
    assert presentation.logs._log_text.value.count("extended translation") == 1
    assert len(presentation.runtime._conversation_record_keys) == 1


@pytest.mark.asyncio
async def test_old_final_only_updates_own_row_and_legacy_events_cannot_regress_managed_turn(
    presentation,
):
    old = ready(draft())
    await publish(presentation, old)
    successor = ready(draft(order=2), "successor")
    await publish(presentation, successor)
    final = replace(old, revision=3, source_closed=True, status="complete")
    accept(presentation, final)
    await publish(presentation, final)
    assert presentation.dashboard.display_card._display_text.value == "successor"
    assert len(presentation.logs.conversation_records) == 2
    assert presentation.logs.conversation_records[0].turn_status == "complete"
    await presentation.bridge._handle_event(
        UIEvent(
            UIEventType.TRANSCRIPT_FINAL, payload=Transcript(old.utterance_id, "obsolete", True)
        )
    )
    await presentation.bridge._handle_event(
        UIEvent(UIEventType.TRANSLATION_DONE, payload=old.targets[0].translation)
    )
    assert presentation.dashboard.display_card._display_text.value == "successor"
    assert len(presentation.logs.conversation_records) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["cancelled", "failed", "expired"])
@pytest.mark.parametrize("retain_ready_target", [False, True])
async def test_terminal_disposition_fences_revival_and_preserves_valid_partial_success(
    presentation, terminal, retain_ready_target
):
    preview = ready(draft())
    await publish(presentation, preview)
    assert t("display.turn.complete") not in presentation.logs._log_text.value
    ended = replace(
        preview,
        revision=3,
        status=terminal,
        targets=(
            (
                preview.targets[0]
                if retain_ready_target
                else replace(preview.targets[0], status=terminal)
            ),
        ),
    )
    await publish(presentation, ended)
    expected = (
        preview.targets[0].translation.text
        if terminal == "failed" and retain_ready_target
        else ended.source_text
    )
    assert presentation.dashboard.display_card._display_text.value == expected
    assert presentation.dashboard.display_card._turn_state_text.value == t(
        f"display.turn.{terminal}"
    )
    assert presentation.logs.conversation_records[0].turn_status == terminal
    assert not presentation.logs.conversation_records[0].accepted
    row_status = "complete" if terminal == "failed" and retain_ready_target else terminal
    assert t(f"display.turn.{row_status}") in presentation.logs._log_text.value.splitlines()[0]
    if row_status != "complete":
        assert t("display.turn.complete") not in presentation.logs._log_text.value
    await publish(presentation, replace(preview, revision=4))
    await publish(
        presentation,
        replace(preview, revision=5, source_closed=True, status="complete"),
    )
    assert presentation.logs.conversation_records[0].turn_status == terminal
    assert presentation.dashboard.display_card._display_text.value == expected
    assert len(presentation.runtime._conversation_record_keys) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted_before_snapshots", [False, True])
async def test_real_dual_target_logs_correlate_child_records_without_duplicate_rows(
    presentation, accepted_before_snapshots
):
    async with OwnerPresentation(
        presentation.dashboard, presentation.logs, presentation.runtime, targets=("en", "ja")
    ) as owner:
        primary = owner.provider.holds[("dual source", "en")] = asyncio.Event()
        secondary = owner.provider.holds[("dual source", "ja")] = asyncio.Event()
        if accepted_before_snapshots:
            owner.delivery_gate.clear()
        source_id = await owner.source("dual source", close=False)
        await eventually(lambda: len(owner.provider.calls) == 1)
        if not accepted_before_snapshots:
            await eventually(lambda: len(presentation.logs.conversation_records) == 2)
            assert not presentation.runtime._conversation_record_keys
            rows = presentation.logs._log_text.value.split("\n\n")
            assert "Mic → English" in rows[0].splitlines()[0]
            assert "Mic → Japanese" in rows[1].splitlines()[0]
            assert all(t("logs.conversation.target.pending") in row for row in rows)
        await owner.close_source(source_id)
        await eventually(
            lambda: {call["target_language"] for call in owner.provider.calls} == {"en", "ja"}
        )
        secondary.set()
        if not accepted_before_snapshots:
            await eventually(
                lambda: "ja:dual source" in presentation.dashboard.display_card._display_text.value
            )
            assert "en:dual source" not in presentation.dashboard.display_card._display_text.value
            rows = presentation.logs._log_text.value.split("\n\n")
            assert "Mic → English" in rows[0].splitlines()[0]
            assert t("logs.conversation.target.pending") in rows[0]
            assert "Mic → Japanese" in rows[1].splitlines()[0]
            assert "ja:dual source" in rows[1]
            assert t("display.turn.complete") not in rows[1].splitlines()[0]
        primary.set()
        await owner.idle()
        if accepted_before_snapshots:
            assert len(presentation.runtime._conversation_record_keys) == 3
            owner.delivery_gate.set()
        await eventually(
            lambda: len(presentation.logs.conversation_records) == 2
            and all(
                record.turn_status == "complete"
                for record in presentation.logs.conversation_records
            )
        )
        final = next(
            snapshot for snapshot in owner.snapshots.values() if snapshot.status == "complete"
        )
        secondary_id = final.targets[1].translation.utterance_id
        assert secondary_id != final.utterance_id
        assert any(
            key[1] == str(secondary_id) and key[3] == 1 and key[4] == "translation"
            for key in presentation.runtime._conversation_record_keys
        )
        assert len(presentation.runtime._conversation_record_keys) == 3
        assert {record.utterance_id for record in presentation.logs.conversation_records} == {
            str(final.utterance_id)
        }
        assert all(record.accepted for record in presentation.logs.conversation_records)
        assert [record.target_language for record in presentation.logs.conversation_records] == [
            "en",
            "ja",
        ]
        assert presentation.logs._log_text.value.count("ja:dual source") == 1
        assert presentation.logs._log_text.value.count("en:dual source") == 1
        headers = [row.splitlines()[0] for row in presentation.logs._log_text.value.split("\n\n")]
        assert "Mic → English" in headers[0]
        assert "Mic → Japanese" in headers[1]
        text = presentation.dashboard.display_card._display_text.value
        assert text.index("en:dual source") < text.index("ja:dual source")
        assert presentation.dashboard.display_card._display_text.size <= 28


class HistoryOnlyAuthority:
    def claim_delivery(self):
        return True

    def claim_presentation(self):
        return False

    def is_current(self):
        return False


@pytest.mark.asyncio
async def test_delivery_authority_retains_history_without_stealing_current_slot(presentation):
    preview = ready(draft())
    await publish(presentation, preview)
    blocked = ready(draft(order=2), "history only")
    await publish(presentation, blocked, authority=HistoryOnlyAuthority())
    assert presentation.dashboard.display_card._display_text.value == "first"
    assert len(presentation.logs.conversation_records) == 2
    assert "history only" in presentation.logs._log_text.value


@pytest.mark.asyncio
async def test_queue_consumer_applies_newest_revision_after_blocked_delivery(presentation):
    entered = asyncio.Event()
    release = asyncio.Event()
    original = presentation.bridge._handle_event

    async def blocked(event):
        entered.set()
        await release.wait()
        await original(event)

    presentation.bridge._handle_event = blocked
    task = asyncio.create_task(presentation.bridge.run())
    try:
        source = draft()
        queue = presentation.bridge.event_queue
        await queue.put(UIEvent(UIEventType.SELF_TURN_UPDATED, payload=source))
        await entered.wait()
        preview = ready(source)
        final = replace(preview, revision=3, source_closed=True, status="complete")
        await queue.put(UIEvent(UIEventType.SELF_TURN_UPDATED, payload=final))
        await queue.put(UIEvent(UIEventType.SELF_TURN_UPDATED, payload=preview))
        release.set()
        await queue.join()
        assert len(presentation.logs.conversation_records) == 1
        assert presentation.logs.conversation_records[0].turn_status == "complete"
        assert presentation.dashboard.display_card._display_text.value == "first"
    finally:
        presentation.bridge.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_generation_revision_fences_and_retention_do_not_reinsert_evicted_turn(presentation):
    presentation.bridge.projection_service = EventProjectionService(max_turns=2)
    presentation.logs._conversation_model = ConversationViewModel(max_entries=2, cleanup_batch=1)
    old = ready(draft())
    await publish(presentation, old)
    for changed in (
        replace(old, revision=1),
        replace(old, revision=3, source_revision=0),
        replace(old, revision=3, turn_generation=2),
    ):
        await publish(presentation, changed)
    assert len(presentation.logs.conversation_records) == 1
    for order in (2, 3, 4):
        await publish(presentation, ready(draft(order=order), f"turn {order}"))
    assert len(presentation.logs.conversation_records) == 3
    assert all(
        record.utterance_id != str(old.utterance_id)
        for record in presentation.logs.conversation_records
    )
    await publish(presentation, replace(old, revision=100, status="complete"))
    assert len(presentation.logs.conversation_records) == 3
    assert presentation.dashboard.display_card._display_text.value == "turn 4"
    successor = ready(draft(order=1, generation=2), "new generation")
    await publish(presentation, successor)
    assert presentation.dashboard.display_card._display_text.value == "new generation"
    retained_count = len(presentation.logs.conversation_records)
    await publish(presentation, ready(draft(order=5, generation=1), "retired generation"))
    assert len(presentation.logs.conversation_records) == retained_count
    assert presentation.dashboard.display_card._display_text.value == "new generation"


@pytest.mark.asyncio
async def test_unmanaged_final_manual_and_peer_keep_existing_dashboard_and_runtime_log_paths(
    presentation,
):
    for channel, source in (("self", "You"), ("peer", "Listen")):
        utterance_id = uuid4()
        translation = Translation(
            utterance_id, f"{source} translated", target_language="en", channel=channel
        )
        presentation.runtime.record_conversation_observation(
            utterance_id=str(utterance_id),
            speaker_channel=channel,
            transcript_text=f"{source} original",
            translation_text=translation.text,
            source_language="ja",
            target_language="en",
            metadata={"source": source},
        )
        await presentation.bridge._handle_event(
            UIEvent(
                UIEventType.TRANSCRIPT_FINAL,
                payload=Transcript(utterance_id, f"{source} original", True, channel=channel),
                source=source,
            )
        )
        await presentation.bridge._handle_event(
            UIEvent(UIEventType.TRANSLATION_DONE, payload=translation, source=source)
        )
        assert presentation.dashboard.display_card._display_text.value == translation.text
    assert len(presentation.logs.conversation_records) == 2
    assert {record.source for record in presentation.logs.conversation_records} == {"You", "Listen"}


@pytest.mark.asyncio
async def test_source_only_final_and_late_accepted_record_update_one_provisional_row(presentation):
    source = draft()
    await publish(presentation, source)
    final = replace(
        source,
        revision=2,
        source_closed=True,
        status="complete",
        targets=(replace(source.targets[0], status="source_only"),),
    )
    await publish(presentation, final)
    assert presentation.dashboard.display_card._display_text.value == source.source_text
    assert "Source only" in presentation.logs._log_text.value
    assert len(presentation.logs.conversation_records) == 1
    assert not presentation.logs.conversation_records[0].accepted
    accept(presentation, final)
    accept(presentation, final)
    assert len(presentation.logs.conversation_records) == 1
    assert presentation.logs.conversation_records[0].accepted
    assert len(presentation.runtime._conversation_record_keys) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_older_final_first", [False, True])
async def test_manual_owns_dashboard_while_late_speech_final_updates_history(
    presentation, monkeypatch, queue_older_final_first
):
    async with OwnerPresentation(
        presentation.dashboard, presentation.logs, presentation.runtime
    ) as owner:
        release = owner.provider.holds[("older speech", "en")] = asyncio.Event()
        await owner.source("older speech")
        await eventually(lambda: any(value.source_closed for value in owner.snapshots.values()))
        older_id = next(iter(owner.snapshots))
        applied = []
        original = presentation.dashboard.set_self_turn

        def present(snapshot):
            applied.append((snapshot.utterance_id, snapshot.status))
            original(snapshot)

        monkeypatch.setattr(presentation.dashboard, "set_self_turn", present)
        if queue_older_final_first:
            owner.delivery_gate.clear()
            release.set()
            await owner.idle()
        await owner.harness.self_owner.submit_text("manual", source="You")
        if not queue_older_final_first:
            await eventually(
                lambda: presentation.dashboard.display_card._display_text.value == "en:manual"
            )
            release.set()
        await owner.idle()
        owner.delivery_gate.set()
        await eventually(
            lambda: presentation.dashboard.display_card._display_text.value == "en:manual"
            and any(
                record.utterance_id == str(older_id) and record.turn_status == "complete"
                for record in presentation.logs.conversation_records
            )
        )
        assert (older_id, "complete") not in applied
        assert "en:older speech" in presentation.logs._log_text.value
        assert presentation.dashboard.display_card._display_text.value == "en:manual"
        assert presentation.dashboard.display_card._turn_status is None
        assert presentation.dashboard.display_card._turn_state_text.value == ""
        assert not presentation.dashboard.display_card._turn_state_text.visible
        await owner.source("successor speech")
        await owner.idle()
        await eventually(
            lambda: presentation.dashboard.display_card._display_text.value == "en:successor speech"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("targets", [("en",), ("en", "ja")])
@pytest.mark.parametrize("queue_older_source_and_final", [False, True])
async def test_newer_speech_keeps_dashboard_when_older_manual_completes(
    presentation, monkeypatch, targets, queue_older_source_and_final
):
    async with OwnerPresentation(
        presentation.dashboard, presentation.logs, presentation.runtime, targets=targets
    ) as owner:
        source_applied = []
        translation_applied = []
        original_source = presentation.dashboard.set_display_text
        original_translation = presentation.dashboard.set_display_translation_text

        def present_source(text, **kwargs):
            source_applied.append(text)
            return original_source(text, **kwargs)

        def present_translation(text, **kwargs):
            translation_applied.append(text)
            return original_translation(text, **kwargs)

        monkeypatch.setattr(presentation.dashboard, "set_display_text", present_source)
        monkeypatch.setattr(
            presentation.dashboard, "set_display_translation_text", present_translation
        )
        release = owner.provider.holds[("older manual", "en")] = asyncio.Event()
        if queue_older_source_and_final:
            owner.delivery_gate.clear()
        admission = await owner.harness.self_owner.submit_text("older manual", source="You")
        if queue_older_source_and_final:
            release.set()
            await owner.idle()
        else:
            await eventually(
                lambda: presentation.dashboard.display_card._display_text.value == "older manual"
            )
        await owner.source("newer speech")
        if queue_older_source_and_final:
            await owner.idle()
        expected = (
            "en:newer speech"
            if len(targets) == 1
            else "English: en:newer speech\nJapanese: ja:newer speech"
        )
        if not queue_older_source_and_final:
            await eventually(
                lambda: presentation.dashboard.display_card._display_text.value == expected
                and any("en:newer speech" in message.text for message in owner.chatbox.messages)
            )
            release.set()
            await owner.idle()
        owner.delivery_gate.set()
        await eventually(
            lambda: presentation.dashboard.display_card._display_text.value == expected
            and "en:older manual" in presentation.logs._log_text.value
            and any("en:older manual" in message.text for message in owner.chatbox.messages)
        )
        await owner.harness.ui_events.join()
        await eventually(
            lambda: any(value.status == "complete" for value in owner.snapshots.values())
            and presentation.dashboard.display_card._turn_state_text.value
            == t("display.turn.complete")
        )
        assert presentation.dashboard.display_card._turn_state_text.visible
        assert presentation.dashboard.display_card._turn_state_text.value == t(
            "display.turn.complete"
        )
        assert "en:older manual" not in translation_applied
        if queue_older_source_and_final:
            assert "older manual" not in source_applied
        else:
            speech_index = next(
                index
                for index, message in enumerate(owner.chatbox.messages)
                if "en:newer speech" in message.text
            )
            manual_index = next(
                index
                for index, message in enumerate(owner.chatbox.messages)
                if "en:older manual" in message.text
            )
            assert manual_index > speech_index
        manual_messages = [
            message for message in owner.chatbox.messages if "en:older manual" in message.text
        ]
        assert all(not message.self_speech for message in manual_messages)
        manual_records = [
            record
            for record in presentation.logs.conversation_records
            if (record.parent_utterance_id or record.utterance_id) == str(admission.turn_id)
        ]
        assert len(manual_records) == 1 + len(targets)
        assert all(record.accepted for record in manual_records)
        assert presentation.logs._log_text.value.count("en:older manual") == 1
        assert len(presentation.logs.conversation_records) == 1 + 2 * len(targets)
        assert len(presentation.runtime._conversation_record_keys) == 2 * (1 + len(targets))
        await owner.harness.self_owner.submit_text("future manual", source="You")
        await owner.idle()
        await eventually(
            lambda: presentation.dashboard.display_card._display_text.value == "en:future manual"
        )
        assert presentation.dashboard.display_card._turn_state_text.value == ""
        assert not presentation.dashboard.display_card._turn_state_text.visible


@pytest.mark.asyncio
async def test_queued_new_manual_source_survives_its_own_admission(presentation):
    async with OwnerPresentation(
        presentation.dashboard, presentation.logs, presentation.runtime
    ) as owner:
        await owner.source("prior speech")
        await owner.idle()
        await eventually(
            lambda: presentation.dashboard.display_card._display_text.value == "en:prior speech"
        )
        owner.delivery_gate.clear()
        release = owner.provider.holds[("queued manual", "en")] = asyncio.Event()
        await owner.harness.self_owner.submit_text("queued manual", source="You")
        await eventually(
            lambda: any(call["text"] == "queued manual" for call in owner.provider.calls)
        )
        owner.delivery_gate.set()
        await eventually(
            lambda: presentation.dashboard.display_card._display_text.value == "queued manual"
        )
        release.set()
        await owner.idle()
        await eventually(
            lambda: presentation.dashboard.display_card._display_text.value == "en:queued manual"
        )
        assert presentation.logs._log_text.value.count("en:queued manual") == 1


@pytest.mark.asyncio
async def test_pending_manual_source_rejects_old_completion_before_its_admission(presentation):
    overlay = ControlledSourceOverlay("pending manual")
    pending = None
    async with OwnerPresentation(
        presentation.dashboard,
        presentation.logs,
        presentation.runtime,
        source_overlay=overlay,
    ) as owner:
        try:
            old_release = owner.provider.holds[("older manual", "en")] = asyncio.Event()
            await owner.harness.self_owner.submit_text("older manual", source="You")
            await eventually(
                lambda: presentation.dashboard.display_card._display_text.value == "older manual"
            )
            pending = asyncio.create_task(
                owner.harness.self_owner.submit_text("pending manual", source="You")
            )
            await asyncio.wait_for(overlay.started.wait(), timeout=3)
            await eventually(
                lambda: presentation.dashboard.display_card._display_text.value == "pending manual"
            )
            assert not pending.done()
            assert not any(call["text"] == "pending manual" for call in owner.provider.calls)
            old_release.set()
            await owner.idle()
            await owner.harness.ui_events.join()
            assert presentation.dashboard.display_card._display_text.value == "pending manual"
            assert "en:older manual" in presentation.logs._log_text.value
            assert any("en:older manual" in message.text for message in owner.chatbox.messages)
            assert not pending.done()
            overlay.release.set()
            await pending
            await owner.idle()
            await eventually(
                lambda: presentation.dashboard.display_card._display_text.value
                == "en:pending manual"
            )
            assert len(presentation.runtime._conversation_record_keys) == 4
            assert len(presentation.logs.conversation_records) == 4
            await owner.source("future speech")
            await owner.idle()
            await eventually(
                lambda: presentation.dashboard.display_card._display_text.value
                == "en:future speech"
            )
        finally:
            overlay.release.set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_real_secondary_fallback_failure_keeps_ready_primary_with_failed_status(presentation):
    async with OwnerPresentation(
        presentation.dashboard,
        presentation.logs,
        presentation.runtime,
        targets=("en", "ja"),
        finalize_wait_ms=50,
    ) as owner:
        owner.provider.failures.add(("partial source", "ja"))
        source_id = await owner.source("partial source", close=False)
        await eventually(
            lambda: "en:partial source" in presentation.dashboard.display_card._display_text.value
        )
        assert not presentation.runtime._conversation_record_keys
        assert t("display.turn.complete") not in presentation.logs._log_text.value
        await owner.close_source(source_id)
        await owner.idle()
        await eventually(
            lambda: any(snapshot.status == "failed" for snapshot in owner.snapshots.values())
        )
        final = next(
            snapshot for snapshot in owner.snapshots.values() if snapshot.status == "failed"
        )
        assert [(target.target_language, target.status) for target in final.targets] == [
            ("en", "ready"),
            ("ja", "failed"),
        ]
        assert sum(call["target_language"] == "ja" for call in owner.provider.calls) == 2
        assert "en:partial source" in presentation.dashboard.display_card._display_text.value
        assert presentation.dashboard.display_card._turn_state_text.value == t(
            "display.turn.failed"
        )
        assert len(presentation.logs.conversation_records) == 2
        assert presentation.logs._log_text.value.count("en:partial source") == 1
        assert t("logs.conversation.target.failed") in presentation.logs._log_text.value
        rows = presentation.logs._log_text.value.split("\n\n")
        assert "Mic → English" in rows[0].splitlines()[0]
        assert "en:partial source" in rows[0]
        assert t("display.turn.complete") in rows[0].splitlines()[0]
        assert t("display.turn.failed") not in rows[0].splitlines()[0]
        assert "Mic → Japanese" in rows[1].splitlines()[0]
        assert t("logs.conversation.target.failed") in rows[1]
        assert t("display.turn.failed") in rows[1].splitlines()[0]
        assert all(
            record.turn_status == "failed" for record in presentation.logs.conversation_records
        )


@pytest.mark.asyncio
async def test_delayed_segmented_child_source_keeps_newer_managed_dashboard(
    presentation, monkeypatch
):
    overlay = ControlledSourceOverlay("first ")
    pending = None
    async with OwnerPresentation(
        presentation.dashboard,
        presentation.logs,
        presentation.runtime,
        source_overlay=overlay,
    ) as owner:
        try:
            unit, parent = independent_recognition((("first ", "en"), ("second", "ja")))
            pending = asyncio.create_task(owner.harness.self_owner.handle_recognition_unit(unit))
            await asyncio.wait_for(overlay.started.wait(), timeout=3)
            await eventually(
                lambda: presentation.dashboard.display_card._display_text.value.strip() == "first"
            )
            assert not pending.done()
            assert owner.harness.self_owner.merge_buffer is None
            applied_sources = []
            original = presentation.dashboard.set_display_text

            def present_source(text, **kwargs):
                applied_sources.append(text)
                return original(text, **kwargs)

            monkeypatch.setattr(presentation.dashboard, "set_display_text", present_source)
            await owner.source("newer managed")
            await eventually(
                lambda: presentation.dashboard.display_card._display_text.value
                == "en:newer managed"
                and any(value.status == "complete" for value in owner.snapshots.values())
            )
            overlay.release.set()
            await pending
            pending = None
            await owner.idle()
            await owner.harness.ui_events.join()
            assert not applied_sources
            assert presentation.dashboard.display_card._display_text.value == "en:newer managed"
            assert presentation.dashboard.display_card._turn_state_text.visible
            assert presentation.dashboard.display_card._turn_state_text.value == t(
                "display.turn.complete"
            )
            records = [
                record
                for record in presentation.logs.conversation_records
                if (record.parent_utterance_id or record.utterance_id) == str(parent)
            ]
            assert len(records) == 4
            assert all(record.accepted for record in records)
            translations = [record for record in records if record.translated_text is not None]
            assert len({record.utterance_id for record in translations}) == 2
            assert all(record.utterance_id != str(parent) for record in translations)
            assert presentation.logs._log_text.value.count("en:first") == 1
            assert presentation.logs._log_text.value.count("en:second") == 1
            assert len(presentation.runtime._conversation_record_keys) == 6
            assert parent not in owner.snapshots
            assert len(owner.snapshots) == 1
            for text in ("first", "second", "newer managed"):
                assert sum(call["text"].strip() == text for call in owner.provider.calls) == 1
            future, future_parent = independent_recognition(
                (("future first ", "en"), ("future second", "ja"))
            )
            await owner.harness.self_owner.handle_recognition_unit(future)
            await owner.idle()
            await eventually(
                lambda: presentation.dashboard.display_card._display_text.value
                == "en:future second"
            )
            assert future_parent not in owner.snapshots
            assert presentation.logs._log_text.value.count("en:future first") == 1
            assert presentation.logs._log_text.value.count("en:future second") == 1
            await owner.source("future managed")
            await owner.idle()
            await eventually(
                lambda: presentation.dashboard.display_card._display_text.value
                == "en:future managed"
            )
        finally:
            overlay.release.set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_current_manual_result_retires_managed_status_after_source_delivery_timeout(
    presentation,
):
    async with OwnerPresentation(
        presentation.dashboard, presentation.logs, presentation.runtime
    ) as owner:
        old_release = owner.provider.holds[("old speech", "en")] = asyncio.Event()
        await owner.source("old speech")
        card = presentation.dashboard.display_card
        card._input_field.value = "unsubmitted draft"
        await eventually(
            lambda: card._display_text.value == "old speech"
            and card._turn_state_text.visible
            and card._turn_state_text.value == t("display.turn.translating")
        )
        previous_timeout = await owner.fill_ui_queue_for_source_timeout()
        manual_release = owner.provider.holds[("manual after dropped source", "en")] = (
            asyncio.Event()
        )
        admission = await owner.harness.self_owner.submit_text(
            "manual after dropped source", source="You"
        )
        writer = owner.harness.output_projection.ui_messages
        await asyncio.wait_for(writer.wait_for_idle(), timeout=3)
        writer.write_timeout_s = previous_timeout
        receipts = [
            decision
            for decision in owner.harness.output_runtime.routing_decisions
            if decision.metadata.get("parent_utterance_id") == str(admission.turn_id)
            and decision.metadata.get("event_type") == UIEventType.TRANSCRIPT_FINAL.value
        ]
        timeouts = [
            decision for decision in receipts if decision.reason == "destination_write_timeout"
        ]
        assert len(timeouts) == 1
        assert timeouts[0].metadata["accepted_handoff"] is False
        assert timeouts[0].metadata["ui_queue_submitted"] is False
        assert not any(decision.metadata["ui_queue_submitted"] for decision in receipts)
        owner.delivery_gate.set()
        await owner.harness.ui_events.join()
        assert card._display_text.value == "old speech"
        assert card._turn_state_text.visible
        assert not old_release.is_set()
        manual_release.set()
        await eventually(
            lambda: card._display_text.value == "en:manual after dropped source"
            and len(presentation.runtime._conversation_record_keys) == 2
            and any(
                "en:manual after dropped source" in message.text
                for message in owner.chatbox.messages
            )
        )
        assert not old_release.is_set()
        assert card._turn_status is None
        assert card._turn_state_text.value == ""
        assert not card._turn_state_text.visible
        assert card._input_field.value == "unsubmitted draft"
        assert presentation.logs._log_text.value.count("en:manual after dropped source") == 1
        assert len(presentation.logs.conversation_records) == 3
        assert (
            sum(call["text"] == "manual after dropped source" for call in owner.provider.calls) == 1
        )
        manual_messages = [
            message
            for message in owner.chatbox.messages
            if "en:manual after dropped source" in message.text
        ]
        assert len(manual_messages) == 1
        assert not manual_messages[0].self_speech
        await owner.source("pressure successor")
        await eventually(
            lambda: card._display_text.value == "en:pressure successor"
            and card._turn_state_text.visible
            and card._turn_state_text.value == t("display.turn.complete")
        )
        old_release.set()
        await owner.idle()
        await owner.harness.ui_events.join()
        assert card._display_text.value == "en:pressure successor"
        assert card._turn_state_text.visible
        assert card._turn_state_text.value == t("display.turn.complete")
        assert card._input_field.value == "unsubmitted draft"
        assert presentation.logs._log_text.value.count("en:manual after dropped source") == 1
        assert (
            sum(
                "en:manual after dropped source" in message.text
                for message in owner.chatbox.messages
            )
            == 1
        )
        assert len(presentation.runtime._conversation_record_keys) == 6
        assert not any(
            decision.metadata.get("parent_utterance_id") == str(admission.turn_id)
            and decision.metadata.get("event_type") == UIEventType.TRANSCRIPT_FINAL.value
            and decision.metadata["ui_queue_submitted"]
            for decision in owner.harness.output_runtime.routing_decisions
        )


@pytest.mark.asyncio
async def test_true_source_only_and_legacy_original_headers_keep_source_language(presentation):
    async with OwnerPresentation(
        presentation.dashboard, presentation.logs, presentation.runtime
    ) as owner:
        configuration = owner.harness.configuration
        configuration.replace(replace(configuration.snapshot().value, translation_enabled=False))
        await owner.source("source only")
        await owner.idle()
        await eventually(
            lambda: len(presentation.logs.conversation_records) == 1
            and presentation.logs.conversation_records[0].turn_status == "complete"
        )
        managed = presentation.logs._log_text.value
        assert "Mic · French" in managed.splitlines()[0]
        assert "→ English" not in managed
        assert "source only" in managed
        await owner.harness.self_owner.submit_text("manual original", source="You")
        await owner.idle()
        await eventually(lambda: len(presentation.logs.conversation_records) == 2)
        legacy = presentation.logs._log_text.value.split("\n\n")[1]
        assert "You · French" in legacy.splitlines()[0]
        assert "→ English" not in legacy
        assert "manual original" in legacy
        assert not owner.provider.calls
        assert len(presentation.runtime._conversation_record_keys) == 2


@pytest.mark.asyncio
async def test_failed_aggregate_does_not_complete_an_earlier_source_translation(presentation):
    source = replace(draft(targets=("en", "ja")), source_language="fr")
    preview = ready(source, "earlier English")
    await publish(presentation, preview)
    ended = replace(
        preview,
        revision=3,
        source_revision=2,
        source_text="extended source",
        source_closed=True,
        status="failed",
        targets=(
            preview.targets[0],
            replace(preview.targets[1], source_revision=2, status="failed"),
        ),
    )
    await publish(presentation, ended)
    rows = presentation.logs._log_text.value.split("\n\n")
    assert "Mic → English" in rows[0].splitlines()[0]
    assert t("display.turn.failed") in rows[0].splitlines()[0]
    assert t("display.turn.complete") not in rows[0].splitlines()[0]
    assert "earlier English" in rows[0]
    assert "earlier source r1 (current source r2)" in rows[0]
    assert "Mic → Japanese" in rows[1].splitlines()[0]
    assert presentation.dashboard.display_card._display_text.value == "extended source"
    assert presentation.dashboard.display_card._turn_state_text.value == t("display.turn.failed")
    assert all(record.turn_status == "failed" for record in presentation.logs.conversation_records)
    assert not presentation.runtime._conversation_record_keys
