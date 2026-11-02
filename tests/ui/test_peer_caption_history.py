from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from puripuly_heart.composition.headless_application import HeadlessApplicationPresentation
from puripuly_heart.config.prompts import get_default_prompt
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.runtime.logging import RuntimeLoggingService
from puripuly_heart.core.runtime_logging import RuntimeLoggingSinks, SessionRuntimeLoggingService
from puripuly_heart.core.translation_backend import TranslationBackend
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.models import Transcript, Translation
from puripuly_heart.ui.event_mapping import map_ui_event
from puripuly_heart.ui.event_projection import EventProjectionContext, EventProjectionService
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from puripuly_heart.ui.views.dashboard import DashboardView
from puripuly_heart.ui.views.logs import ConversationRecord, ConversationViewModel, LogsView
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness


@pytest.fixture
def history_destination(tmp_path):
    records = []
    pending = []

    async def record_conversation(record):
        records.append(record)

    log_file = tmp_path / "peer-history.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    session = SessionRuntimeLoggingService(
        root_logger=logging.Logger(f"peer-history-root-{uuid4()}"),
        session_logger=logging.Logger(f"peer-history-session-{uuid4()}"),
        sinks=RuntimeLoggingSinks(
            stream_handler=logging.StreamHandler(io.StringIO()),
            file_handler=handler,
            log_file=log_file,
        ),
        conversation_record_sink=SimpleNamespace(record_conversation=record_conversation),
        observability_runner=pending.append,
    )
    runtime = RuntimeLoggingService(session_service=session)
    logs = LogsView()
    legacy_entries = []
    host = SimpleNamespace(
        view_logs=logs,
        view_dashboard=DashboardView(),
        add_history_entry=lambda *args, **kwargs: legacy_entries.append((args, kwargs)),
        get_event_language_codes=lambda: ("en", "ko"),
    )
    adapter = FletUiPresentationAdapter(host)
    adapter.attach_runtime_log_sink(runtime)
    bridge = adapter.create_ui_event_bridge(event_queue=asyncio.Queue(), runtime_logging=runtime)
    logs._on_conversation_button_click(None)
    yield SimpleNamespace(
        runtime=runtime,
        session=session,
        logs=logs,
        bridge=bridge,
        audit=records,
        pending=pending,
        log_file=log_file,
        handler=handler,
        legacy_entries=legacy_entries,
    )
    bridge.close()
    runtime.close()
    logs.will_unmount()


def caption_event(unit_id, revision, text, *, translated=False, final=True):
    if translated:
        payload = Translation(unit_id, text, target_language="ko", channel="peer")
        kind = UIEventType.TRANSLATION_DONE
    else:
        payload = Transcript(unit_id, text, final, channel="peer")
        kind = UIEventType.TRANSCRIPT_FINAL if final else UIEventType.TRANSCRIPT_PARTIAL
    return UIEvent(
        kind,
        utterance_id=unit_id,
        payload=payload,
        source="Peer",
        peer_caption_revision=revision,
    )


def observe(destination, unit_id, revision, text, *, translated=False):
    destination.runtime.record_conversation_observation(
        utterance_id=str(unit_id),
        speaker_channel="peer",
        transcript_text=None if translated else text,
        translation_text=text if translated else None,
        source_language="en",
        target_language="ko" if translated else None,
        metadata={
            "source": "Peer",
            "parent_utterance_id": str(unit_id),
            "segment_index": 0,
            "target_index": 0 if translated else None,
            "peer_caption_revision": revision,
            "disposition": "translated" if translated else "source_only",
        },
    )


async def drain_audit(destination):
    while destination.pending:
        await destination.pending.pop(0)
    destination.handler.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_first", [True, False])
async def test_dual_history_delivery_replaces_chained_repairs_once(
    history_destination, runtime_first
):
    destination = history_destination
    unit_id = uuid4()
    for revision, text in ((1, "A,"), (2, "A, second,"), (3, "A, second, third.")):
        for translated in (False, True):
            value = f"translated:{text}" if translated else text
            event = caption_event(unit_id, revision, value, translated=translated)
            if runtime_first:
                observe(destination, unit_id, revision, value, translated=translated)
            await destination.bridge._handle_event(event)
            if not runtime_first:
                observe(destination, unit_id, revision, value, translated=translated)
            observe(destination, unit_id, revision, "duplicate", translated=translated)
            await destination.bridge._handle_event(event)
        assert len(destination.logs.conversation_records) == 1
        record = destination.logs.conversation_records[0]
        assert (record.source_text, record.translated_text, record.peer_caption_revision) == (
            text,
            f"translated:{text}",
            revision,
        )
    for revision in (1, 2):
        for translated in (False, True):
            observe(destination, unit_id, revision, "stale", translated=translated)
            await destination.bridge._handle_event(
                caption_event(unit_id, revision, "stale", translated=translated)
            )
    await drain_audit(destination)
    assert [record.metadata["peer_caption_revision"] for record in destination.audit] == [
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert [record.metadata["peer_caption_history_action"] for record in destination.audit] == [
        "commit",
        "commit",
        "replace",
        "replace",
        "replace",
        "replace",
    ]
    assert {record.metadata["peer_caption_unit_id"] for record in destination.audit} == {
        str(unit_id)
    }
    persisted = destination.log_file.read_text(encoding="utf-8")
    assert persisted.count("Caption commit") == 2
    assert persisted.count("Caption replace") == 4
    assert f'unit="{unit_id}" revision=3' in persisted
    assert 'Original "A, second, third."' in persisted
    assert 'Translation "translated:A, second, third."' in persisted
    assert "stale" not in persisted
    assert "duplicate" not in persisted
    assert destination.legacy_entries == []
    assert destination.logs._log_text.value.count("A, second, third.") == 2
    assert "translated:A,\n" not in destination.logs._log_text.value


@pytest.mark.asyncio
async def test_source_only_revision_clears_old_translation_and_allows_recovery(history_destination):
    destination = history_destination
    unit_id = uuid4()
    for translated in (False, True):
        value = "old translation" if translated else "old source"
        observe(destination, unit_id, 1, value, translated=translated)
        await destination.bridge._handle_event(
            caption_event(unit_id, 1, value, translated=translated)
        )
    observe(destination, unit_id, 3, "authoritative merged source")
    await destination.bridge._handle_event(caption_event(unit_id, 3, "authoritative merged source"))
    observe(destination, unit_id, 2, "late translation", translated=True)
    await destination.bridge._handle_event(
        caption_event(unit_id, 2, "late translation", translated=True)
    )
    record = destination.logs.conversation_records[0]
    assert record.source_text == "authoritative merged source"
    assert record.translated_text is None
    assert "old translation" not in destination.logs._log_text.value
    for translated in (True, False):
        value = "recovered translation" if translated else "recovered source"
        await destination.bridge._handle_event(
            caption_event(unit_id, 4, value, translated=translated)
        )
        observe(destination, unit_id, 4, value, translated=translated)
    assert len(destination.logs.conversation_records) == 1
    record = destination.logs.conversation_records[0]
    assert (record.source_text, record.translated_text) == (
        "recovered source",
        "recovered translation",
    )
    await drain_audit(destination)
    assert [record.metadata["peer_caption_revision"] for record in destination.audit] == [
        1,
        1,
        3,
        4,
        4,
    ]


def test_projection_preserves_identity_rejects_stale_and_bounds_revision_state():
    projection = EventProjectionService(max_turns=2)
    context = EventProjectionContext(source_language="en", target_language="ko")
    first = uuid4()
    for revision, translated in ((1, False), (1, True), (3, True), (3, False)):
        mapped = map_ui_event(caption_event(first, revision, "content", translated=translated))
        batch = projection.project(mapped, context)
        assert len(batch.history) == 1
        assert batch.history[0].utterance_id == first
        assert batch.history[0].peer_caption_revision == revision
        assert projection.project(mapped, context).history == ()
    for translated in (False, True):
        mapped = map_ui_event(caption_event(first, 2, "stale", translated=translated))
        assert projection.project(mapped, context).history == ()
    preview = map_ui_event(caption_event(uuid4(), 1, "preview", final=False))
    assert projection.project(preview, context).history == ()
    assert len(projection._peer_revisions) == 1
    for _ in range(3):
        mapped = map_ui_event(caption_event(uuid4(), 1, "genuine repetition"))
        assert projection.project(mapped, context).history
    assert len(projection._peer_revisions) == 2


@pytest.mark.parametrize("translation_first", [True, False])
def test_history_model_merges_each_revision_pair_and_retains_genuine_repetition(translation_first):
    model = ConversationViewModel(max_entries=2, cleanup_batch=1)
    unit_id = str(uuid4())
    source = ConversationRecord(
        "10:00:00",
        "Peer",
        "peer",
        "A,",
        None,
        utterance_id=unit_id,
        peer_caption_revision=1,
    )
    translation = replace(
        source,
        source_text=None,
        translated_text="translated:A,",
        utterance_id=str(uuid4()),
        parent_utterance_id=unit_id,
    )
    for record in ((translation, source) if translation_first else (source, translation)):
        model.append(record)
        model.append(
            replace(
                record,
                source_text="duplicate" if record.source_text is not None else None,
                translated_text="duplicate" if record.translated_text is not None else None,
            )
        )
    assert len(model.records) == 1
    assert (model.records[0].source_text, model.records[0].translated_text) == (
        "A,",
        "translated:A,",
    )
    model.append(replace(source, source_text="merged", peer_caption_revision=3))
    model.append(replace(translation, translated_text="stale", peer_caption_revision=2))
    assert model.records[0].translated_text is None
    for _ in range(3):
        model.append(replace(source, utterance_id=str(uuid4())))
    assert all(record.utterance_id != unit_id for record in model.records)
    model.append(source)
    assert all(record.utterance_id != unit_id for record in model.records)
    for _ in range(4):
        model.append(replace(source, utterance_id=str(uuid4())))
    assert len(model.records) <= 3
    assert len(model._peer_revisions) <= 6
    assert [record.source_text for record in model.records] == ["A,"] * len(model.records)
    assert len({record.utterance_id for record in model.records}) == len(model.records)


class HistoryBackend(TranslationBackend):
    def __init__(self):
        self.fail_next = False

    async def translate(self, request):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("controlled history translation failure")
        return Translation(request.utterance_id, f"translated:{request.text}")

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_composed_cloud_caption_history_smoke(history_destination):
    destination = history_destination
    clock = FakeClock()
    backend = HistoryBackend()
    harness = compose_translation_test_harness(
        stt=None,
        llm=backend,
        osc=RecordingOscQueue(),
        clock=clock,
        runtime_logging=destination.runtime,
        source_language="en",
        target_language="ko",
        system_prompt=get_default_prompt(),
        peer_translation_enabled=True,
    )

    async def flush():
        await harness.peer_owner.cloud_captions.wait_for_idle()
        await harness.output_runtime.wait_for_peer_output_idle()
        while not harness.ui_events.empty():
            await destination.bridge._handle_event(harness.ui_events.get_nowait())
        await drain_audit(destination)

    async def partial(item):
        await harness.peer_owner.cloud_captions.partial(
            item,
            provider_id="soniox",
            scope=(1, 0, "history-smoke"),
            config=harness.configuration.snapshot(),
        )

    async def final(item):
        await harness.peer_owner.cloud_captions.final(
            replace(item, is_final=True),
            provider_id="soniox",
            scope=(1, 0, "history-smoke"),
            config=harness.configuration.snapshot(),
            sealed=None,
        )
        await flush()

    first = Transcript(
        uuid4(), "A,", False, channel="peer", publication_generation=1, source_order=1
    )
    try:
        await harness.start()
        await partial(first)
        clock.advance(5)
        await harness.peer_owner.cloud_captions.check_previews()
        await flush()
        assert destination.audit == []
        assert destination.logs.conversation_records == ()
        await final(first)
        for order, source in ((2, "b,"), (3, "c.")):
            clock.advance(0.5)
            item = replace(first, utterance_id=uuid4(), text=source, source_order=order)
            await partial(item)
            await final(item)
        assert len(destination.logs.conversation_records) == 1
        record = destination.logs.conversation_records[0]
        assert (record.source_text, record.translated_text, record.peer_caption_revision) == (
            "A, b, c.",
            "translated:A, b, c.",
            4,
        )
        assert [record.metadata["peer_caption_revision"] for record in destination.audit] == [
            2,
            2,
            3,
            3,
            4,
            4,
        ]
        clock.advance(0.5)
        backend.fail_next = True
        failed = replace(first, utterance_id=uuid4(), text="and d.", source_order=4)
        await partial(failed)
        await final(failed)
        assert len(destination.logs.conversation_records) == 1
        record = destination.logs.conversation_records[0]
        assert (record.source_text, record.translated_text, record.peer_caption_revision) == (
            "A, b, c. and d.",
            None,
            5,
        )
        assert "translated:A, b, c." not in destination.logs._log_text.value
        assert destination.audit[-1].transcript_text == "A, b, c. and d."
        assert destination.audit[-1].translation_text is None
        clock.advance(2)
        repeated = replace(first, utterance_id=uuid4(), text="A, b, c. and d.", source_order=5)
        await partial(repeated)
        await final(repeated)
        assert [record.source_text for record in destination.logs.conversation_records] == [
            "A, b, c. and d.",
            "A, b, c. and d.",
        ]
        assert len({record.utterance_id for record in destination.logs.conversation_records}) == 2
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "A, b, c. and d.",
            "A, b, c. and d.",
        ]
        assert destination.legacy_entries == []
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_headless_null_history_preserves_independent_runtime_records(history_destination):
    destination = history_destination
    presentation = HeadlessApplicationPresentation()
    presentation.application = SimpleNamespace(
        get_event_language_codes=lambda: ("en", "ko"),
        state=lambda: SimpleNamespace(translation_enabled=True, stt_state=None),
        clear_managed_auth_pending_state=lambda: None,
        schedule_github_star_prompt_translation_success_observed=lambda: None,
    )
    bridge = presentation.create_ui_event_bridge(
        event_queue=asyncio.Queue(), runtime_logging=destination.runtime
    )
    failures = []
    bridge._record_output_failure = lambda *args: failures.append(args)
    unit_id = uuid4()
    try:
        for translated in (False, True):
            value = "translated source" if translated else "source"
            observe(destination, unit_id, 1, value, translated=translated)
            await bridge._handle_event(caption_event(unit_id, 1, value, translated=translated))
        await drain_audit(destination)
        assert failures == []
        assert len(destination.audit) == 2
        assert presentation.dashboard.transcript == "source"
        assert presentation.dashboard.translation == "translated source"
        assert not hasattr(bridge.history_destination, "records")
    finally:
        bridge.close()
