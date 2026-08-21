from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np
import pytest
from puripuly_heart.core.managed_openrouter_release import (
    ManagedOpenRouterReleaseDiagnostics,
    ManagedOpenRouterUserFacingError,
)

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.messages import UserErrorReport
from puripuly_heart.core.orchestrator.channel_runtime import (
    ContextEntry,
    _MergeBuffer,
    _SpeculativeAttemptStatus,
)
from puripuly_heart.core.orchestrator.self_translation_channel import (
    SelfTranslationChannelOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import (
    ChatboxProjection,
)
from puripuly_heart.core.orchestrator.translation_request import DirectTranslationRequest
from puripuly_heart.core.overlay.state import ActiveSelfOverlayMetadata
from puripuly_heart.core.runtime_logging import SessionLoggingMode, SessionRuntimeLoggingService
from puripuly_heart.core.stt.backend import STTBackendTranscriptEvent
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd
from puripuly_heart.domain.events import (
    STTErrorEvent,
    STTFinalEvent,
    STTPartialEvent,
    STTSessionState,
    STTSessionStateEvent,
    UIEventType,
)
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import (
    compose_translation_test_harness,
    make_speculative_attempt,
)


@dataclass(slots=True)
class StubLLM:
    should_fail: bool = False
    calls: list[tuple[UUID, str, str]] = field(default_factory=list)
    closed: bool = False

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        _ = (system_prompt, source_language, target_language)
        self.calls.append((utterance_id, text, context))
        if self.should_fail:
            raise RuntimeError("llm failed")
        return Translation(utterance_id=utterance_id, text=f"T:{text}")

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class RecordingLanguageLLM:
    calls: list[dict[str, str]] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        _ = system_prompt
        self.calls.append(
            {
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
            }
        )
        return Translation(utterance_id=utterance_id, text=f"{target_language}:{text}")

    async def close(self) -> None:
        return None


@dataclass(slots=True)
class ManagedAuthFailingLLM:
    diagnostics: ManagedOpenRouterReleaseDiagnostics
    closed: bool = False

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        _ = (utterance_id, text, system_prompt, source_language, target_language, context)
        raise ManagedOpenRouterUserFacingError(
            message_key="managed_release.retry_after_ms",
            message_kwargs={"retry_after_ms": 9000},
            diagnostics=self.diagnostics,
        )

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class StubSTT:
    handled: list[object] = field(default_factory=list)
    closed: bool = False

    async def handle_vad_event(self, event: object) -> None:
        self.handled.append(event)

    async def close(self) -> None:
        self.closed = True

    async def events(self):
        while True:
            await asyncio.sleep(60.0)
            yield STTBackendTranscriptEvent(text="", is_final=False)


@dataclass(slots=True)
class QueueingSTT:
    handled: list[object] = field(default_factory=list)
    closed: bool = False
    queue: asyncio.Queue[object | None] = field(default_factory=asyncio.Queue)

    async def handle_vad_event(self, event: object) -> None:
        self.handled.append(event)

    async def close(self) -> None:
        self.closed = True
        await self.queue.put(None)

    async def emit(self, event: object) -> None:
        await self.queue.put(event)

    async def events(self):
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item


@dataclass(slots=True)
class BlockingOverlaySink:
    events: list[object] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    active_self_metadata: ActiveSelfOverlayMetadata | None = None

    async def emit(self, event: object) -> None:
        self.events.append(event)
        self._capture_active_self_metadata(event)
        self.started.set()
        await self.release.wait()

    def _capture_active_self_metadata(self, event: object) -> None:
        if getattr(event, "type", None) != "self_active_update":
            return
        utterance_id = getattr(event, "utterance_id", None)
        if not isinstance(utterance_id, UUID):
            return
        self.active_self_metadata = ActiveSelfOverlayMetadata(
            text=getattr(event, "text", ""),
            secondary_text=getattr(event, "secondary_text", ""),
            utterance_id=utterance_id,
            occupant_key=getattr(event, "occupant_key", ""),
            update_id=getattr(event, "update_id", None),
            origin_wall_clock_ms=getattr(event, "origin_wall_clock_ms", None),
            session_scope=getattr(event, "session_scope", None),
            source_text_hash=getattr(event, "source_text_hash", None),
            source_text_len=getattr(event, "source_text_len", None),
            logical_turn_key=getattr(event, "logical_turn_key", None),
        )

    def active_self_overlay_metadata(self) -> ActiveSelfOverlayMetadata | None:
        return self.active_self_metadata


@dataclass(slots=True)
class MetadataOverlaySink:
    active_self_metadata: ActiveSelfOverlayMetadata | None = None
    events: list[object] = field(default_factory=list)

    async def emit(self, event: object) -> None:
        self.events.append(event)
        event_type = getattr(event, "type", None)
        if event_type == "self_active_clear":
            self.active_self_metadata = None
        elif event_type == "self_transcript_final" and self.active_self_metadata is not None:
            if self.active_self_metadata.utterance_id == getattr(event, "utterance_id", None):
                self.active_self_metadata = None

    def active_self_overlay_metadata(self) -> ActiveSelfOverlayMetadata | None:
        return self.active_self_metadata


def active_self_metadata_for_merge(
    merge_id: UUID,
    *,
    text: str,
    secondary_text: str,
) -> ActiveSelfOverlayMetadata:
    return ActiveSelfOverlayMetadata(
        text=text,
        secondary_text=secondary_text,
        utterance_id=merge_id,
        occupant_key=f"self:{merge_id}",
        update_id=None,
        origin_wall_clock_ms=None,
        session_scope=None,
        source_text_hash=None,
        source_text_len=None,
        logical_turn_key=None,
    )


@dataclass(slots=True)
class RaisingOverlaySink:
    error: Exception = field(default_factory=lambda: RuntimeError("overlay down"))

    async def emit(self, event: object) -> None:
        _ = event
        raise self.error


@dataclass(slots=True)
class _RuntimeLogSinks:
    stream_handler: logging.Handler
    file_handler: logging.Handler
    log_file: object


def _make_runtime_logging_capture() -> tuple[SessionRuntimeLoggingService, io.StringIO]:
    stream = io.StringIO()
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger = logging.getLogger(f"test.translation.runtime.root.{uuid4()}")
    root_logger.handlers.clear()
    root_logger.propagate = False

    session_logger = logging.getLogger(f"test.translation.runtime.session.{uuid4()}")
    session_logger.handlers.clear()
    session_logger.propagate = False

    runtime_logging = SessionRuntimeLoggingService(
        root_logger=root_logger,
        session_logger=session_logger,
        sinks=_RuntimeLogSinks(
            stream_handler=stream_handler,
            file_handler=logging.NullHandler(),
            log_file="runtime.log",
        ),
    )
    return runtime_logging, stream


def _runtime_log_messages(stream: io.StringIO) -> list[str]:
    return [line for line in stream.getvalue().splitlines() if line]


def test_peer_translation_disclosure_enqueues_chatbox_notice_without_context_history() -> None:
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(stt=None, llm=None, osc=osc, clock=FakeClock(12.0))
    harness.self_runtime.remember_context(
        "existing context",
        timestamp=10.0,
        source_language="ko",
        target_language="en",
    )
    before_history = list(harness.self_runtime.translation_history)

    harness.output_projection.publish_system_disclosure("Peer translation is on")

    assert [message.text for message in osc.messages] == ["Peer translation is on"]
    assert osc.messages[0].created_at == 12.0
    assert harness.self_runtime.translation_history == before_history


@pytest.mark.asyncio
async def test_translation_drops_stale_partial_and_keeps_final_order() -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock()
    )
    buffer = _MergeBuffer(merge_id=uuid4())
    utterance_id = uuid4()

    harness.self_owner._upsert_merge_part(buffer, utterance_id, "hello world")
    harness.self_owner._upsert_merge_part(buffer, utterance_id, "hello")
    harness.self_owner._upsert_merge_part(buffer, utterance_id, "hello world!!!")

    partial = Transcript(utterance_id=utterance_id, text="he", is_final=False, created_at=1.0)
    final = Transcript(
        utterance_id=utterance_id, text="hello world!!!", is_final=True, created_at=2.0
    )

    await harness.dispatch_transcript(partial, is_final=False, source="Mic")
    await harness.dispatch_transcript(final, is_final=True, source="Mic")
    await harness.dispatch_transcript(partial, is_final=False, source="Mic")

    bundle = harness.bundle_for(utterance_id)
    assert buffer.parts == ["hello world!!!"]
    assert harness.self_owner._merge_text(buffer.parts) == "hello world!!!"
    assert bundle.final is not None
    assert bundle.final.text == "hello world!!!"
    assert bundle.partial is None


@pytest.mark.asyncio
async def test_stop_cancels_pending_tasks_and_closes_providers() -> None:
    stt = StubSTT()
    llm = StubLLM()
    harness = compose_translation_test_harness(
        stt=stt, llm=llm, osc=RecordingOscQueue(), clock=FakeClock()
    )
    harness.set_started_for_test(True)

    harness.self_runtime.translation_tasks[uuid4()] = asyncio.create_task(asyncio.sleep(60.0))
    buffer = _MergeBuffer(
        merge_id=uuid4(),
        speculative_attempt=make_speculative_attempt(task=asyncio.create_task(asyncio.sleep(60.0))),
    )
    buffer.finalize_wait_task = asyncio.create_task(asyncio.sleep(60.0))
    buffer.awaiting_vad_timeout_task = asyncio.create_task(asyncio.sleep(60.0))
    buffer.resume_end_timeout_task = asyncio.create_task(asyncio.sleep(60.0))
    harness.self_owner.merge_buffer = buffer

    await harness.stop()

    assert harness.self_runtime.translation_tasks == {}
    assert harness.self_owner.merge_buffer is None
    assert stt.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_start_is_idempotent_and_creates_background_tasks() -> None:
    stt = StubSTT()
    harness = compose_translation_test_harness(
        stt=stt, llm=StubLLM(), osc=RecordingOscQueue(), clock=FakeClock()
    )

    await harness.start(auto_flush_osc=True)
    stt_task = harness.self_runtime.stt_task
    osc_task = harness.output_runtime.chatbox_flush_task
    await harness.start(auto_flush_osc=True)

    assert harness.self_runtime.stt_task is stt_task
    assert harness.output_runtime.chatbox_flush_task is osc_task
    await harness.stop()


@pytest.mark.asyncio
async def test_clear_language_runtime_state_self_preserves_stt_task_and_clears_overlay_preview() -> (
    None
):
    preview_merge_id = uuid4()
    overlay_sink = MetadataOverlaySink(
        active_self_metadata=active_self_metadata_for_merge(
            preview_merge_id,
            text="preview",
            secondary_text="secondary",
        )
    )
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=overlay_sink,
        clock=FakeClock(),
    )
    self_id = uuid4()
    standalone_id = uuid4()
    peer_id = uuid4()
    stt_task = asyncio.create_task(asyncio.sleep(60.0))
    translation_task = asyncio.create_task(asyncio.sleep(60.0))
    standalone_translation_task = asyncio.create_task(asyncio.sleep(60.0))
    spec_task = asyncio.create_task(asyncio.sleep(60.0))
    finalize_wait_task = asyncio.create_task(asyncio.sleep(60.0))
    awaiting_vad_timeout_task = asyncio.create_task(asyncio.sleep(60.0))
    resume_end_timeout_task = asyncio.create_task(asyncio.sleep(60.0))
    all_tasks = [
        stt_task,
        translation_task,
        standalone_translation_task,
        spec_task,
        finalize_wait_task,
        awaiting_vad_timeout_task,
        resume_end_timeout_task,
    ]

    harness.self_runtime.stt_task = stt_task
    harness.self_runtime.translation_tasks[self_id] = translation_task
    harness.self_runtime.translation_tasks[standalone_id] = standalone_translation_task
    harness.self_runtime.get_or_create_bundle(self_id)
    harness.self_runtime.get_or_create_bundle(standalone_id)
    harness.self_runtime.utterance_sources[self_id] = "Mic"
    harness.self_runtime.utterance_sources[standalone_id] = "Mic"
    harness.self_runtime.utterance_start_times[self_id] = 1.0
    harness.self_runtime.utterance_start_times[standalone_id] = 1.5
    harness.self_runtime.speech_ended_ids.add(self_id)
    harness.self_runtime.speech_ended_ids.add(standalone_id)
    harness.self_runtime.translation_history.append(ContextEntry("history", "ko", "en", 1.0))
    harness.self_runtime.merge_buffer = _MergeBuffer(
        merge_id=preview_merge_id,
        utterance_ids=[self_id],
        speculative_attempt=make_speculative_attempt(task=spec_task),
        finalize_wait_task=finalize_wait_task,
        awaiting_vad_timeout_task=awaiting_vad_timeout_task,
        resume_end_timeout_task=resume_end_timeout_task,
    )
    harness.peer_owner._record_latency_stage(
        channel="self",
        utterance_id=self_id,
        stage="speech_end",
        timestamp=1.0,
        publish_now=False,
    )
    harness.peer_owner._record_latency_stage(
        channel="peer",
        utterance_id=peer_id,
        stage="speech_end",
        timestamp=2.0,
        publish_now=False,
    )

    try:
        await harness.clear_channel_language_state(channel="self")

        assert harness.self_runtime.stt_task is stt_task
        assert harness.self_owner.runtime.stt_task is stt_task
        assert harness.self_runtime.translation_tasks == {}
        assert harness.self_runtime.merge_buffer is None
        assert standalone_id in harness.self_runtime.utterances
        assert harness.self_runtime.utterance_sources == {standalone_id: "Mic"}
        assert harness.self_runtime.utterance_start_times == {}
        assert harness.self_runtime.speech_ended_ids == set()
        assert harness.self_runtime.translation_history == [
            ContextEntry("history", "ko", "en", 1.0)
        ]
        assert overlay_sink.active_self_overlay_metadata() is None
        timeline_keys = harness.translation_diagnostics.snapshot().timeline_keys
        assert ("self", self_id) not in timeline_keys
        assert ("peer", peer_id) in timeline_keys
        assert translation_task.cancelled() is True
        assert standalone_translation_task.cancelled() is True
        assert spec_task.cancelled() is True
        assert finalize_wait_task.cancelled() is True
        assert awaiting_vad_timeout_task.cancelled() is True
        assert resume_end_timeout_task.cancelled() is True
        assert stt_task.done() is False
    finally:
        for task in all_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_language_change_updates_next_self_translation_request_target() -> None:
    llm = RecordingLanguageLLM()
    harness = compose_translation_test_harness(
        stt=None, llm=llm, osc=RecordingOscQueue(), clock=FakeClock()
    )

    await harness.translation_requests.translate(
        DirectTranslationRequest(utterance_id=uuid4(), text="hello")
    )
    harness.replace_configuration(target_language="ja")
    await harness.clear_channel_language_state(channel="self")
    await harness.translation_requests.translate(
        DirectTranslationRequest(utterance_id=uuid4(), text="world")
    )

    assert llm.calls == [
        {
            "text": "hello",
            "source_language": "ko",
            "target_language": "en",
            "context": "",
        },
        {
            "text": "world",
            "source_language": "ko",
            "target_language": "ja",
            "context": "",
        },
    ]


def test_send_stt_connected_notification_respects_eligibility_and_interval() -> None:
    clock = FakeClock()
    osc = RecordingOscQueue(immediate_result=True)
    harness = compose_translation_test_harness(stt=None, llm=None, osc=osc, clock=clock)

    harness.self_owner._send_stt_connected_notification()
    assert osc.immediate_messages == []

    harness.self_owner.mark_promo_eligible()
    harness.self_owner._send_stt_connected_notification()
    assert osc.immediate_messages == ["PuriPuly ON!"]
    assert harness.self_owner._last_promo_time == 0.0

    clock.advance(30.0)
    harness.self_owner.mark_promo_eligible()
    harness.self_owner._send_stt_connected_notification()
    assert osc.immediate_messages == ["PuriPuly ON!"]

    clock.advance(301.0)
    harness.self_owner.mark_promo_eligible()
    harness.self_owner._send_stt_connected_notification()
    assert osc.immediate_messages == ["PuriPuly ON!", "PuriPuly ON!"]


def test_send_stt_connected_notification_does_not_update_time_on_failed_send() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(immediate_result=False),
        clock=FakeClock(),
    )

    harness.self_owner.mark_promo_eligible()
    harness.self_owner._send_stt_connected_notification()
    assert harness.self_owner._last_promo_time is None


def test_prepare_llm_request_routes_context_logs_by_runtime_visibility() -> None:
    basic_runtime_logging, basic_stream = _make_runtime_logging_capture()
    detailed_runtime_logging, detailed_stream = _make_runtime_logging_capture()
    detailed_runtime_logging.set_mode(SessionLoggingMode.DETAILED)

    basic_harness = compose_translation_test_harness(
        stt=None,
        llm=StubLLM(),
        osc=RecordingOscQueue(),
        clock=FakeClock(_now=10.0),
        runtime_logging=basic_runtime_logging,
    )
    detailed_harness = compose_translation_test_harness(
        stt=None,
        llm=StubLLM(),
        osc=RecordingOscQueue(),
        clock=FakeClock(_now=10.0),
        runtime_logging=detailed_runtime_logging,
    )

    try:
        basic_harness.remember_context("안녕", 9.0)
        detailed_harness.remember_context("안녕", 9.0)

        basic_harness.prepare_translation_request_with_mode("입력")
        detailed_harness.prepare_translation_request_with_mode("입력")

        basic_messages = _runtime_log_messages(basic_stream)
        detailed_messages = _runtime_log_messages(detailed_stream)
        expected_context_chars = len('- [self] "안녕"')
        expected_context_apply_log = (
            "[Translation] Context apply: channel=self mode=local "
            "request_chars=2 entries=1 self_entries=1 peer_entries=0 "
            f"context_chars={expected_context_chars}"
        )

        assert "[Translation] Context mode: channel=self mode=local" in basic_messages
        assert expected_context_apply_log in basic_messages
        assert not any("입력" in message for message in basic_messages)
        assert not any("안녕" in message for message in basic_messages)

        assert "[Translation] Context mode: channel=self mode=local" in detailed_messages
        assert expected_context_apply_log in detailed_messages
        assert not any("입력" in message for message in detailed_messages)
        assert not any("안녕" in message for message in detailed_messages)
    finally:
        basic_runtime_logging.close()
        detailed_runtime_logging.close()


@pytest.mark.asyncio
async def test_handle_stt_event_logs_basic_channel_state_breadcrumb() -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        runtime_logging=runtime_logging,
    )

    try:
        await harness.dispatch_stt_event(
            STTSessionStateEvent(state=STTSessionState.STREAMING, channel="peer")
        )

        event = await harness.ui_events.get()
        assert event.type == UIEventType.SESSION_STATE_CHANGED
        assert event.channel == "peer"
        assert "[Translation] STT state: channel=peer state=STREAMING" in _runtime_log_messages(
            log_stream
        )
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_retired_stt_ingress_forwards_only_final_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handled: list[object] = []

    async def record_event(_self: SelfTranslationChannelOwner, event: object) -> None:
        handled.append(event)

    monkeypatch.setattr(SelfTranslationChannelOwner, "handle_stt_event", record_event)
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock()
    )
    utterance_id = uuid4()
    transcript = Transcript(
        utterance_id=utterance_id,
        text="retired final",
        is_final=True,
        created_at=1.0,
    )
    partial = Transcript(
        utterance_id=utterance_id,
        text="retired partial",
        is_final=False,
        created_at=0.5,
    )
    final_event = STTFinalEvent(utterance_id=utterance_id, transcript=transcript)

    await harness.dispatch_retired_stt_event(
        STTSessionStateEvent(state=STTSessionState.DISCONNECTED)
    )
    await harness.dispatch_retired_stt_event(STTErrorEvent(message="retired failure"))
    await harness.dispatch_retired_stt_event(
        STTPartialEvent(utterance_id=utterance_id, transcript=partial)
    )
    await harness.dispatch_retired_stt_event(final_event)

    assert handled == [final_event]


@pytest.mark.asyncio
async def test_handle_stt_partial_runtime_log_uses_metadata_without_transcript_text() -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    utterance_id = uuid4()
    raw_partial = "raw partial transcript should not enter runtime logs"
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        runtime_logging=runtime_logging,
    )

    try:
        await harness.dispatch_stt_event(
            STTPartialEvent(
                utterance_id=utterance_id,
                transcript=Transcript(
                    utterance_id=utterance_id,
                    text=raw_partial,
                    is_final=False,
                    created_at=1.0,
                ),
            )
        )

        event = await harness.ui_events.get()
        messages = _runtime_log_messages(log_stream)

        assert event.type == UIEventType.TRANSCRIPT_PARTIAL
        assert any(
            message.startswith("[Translation] STT Partial:")
            and "channel=self" in message
            and f"utterance_id={utterance_id}" in message
            and f"text_len={len(raw_partial)}" in message
            for message in messages
        )
        assert not any(raw_partial in message for message in messages)
        assert not any(raw_partial[:20] in message for message in messages)
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_publish_chatbox_candidate_emits_metadata_preview_only_in_detailed_runtime_logs() -> (
    None
):
    basic_runtime_logging, basic_stream = _make_runtime_logging_capture()
    detailed_runtime_logging, detailed_stream = _make_runtime_logging_capture()
    detailed_runtime_logging.set_mode(SessionLoggingMode.DETAILED)

    basic_harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        runtime_logging=basic_runtime_logging,
    )
    detailed_harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        runtime_logging=detailed_runtime_logging,
    )
    utterance_id = uuid4()

    try:
        await basic_harness.output_projection.publish_chatbox(
            ChatboxProjection(
                utterance_id=utterance_id,
                channel="self",
                transcript_text="hello world from transcript",
                translation_text="hello world translated",
                include_source=True,
                source=None,
            )
        )
        await detailed_harness.output_projection.publish_chatbox(
            ChatboxProjection(
                utterance_id=utterance_id,
                channel="self",
                transcript_text="hello world from transcript",
                translation_text="hello world translated",
                include_source=True,
                source=None,
            )
        )

        basic_event = await basic_harness.ui_events.get()
        detailed_event = await detailed_harness.ui_events.get()
        assert basic_event.type == UIEventType.OSC_SENT
        assert detailed_event.type == UIEventType.OSC_SENT

        basic_messages = _runtime_log_messages(basic_stream)
        detailed_messages = _runtime_log_messages(detailed_stream)

        assert not any("OSC enqueue preview" in message for message in basic_messages)
        assert any(
            message.startswith("[Translation] OSC enqueue preview:")
            and "text_len=" in message
            and "translation_text_present=True" in message
            for message in detailed_messages
        )
        assert not any("hello world from transcript" in message for message in detailed_messages)
        assert not any("hello world translated" in message for message in detailed_messages)
    finally:
        basic_runtime_logging.close()
        detailed_runtime_logging.close()


@pytest.mark.asyncio
async def test_publish_chatbox_candidate_after_translation_stop_skips_without_user_text() -> None:
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=osc,
        clock=FakeClock(),
    )
    utterance_id = uuid4()

    await harness.start(auto_flush_osc=True)
    await harness.stop()
    await harness.output_projection.publish_chatbox(
        ChatboxProjection(
            utterance_id=utterance_id,
            channel="self",
            transcript_text="closed secret transcript",
            translation_text="closed secret translation",
            include_source=True,
            source=None,
        )
    )

    assert osc.messages == []
    assert harness.ui_events.empty()
    decision = harness.output_runtime.routing_decisions[-1]
    assert decision.decision == "skipped"
    assert decision.reason == "output_runtime_closed"
    assert "closed secret transcript" not in repr(decision)
    assert "closed secret translation" not in repr(decision)


@pytest.mark.asyncio
async def test_stop_without_start_closes_output_runtime_ingress() -> None:
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=osc,
        clock=FakeClock(),
    )
    utterance_id = uuid4()

    await harness.stop()
    await harness.output_projection.publish_chatbox(
        ChatboxProjection(
            utterance_id=utterance_id,
            channel="self",
            transcript_text="never started secret transcript",
            translation_text=None,
            include_source=True,
            source=None,
        )
    )

    assert osc.messages == []
    decision = harness.output_runtime.routing_decisions[-1]
    assert decision.decision == "skipped"
    assert decision.reason == "output_runtime_closed"
    assert "never started secret transcript" not in repr(decision)


@pytest.mark.asyncio
async def test_restart_after_output_runtime_closed_failure_keeps_owners_not_running() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
    )

    await harness.start(auto_flush_osc=False)
    await harness.stop()

    with pytest.raises(RuntimeError, match="closed"):
        await harness.start(auto_flush_osc=False)

    assert harness.started is False


@pytest.mark.asyncio
async def test_restart_after_failed_output_runtime_close_keeps_owners_not_running() -> None:
    class DropPendingFailsOnceOsc(RecordingOscQueue):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next_drop = True

        def drop_pending(self) -> None:
            if self.fail_next_drop:
                self.fail_next_drop = False
                raise RuntimeError("drop pending failed")

    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=DropPendingFailsOnceOsc(),
        clock=FakeClock(),
    )

    await harness.start(auto_flush_osc=False)
    with pytest.raises(RuntimeError, match="drop pending failed"):
        await harness.stop()

    with pytest.raises(RuntimeError, match="closing"):
        await harness.start(auto_flush_osc=False)

    assert harness.started is False


@pytest.mark.asyncio
async def test_handle_stt_event_routes_non_low_latency_events() -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        runtime_logging=runtime_logging,
    )
    harness.self_owner.mark_promo_eligible()
    utterance_id = uuid4()
    partial = Transcript(utterance_id=utterance_id, text="hel", is_final=False, created_at=1.0)
    final = Transcript(utterance_id=utterance_id, text="hello", is_final=True, created_at=2.0)

    try:
        await harness.dispatch_stt_event(STTSessionStateEvent(state=STTSessionState.STREAMING))
        await harness.dispatch_stt_event(STTErrorEvent(message="boom"))
        await harness.dispatch_stt_event(
            STTPartialEvent(utterance_id=utterance_id, transcript=partial)
        )
        await harness.dispatch_stt_event(STTFinalEvent(utterance_id=utterance_id, transcript=final))

        events = [await harness.ui_events.get() for _ in range(5)]
        assert [event.type for event in events] == [
            UIEventType.SESSION_STATE_CHANGED,
            UIEventType.ERROR,
            UIEventType.TRANSCRIPT_PARTIAL,
            UIEventType.TRANSCRIPT_FINAL,
            UIEventType.OSC_SENT,
        ]
        assert events[1].runtime_log_handled is False
        assert harness.osc.immediate_messages == ["PuriPuly ON!"]
        assert len(harness.osc.messages) == 1
        assert harness.osc.messages[0].text == "hello"
        assert (
            "[Translation] Translation skipped (stage=final, channel=self, publish_chatbox=True): "
            "llm unavailable"
        ) in _runtime_log_messages(log_stream)
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_handle_stt_event_ignores_partial_in_low_latency_mode() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        low_latency_mode=True,
    )
    utterance_id = uuid4()
    partial = Transcript(utterance_id=utterance_id, text="hel", is_final=False, created_at=1.0)

    await harness.dispatch_stt_event(STTPartialEvent(utterance_id=utterance_id, transcript=partial))

    assert harness.ui_events.empty()


@pytest.mark.asyncio
async def test_translate_and_enqueue_emits_error_and_fallback_transcript() -> None:
    llm = StubLLM(should_fail=True)
    runtime_logging, log_stream = _make_runtime_logging_capture()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        fallback_transcript_only=True,
        runtime_logging=runtime_logging,
    )
    utterance_id = uuid4()

    try:
        await harness.process_translation(utterance_id, "hello")

        events = [await harness.ui_events.get() for _ in range(2)]
        assert [event.type for event in events] == [UIEventType.ERROR, UIEventType.OSC_SENT]
        assert events[0].runtime_log_handled is True
        assert harness.osc.messages[0].text == "hello"
        assert (
            "[Translation] Translation failed (stage=final, channel=self): "
            "category=unknown code=provider.unknown" in _runtime_log_messages(log_stream)
        )
        assert "llm failed" not in "\n".join(_runtime_log_messages(log_stream))
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_translate_and_enqueue_logs_managed_auth_diagnostics() -> None:
    runtime_logging, log_stream = _make_runtime_logging_capture()
    harness = compose_translation_test_harness(
        stt=None,
        llm=ManagedAuthFailingLLM(
            diagnostics=ManagedOpenRouterReleaseDiagnostics(
                operation="issue",
                code="trial_unavailable",
                error_class="retryable",
                subcode="broker_backoff",
                retry_after_ms=9000,
                message="broker is temporarily unavailable",
            )
        ),
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        fallback_transcript_only=True,
        runtime_logging=runtime_logging,
    )
    utterance_id = uuid4()

    try:
        await harness.process_translation(utterance_id, "hello")

        events = [await harness.ui_events.get() for _ in range(2)]
        assert [event.type for event in events] == [UIEventType.ERROR, UIEventType.OSC_SENT]
        assert events[0].runtime_log_handled is True
        payload = events[0].payload
        assert isinstance(payload, UserErrorReport)
        assert payload.message.key == "managed_release.retry_after_ms"
        assert dict(payload.message.params) == {"retry_after_ms": 9000}
        assert payload.diagnostics.code == "provider.service_unavailable"
        assert payload.diagnostics.fields["exception_type"] == "ManagedOpenRouterUserFacingError"
        assert payload.diagnostics.fields["managed_operation"] == "issue"
        assert payload.diagnostics.fields["managed_code"] == "trial_unavailable"
        assert payload.diagnostics.fields["managed_error_class"] == "retryable"
        assert payload.diagnostics.fields["managed_subcode"] == "broker_backoff"
        assert "broker is temporarily unavailable" not in repr(payload)
        messages = _runtime_log_messages(log_stream)
        assert any(
            "managed_operation=issue managed_code=trial_unavailable "
            "managed_error_class=retryable managed_subcode=broker_backoff retry_after_ms=9000"
            in message
            for message in messages
        )
        assert "broker is temporarily unavailable" not in "\n".join(messages)
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_next_action_evaluator_starts_failed_fallback_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=StubLLM(), osc=RecordingOscQueue(), clock=FakeClock()
    )
    buffer = _MergeBuffer(
        merge_id=uuid4(),
        parts=["text"],
        speculative_attempt=make_speculative_attempt(
            source_text="text",
            status=_SpeculativeAttemptStatus.FAILED,
        ),
    )
    harness.self_owner.merge_buffer = buffer
    called: list[str] = []

    async def fake_commit(
        _self: SelfTranslationChannelOwner,
        _buffer: _MergeBuffer,
        *,
        reason: str,
    ) -> None:
        called.append(reason)

    monkeypatch.setattr(SelfTranslationChannelOwner, "_commit_merge", fake_commit)

    await harness.self_owner._evaluate_speculative_next_action(buffer, reason="spec_failed")
    await harness.self_owner._evaluate_speculative_next_action(buffer, reason="spec_failed")

    assert called == ["spec_failed"]
    assert buffer.speculative_attempt is not None
    assert buffer.speculative_attempt.terminal_action_started is True


@pytest.mark.asyncio
async def test_run_spec_translation_logs_spec_failure_only_in_detailed_mode() -> None:
    basic_runtime_logging, basic_stream = _make_runtime_logging_capture()
    detailed_runtime_logging, detailed_stream = _make_runtime_logging_capture()
    detailed_runtime_logging.set_mode(SessionLoggingMode.DETAILED)

    basic_harness = compose_translation_test_harness(
        stt=None,
        llm=StubLLM(should_fail=True),
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        runtime_logging=basic_runtime_logging,
        low_latency_mode=True,
    )
    detailed_harness = compose_translation_test_harness(
        stt=None,
        llm=StubLLM(should_fail=True),
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        runtime_logging=detailed_runtime_logging,
        low_latency_mode=True,
    )
    basic_buffer = _MergeBuffer(
        merge_id=uuid4(),
        parts=["hello"],
        speculative_attempt=make_speculative_attempt(source_text="hello", sequence=1),
    )
    detailed_buffer = _MergeBuffer(
        merge_id=uuid4(),
        parts=["hello"],
        speculative_attempt=make_speculative_attempt(source_text="hello", sequence=1),
    )
    basic_harness.self_owner.merge_buffer = basic_buffer
    detailed_harness.self_owner.merge_buffer = detailed_buffer

    try:
        await basic_harness.self_owner._run_spec_translation(basic_buffer.merge_id, "hello", 1)
        await detailed_harness.self_owner._run_spec_translation(
            detailed_buffer.merge_id, "hello", 1
        )
        assert basic_buffer.speculative_attempt is not None
        assert detailed_buffer.speculative_attempt is not None
        assert basic_buffer.speculative_attempt.status is _SpeculativeAttemptStatus.FAILED
        assert detailed_buffer.speculative_attempt.status is _SpeculativeAttemptStatus.FAILED

        assert not any(
            "[Translation] Translation failed (stage=spec, channel=self): "
            "category=unknown code=provider.unknown" in message
            for message in _runtime_log_messages(basic_stream)
        )
        assert any(
            "[Translation] Translation failed (stage=spec, channel=self): "
            "category=unknown code=provider.unknown" in message
            for message in _runtime_log_messages(detailed_stream)
        )
        assert "llm failed" not in "\n".join(_runtime_log_messages(detailed_stream))
    finally:
        basic_runtime_logging.close()
        detailed_runtime_logging.close()


@pytest.mark.asyncio
async def test_handle_stt_event_preserves_runtime_logged_flag_from_stt_errors() -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock()
    )

    await harness.dispatch_stt_event(
        STTErrorEvent(message="session failed", channel="peer", runtime_log_handled=True)
    )

    event = await harness.ui_events.get()
    assert event.type == UIEventType.ERROR
    assert event.channel == "peer"
    assert event.runtime_log_handled is True


@pytest.mark.asyncio
async def test_peer_stt_event_loop_failure_without_runtime_logging_is_safe(
    caplog,
) -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock()
    )

    with caplog.at_level(logging.ERROR, logger="puripuly_heart.core.orchestrator.translation"):
        await harness.dispatch_stt_failure(
            RuntimeError("loop boom"),
            channel="peer",
        )

    assert "[Translation] STT event loop crashed: RuntimeError" in caplog.messages
    assert "loop boom" not in "\n".join(caplog.messages)
    assert any(record.levelno == logging.ERROR for record in caplog.records)


@pytest.mark.asyncio
async def test_peer_stt_event_loop_failure_with_runtime_logging_is_safe() -> None:
    raw_detail = "stt event loop socket failed token=translation-stt-secret-123"
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        runtime_logging=runtime_logging,
    )

    try:
        await harness.dispatch_stt_failure(
            ConnectionError(raw_detail),
            channel="peer",
        )

        runtime_log = "\n".join(_runtime_log_messages(log_stream))
        assert (
            "[Translation] STT event loop crashed: category=network code=stt.network" in runtime_log
        )
        assert raw_detail not in runtime_log
        assert "translation-stt-secret-123" not in runtime_log
        assert "Traceback (most recent call last):" not in runtime_log
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_handle_stt_event_loop_exception_with_runtime_logging_uses_safe_stt_report() -> None:
    raw_detail = "stt owner task socket failed token=translation-stt-secret-456"
    runtime_logging, log_stream = _make_runtime_logging_capture()
    runtime_logging.set_mode(SessionLoggingMode.DETAILED)
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        runtime_logging=runtime_logging,
    )

    try:
        await harness.dispatch_stt_failure(ConnectionError(raw_detail))

        runtime_log = "\n".join(_runtime_log_messages(log_stream))
        assert (
            "[Translation] STT event loop crashed: category=network code=stt.network" in runtime_log
        )
        assert raw_detail not in runtime_log
        assert "translation-stt-secret-456" not in runtime_log
        assert "Traceback (most recent call last):" not in runtime_log
    finally:
        runtime_logging.close()


@pytest.mark.asyncio
async def test_emit_overlay_event_logs_safe_exception_metadata() -> None:
    basic_runtime_logging, basic_stream = _make_runtime_logging_capture()
    detailed_runtime_logging, detailed_stream = _make_runtime_logging_capture()
    detailed_runtime_logging.set_mode(SessionLoggingMode.DETAILED)

    basic_harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=RaisingOverlaySink(),
        clock=FakeClock(),
        runtime_logging=basic_runtime_logging,
    )
    detailed_harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=RaisingOverlaySink(),
        clock=FakeClock(),
        runtime_logging=detailed_runtime_logging,
    )

    try:
        await basic_harness.output_projection.publish_overlay_event(
            basic_harness.output_projection.overlay_event_adapter.utterance_closed(
                utterance_id=uuid4(),
                channel="self",
                is_final=True,
            )
        )
        await detailed_harness.output_projection.publish_overlay_event(
            detailed_harness.output_projection.overlay_event_adapter.utterance_closed(
                utterance_id=uuid4(),
                channel="self",
                is_final=True,
            )
        )

        basic_messages = _runtime_log_messages(basic_stream)
        detailed_messages = _runtime_log_messages(detailed_stream)

        assert "[Translation] Overlay sink emit failed: RuntimeError" in basic_messages
        assert not any(
            "Traceback (most recent call last):" in message for message in basic_messages
        )

        assert "[Translation] Overlay sink emit failed: RuntimeError" in detailed_messages
        assert not any(
            "Traceback (most recent call last):" in message for message in detailed_messages
        )
        assert not any("overlay down" in message for message in detailed_messages)
    finally:
        basic_runtime_logging.close()
        detailed_runtime_logging.close()


@pytest.mark.asyncio
async def test_maybe_restart_spec_replaces_previous_task_and_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=StubLLM(), osc=RecordingOscQueue(), clock=FakeClock()
    )
    old_task = asyncio.create_task(asyncio.sleep(60.0))
    buffer = _MergeBuffer(
        merge_id=uuid4(),
        parts=["final text"],
        speculative_attempt=make_speculative_attempt(
            source_text="old",
            task=old_task,
            result=Translation(utterance_id=uuid4(), text="old"),
        ),
    )
    harness.self_owner.merge_buffer = buffer
    assert buffer.speculative_attempt is not None
    buffer.speculative_attempt.result = Translation(utterance_id=buffer.merge_id, text="old")
    seen: list[tuple[UUID, str, int]] = []

    async def fake_run_spec(
        _self: SelfTranslationChannelOwner,
        merge_id: UUID,
        text: str,
        attempt: int,
    ) -> None:
        seen.append((merge_id, text, attempt))

    monkeypatch.setattr(SelfTranslationChannelOwner, "_run_spec_translation", fake_run_spec)
    await harness.self_owner._maybe_restart_spec(buffer)
    await asyncio.sleep(0)

    assert old_task.done() is True
    assert buffer.speculative_attempt is not None
    assert buffer.speculative_attempt.sequence == 2
    assert buffer.speculative_attempt.source_text == "final text"
    assert seen == [(buffer.merge_id, "final text", 2)]


@pytest.mark.asyncio
async def test_handle_vad_event_speech_end_tracks_timing_and_forwards_to_stt() -> None:
    stt = StubSTT()
    clock = FakeClock(_now=10.0)
    harness = compose_translation_test_harness(
        stt=stt, llm=None, osc=RecordingOscQueue(), clock=clock, low_latency_mode=True
    )
    utterance_id = uuid4()

    await harness.self_owner.handle_vad_event(SpeechEnd(utterance_id))

    assert harness.osc.typing == [True]
    assert harness.self_runtime.utterance_start_times[utterance_id] == 10.0
    assert utterance_id in harness.self_runtime.speech_ended_ids
    assert stt.handled == [SpeechEnd(utterance_id)]


@pytest.mark.asyncio
async def test_handle_vad_event_forwards_resume_confirming_chunk_before_overlay_resync() -> None:
    stt = StubSTT()
    sink = BlockingOverlaySink()
    clock = FakeClock(_now=10.0)
    harness = compose_translation_test_harness(
        stt=stt,
        llm=None,
        osc=RecordingOscQueue(),
        overlay_sink=sink,
        clock=clock,
        low_latency_mode=True,
    )
    first_utterance_id = uuid4()
    resumed_utterance_id = uuid4()
    merge_id = uuid4()
    chunk = SpeechChunk(resumed_utterance_id, chunk=np.zeros((1,), dtype=np.float32))

    harness.self_owner.merge_buffer = _MergeBuffer(
        merge_id=merge_id,
        parts=["hello live"],
        utterance_ids=[first_utterance_id],
        speculative_attempt=make_speculative_attempt(
            source_text="hello live",
            result=Translation(utterance_id=merge_id, text="translated live"),
        ),
        resume_pending=True,
        resume_utterance_id=resumed_utterance_id,
        resume_chunk_count=2,
    )
    sink.active_self_metadata = active_self_metadata_for_merge(
        merge_id,
        text="stale preview",
        secondary_text="translated live",
    )

    task = asyncio.create_task(harness.self_owner.handle_vad_event(chunk))
    await sink.started.wait()

    assert len(stt.handled) == 1
    assert stt.handled[0] is chunk
    assert task.done() is False

    sink.release.set()
    await task

    assert sink.events[-1].type == "self_active_update"
    assert sink.events[-1].text == "hello live"
    assert sink.events[-1].secondary_text == "translated live"
    assert sink.events[-1].utterance_id == merge_id


@pytest.mark.asyncio
async def test_submit_text_validates_input_and_enqueues_without_llm() -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock()
    )

    with pytest.raises(ValueError, match="text must be non-empty"):
        await harness.self_owner.submit_text("   ")

    utterance_id = await harness.self_owner.submit_text("hello", source="You")
    events = [await harness.ui_events.get(), await harness.ui_events.get()]
    assert [event.type for event in events] == [UIEventType.TRANSCRIPT_FINAL, UIEventType.OSC_SENT]
    assert harness.osc.messages[-1].utterance_id == utterance_id
    assert harness.osc.messages[-1].text == "hello"


@pytest.mark.asyncio
async def test_submit_text_clipboard_source_uses_manual_fallback_without_llm() -> None:
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(stt=None, llm=None, osc=osc, clock=FakeClock())
    harness.replace_configuration(translation_enabled=False)

    utterance_id = await harness.self_owner.submit_text("clipboard fallback", source="Clipboard")
    events = [await harness.ui_events.get(), await harness.ui_events.get()]

    assert [event.type for event in events] == [UIEventType.TRANSCRIPT_FINAL, UIEventType.OSC_SENT]
    assert events[0].source == "Clipboard"
    assert osc.messages[-1].utterance_id == utterance_id
    assert osc.messages[-1].text == "clipboard fallback"


def test_merge_helpers_cover_overlap_and_spacing_paths() -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock()
    )

    assert harness.self_owner._merge_with_overlap("same text", "text done") == "same text done"
    assert harness.self_owner._merge_with_overlap("go", "home") == "go home"
    assert harness.self_owner._merge_with_overlap("abc", "...abc") == "abc"
    assert harness.self_owner._merge_with_overlap("가다.", "가다고") == "가다.가다고"
    assert harness.self_owner._strip_trailing_boundary("abc. ") == ("abc", 2)
    assert harness.self_owner._strip_leading_boundary(" ..abc") == ("abc", 3)
