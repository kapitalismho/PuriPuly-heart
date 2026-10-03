from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Literal
from uuid import uuid4

import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.messages import UserErrorReport
from puripuly_heart.core.orchestrator.channel_runtime import ContextEntry, _MergeBuffer
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.orchestrator.translation_request import PreparedTranslationRequest
from puripuly_heart.core.orchestrator.translation_turn import TranslationOutputSubmission
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
    STTTextContribution,
)
from puripuly_heart.core.vad.gating import SpeechEnd
from puripuly_heart.domain.events import (
    STTFinalEvent,
    STTSessionState,
    STTSessionStateEvent,
    UIEventType,
)
from puripuly_heart.domain.models import OSCMessage, Transcript
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity
from tests.core.test_self_translation_low_latency import BlockingLLMProvider, FakeLLMProvider
from tests.core.test_translation_owner_branch_coverage import (
    _make_runtime_logging_capture,
    _runtime_log_messages,
)
from tests.helpers.fakes import FakeSender, RecordingOscQueue
from tests.helpers.translation_owners import (
    compose_translation_test_harness,
    make_speculative_attempt,
)


def test_self_waiting_output_attaches_admitted_context_texts() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    child_id = uuid4()
    harness.self_owner._admitted_requests[child_id] = PreparedTranslationRequest(
        system_prompt="prompt",
        context="",
        requested_at=1.0,
        applied_context_mode="integrated",
        source_language="ko",
        target_language="en",
        context_texts=("어제 뭐 했어",),
    )
    submission = TranslationOutputSubmission(
        parent_utterance_id=uuid4(),
        child_utterance_id=child_id,
        sequence=0,
        channel="self",
        source="Mic",
        source_text="오늘 뭐 해",
        source_language="ko",
        target_language="en",
        outcome="source_only",
        config_snapshot=harness.configuration.snapshot(),
        failure_code="translation_timeout",
        turn_generation=0,
        turn_order=0,
        turn_kind="self",
    )

    filled = harness.self_owner._with_prepared_context(submission)
    already = harness.self_owner._with_prepared_context(replace(submission, context_texts=()))

    assert filled.context_texts == ("어제 뭐 했어",)
    assert already.context_texts == ()


@pytest.mark.asyncio
async def test_activation_notice_preference_preserves_self_cooldown_without_replay() -> None:
    clock = FakeClock()
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(stt=None, llm=None, osc=osc, clock=clock)
    owner = harness.self_owner
    ready = STTSessionStateEvent(state=STTSessionState.STREAMING)
    try:
        await harness.start()
        harness.output_runtime.activation_notice_enabled = False
        owner.mark_promo_eligible()
        await owner.handle_stt_event(ready)

        assert osc.immediate_messages == []
        assert owner._last_promo_time is None
        assert harness.output_runtime.routing_decisions[-1].reason == "activation_notice_disabled"

        harness.output_runtime.activation_notice_enabled = True
        await owner.handle_stt_event(ready)
        assert osc.immediate_messages == []

        clock.advance(5.0)
        owner.mark_promo_eligible()
        await owner.handle_stt_event(ready)
        assert osc.immediate_messages == ["PuriPuly ON!"]
        assert owner._last_promo_time == 5.0

        clock.advance(30.0)
        harness.output_runtime.activation_notice_enabled = False
        owner.mark_promo_eligible()
        await owner.handle_stt_event(ready)
        harness.output_runtime.activation_notice_enabled = True
        owner.mark_promo_eligible()
        await owner.handle_stt_event(ready)
        assert osc.immediate_messages == ["PuriPuly ON!"]
        assert owner._last_promo_time == 5.0

        clock.advance(301.0)
        harness.output_runtime.activation_notice_enabled = False
        owner.mark_promo_eligible()
        await owner.handle_stt_event(ready)
        assert owner._last_promo_time == 5.0
        assert harness.output_runtime.routing_decisions[-1].reason == "activation_notice_disabled"

        harness.output_runtime.activation_notice_enabled = True
        await owner.handle_stt_event(ready)
        assert osc.immediate_messages == ["PuriPuly ON!"]
        owner.mark_promo_eligible()
        await owner.handle_stt_event(ready)
        assert osc.immediate_messages == ["PuriPuly ON!", "PuriPuly ON!"]
        assert owner._last_promo_time == clock.now()
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_self_owner_rejects_closed_ingress_and_non_self_events() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.self_owner
    utterance_id = uuid4()

    await owner.close_ingress()
    with pytest.raises(RuntimeError, match="closed"):
        await owner.submit_text("closed")

    await owner.open_ingress()
    with pytest.raises(ValueError, match="non-Self"):
        await owner.handle_stt_event(
            STTFinalEvent(
                utterance_id,
                Transcript(
                    utterance_id,
                    "peer",
                    is_final=True,
                    channel="peer",
                ),
            )
        )


@pytest.mark.asyncio
async def test_self_provider_reset_cancels_speech_without_erasing_manual_or_peer_state() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.self_owner
    speech_id = uuid4()
    manual_id = uuid4()
    peer_id = uuid4()
    owner.runtime.get_or_create_bundle(speech_id)
    owner.runtime.remember_source(speech_id, "Mic")
    owner.runtime.get_or_create_bundle(manual_id)
    owner.runtime.remember_source(manual_id, "You")
    owner.runtime.translation_history.append(ContextEntry("manual", "ko", "en", 1.0))
    harness.peer_runtime.get_or_create_bundle(peer_id)
    harness.peer_runtime.translation_history.append(
        ContextEntry("peer", "ko", "en", 1.0, channel="peer")
    )

    await owner.reset_provider_channel("self")

    assert speech_id not in owner.runtime.utterances
    assert manual_id in owner.runtime.utterances
    assert [entry.text for entry in owner.runtime.translation_history] == ["manual"]
    assert peer_id in harness.peer_runtime.utterances
    assert [entry.text for entry in harness.peer_runtime.translation_history] == ["peer"]


@pytest.mark.asyncio
async def test_independent_self_final_in_low_latency_mode_keeps_origin_without_local_turn() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        low_latency_mode=True,
    )
    stream = RecognitionStreamIdentity("self", 1, 2, "epoch", ("gemini_transcribe",))
    unit = STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 1), "early final")
    try:
        await harness.start()
        await harness.self_owner.handle_recognition_unit(unit)
        await harness.translation_turns.wait_for_idle()

        finals = []
        while not harness.ui_events.empty():
            event = harness.ui_events.get_nowait()
            if event.channel == "self" and event.type is UIEventType.TRANSCRIPT_FINAL:
                finals.append(event.payload)
        assert [transcript.text for transcript in finals] == ["early final"]
        assert finals[0].recognition_origins[0].identity == unit.identity
        assert harness.self_owner.merge_buffer is None
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_independent_final_admission_rejects_stale_and_duplicate_but_not_repeated_text() -> (
    None
):
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_self(harness.self_owner)
    stream = RecognitionStreamIdentity("self", 1, 2, "epoch", ("gemini_transcribe",))
    stale = RecognitionStreamIdentity("self", 1, 1, "old", ("gemini_transcribe",))
    empty = STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 1), "")
    accepted = STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 2), "same words")
    repeated = STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 3), "same words")
    obsolete = STTRecognitionUnit(RecognitionUnitIdentity(stale, uuid4(), 1), "obsolete")
    callbacks._self_capture = SimpleNamespace(
        is_current_recognition_stream=lambda identity: identity == stream,
    )
    harness.self_owner.local_asr_runtime = SimpleNamespace(
        is_current_recognition_stream=lambda _channel, identity: identity == stream,
    )
    try:
        await harness.start()
        harness.self_owner.mark_promo_eligible()
        await callbacks.self_event_handler(STTRecognitionUnitTerminal(obsolete, "final"))
        assert harness.stt_session_state() is None
        await callbacks.self_event_handler(STTRecognitionUnitTerminal(empty, "empty"))
        assert harness.stt_session_state() is STTSessionState.STREAMING
        for unit in (accepted, accepted, repeated):
            await callbacks.self_event_handler(STTRecognitionUnitTerminal(unit, "final"))
        await harness.translation_turns.wait_for_idle()
        finals = []
        while not harness.ui_events.empty():
            event = harness.ui_events.get_nowait()
            if event.channel == "self" and event.type is UIEventType.TRANSCRIPT_FINAL:
                finals.append(event.payload)
        assert [transcript.text for transcript in finals] == ["same words", "same words"]
        assert [transcript.recognition_origins[0].identity for transcript in finals] == [
            accepted.identity,
            repeated.identity,
        ]
        assert harness.osc.immediate_messages == ["PuriPuly ON!"]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_self_native_input_failure_disconnects_only_current_stream() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_self(harness.self_owner)
    current = RecognitionStreamIdentity("self", 2, 3, "epoch-a", ("gemini_transcribe",))
    callbacks._self_capture = SimpleNamespace(
        snapshot=SimpleNamespace(generation=2),
        note_input_terminal=lambda _event: None,
        is_current_recognition_stream=lambda stream: (
            stream.activation_generation == 2
            and stream.capture_epoch == 3
            and stream.settings_scope == current.settings_scope
        ),
    )
    harness.self_owner.local_asr_runtime = SimpleNamespace(
        is_current_recognition_stream=lambda _channel, stream: stream in (current, newer)
    )
    newer = replace(current, provider_epoch_id="epoch-b")

    await callbacks.self_event_handler(
        STTRecognitionUnitTerminal(
            STTRecognitionUnit(RecognitionUnitIdentity(current, uuid4(), 1), "ready"),
            "final",
        )
    )
    assert harness.stt_session_state() is STTSessionState.STREAMING
    failed = STTProviderInputTerminal(
        STTProviderTurnIdentity(
            AudioSegmentIdentity(2, 1, uuid4(), 3), "epoch-a", "input", current.settings_scope
        ),
        "failed",
        "self",
        "provider_send_failed",
        "retire",
    )
    await callbacks.self_event_handler(failed)
    assert harness.stt_session_state() is STTSessionState.DISCONNECTED

    await callbacks.self_event_handler(
        STTRecognitionUnitTerminal(
            STTRecognitionUnit(RecognitionUnitIdentity(newer, uuid4(), 1), "reconnected"),
            "final",
        )
    )
    assert harness.stt_session_state() is STTSessionState.STREAMING
    await callbacks.self_event_handler(failed)
    await callbacks.self_event_handler(
        replace(
            failed,
            identity=replace(
                failed.identity,
                segment=replace(failed.identity.segment, activation_generation=1),
            ),
        )
    )
    await callbacks.self_event_handler(
        replace(failed, identity=replace(failed.identity, settings_scope=("other_provider",)))
    )
    await callbacks.self_event_handler(
        replace(
            failed,
            identity=replace(
                failed.identity,
                segment=replace(failed.identity.segment, capture_epoch=2),
            ),
        )
    )
    await callbacks.self_event_handler(STTProviderEpochEnded("epoch-a", True, "native_idle_end"))
    assert harness.stt_session_state() is STTSessionState.STREAMING
    await callbacks.self_event_handler(STTProviderEpochEnded("epoch-b", True, "native_idle_end"))
    assert harness.stt_session_state() is STTSessionState.DISCONNECTED


@pytest.mark.asyncio
@pytest.mark.parametrize("native_before_input", [False, True])
async def test_self_latest_submitted_epoch_end_disconnects_before_its_first_final(
    native_before_input: bool,
) -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_self(harness.self_owner)
    first = RecognitionStreamIdentity("self", 1, 0, "epoch-a", ("gemini_transcribe",))
    second = replace(first, provider_epoch_id="epoch-b")
    third = replace(first, provider_epoch_id="epoch-c")
    streams = (first, second, third)
    callbacks._self_capture = SimpleNamespace(
        snapshot=SimpleNamespace(generation=1),
        note_input_terminal=lambda _event: None,
        is_current_recognition_stream=lambda stream: stream in streams,
    )
    harness.self_owner.local_asr_runtime = SimpleNamespace(
        is_current_recognition_stream=lambda _channel, stream: stream in streams
    )

    def submitted(stream: RecognitionStreamIdentity) -> STTProviderInputTerminal:
        return STTProviderInputTerminal(
            STTProviderTurnIdentity(
                AudioSegmentIdentity(1, 1, uuid4(), 0),
                stream.provider_epoch_id,
                uuid4().hex,
                stream.settings_scope,
            ),
            "submitted",
            "self",
        )

    def native(
        stream: RecognitionStreamIdentity, text: str, sequence: int = 1
    ) -> STTRecognitionUnitTerminal:
        return STTRecognitionUnitTerminal(
            STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), sequence), text),
            "final" if text else "empty",
        )

    try:
        await harness.start()
        await callbacks.self_event_handler(submitted(first))
        await callbacks.self_event_handler(native(first, "first ready"))
        if native_before_input:
            await callbacks.self_event_handler(native(second, ""))
        await callbacks.self_event_handler(submitted(second))
        await callbacks.self_event_handler(
            STTProviderEpochEnded("epoch-a", True, "native_idle_end")
        )
        assert harness.stt_session_state() is STTSessionState.STREAMING

        await callbacks.self_event_handler(
            STTProviderEpochEnded("epoch-b", True, "native_idle_end")
        )
        assert harness.stt_session_state() is STTSessionState.DISCONNECTED
        await callbacks.self_event_handler(native(second, "accepted before epoch end", sequence=2))
        assert harness.stt_session_state() is STTSessionState.DISCONNECTED
        await harness.translation_turns.wait_for_idle()
        finals = []
        while not harness.ui_events.empty():
            item = harness.ui_events.get_nowait()
            if item.channel == "self" and item.type is UIEventType.TRANSCRIPT_FINAL:
                finals.append(item.payload.text)
        assert finals == ["first ready", "accepted before epoch end"]

        await callbacks.self_event_handler(native(third, ""))
        assert harness.stt_session_state() is STTSessionState.STREAMING
        await callbacks.self_event_handler(
            STTProviderEpochEnded("epoch-b", True, "native_idle_end")
        )
        assert harness.stt_session_state() is STTSessionState.STREAMING
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_self_late_accepted_native_finals_preserve_text_without_reviving_old_readiness() -> (
    None
):
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_self(harness.self_owner)
    first = RecognitionStreamIdentity("self", 1, 0, "epoch-a", ("gemini_transcribe",))
    second = replace(first, provider_epoch_id="epoch-b")
    callbacks._self_capture = SimpleNamespace(
        snapshot=SimpleNamespace(generation=1),
        note_input_terminal=lambda _event: None,
        is_current_recognition_stream=lambda stream: (
            stream.activation_generation == 1
            and stream.capture_epoch == 0
            and stream.settings_scope == first.settings_scope
        ),
    )
    harness.self_owner.local_asr_runtime = SimpleNamespace(
        is_current_recognition_stream=lambda _channel, stream: stream in (first, second)
    )

    def input_terminal(
        stream: RecognitionStreamIdentity,
        outcome: Literal["submitted", "failed"],
    ) -> STTProviderInputTerminal:
        return STTProviderInputTerminal(
            STTProviderTurnIdentity(
                AudioSegmentIdentity(1, 1, uuid4(), 0),
                stream.provider_epoch_id,
                uuid4().hex,
                stream.settings_scope,
            ),
            outcome,
            "self",
        )

    try:
        await harness.start()
        await callbacks.self_event_handler(input_terminal(first, "submitted"))
        await callbacks.self_event_handler(
            STTRecognitionUnitTerminal(
                STTRecognitionUnit(RecognitionUnitIdentity(first, uuid4(), 1), "first ready"),
                "final",
            )
        )
        await callbacks.self_event_handler(input_terminal(second, "submitted"))
        await callbacks.self_event_handler(
            STTRecognitionUnitTerminal(
                STTRecognitionUnit(RecognitionUnitIdentity(second, uuid4(), 1), "second ready"),
                "final",
            )
        )
        late_first = STTRecognitionUnit(
            RecognitionUnitIdentity(first, uuid4(), 2), "late first text"
        )
        await callbacks.self_event_handler(STTRecognitionUnitTerminal(late_first, "final"))
        assert harness.stt_session_state() is STTSessionState.STREAMING

        await callbacks.self_event_handler(input_terminal(second, "failed"))
        assert harness.stt_session_state() is STTSessionState.DISCONNECTED
        queued_second = STTRecognitionUnit(
            RecognitionUnitIdentity(second, uuid4(), 2), "queued after failure"
        )
        await callbacks.self_event_handler(STTRecognitionUnitTerminal(queued_second, "final"))
        assert harness.stt_session_state() is STTSessionState.DISCONNECTED
        await harness.translation_turns.wait_for_idle()
        finals = []
        while not harness.ui_events.empty():
            item = harness.ui_events.get_nowait()
            if item.channel == "self" and item.type is UIEventType.TRANSCRIPT_FINAL:
                finals.append(item.payload)
        assert [item.text for item in finals] == [
            "first ready",
            "second ready",
            "late first text",
            "queued after failure",
        ]
        assert [item.recognition_origins[0].identity for item in finals[2:]] == [
            late_first.identity,
            queued_second.identity,
        ]
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_self_native_input_terminal_retires_only_local_vad_timing() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_self(harness.self_owner)
    stream = RecognitionStreamIdentity("self", 1, 0, "epoch", ("gemini_transcribe",))
    callbacks._self_capture = SimpleNamespace(
        snapshot=SimpleNamespace(generation=1),
        note_input_terminal=lambda _event: None,
        is_current_recognition_stream=lambda identity: identity == stream,
    )

    async def accept_vad(_channel: str, _event: object) -> None:
        return None

    harness.self_owner.local_asr_runtime = SimpleNamespace(
        handle_vad_event=accept_vad,
        commit_handoff=lambda _channel: accept_vad(_channel, None),
        is_current_recognition_stream=lambda _channel, identity: identity == stream,
    )
    local_ids = [uuid4() for _ in range(24)]
    native_units = [
        STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), index), "same words")
        for index in (1, 2)
    ]
    try:
        await harness.start()
        for order, segment_id in enumerate(local_ids, 1):
            if order != 1:
                await harness.self_owner.handle_vad_event(SpeechEnd(segment_id))
                assert segment_id in harness.self_runtime.speech_ended_ids
            await callbacks.self_event_handler(
                STTProviderInputTerminal(
                    STTProviderTurnIdentity(
                        AudioSegmentIdentity(1, order, segment_id, 0),
                        "epoch",
                        f"input-{order}",
                        stream.settings_scope,
                    ),
                    "failed" if order == 1 else "submitted",
                    "self",
                )
            )
            if order == 1:
                await harness.self_owner.handle_vad_event(SpeechEnd(segment_id))
        assert harness.self_runtime.utterance_start_times == {}
        assert not (set(local_ids) & harness.self_runtime.speech_ended_ids)
        assert not (
            set(local_ids)
            & {key for _, key in harness.self_owner.diagnostics.snapshot().timeline_keys}
        )
        for unit in native_units:
            await callbacks.self_event_handler(STTRecognitionUnitTerminal(unit, "final"))
        await harness.translation_turns.wait_for_idle()
        finals = []
        while not harness.ui_events.empty():
            item = harness.ui_events.get_nowait()
            if item.channel == "self" and item.type is UIEventType.TRANSCRIPT_FINAL:
                finals.append(item.payload)
        assert [item.text for item in finals] == ["same words", "same words"]
        assert [item.recognition_origins[0].identity for item in finals] == [
            unit.identity for unit in native_units
        ]
        assert all(unit.identity.unit_id not in local_ids for unit in native_units)
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_scoped_request_keeps_suffix_after_early_publication() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    segment_id = uuid4()
    identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(1, 1, segment_id, 1),
        provider_epoch_id="epoch",
        provider_turn_id="turn",
    )
    contribution_a = STTTextContribution("a", 0, 1)
    contribution_b = STTTextContribution("b", 1, 2)

    await harness.self_owner.handle_stt_event(
        STTProviderTurnUpdate(identity, 1, "stable", "append", "A", contribution=contribution_a)
    )
    first = harness.self_owner.merge_buffer
    assert first is not None
    first.awaiting_vad_end = False
    first.awaiting_vad_utterance_id = None
    await harness.self_owner._commit_merge(first, reason="awaiting_vad_timeout")

    await harness.self_owner.handle_stt_event(
        STTProviderTurnUpdate(identity, 2, "stable", "append", "AB", contribution=contribution_b)
    )
    successor = harness.self_owner.merge_buffer
    assert successor is not None
    assert successor.parts == ["B"]
    assert successor.utterance_ids != first.utterance_ids

    await harness.self_owner.handle_stt_event(
        STTProviderTurnTerminal(
            identity,
            "final",
            text="AB",
            text_authority="authoritative",
            included_contributions=(contribution_a, contribution_b),
        )
    )
    await harness.self_owner.handle_stt_event(
        STTProviderTurnTerminal(
            identity,
            "final",
            text="AB",
            text_authority="authoritative",
            included_contributions=(contribution_a, contribution_b),
        )
    )

    assert successor.parts == ["B"]
    assert identity not in harness.self_owner._scoped_publication_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("speech_at", [8.0, None])
async def test_native_latency_uses_frozen_anchor_without_waiting_for_local_end(
    channel: Literal["self", "peer"], speech_at: float | None
) -> None:
    clock = FakeClock(_now=10.0)
    logging, log_stream = _make_runtime_logging_capture()
    sender = FakeSender()
    paginator = ChatboxPaginator(sender=sender, clock=clock)
    overlay = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    llm = BlockingLLMProvider(response_text="translated")
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=paginator,
        clock=clock,
        overlay_sink=overlay,
        runtime_logging=logging,
        peer_translation_enabled=True,
        low_latency_mode=True,
    )
    paginator.stage_recorder = harness.translation_diagnostics.record_chatbox_stage
    stream = RecognitionStreamIdentity(channel, 1, 0, "epoch", ("gemini_transcribe",))
    unit = STTRecognitionUnit(
        RecognitionUnitIdentity(stream, uuid4(), 1),
        "source",
        estimated_last_speech_at=speech_at,
    )
    owner = harness.self_owner if channel == "self" else harness.peer_owner
    try:
        await harness.start()
        harness.output_runtime.activate_peer_generation(1)
        await owner.handle_recognition_unit(unit)
        await asyncio.wait_for(llm.started.wait(), timeout=1.0)
        assert not any(
            "[Basic][Latency]" in message for message in _runtime_log_messages(log_stream)
        )
        clock.advance(1.0)
        llm.release.set()
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
        await asyncio.wait_for(harness.output_runtime.wait_for_peer_output_idle(), timeout=1.0)
        if channel == "self":
            assert sender.sent == ["source (translated)"]
            endpoint = "chatbox_send"
        else:
            assert sender.sent == []
            assert [
                (block.primary_text, block.secondary_text) for block in overlay.snapshot().blocks
            ] == [("translated", "source")]
            endpoint = "overlay_applied"
        summaries = [
            dict(field.split("=", 1) for field in message.split()[1:])
            for message in _runtime_log_messages(log_stream)
            if message.startswith("[Basic][Latency]")
        ]
        expected = (
            []
            if speech_at is None
            else [
                {
                    "channel": channel,
                    "endpoint": endpoint,
                    f"last_speech_to_{endpoint}_ms": "3000",
                    "estimated": "true",
                }
            ]
        )
        assert summaries == expected
        assert not harness.translation_diagnostics.snapshot().timeline_keys
    finally:
        llm.release.set()
        await harness.stop()
        await overlay.close()
        logging.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scoped", [False, True])
async def test_late_speech_end_measures_already_committed_translation(scoped: bool) -> None:
    clock = FakeClock(_now=10.0)
    logging, stream = _make_runtime_logging_capture()
    sender = FakeSender()
    paginator = ChatboxPaginator(sender=sender, clock=clock, max_chars=24)
    harness = compose_translation_test_harness(
        stt=None,
        llm=FakeLLMProvider(delay_s=0),
        osc=paginator,
        clock=clock,
        runtime_logging=logging,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        low_latency_awaiting_vad_timeout_s=0.01,
    )
    paginator.stage_recorder = harness.translation_diagnostics.record_chatbox_stage
    source_id = uuid4()
    try:
        paginator.enqueue(OSCMessage(uuid4(), "x" * 48, created_at=clock.now()))
        if scoped:
            identity = STTProviderTurnIdentity(
                segment=AudioSegmentIdentity(1, 1, source_id, 1),
                provider_epoch_id="epoch",
                provider_turn_id="turn",
            )
            event = STTProviderTurnUpdate(
                identity,
                1,
                "stable",
                "append",
                "hello",
                contribution=STTTextContribution("hello", 0, 5),
            )
        else:
            event = STTFinalEvent(source_id, Transcript(source_id, "hello", is_final=True))
        await harness.self_owner.handle_stt_event(event)
        buffer = harness.self_owner.merge_buffer
        assert buffer is not None
        assert buffer.awaiting_vad_timeout_task is not None
        await asyncio.wait_for(buffer.awaiting_vad_timeout_task, timeout=1.0)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)
        assert harness.self_owner.merge_buffer is None
        assert sender.sent == ["x" * 24]
        if scoped:
            await harness.self_owner.handle_stt_event(
                STTProviderTurnUpdate(
                    identity,
                    2,
                    "stable",
                    "append",
                    "hello world",
                    contribution=STTTextContribution("world", 5, 11),
                )
            )
            successor = harness.self_owner.merge_buffer
            assert successor is not None
            assert successor.awaiting_vad_timeout_task is not None
            await asyncio.wait_for(successor.awaiting_vad_timeout_task, timeout=1.0)
            await asyncio.wait_for(harness.translation_turns.wait_for_idle(), timeout=1.0)

        clock.advance(0.1)
        await harness.self_owner.handle_vad_event(SpeechEnd(source_id, trailing_silence_ms=600))
        clock.advance(2.9)
        paginator.process_due()

        expected_text = "world (translated)" if scoped else "hello (translated)"
        assert sender.sent[-1] == expected_text
        summaries = [
            message
            for message in _runtime_log_messages(stream)
            if message.startswith("[Basic][Latency]")
        ]
        assert len(summaries) == 1
        assert "last_speech_to_chatbox_send_ms=3500" in summaries[0]
        assert not any("[Metric]" in message for message in _runtime_log_messages(stream))
        assert not harness.translation_diagnostics.snapshot().timeline_keys
    finally:
        await harness.stop()
        logging.close()


@pytest.mark.asyncio
async def test_scoped_successor_before_real_end_uses_post_end_grace() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=10,
        low_latency_awaiting_vad_timeout_s=5.0,
    )
    segment_id = uuid4()
    identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(1, 1, segment_id, 1),
        provider_epoch_id="epoch",
        provider_turn_id="turn",
        settings_scope=("provider", "configuration"),
    )
    contribution_a = STTTextContribution("a", 0, 1)
    contribution_b = STTTextContribution("b", 1, 2)

    await harness.self_owner.handle_stt_event(
        STTProviderTurnUpdate(identity, 1, "stable", "append", "A", contribution=contribution_a)
    )
    first = harness.self_owner.merge_buffer
    assert first is not None
    first.awaiting_vad_end = False
    first.awaiting_vad_utterance_id = None
    await harness.self_owner._commit_merge(first, reason="awaiting_vad_timeout")
    await harness.self_owner.handle_stt_event(
        STTProviderTurnUpdate(identity, 2, "stable", "append", "AB", contribution=contribution_b)
    )
    successor = harness.self_owner.merge_buffer
    assert successor is not None
    await harness.self_owner.handle_stt_event(
        STTProviderTurnTerminal(
            identity,
            "final",
            text="AB",
            text_authority="authoritative",
            included_contributions=(contribution_a, contribution_b),
        )
    )

    await harness.self_owner.handle_vad_event(SpeechEnd(segment_id))

    assert successor.awaiting_vad_end is False
    assert successor.finalize_wait_task is not None
    await asyncio.sleep(0.03)
    assert harness.self_owner.merge_buffer is None


@pytest.mark.asyncio
async def test_recognition_configuration_change_is_merge_barrier_but_epoch_rotation_is_not() -> (
    None
):
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
    )
    first_identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(1, 1, uuid4(), 1),
        provider_epoch_id="epoch-a",
        provider_turn_id="turn-a",
        settings_scope=("provider-a", "configuration-a"),
    )
    same_config_identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(1, 2, uuid4(), 1),
        provider_epoch_id="epoch-b",
        provider_turn_id="turn-b",
        settings_scope=first_identity.settings_scope,
    )
    changed_identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(1, 3, uuid4(), 1),
        provider_epoch_id="epoch-c",
        provider_turn_id="turn-c",
        settings_scope=("provider-b", "configuration-b"),
    )

    await harness.self_owner.handle_stt_event(
        STTProviderTurnUpdate(
            first_identity,
            1,
            "stable",
            "append",
            "one",
            contribution=STTTextContribution("first", 0, 3),
        )
    )
    original = harness.self_owner.merge_buffer
    assert original is not None
    await harness.self_owner.handle_stt_event(
        STTProviderTurnUpdate(
            same_config_identity,
            1,
            "stable",
            "append",
            "two",
            contribution=STTTextContribution("same", 0, 3),
        )
    )
    assert harness.self_owner.merge_buffer is original
    assert original.parts == ["one", "two"]

    await harness.self_owner.handle_stt_event(
        STTProviderTurnUpdate(
            changed_identity,
            1,
            "stable",
            "append",
            "three",
            contribution=STTTextContribution("changed", 0, 5),
        )
    )

    replacement = harness.self_owner.merge_buffer
    assert replacement is not None
    assert replacement is not original
    assert replacement.parts == ["three"]
    assert original.parts == ["one", "two"]


@pytest.mark.asyncio
async def test_self_owner_close_cancels_and_awaits_owned_runtime_tasks() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.self_owner
    utterance_id = uuid4()
    translation_task = asyncio.create_task(asyncio.sleep(60.0))
    spec_task = asyncio.create_task(asyncio.sleep(60.0))
    finalize_task = asyncio.create_task(asyncio.sleep(60.0))
    owner.runtime.translation_tasks[utterance_id] = translation_task
    owner.merge_buffer = _MergeBuffer(
        merge_id=uuid4(),
        utterance_ids=[utterance_id],
        speculative_attempt=make_speculative_attempt(task=spec_task),
        finalize_wait_task=finalize_task,
    )

    await owner.close()

    assert translation_task.done()
    assert spec_task.done()
    assert finalize_task.done()
    assert owner.merge_buffer is None
    assert not owner.accepting_events


@pytest.mark.asyncio
async def test_scoped_readiness_restores_session_state_disclosure_and_failure_status() -> None:
    osc = RecordingOscQueue()
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=osc,
        low_latency_mode=True,
        ui_queue_maxsize=10,
    )
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_self(harness.self_owner)

    class Capture:
        snapshot = SimpleNamespace(generation=1)
        terminals: list[STTProviderTurnTerminal] = []

        def note_recognition_terminal(self, terminal: STTProviderTurnTerminal) -> None:
            self.terminals.append(terminal)

    capture = Capture()
    callbacks._self_capture = capture
    harness.self_owner.mark_promo_eligible()
    peer_id = uuid4()
    peer_bundle = harness.peer_runtime.get_or_create_bundle(peer_id)
    identity = STTProviderTurnIdentity(
        AudioSegmentIdentity(1, 1, uuid4(), 1),
        "epoch",
        "turn",
        ("provider",),
    )
    await callbacks.self_event_handler(
        STTProviderTurnUpdate(
            identity,
            1,
            "stable",
            "append",
            "usable",
            contribution=STTTextContribution("usable", 0, 6),
        )
    )
    await callbacks.self_event_handler(
        STTProviderTurnUpdate(
            identity,
            2,
            "stable",
            "append",
            "usable",
            contribution=STTTextContribution("usable", 0, 6),
        )
    )

    assert harness.stt_session_state() is STTSessionState.STREAMING
    assert osc.immediate_messages == ["PuriPuly ON!"]
    terminal = STTProviderTurnTerminal(
        identity,
        "degraded",
        text="usable",
        text_authority="degraded",
        failure_reason="provider_final_timeout",
        epoch_disposition="retire",
    )
    await callbacks.self_event_handler(terminal)

    assert capture.terminals == [terminal]
    assert harness.stt_session_state() is STTSessionState.DISCONNECTED
    ui_events = []
    while not harness.ui_events.empty():
        ui_events.append(harness.ui_events.get_nowait())
    assert UIEventType.SESSION_STATE_CHANGED in {event.type for event in ui_events}
    error_events = [event for event in ui_events if event.type is UIEventType.ERROR]
    assert len(error_events) == 1
    assert isinstance(error_events[0].payload, UserErrorReport)
    assert error_events[0].payload.message.key == "stt.failure"
    assert osc.immediate_messages == ["PuriPuly ON!"]
    await harness.self_owner.submit_text("manual-after-recognition-failure")
    assert harness.peer_runtime.get_or_create_bundle(peer_id) is peer_bundle
    assert any(message.text == "manual-after-recognition-failure" for message in osc.messages)


@pytest.mark.asyncio
async def test_recoverable_self_terminal_restores_streaming_without_user_error() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        ui_queue_maxsize=20,
    )
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_self(harness.self_owner)
    failed_identity = STTProviderTurnIdentity(
        AudioSegmentIdentity(1, 1, uuid4(), 1), "failed-epoch", "failed-turn"
    )
    await callbacks.self_event_handler(
        STTProviderTurnTerminal(
            failed_identity,
            "failed",
            failure_reason="soniox_receive_failed",
            failure_retryable=True,
            recovery_pending=True,
            epoch_disposition="retire",
        )
    )
    assert harness.stt_session_state() is STTSessionState.DISCONNECTED
    transient_events = []
    while not harness.ui_events.empty():
        transient_events.append(harness.ui_events.get_nowait())
    assert not [event for event in transient_events if event.type is UIEventType.ERROR]

    success_id = uuid4()
    await callbacks.self_event_handler(
        STTProviderTurnTerminal(
            STTProviderTurnIdentity(
                AudioSegmentIdentity(1, 2, success_id, 1), "new-epoch", "new-turn"
            ),
            "final",
            text="later success",
            text_authority="authoritative",
        )
    )
    assert harness.stt_session_state() is STTSessionState.STREAMING
    events = []
    while not harness.ui_events.empty():
        events.append(harness.ui_events.get_nowait())
    assert not [event for event in events if event.type is UIEventType.ERROR]
    assert any(
        event.type is UIEventType.TRANSCRIPT_FINAL
        and event.payload.utterance_id == success_id
        and event.payload.text == "later success"
        for event in events
    )

    await callbacks.self_event_handler(
        STTProviderTurnTerminal(
            STTProviderTurnIdentity(
                AudioSegmentIdentity(1, 3, uuid4(), 1), "exhausted-epoch", "last-turn"
            ),
            "failed",
            failure_reason="soniox_receive_failed",
            failure_retryable=True,
            epoch_disposition="retire",
        )
    )
    final_events = []
    while not harness.ui_events.empty():
        final_events.append(harness.ui_events.get_nowait())
    assert len([event for event in final_events if event.type is UIEventType.ERROR]) == 1


@pytest.mark.asyncio
async def test_old_self_terminal_cannot_disconnect_reactivated_session_state() -> None:
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), ui_queue_maxsize=10
    )
    callbacks = TranslationChannelOwnerCallbacks(harness.stt_sessions)
    callbacks.bind_self(harness.self_owner)

    class Capture:
        snapshot = SimpleNamespace(generation=2)
        terminals: list[STTProviderTurnTerminal] = []

        def note_recognition_terminal(self, terminal: STTProviderTurnTerminal) -> None:
            self.terminals.append(terminal)

    capture = Capture()
    callbacks._self_capture = capture
    await callbacks.self_event_handler(
        STTProviderTurnUpdate(
            STTProviderTurnIdentity(
                AudioSegmentIdentity(2, 1, uuid4(), 1), "new-epoch", "new-turn"
            ),
            1,
            "stable",
            "append",
            "usable",
        )
    )
    assert harness.stt_session_state() is STTSessionState.STREAMING
    old_terminal = STTProviderTurnTerminal(
        STTProviderTurnIdentity(AudioSegmentIdentity(1, 1, uuid4(), 1), "old-epoch", "old-turn"),
        "failed",
        failure_reason="soniox_receive_failed",
        recovery_pending=True,
    )
    await callbacks.self_event_handler(old_terminal)
    assert capture.terminals == [old_terminal]
    assert harness.stt_session_state() is STTSessionState.STREAMING


def test_self_owner_requires_self_runtime() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=RecordingOscQueue())
    owner = harness.self_owner

    with pytest.raises(ValueError, match="Self channel"):
        type(owner)(
            runtime=harness.peer_runtime,
            config_snapshot=owner.config_snapshot,
            translation_turns=owner.translation_turns,
            local_asr_runtime=owner.local_asr_runtime,
            translation_requests=owner.translation_requests,
            output_projection=owner.output_projection,
            diagnostics=owner.diagnostics,
            clock=owner.clock,
        )
