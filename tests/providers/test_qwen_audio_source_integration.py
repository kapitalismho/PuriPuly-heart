from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.stt.backend import (
    STTProviderAudioBoundary,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
)
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.recognition import validate_recognition_origins
from puripuly_heart.providers.stt.qwen_audio import QWEN_AUDIO_MODEL
from tests.core.test_source_context_contract import ControlledProvider, harness_for
from tests.core.test_streaming_speaker import evidence
from tests.core.test_stt_scoped_engine import segment_events, settings, span, wait_until
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.providers.test_qwen_audio import control, native, open_session, request, seal, send
from tests.providers.test_qwen_audio_delivery_recovery import SelfSourceObserver
from tests.providers.test_qwen_audio_peer import make_engine


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_mapping", [True, False])
async def test_wrapped_two_slot_peer_retains_native_regions_short_no_and_order(complete_mapping):
    harness, trace = harness_for(ControlledProvider())
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("qwen_audio"))
    terminals = []
    admitted = []

    async def sink(event):
        assert not isinstance(event, STTProviderTurnUpdate)
        if isinstance(event, STTProviderAudioBoundary):
            harness.peer_owner.note_audio_boundary(event.identity, event.boundary)
            return
        if not isinstance(event, STTProviderTurnTerminal):
            return
        terminals.append(event)
        receipt = ledger.terminalize(
            event.identity.segment.segment_id,
            outcome=event.outcome,
            now_monotonic_s=3,
            provider_epoch_id=event.identity.provider_epoch_id,
            provider_turn_id=event.identity.provider_turn_id,
            text_authority=event.text_authority,
        )
        result = await harness.peer_owner.handle_provider_turn_terminal(receipt, event)
        if result is not None:
            admitted.append(result.transcript)

    engine, factory, _ = await make_engine("peer", sink=sink)
    original_factory = engine.session_factory
    wrappers = []

    async def wrapped_factory(snapshot, epoch):
        inner = await original_factory(snapshot, epoch)
        inner.backend.endpoint = "wss://controlled.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
        wrapper = StreamingSpeakerSession(inner)
        wrappers.append(wrapper)
        return wrapper

    engine.session_factory = wrapped_factory
    try:
        await harness.start()
        for start_sample in (0, 16000):
            segment_id = uuid4()
            start = ledger.observe_vad_event(
                SpeechStart(
                    segment_id,
                    np.empty(0, dtype=np.float32),
                    np.full(16000, 0.1, dtype=np.float32),
                    chunk_capture=(span(start_sample, start_sample, start_sample + 16000),),
                ),
                now_monotonic_s=start_sample / 16000,
            )
            end = ledger.observe_vad_event(
                SpeechEnd(segment_id), now_monotonic_s=start_sample / 16000 + 1
            )
            await engine.handle_owned_vad_event(start)
            await engine.handle_owned_vad_event(end)
        assert len(factory.sockets) == 2
        await wrappers[0].observe_speaker_evidence(
            tuple(
                replace(
                    item,
                    source_map=tuple(
                        replace(piece, capture=replace(piece.capture, capture_epoch=1))
                        for piece in item.source_map
                    ),
                )
                for item in (
                    evidence(0, 8000, 0),
                    evidence(8000, 16000, 1),
                    evidence(16000, 32000, 0),
                )
            )
        )
        first, second = factory.sockets
        await second.push(native(second.task_ids[0], text="tail", begin_time=0, end_time=1000))
        await second.push(control(second.task_ids[0], "task-finished"))
        await wait_until(lambda: any(turn.terminal_ready.done() for turn in engine._turns.values()))
        assert admitted == []
        await first.push(
            native(
                first.task_ids[0],
                1,
                "A",
                begin_time=0,
                end_time=500,
                words=[{"text": "A", "begin_time": 0, "end_time": 500}],
            )
        )
        await first.push(
            native(
                first.task_ids[0],
                2,
                "No.",
                begin_time=500,
                end_time=1000 if complete_mapping else 1100,
                words=[
                    {
                        "text": "No",
                        "punctuation": ".",
                        "begin_time": 500,
                        "end_time": 1000 if complete_mapping else 1100,
                    }
                ],
            )
        )
        await first.push(control(first.task_ids[0], "task-finished"))
        await wait_until(lambda: len(admitted) == 2)
        await harness.translation_turns.wait_for_idle()
        assert [event.text for event in terminals] == ["A No.", "tail"]
        assert [transcript.text for transcript in admitted] == ["A No.", "tail"]
        for transcript in admitted:
            validate_recognition_origins(transcript.recognition_origins, transcript.text)
        first_origins = admitted[0].recognition_origins
        assert [item.source_binding for item in first_origins] == [
            "local_segment",
            "composition",
            "local_segment",
        ]
        assert [
            item.native.native_sentence_id for item in first_origins if item.native is not None
        ] == ["1", "2"]
        assert first_origins[0].request_id == terminals[0].identity.provider_turn_id
        assert (
            first_origins[0].ownership == "request"
            and first_origins[0].order_basis == "source_order"
        )
        assert all(not item.audio_clocks for item in terminals[0].provenance)
        timings = [timing for item in terminals[0].provenance for timing in item.text_timing]
        assert [
            (timing.raw_start, timing.raw_end)
            for timing in timings
            if timing.semantics == "word_interval"
        ] == [(0, 500), (500, 1000 if complete_mapping else 1100)]
        assert [item.source_text for item in trace.items] == ["A", " No.", "tail"]
        assert trace.items[0].source_child.speaker.speaker_id == "0"
        correction = trace.items[1].source_child.speaker
        assert correction.speaker_id == ("1" if complete_mapping else None)
        assert correction.attribution_state == ("identified" if complete_mapping else "unavailable")
        assert [entry.text for entry in harness.peer_runtime.translation_history] == [
            "A",
            " No.",
            "tail",
        ]
        assert [item.context_texts for item in trace.items] == [(), ("A",), ("A", " No.")]
        assert harness.osc.messages == []
    finally:
        await engine.close()
        await harness.stop()
    assert all(socket.closed for socket in factory.sockets)


@pytest.mark.asyncio
async def test_qwen_self_task_and_cross_source_join_separators_keep_real_origins():
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=100,
    )
    observer = SelfSourceObserver(harness)
    engine, factory, _ = await make_engine("self", sink=harness.self_owner.handle_stt_event)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("qwen_audio"))
    try:
        await harness.start()
        for index in range(2):
            start, _, end = segment_events(ledger, start_sample=index * 100, now=index)
            await engine.handle_owned_vad_event(start)
            await engine.handle_owned_vad_event(end)
            await harness.self_owner.handle_vad_event(end.event)
        first, second = factory.sockets
        await second.push(native(second.task_ids[0], 1, "Three"))
        await second.push(control(second.task_ids[0], "task-finished"))
        await first.push(native(first.task_ids[0], 1, "One"))
        await first.push(native(first.task_ids[0], 2, "Two"))
        await first.push(control(first.task_ids[0], "task-finished"))
        await wait_until(lambda: bool(observer.final_sources()), timeout=2)
        (source,) = observer.final_sources()
        assert source.text == "One Two Three"
        validate_recognition_origins(source.recognition_origins, source.text)
        assert [
            source.text[item.text_start : item.text_end] for item in source.recognition_origins
        ] == ["One", " ", "Two", " ", "Three"]
        assert [
            item.native.native_task_id
            for item in source.recognition_origins
            if item.native is not None
        ] == [first.task_ids[0], first.task_ids[0], second.task_ids[0]]
    finally:
        await engine.close()
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "model", "policy"),
    [
        ("wss://owned.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference", QWEN_AUDIO_MODEL, "timed"),
        ("wss://dashscope.aliyuncs.com/api-ws/v1/inference", QWEN_AUDIO_MODEL, "unknown"),
        (
            "wss://owned.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/inference",
            QWEN_AUDIO_MODEL,
            "unknown",
        ),
        ("wss://owned.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference", "other-model", "unknown"),
        (
            "ws://owned.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
            QWEN_AUDIO_MODEL,
            "unknown",
        ),
        (
            "wss://owned.cn-beijing.maas.aliyuncs.com:443/api-ws/v1/inference",
            QWEN_AUDIO_MODEL,
            "unknown",
        ),
        (
            "wss://owned.cn-beijing.maas.aliyuncs.com:bad/api-ws/v1/inference",
            QWEN_AUDIO_MODEL,
            "unknown",
        ),
        (
            "wss://owned.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference?x=1",
            QWEN_AUDIO_MODEL,
            "unknown",
        ),
        (
            "wss://user@owned.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
            QWEN_AUDIO_MODEL,
            "unknown",
        ),
    ],
)
async def test_timing_policy_snapshots_actual_task_model_and_connection_region(
    endpoint, model, policy
):
    session, factory = await open_session("peer", endpoint=endpoint, model=model)
    req = request("peer")
    try:
        await session.begin_turn(req)
        await send(session, req)
        session.backend.endpoint = (
            "wss://changed.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/inference"
        )
        session.backend.model = "changed-model"
        await seal(session, req)
        socket = factory.sockets[0]
        await socket.push(native(socket.task_ids[0], text="No.", begin_time=0, end_time=500))
        await socket.push(control(socket.task_ids[0], "task-finished"))
        events = session.turn_events()
        terminal = await anext(events)
        assert isinstance(terminal, STTProviderTurnTerminal)
        provenance = next(item for item in terminal.provenance if item.transcription is not None)
        assert provenance.capabilities.policy == policy
        assert provenance.capabilities.model_id == model
        assert provenance.capabilities.optional_progress is False
        assert provenance.audio_clocks == ()
        assert provenance.transcription.audio_coverage
        assert provenance.text_timing[0].clock_id == socket.task_ids[0]
        assert factory.options[0][0] == (endpoint,)
        await events.aclose()
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intervals", "policies"),
    [
        (((0, 500), (500, 1000), (1000, 1500)), ("timed", "timed", "timed")),
        (((500, 1000), (0, 500), (1000, 1500)), ("timed", "unknown", "unknown")),
        (((0, 1000), (500, 750), (1000, 1500)), ("timed", "unknown", "unknown")),
        (((0, 500), (500, 500), (1000, 1500)), ("timed", "unknown", "unknown")),
        (((0, 500), (500, None), (1000, 1500)), ("timed", "unknown", "unknown")),
    ],
    ids=["valid", "regressed-start", "regressed-end", "invalid-interval", "missing-end"],
)
async def test_task_clock_authority_latches_until_new_task_with_useful_recovery(
    intervals, policies
):
    inner, factory = await open_session(
        "peer", endpoint="wss://controlled.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
    )
    wrapper = StreamingSpeakerSession(inner)
    stream = wrapper.turn_events()

    async def next_terminal():
        while True:
            event = await asyncio.wait_for(anext(stream), 1)
            if isinstance(event, STTProviderTurnTerminal):
                return event

    try:
        first = request("peer")
        await wrapper.begin_turn(first)
        await send(wrapper, first, count=24000)
        await wrapper.observe_speaker_evidence(
            tuple(
                replace(
                    item,
                    source_map=tuple(
                        replace(piece, capture=replace(piece.capture, capture_epoch=1))
                        for piece in item.source_map
                    ),
                )
                for item in (
                    evidence(0, 8000, 0),
                    evidence(8000, 16000, 1),
                    evidence(16000, 32000, 0),
                    evidence(32000, 40000, 1),
                )
            )
        )
        await seal(wrapper, first)
        socket = factory.sockets[0]
        for sentence_id, (begin, end) in enumerate(intervals, 1):
            await socket.push(
                native(
                    socket.task_ids[0],
                    sentence_id,
                    f"Sentence{sentence_id}",
                    begin_time=begin,
                    end_time=end,
                )
            )
        await socket.push(control(socket.task_ids[0], "task-finished"))
        terminal = await next_terminal()
        assert terminal.text == "Sentence1 Sentence2 Sentence3"
        assert terminal.outcome == "final" and terminal.text_authority == "authoritative"
        native_events = [item for item in terminal.provenance if item.transcription is not None]
        assert tuple(item.capabilities.policy for item in native_events) == policies
        assert [
            (item.transcription.begin_time_ms, item.transcription.end_time_ms)
            for item in native_events
        ] == list(intervals)
        assert [item.transcription.native_sentence_id for item in native_events] == ["1", "2", "3"]
        validate_recognition_origins(terminal.recognition_origins, terminal.text)
        if policies[-1] == "timed":
            assert [run.speaker_id for run in terminal.final_speaker_runs] == ["0", "1", "0"]
        else:
            assert all(run.speaker_id is None for run in terminal.final_speaker_runs)
        second = request("peer", order=2)
        await wrapper.begin_turn(second)
        await send(wrapper, second, start=24000, count=16000)
        await seal(wrapper, second)
        successor = factory.sockets[1]
        for sentence_id, begin in ((1, 0), (2, 500)):
            await successor.push(
                native(
                    successor.task_ids[0],
                    sentence_id,
                    f"Next{sentence_id}",
                    begin_time=begin,
                    end_time=begin + 500,
                )
            )
        await successor.push(control(successor.task_ids[0], "task-finished"))
        recovered = await next_terminal()
        assert recovered.text == "Next1 Next2" and recovered.outcome == "final"
        assert [
            item.capabilities.policy
            for item in recovered.provenance
            if item.transcription is not None
        ] == ["timed", "timed"]
        assert [run.speaker_id for run in recovered.final_speaker_runs] == ["0", "1"]
        assert inner.task_finished_count == 2
    finally:
        await wrapper.close()
        await stream.aclose()
    assert all(socket.closed for socket in factory.sockets)
