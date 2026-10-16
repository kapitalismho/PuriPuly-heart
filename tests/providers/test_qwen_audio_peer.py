from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.stt.backend import (
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.providers.stt.qwen_audio import QwenAudioStreamingSTTBackend
from tests.core.test_stt_scoped_engine import (
    ControlledMonotonicClock,
    segment_events,
    settings,
    wait_until,
    watchdogs,
)
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.providers.test_qwen_audio import SocketFactory, control, native


async def make_engine(channel, *, sink=None, factory=None, clock=None):
    factory = factory or SocketFactory()
    sessions = []

    async def open_owned(_settings, epoch):
        backend = QwenAudioStreamingSTTBackend(
            api_key="offline",
            channel=channel,
            language_hints=("en",),
            websocket_factory=factory,
            task_start_timeout_s=0.2,
            task_finish_timeout_s=0.5,
            send_timeout_s=0.2,
        )
        session = await backend.open_session(projection=STTSessionProjection("scoped", epoch))
        sessions.append(session)
        return session

    options = {"monotonic_clock": clock.now, "sleep": clock.sleep} if clock else {}
    engine = ScopedRecognitionEngine(
        channel=channel,
        session_factory=open_owned,
        event_sink=sink,
        watchdog_resolver=lambda _: watchdogs(final_timeout_s=0.8),
        **options,
    )
    return engine, factory, sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", [False, True])
async def test_self_successor_stable_waits_for_predecessor_and_merge_preserves_native_order(
    prefix, monkeypatch
):
    harness = compose_translation_test_harness(
        stt=None,
        llm=None,
        osc=RecordingOscQueue(),
        clock=FakeClock(),
        low_latency_mode=True,
        low_latency_awaiting_vad_timeout_s=60,
        low_latency_finalize_wait_ms=500,
    )
    owner = harness.self_owner
    emitted = []

    finals = []
    original_transcript = type(owner)._handle_transcript

    async def observe_transcript(current_owner, transcript, **kwargs):
        if current_owner is owner and kwargs.get("is_final"):
            finals.append(transcript)
        await original_transcript(current_owner, transcript, **kwargs)

    monkeypatch.setattr(type(owner), "_handle_transcript", observe_transcript)

    async def sink(event):
        emitted.append(event)
        await owner.handle_stt_event(event)

    engine, factory, _sessions = await make_engine("self", sink=sink)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("qwen_audio"))
    a_start, _, a_end = segment_events(ledger, start_sample=0, now=0)
    b_start, _, b_end = segment_events(ledger, start_sample=100, now=1)
    try:
        await harness.start()
        await engine.handle_owned_vad_event(a_start)
        a = factory.sockets[0]
        if prefix:
            await a.push(native(a.task_ids[0], 1, " A prefix "))
            await wait_until(lambda: owner.merge_buffer is not None)
            assert owner.merge_buffer.parts == [" A prefix "]
        await engine.handle_owned_vad_event(a_end)
        await owner.handle_vad_event(a_end.event)
        await engine.handle_owned_vad_event(b_start)
        b = factory.sockets[1]
        await b.push(native(b.task_ids[0], 1, " B stable "))
        await b.push(native(b.task_ids[0], 2, " B tail "))
        await engine.handle_owned_vad_event(b_end)
        await owner.handle_vad_event(b_end.event)
        await b.push(control(b.task_ids[0], "task-finished"))
        await wait_until(lambda: any(turn.terminal_ready.done() for turn in engine._turns.values()))
        assert not any(
            event.identity.segment.segment_id == b_start.event.utterance_id for event in emitted
        )
        assert (owner.merge_buffer.parts if owner.merge_buffer is not None else []) == (
            [" A prefix "] if prefix else []
        )
        await a.push(native(a.task_ids[0], 2 if prefix else 1, " A drain "))
        await a.push(control(a.task_ids[0], "task-finished"))
        await wait_until(
            lambda: sum(isinstance(event, STTProviderTurnTerminal) for event in emitted) == 2
        )
        expected_a = " A prefix  A drain " if prefix else " A drain "
        assert owner.merge_buffer.parts == [expected_a, " B stable  B tail "]
        assert [event.text for event in emitted if isinstance(event, STTProviderTurnTerminal)] == [
            expected_a,
            " B stable  B tail ",
        ]
        await wait_until(lambda: len(finals) == 1, timeout=2)
        assert len(finals) == 1
        final = finals[0]
        assert final.text == expected_a + " B stable  B tail "
        assert [
            final.text[origin.text_start : origin.text_end] for origin in final.recognition_origins
        ] == ([" A prefix "] if prefix else []) + [" A drain ", " B stable ", " B tail "]
    finally:
        await engine.close()
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_first", [False, True])
async def test_peer_completed_segments_only_reach_real_owner_in_source_order_with_origins(
    fail_first,
):
    harness = compose_translation_test_harness(
        stt=None, llm=None, osc=RecordingOscQueue(), clock=FakeClock()
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("qwen_audio"))
    admitted = []

    async def sink(event):
        if not isinstance(event, STTProviderTurnTerminal):
            pytest.fail("Peer native sentences must not reach translation admission")
        receipt = ledger.terminalize(
            event.identity.segment.segment_id,
            outcome=event.outcome,
            now_monotonic_s=2,
            provider_epoch_id=event.identity.provider_epoch_id,
            provider_turn_id=event.identity.provider_turn_id,
            text_authority=event.text_authority,
            failure_reason=event.failure_reason,
        )
        result = await harness.peer_owner.handle_provider_turn_terminal(receipt, event)
        if result is not None:
            admitted.append(result.transcript)

    engine, factory, _ = await make_engine("peer", sink=sink)
    a_start, _, a_end = segment_events(ledger, start_sample=0, now=0)
    b_start, _, b_end = segment_events(ledger, start_sample=100, now=1)
    try:
        await harness.start()
        harness.output_runtime.activate_peer_generation(1)
        await engine.handle_owned_vad_event(a_start)
        a = factory.sockets[0]
        if not fail_first:
            await a.push(native(a.task_ids[0], 1, " equal "))
        await engine.handle_owned_vad_event(a_end)
        await engine.handle_owned_vad_event(b_start)
        b = factory.sockets[1]
        await b.push(native(b.task_ids[0], 1, " equal "))
        await b.push(native(b.task_ids[0], 2, " equal "))
        await engine.handle_owned_vad_event(b_end)
        await b.push(control(b.task_ids[0], "task-finished"))
        await wait_until(lambda: any(turn.terminal_ready.done() for turn in engine._turns.values()))
        assert admitted == []
        if fail_first:
            await a.push(control(a.task_ids[0], "task-failed", error_code="500"))
        else:
            await a.push(native(a.task_ids[0], 2, " tail "))
            await a.push(control(a.task_ids[0], "task-finished"))
        await wait_until(lambda: len(admitted) == (1 if fail_first else 2))
        assert [transcript.text for transcript in admitted] == (
            [" equal  equal "] if fail_first else [" equal  tail ", " equal  equal "]
        )
        assert [transcript.source_order for transcript in admitted] == (
            [2] if fail_first else [1, 2]
        )
        assert all(len(transcript.recognition_origins) == 2 for transcript in admitted)
        assert harness.osc.messages == [] and harness.osc.immediate_messages == []
    finally:
        await engine.close()
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_two_logical_turn_limit_holds_completed_successor_until_predecessor_resolves(channel):
    emitted = []
    engine, factory, _ = await make_engine(channel, sink=emitted.append)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("qwen_audio"))
    segments = [
        segment_events(ledger, start_sample=index * 100, now=float(index)) for index in range(3)
    ]
    third_start = None
    try:
        for start, _, end in segments[:2]:
            await engine.handle_owned_vad_event(start)
            await engine.handle_owned_vad_event(end)
        third_start = asyncio.create_task(engine.handle_owned_vad_event(segments[2][0]))
        await asyncio.sleep(0.01)
        a, b = factory.sockets
        await b.push(native(b.task_ids[0], text="B"))
        await b.push(control(b.task_ids[0], "task-finished"))
        await asyncio.sleep(0.01)
        assert not third_start.done() and len(engine._turns) == 2
        assert (
            len(factory.sockets) == 2
            and sum(len(socket.task_ids) for socket in factory.sockets) == 2
        )
        await a.push(control(a.task_ids[0], "task-finished"))
        await third_start
        await engine.handle_owned_vad_event(segments[2][2])
        assert len(factory.sockets) == 2 and len(a.task_ids) == 2
        await a.push(control(a.task_ids[-1], "task-finished"))
        await wait_until(
            lambda: sum(isinstance(event, STTProviderTurnTerminal) for event in emitted) == 3
        )
    finally:
        if third_start is not None and not third_start.done():
            third_start.cancel()
            await asyncio.gather(third_start, return_exceptions=True)
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("reason", ["toggle_off", "configuration_replaced", "source_discontinuity"])
async def test_abort_invalidates_both_slots_then_new_scope_recovers_without_old_audio(
    channel, reason
):
    emitted = []
    engine, factory, _ = await make_engine(channel, sink=emitted.append)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("qwen_audio"))
    try:
        for index in range(2):
            start, _, end = segment_events(ledger, start_sample=index * 100, now=index)
            await engine.handle_owned_vad_event(start)
            await engine.handle_owned_vad_event(end)
        old = list(factory.sockets)
        if channel == "self":
            await old[1].push(native(old[1].task_ids[0], text="queued revoked update"))
            await wait_until(
                lambda: any(turn.pending_update is not None for turn in engine._turns.values())
            )
        await engine.abort(reason=reason)
        await wait_until(lambda: all(socket.closed for socket in old))
        for socket in old:
            await socket.push(native(socket.task_ids[0], text="obsolete"))
            await socket.push(control(socket.task_ids[0], "task-finished"))
        fresh_ledger = PeerAudioSegmentLedger(
            activation_generation=2, settings=settings("qwen_audio", signature="new")
        )
        start, _, end = segment_events(fresh_ledger, start_sample=1000, now=3)
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(end)
        fresh = factory.sockets[-1]
        await fresh.push(native(fresh.task_ids[0], text="fresh"))
        await fresh.push(control(fresh.task_ids[0], "task-finished"))
        await wait_until(
            lambda: any(
                isinstance(event, STTProviderTurnTerminal) and event.text == "fresh"
                for event in emitted
            )
        )
        assert not any(
            "obsolete" in event.text
            for event in emitted
            if isinstance(event, (STTProviderTurnTerminal, STTProviderTurnUpdate))
        )
        assert all(
            event.text == ""
            for event in emitted
            if isinstance(event, STTProviderTurnTerminal) and event.outcome == "cancelled"
        )
        assert not any(
            "queued revoked update" in event.text
            for event in emitted
            if isinstance(event, STTProviderTurnUpdate)
        )
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_close_drains_delayed_tail_before_retiring_authority(channel):
    events = []
    engine, factory, _ = await make_engine(channel, sink=events.append)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("qwen_audio"))
    for index in range(2):
        start, _, end = segment_events(ledger, start_sample=index * 100, now=index)
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(end)
    closing = asyncio.create_task(engine.close())
    await asyncio.sleep(0)
    for socket in reversed(factory.sockets):
        await socket.push(native(socket.task_ids[0], text=" tail "))
        await socket.push(control(socket.task_ids[0], "task-finished"))
    await closing
    assert [event.text for event in events if isinstance(event, STTProviderTurnTerminal)] == [
        " tail ",
        " tail ",
    ]
    assert all(socket.closed for socket in factory.sockets)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_source_idle_retires_without_empty_tasks_or_filler_audio(channel):
    clock = ControlledMonotonicClock()
    events = []
    engine, factory, _ = await make_engine(channel, sink=events.append, clock=clock)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("qwen_audio"))
    start, _, end = segment_events(ledger, start_sample=0, now=0)
    try:
        await engine.observe_source_activity(speech_observed=True, observed_at_monotonic_s=0)
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(end)
        socket = factory.sockets[0]
        await socket.push(native(socket.task_ids[0], text="last"))
        await socket.push(control(socket.task_ids[0], "task-finished"))
        await wait_until(lambda: len(engine._turns) == 0)
        await engine.observe_source_activity(speech_observed=False, observed_at_monotonic_s=0)
        pcm = [value for value in socket.sent if isinstance(value, bytes)]
        await clock.advance_to(59.9)
        assert not socket.closed
        await clock.advance_to(60.0)
        await wait_until(lambda: socket.closed)
        assert [value for value in socket.sent if isinstance(value, bytes)] == pcm
        assert len(socket.task_ids) == 1
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_self_ordered_pending_stable_survives_engine_timeout_as_degraded_once():
    events = []
    engine, factory, _ = await make_engine("self", sink=events.append)
    engine.watchdog_resolver = lambda _: watchdogs(final_timeout_s=0.03)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("qwen_audio"))
    try:
        for index in range(2):
            start, _, end = segment_events(ledger, start_sample=index * 100, now=index)
            await engine.handle_owned_vad_event(start)
            await engine.handle_owned_vad_event(end)
        successor = factory.sockets[1]
        await successor.push(native(successor.task_ids[0], text=" retained stable "))
        await wait_until(
            lambda: any(turn.pending_update is not None for turn in engine._turns.values())
        )
        assert events == []
        await wait_until(
            lambda: sum(isinstance(event, STTProviderTurnTerminal) for event in events) == 2
        )
        terminals = [event for event in events if isinstance(event, STTProviderTurnTerminal)]
        assert [(event.outcome, event.text) for event in terminals] == [
            ("failed", ""),
            ("degraded", " retained stable "),
        ]
        from puripuly_heart.core.stt.backend import STTContributionConsumptionLedger

        ledger_consumption = STTContributionConsumptionLedger()
        assert "".join(ledger_consumption.consume(event) for event in events) == " retained stable "
        for socket in factory.sockets:
            await socket.push(control(socket.task_ids[0], "task-finished"))
    finally:
        await engine.close()
