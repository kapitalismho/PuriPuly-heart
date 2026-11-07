from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTProviderEpochEnded,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTScopedWriteFailure,
    scoped_event_for_delivery,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.providers.stt import soniox
from tests.core.test_stt_scoped_engine import segment_events, settings, wait_until
from tests.helpers.scoped_provider_sessions import _soniox_session
from tests.providers.test_soniox_reuse import _next, _request, _seal, _session


def tokens(count: int, *, width: int = 6):
    return [
        {
            "text": f"{index:04d} ".ljust(width),
            "is_final": True,
            "language": "en",
            "speaker": "A",
            "start_ms": index * 10,
            "end_ms": (index + 1) * 10,
        }
        for index in range(count)
    ]


def message(items):
    return json.dumps({"request_id": "bounded-tape", "tokens": items})


FIN = {"text": "<fin>", "is_final": True}


async def engine_fixture(sink, *, channel="peer"):
    sessions = []
    sockets = []

    async def factory(_settings, epoch):
        session, socket = _soniox_session(epoch=epoch)
        sessions.append(session)
        sockets.append(socket)
        return session

    engine = ScopedRecognitionEngine(session_factory=factory, channel=channel)
    engine.bind_event_sink(sink)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
    start, chunk, end = segment_events(ledger, start_sample=0, now=100.0)
    await engine.handle_owned_vad_event(start)
    await engine.handle_owned_vad_event(chunk)
    return engine, sessions, sockets, ledger, end


@pytest.mark.asyncio
@pytest.mark.parametrize("buffered", [False, True], ids=["single-burst", "buffered-responses"])
async def test_both_buffers_conserve_token_updates_terminal_and_response_time(
    buffered, monkeypatch
):
    received = []
    clock = [100.0]
    monkeypatch.setattr(soniox, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    engine, sessions, sockets, _, end = await engine_fixture(received.append)
    ending = asyncio.create_task(engine.handle_owned_vad_event(end))
    try:
        await wait_until(lambda: sessions[0]._event_projection.sealed)
        pieces = tokens(600)
        if buffered:
            for item in pieces:
                sockets[0].push(message([item]))
            sockets[0].push(message([FIN]))
        else:
            sockets[0].push(message([*pieces, FIN]))
        await asyncio.wait_for(ending, timeout=3)
        await asyncio.wait_for(engine.wait_for_event_ingress_drain(), timeout=3)
        updates = [event for event in received if isinstance(event, STTProviderTurnUpdate)]
        terminals = [event for event in received if isinstance(event, STTProviderTurnTerminal)]
        expected = "".join(item["text"] for item in pieces).strip()
        assert [event.text for event in updates] == [
            "".join(item["text"] for item in pieces[: index + 1]).strip()
            for index in range(len(pieces))
        ]
        assert len(terminals) == 1
        terminal = terminals[0]
        assert (terminal.outcome, terminal.text, terminal.text_authority) == (
            "final",
            expected,
            "authoritative",
        )
        ledger = STTContributionConsumptionLedger()
        assert "".join(ledger.consume(event) for event in [*updates, terminal]) == expected
        assert ledger.consume(terminal) == ""
        assert len({event.contribution.contribution_id for event in updates}) == len(pieces)
        assert all(event.observed_at == event.first_nonempty_at == 100.0 for event in updates)
        assert all(event.provenance.native_request_id == "bounded-tape" for event in updates)
        assert "".join(run.text for run in terminal.final_language_runs) == expected
        assert "".join(run.text for run in terminal.final_speaker_runs) == expected
        assert {run.session_scope for run in terminal.final_speaker_runs} == {
            sessions[0].speaker_session_scope
        }
        assert terminal.provenance[-1].barrier == "manual_finalize"
        assert sessions[0]._event_projection._scoped_buffer().max_events == 512
        assert engine._event_buffer.max_events == 512
        assert not any(isinstance(event, STTProviderEpochEnded) for event in received)
    finally:
        ending.cancel()
        await asyncio.gather(ending, return_exceptions=True)
        await engine.close()
        assert all(socket.closed for socket in sockets)


@pytest.mark.asyncio
async def test_byte_pressure_waits_for_legitimate_consumer_progress_without_losing_text():
    gate = asyncio.Event()
    entered = asyncio.Event()
    received = []

    async def sink(event):
        entered.set()
        await gate.wait()
        received.append(event)

    engine, sessions, sockets, _, end = await engine_fixture(sink)
    ending = asyncio.create_task(engine.handle_owned_vad_event(end))
    try:
        await wait_until(lambda: sessions[0]._event_projection.sealed)
        pieces = tokens(100, width=1024)
        sockets[0].push(message([*pieces, FIN]))
        await asyncio.wait_for(entered.wait(), timeout=1)
        await wait_until(lambda: engine._event_buffer._recognition_bytes > 900_000)
        assert engine._event_buffer.depth < engine._event_buffer.max_events
        assert (
            engine._event_buffer._recognition_bytes <= STTProviderEventBuffer.MAX_RECOGNITION_BYTES
        )
        assert not ending.done()
        gate.set()
        await asyncio.wait_for(ending, timeout=3)
        await asyncio.wait_for(engine.wait_for_event_ingress_drain(), timeout=3)
        terminal = [event for event in received if isinstance(event, STTProviderTurnTerminal)]
        assert len(terminal) == 1
        assert terminal[0].outcome == "final"
        assert terminal[0].text == "".join(item["text"] for item in pieces).strip()
        assert [
            event.sequence for event in received if isinstance(event, STTProviderTurnUpdate)
        ] == list(range(1, 101))
        assert engine._event_buffer._recognition_bytes == 0
    finally:
        gate.set()
        ending.cancel()
        await asyncio.gather(ending, return_exceptions=True)
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["abort", "close"])
@pytest.mark.parametrize("waiting", ["updates", "terminal"])
async def test_permanently_stopped_consumer_cannot_hold_terminal_drain(operation, waiting):
    never = asyncio.Event()
    entered = asyncio.Event()
    delivered = []

    async def sink(event):
        entered.set()
        await never.wait()
        current = scoped_event_for_delivery(event)
        if current is not None:
            delivered.append(current)

    engine, sessions, sockets, _, end = await engine_fixture(sink)
    ending = asyncio.create_task(engine.handle_owned_vad_event(end))
    try:
        await wait_until(lambda: sessions[0]._event_projection.sealed)
        if waiting == "terminal":
            sessions[0].enable_language_identification = False
            sessions[0].enable_speaker_diarization = False
            pieces = [{"text": "x", "is_final": True} for _ in range(513)]
        else:
            pieces = tokens(1500)
        sockets[0].push(message([*pieces, FIN]))
        await asyncio.wait_for(entered.wait(), timeout=1)
        if waiting == "terminal":
            await wait_until(lambda: engine._terminal_drain_lock.locked())
            assert engine._event_buffer.depth == 512
        else:
            await wait_until(lambda: sessions[0]._event_projection.scoped_event_depth == 512)
        assert not ending.done()
        if operation == "abort":
            await asyncio.wait_for(engine.abort(), timeout=2)
            assert engine.is_at_turn_boundary
            assert sessions[0]._event_projection.scoped_event_depth <= 2
        else:
            await asyncio.wait_for(engine.close(), timeout=8)
            await asyncio.sleep(0)
            assert engine._event_dispatch_task.done()
            assert sockets[0].closed
        assert delivered == []
        assert engine._event_buffer.depth <= 2
    finally:
        ending.cancel()
        await asyncio.gather(ending, return_exceptions=True)
        if operation != "close":
            await asyncio.wait_for(engine.close(), timeout=8)
        assert never.is_set() is False


@pytest.mark.asyncio
async def test_dispatcher_exception_wakes_waiters_and_retires_once():
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def sink(_event):
        entered.set()
        await gate.wait()
        raise RuntimeError("controlled sink failure")

    engine, sessions, sockets, _, end = await engine_fixture(sink)
    ending = asyncio.create_task(engine.handle_owned_vad_event(end))
    try:
        await wait_until(lambda: sessions[0]._event_projection.sealed)
        sockets[0].push(message([*tokens(1500), FIN]))
        await asyncio.wait_for(entered.wait(), timeout=1)
        await wait_until(lambda: sessions[0]._event_projection.scoped_event_depth == 512)
        gate.set()
        await wait_until(lambda: engine._event_dispatch_task.done(), timeout=3)
        with pytest.raises(RuntimeError, match="controlled sink failure"):
            engine._event_dispatch_task.result()
        await asyncio.wait_for(ending, timeout=2)
        await wait_until(lambda: sockets[0].closed, timeout=3)
        assert engine.is_at_turn_boundary
        assert engine._event_buffer.depth <= 2
        assert len(engine._terminal_turn_ids) == 1
    finally:
        gate.set()
        ending.cancel()
        await asyncio.gather(ending, return_exceptions=True)
        await engine.close()


@pytest.mark.asyncio
async def test_fin_before_seal_stays_invalid_when_seal_arrives_during_capacity_wait():
    session, _ = _session()
    request = _request(1)
    await session.begin_turn(request)
    ingestion = asyncio.create_task(session._handle_message(message([*tokens(600), FIN])))
    received = []
    try:
        await wait_until(lambda: session._event_projection.scoped_event_depth == 512)
        observed_at = session._last_rx_at
        await _seal(session, request)
        while not ingestion.done() or session._event_projection.scoped_event_depth:
            event = await _next(session)
            received.append(event)
            if isinstance(event, STTProviderEpochEnded):
                break
        await ingestion
        terminal = [event for event in received if isinstance(event, STTProviderTurnTerminal)]
        assert len(terminal) == 1
        assert terminal[0].outcome == "degraded"
        assert terminal[0].failure_reason == "soniox_protocol_ambiguity"
        assert terminal[0].text == "".join(item["text"] for item in tokens(600))
        assert all(
            event.observed_at == observed_at
            for event in received
            if isinstance(event, STTProviderTurnUpdate)
        )
        assert session._event_projection.retired
    finally:
        ingestion.cancel()
        await asyncio.gather(ingestion, return_exceptions=True)
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["malformed", "error"])
async def test_failure_after_capacity_wait_keeps_confirmed_prefix_not_provisional(failure):
    session, _ = _session()
    request = _request(1)
    await session.begin_turn(request)
    pieces = tokens(600)

    async def ingest():
        await session._handle_message(
            message(
                [
                    *pieces,
                    {"text": "unconfirmed!", "is_final": False},
                    *([None] if failure == "malformed" else []),
                ]
            )
        )
        if failure == "error":
            await session._handle_message(json.dumps({"error_code": 400, "error": "controlled"}))

    ingestion = asyncio.create_task(ingest())
    received = []
    try:
        await wait_until(lambda: session._event_projection.scoped_event_depth == 512)
        while True:
            event = await _next(session)
            received.append(event)
            if isinstance(event, STTProviderEpochEnded):
                break
        await ingestion
        terminal = [event for event in received if isinstance(event, STTProviderTurnTerminal)]
        assert len(terminal) == 1
        assert (terminal[0].outcome, terminal[0].text_authority) == ("degraded", "degraded")
        assert terminal[0].text == "".join(item["text"] for item in pieces)
        assert "unconfirmed" not in terminal[0].text
        assert [
            event.text
            for event in received
            if isinstance(event, STTProviderTurnUpdate) and event.stability == "stable"
        ] == [item["text"] for item in pieces]
    finally:
        ingestion.cancel()
        await asyncio.gather(ingestion, return_exceptions=True)
        await session.close()


@pytest.mark.asyncio
async def test_abort_fences_dequeued_update_and_capture_successor_is_independent():
    gate = asyncio.Event()
    entered = asyncio.Event()
    received = []

    async def sink(event):
        if not entered.is_set():
            entered.set()
            await gate.wait()
        current = scoped_event_for_delivery(event)
        if current is not None:
            received.append(current)

    engine, sessions, sockets, ledger, _ = await engine_fixture(sink)
    try:
        sockets[0].push(message(tokens(1500)))
        await asyncio.wait_for(entered.wait(), timeout=1)
        await wait_until(lambda: sessions[0]._event_projection.scoped_event_depth == 512)
        old_identity = sessions[0]._event_projection.active_identity
        await asyncio.wait_for(engine.abort(reason="stream_input_ownership_changed"), timeout=2)
        gate.set()
        await wait_until(lambda: sockets[0].closed, timeout=3)
        start, chunk, end = segment_events(ledger, start_sample=100, now=101.0)
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(chunk)
        ending = asyncio.create_task(engine.handle_owned_vad_event(end))
        await wait_until(lambda: len(sessions) == 2 and sessions[1]._event_projection.sealed)
        sockets[0].push(message([{"text": "stale", "is_final": True}, FIN]))
        sockets[1].push(message([{"text": "successor", "is_final": True}, FIN]))
        await asyncio.wait_for(ending, timeout=2)
        await asyncio.wait_for(engine.wait_for_event_ingress_drain(), timeout=2)
        assert not any(
            isinstance(event, STTProviderTurnUpdate) and event.identity == old_identity
            for event in received
        )
        terminals = [event for event in received if isinstance(event, STTProviderTurnTerminal)]
        assert [(event.outcome, event.text) for event in terminals] == [
            ("cancelled", ""),
            ("final", "successor"),
        ]
        assert len({event.identity for event in terminals}) == 2
    finally:
        gate.set()
        await engine.close()


@pytest.mark.asyncio
async def test_capacity_admission_counts_text_and_metadata_and_wakes_on_close():
    identity = _request(1).identity
    buffer = STTProviderEventBuffer()
    event = STTProviderTurnUpdate(identity, 1, "stable", "append", "x" * 200_000)
    event = replace(event, provenance=replace(event.provenance, native_request_id="r" * 400_000))
    assert buffer.put(event)
    waiter = asyncio.create_task(buffer.wait_for_capacity(event, is_current=lambda: True))
    await asyncio.sleep(0)
    assert not waiter.done()
    assert buffer.depth == 1
    assert buffer._recognition_bytes == 600_000
    assert await buffer.get() == event
    assert await asyncio.wait_for(waiter, timeout=1)
    assert buffer.put(event)
    waiter = asyncio.create_task(buffer.wait_for_capacity(event, is_current=lambda: True))
    await asyncio.sleep(0)
    buffer.close()
    assert await asyncio.wait_for(waiter, timeout=1) is False


async def owner_smoke(channel, monkeypatch):
    from puripuly_heart.config.prompts import get_default_prompt
    from puripuly_heart.core.clock import FakeClock, SystemClock
    from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
        TranslationChannelOwnerCallbacks,
    )
    from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
    from puripuly_heart.domain.events import SelfTurnPresentation
    from tests.core.runtime.test_peer_capture_session import make_config, make_owner
    from tests.core.test_peer_caption_lifecycle import ControlledBackend, idle, turn_loop
    from tests.helpers.fakes import RecordingOscQueue
    from tests.helpers.translation_owners import compose_translation_test_harness

    clock = FakeClock(_now=100.0) if channel == "peer" else SystemClock()
    observed = [100.0]
    monkeypatch.setattr(soniox, "time", SimpleNamespace(monotonic=lambda: observed[0]))
    sessions, sockets, normalized, logical = [], [], [], []
    capture = ending = None

    async def factory(_settings, epoch):
        session, socket = _soniox_session(epoch=epoch)
        sessions.append(session)
        sockets.append(socket)
        return session

    engine = ScopedRecognitionEngine(session_factory=factory, channel=channel)
    backend = ControlledBackend()

    async def emit(event):
        logical.append(event)

    harness = compose_translation_test_harness(
        stt=engine if channel == "self" else None,
        peer_stt=engine if channel == "peer" else None,
        llm=backend,
        osc=RecordingOscQueue(),
        clock=clock,
        overlay_sink=SimpleNamespace(emit=emit, active_self_overlay_metadata=lambda: None),
        source_language="en",
        target_language="ko",
        system_prompt=get_default_prompt(),
        peer_translation_enabled=True,
        low_latency_mode=True,
    )
    try:
        await harness.start()
        assert harness.configuration.snapshot().value.low_latency_finalize_wait_ms == 400
        original = engine._deferred_event_sink
        assert original is not None and engine.event_sink is None
        if channel == "peer":
            capture, *_ = make_owner(
                sink=SimpleNamespace(
                    handle_owned_vad_event=harness.peer_owner.handle_peer_owned_vad_event,
                    handle_stream_input=harness.peer_owner.handle_stream_input,
                    observe_source_activity=harness.peer_owner.observe_source_activity,
                    observe_pending_source_work=harness.peer_owner.observe_pending_source_work,
                ),
                clock=clock,
            )
            callbacks = TranslationChannelOwnerCallbacks(SttSessionStateProjection())
            callbacks.bind_peer(harness.peer_owner)
            callbacks.bind_peer_capture(capture)
            capture.bind_publication_generation_observer(
                activated=harness.output_runtime.activate_peer_generation,
                retired=harness.output_runtime.retire_peer_generation,
            )
            config = make_config()
            config = replace(config, language=replace(config.language, source_language="en"))
            await capture.apply_intent(config, enabled=True)
            ledger = capture.segment_ledger
            assert ledger is not None
            original = callbacks.peer_event_handler
            controls = harness.peer_owner.handle_peer_owned_vad_event
        else:
            ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
            controls = harness.self_owner.handle_vad_event
        engine.accepted_settings_scope = (
            ledger.settings.provider_id,
            ledger.settings.provider_signature,
            ledger.settings.runtime_signature,
        )

        async def observe(event):
            if isinstance(event, (STTProviderTurnUpdate, STTProviderTurnTerminal)):
                normalized.append(event)
            await original(event)

        engine.bind_event_sink(observe)
        start, chunk, end = segment_events(ledger, start_sample=0, now=clock.now())
        await controls(start)
        await controls(chunk)

        async def deliver(items, at):
            observed[0] = at
            if channel == "peer":
                clock._now = at
            sockets[0].push(message(items))
            await wait_until(lambda: sessions[0]._last_rx_at == at)
            await wait_until(
                lambda: sessions[0]._event_projection.scoped_event_depth == 0
                and engine._event_buffer.depth == 0
                and not engine._event_dispatching
            )
            await turn_loop()

        if channel == "self":
            ending = asyncio.create_task(controls(end))
            await wait_until(lambda: sessions[0]._event_projection.sealed)
            await deliver(
                [{"text": "hello", "is_final": True}, {"text": " world", "is_final": True}, FIN],
                100.0,
            )
            await ending
            await wait_until(lambda: harness.self_owner.merge_buffer is None, timeout=2)
            await harness.translation_turns.wait_for_idle()
        else:
            await deliver(
                [
                    {"text": "I ", "is_final": True},
                    {"text": "agree", "is_final": True},
                    {"text": ". But I do not.", "is_final": False},
                ],
                100.0,
            )
            clock._now = 104.99
            await harness.peer_owner.cloud_captions.check_previews()
            await idle(harness)
            assert backend.calls == []
            clock._now = 105.0
            await harness.peer_owner.cloud_captions.check_previews()
            await idle(harness)
            assert [request.text for request in backend.calls] == ["I agree. But I do not."]
            assert harness.peer_runtime.translation_history == []
            await deliver([{"text": "?", "is_final": False}], 106.0)
            await deliver([], 107.0)
            await deliver([{"text": "?", "is_final": False}], 108.0)
            await deliver([{"text": "?", "is_final": True}], 109.0)
            ending = asyncio.create_task(controls(end))
            await wait_until(lambda: sessions[0]._event_projection.sealed)
            await deliver([FIN], 109.1)
            await ending
            await idle(harness)
            terminal = next(
                event for event in normalized if isinstance(event, STTProviderTurnTerminal)
            )
            await callbacks.peer_event_handler(terminal)
            await idle(harness)
            assert len(capture.segment_ledgers[-1].terminal_receipts) == 1
            assert (
                capture.segment_ledgers[-1].terminal_receipts[0].text_authority == "authoritative"
            )

        updates = [event for event in normalized if isinstance(event, STTProviderTurnUpdate)]
        terminals = [event for event in normalized if isinstance(event, STTProviderTurnTerminal)]
        assert len(terminals) == 1
        terminal = terminals[0]
        history = (
            harness.self_runtime.translation_history
            if channel == "self"
            else harness.peer_runtime.translation_history
        )
        assert [item.text for item in history] == [
            "hello world" if channel == "self" else "I agree?"
        ]
        presentations = []
        while not harness.ui_events.empty():
            event = harness.ui_events.get_nowait()
            if isinstance(event.payload, SelfTurnPresentation):
                presentations.append(event.payload.source_text)
        if channel == "self":
            assert [request.text for request in backend.calls] == ["hello", "hello world"]
            assert "hello" in presentations
            assert all(event.stability == "stable" for event in updates)
        else:
            assert [request.text for request in backend.calls] == [
                "I agree. But I do not.",
                "I agree?",
            ]
            assert harness.osc.messages == []
            assert [event.text for event in updates if event.stability == "provisional"] == [
                "I agree. But I do not.",
                "I agree?",
                "I agree",
                "I agree?",
                "I agree?",
            ]
            assert all(event.first_nonempty_at == 100.0 for event in updates)
        return {
            "channel": channel,
            "finalize_wait_ms": 400,
            "requests": [request.text for request in backend.calls],
            "history": [item.text for item in history],
            "presentations": presentations,
            "observations": [
                [event.stability, event.text, event.observed_at, event.first_nonempty_at]
                for event in updates
            ],
            "terminal": [terminal.outcome, terminal.text, terminal.text_authority],
            "logical": [
                [
                    type(event).__name__,
                    getattr(event, "source_text", None),
                    getattr(event, "peer_caption_revision", None),
                ]
                for event in logical
            ],
            "osc": [item.text for item in harness.osc.messages],
            "ordinary_capture_terminal": capture is not None,
        }
    finally:
        if ending is not None and not ending.done():
            ending.cancel()
            await asyncio.gather(ending, return_exceptions=True)
        if capture is not None:
            await capture.close()
        await harness.stop()
        assert all(socket.closed for socket in sockets)
        assert all(session._send_task.done() and session._recv_task.done() for session in sessions)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_actual_translation_owners_keep_default_intermediate_and_final_contracts(
    channel, monkeypatch
):
    await owner_smoke(channel, monkeypatch)


@pytest.mark.asyncio
async def test_queued_retraction_revokes_running_peer_preview_and_replacement_progresses(
    monkeypatch,
):
    from puripuly_heart.core.clock import FakeClock
    from tests.core.test_peer_caption_lifecycle import idle, setup, turn_loop

    clock = FakeClock(_now=100.0)
    monkeypatch.setattr(soniox, "time", SimpleNamespace(monotonic=clock.now))
    harness, backend, events, _ = setup(clock=clock)
    entered = asyncio.Event()
    release = asyncio.Event()
    stalled = False
    engine = None
    old_source = "Old hypothesis"
    replacement = "A longer replacement hypothesis"

    async def sink(event):
        if isinstance(event, STTProviderTurnUpdate):
            if stalled:
                entered.set()
                await release.wait()
            await harness.peer_owner.handle_provider_update(event)

    try:
        await harness.start()
        engine, sessions, sockets, _, _ = await engine_fixture(sink)

        async def receive(text, at):
            clock._now = at
            sockets[0].push(message([{"text": text, "is_final": False}] if text else []))
            await wait_until(
                lambda: sessions[0]._last_rx_at == at
                and sessions[0]._event_projection.scoped_event_depth == 0
            )

        backend.block = True
        await receive(old_source, 100.0)
        await asyncio.wait_for(engine.wait_for_event_ingress_drain(), timeout=1)
        clock._now = 105.0
        await harness.peer_owner.cloud_captions.check_previews()
        await wait_until(lambda: len(backend.completions) == 1)
        assert [request.text for request in backend.calls] == [old_source]

        stalled = True
        await receive(old_source, 105.0)
        await asyncio.wait_for(entered.wait(), timeout=1)
        await receive("", 106.0)
        await receive(replacement, 107.0)
        release.set()
        await asyncio.wait_for(engine.wait_for_event_ingress_drain(), timeout=1)
        old_translation = backend.completions[0]
        if not old_translation.done():
            old_translation.set_result(None)
        await turn_loop()
        await asyncio.wait_for(idle(harness), timeout=1)
        before_next_preview = [event.source_text for event in events]

        backend.block = False
        clock._now = 109.0
        await harness.peer_owner.cloud_captions.check_previews()
        await asyncio.wait_for(idle(harness), timeout=1)
        assert [request.text for request in backend.calls] == [old_source, replacement]
        assert events[-1].source_text == replacement
        assert events[-1].type == "translation_stream_update"
        assert before_next_preview == []
        assert [event.source_text for event in events] == [replacement]
        assert harness.peer_runtime.translation_history == []
        assert harness.self_runtime.translation_history == []
        assert harness.osc.messages == []
    finally:
        release.set()
        if engine is not None:
            await asyncio.wait_for(engine.close(), timeout=3)
        await asyncio.wait_for(harness.stop(), timeout=3)


@pytest.mark.asyncio
async def test_buffered_responses_yield_before_capacity_is_exhausted():
    received = []
    native_backlog_at_delivery = []
    sockets = []

    async def sink(event):
        received.append(event)
        native_backlog_at_delivery.append(sockets[0].queue.qsize())

    engine, sessions, created, _, end = await engine_fixture(sink)
    sockets.extend(created)
    ending = asyncio.create_task(engine.handle_owned_vad_event(end))
    try:
        await wait_until(lambda: sessions[0]._event_projection.sealed)
        for text in ["one", " two", " three"]:
            sockets[0].push(message([{"text": text, "is_final": True}]))
        sockets[0].push(message([FIN]))
        await ending
        await engine.wait_for_event_ingress_drain()
        assert native_backlog_at_delivery[0] > 0
        assert [event.text for event in received] == [
            "one",
            "one two",
            "one two three",
            "one two three",
        ]
    finally:
        ending.cancel()
        await asyncio.gather(ending, return_exceptions=True)
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("trailer", ["end", "text", "fin"])
async def test_same_response_fin_barrier_never_seeds_a_successor(trailer):
    session, _ = _session()
    request = _request(1)
    await session.begin_turn(request)
    await _seal(session, request)
    trailers = {
        "end": {"text": "<end>", "is_final": True},
        "text": {"text": "not successor text", "is_final": True},
        "fin": FIN,
    }
    try:
        await session._handle_message(
            message([{"text": "accepted", "is_final": True}, FIN, trailers[trailer]])
        )
        stable = await _next(session)
        terminal = await _next(session)
        assert isinstance(stable, STTProviderTurnUpdate)
        assert isinstance(terminal, STTProviderTurnTerminal)
        assert (stable.text, terminal.text, terminal.outcome) == ("accepted", "accepted", "final")
        if trailer == "end":
            assert session._event_projection.scoped_event_depth == 0
            await session.begin_turn(_request(2))
            assert session._scoped_tokens == []
        else:
            ended = await _next(session)
            assert isinstance(ended, STTProviderEpochEnded)
            with pytest.raises(RuntimeError, match="retired"):
                await session.begin_turn(_request(2))
        assert session._event_projection.scoped_event_depth == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_dequeued_terminal_rechecks_authority_after_peer_admission_lock():
    from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
        TranslationChannelOwnerCallbacks,
    )
    from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
    from tests.core.runtime.test_peer_capture_session import make_config, make_owner

    capture, *_ = make_owner()
    callbacks = TranslationChannelOwnerCallbacks(SttSessionStateProjection())
    admitted = []

    async def receive(receipt, terminal):
        admitted.append((receipt, terminal))

    callbacks.bind_peer_capture(capture)
    callbacks.bind_peer(SimpleNamespace(handle_provider_turn_terminal=receive))
    current = [True]
    delivery = None
    lock = callbacks._recognition_locks["peer"]
    try:
        await capture.apply_intent(make_config(), enabled=True)
        ledger = capture.segment_ledger
        _, _, end = segment_events(ledger, start_sample=0, now=100.0)
        identity = replace(_request(1).identity, segment=end.segment.identity)
        terminal = STTProviderTurnTerminal(
            identity,
            "final",
            "stale authoritative text",
            text_authority="authoritative",
            is_current=lambda: current[0],
        )
        await lock.acquire()
        delivery = asyncio.create_task(callbacks.peer_event_handler(terminal))
        await asyncio.sleep(0)
        assert not delivery.done()
        current[0] = False
        lock.release()
        await asyncio.wait_for(delivery, timeout=1)
        await callbacks.peer_event_handler(terminal)
        assert len(admitted) == 1
        assert (admitted[0][1].outcome, admitted[0][1].text) == ("cancelled", "")
        assert ledger.terminal_receipts[0].text_authority == "none"
        _, _, successor = segment_events(ledger, start_sample=100, now=101.0)
        next_identity = replace(_request(2).identity, segment=successor.segment.identity)
        await callbacks.peer_event_handler(
            STTProviderTurnTerminal(
                next_identity, "final", "legitimate successor", text_authority="authoritative"
            )
        )
        assert [(item.outcome, item.text) for _, item in admitted] == [
            ("cancelled", ""),
            ("final", "legitimate successor"),
        ]
        assert len(ledger.terminal_receipts) == 2
    finally:
        if lock.locked():
            lock.release()
        if delivery is not None:
            await asyncio.gather(delivery, return_exceptions=True)
        await capture.close()


async def transport_failure_scenario(operation, *, cancel=None, channel="peer"):
    import hashlib

    import numpy as np

    from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
    from tests.core.test_stt_scoped_engine import span

    gate = asyncio.Event()
    entered = asyncio.Event()
    received = []
    sessions = []
    sockets = []
    work = None

    async def sink(event):
        entered.set()
        await gate.wait()
        current = scoped_event_for_delivery(event)
        if current is not None:
            received.append(current)

    async def factory(_settings, epoch):
        session, socket = _soniox_session(epoch=epoch)
        sessions.append(session)
        sockets.append(socket)
        return session

    engine = ScopedRecognitionEngine(session_factory=factory, channel=channel)
    engine.bind_event_sink(sink)
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
    identifier = _request(1).identity.segment.segment_id
    start = ledger.observe_vad_event(
        SpeechStart(
            identifier,
            np.array([0.1, 0.1], dtype=np.float32),
            np.array([0.2] * 4, dtype=np.float32),
            pre_roll_capture=(span(0, 0, 2),),
            chunk_capture=(span(1, 2, 6),),
        ),
        now_monotonic_s=100.0,
    )
    await engine.handle_owned_vad_event(start)
    session, socket = sessions[0], sockets[0]
    pieces = tokens(1500)
    for index, item in enumerate(pieces):
        item["language"] = "en" if index < 400 else "ko"
        item["speaker"] = "A" if index < 250 else "B"
    original_error = OSError("controlled transport failure")

    async def fail_send(_payload):
        raise original_error

    try:
        socket.push(message(pieces))
        await asyncio.wait_for(entered.wait(), timeout=1)
        await wait_until(lambda: session._event_projection.scoped_event_depth == 512)
        parsed_count = len(session._scoped_tokens)
        expected = "".join(item["text"] for item in pieces[:parsed_count]).strip()
        assert 512 < parsed_count < len(pieces)
        socket.send = fail_send
        if operation == "send":
            chunk = ledger.observe_vad_event(
                SpeechChunk(
                    identifier,
                    np.array([0.3] * 4, dtype=np.float32),
                    chunk_capture=(span(2, 6, 10),),
                ),
                now_monotonic_s=100.1,
            )
            work = asyncio.create_task(engine.handle_owned_vad_event(chunk))
        elif operation == "seal":
            end = ledger.observe_vad_event(
                SpeechEnd(identifier, trailing_silence_ms=224, reason="silence"),
                now_monotonic_s=100.2,
            )
            work = asyncio.create_task(engine.handle_owned_vad_event(end))
        elif operation == "keepalive":
            session.keepalive_interval_s = 0.01
            session._last_send_at = None
            session._keepalive_task = asyncio.create_task(session._keepalive_loop())
        else:
            await session.send_audio(b"controlled transport write")
        await wait_until(lambda: session._failure_terminal is not None)
        captured = session._failure_terminal
        expected_reason = (
            "soniox_keepalive_failed" if operation == "keepalive" else "soniox_write_failed"
        )
        assert captured.text.strip() == expected
        assert captured.failure_reason == expected_reason
        assert captured.failure_diagnostics is session._failure_diagnostics
        assert session._failure_cause is original_error
        socket.push(None)
        if cancel is not None:
            assert not session._send_task.done()
            assert session._event_projection.scoped_event_depth == 512
            if cancel == "abort":
                await asyncio.wait_for(
                    engine.abort(reason="stream_input_ownership_changed"), timeout=2
                )
                gate.set()
                await asyncio.wait_for(engine.wait_for_event_ingress_drain(), timeout=2)
                terminals = [
                    event for event in received if isinstance(event, STTProviderTurnTerminal)
                ]
                assert [(event.outcome, event.text) for event in terminals] == [("cancelled", "")]
            else:
                await asyncio.wait_for(engine.close(), timeout=8)
                await asyncio.sleep(0)
                assert not gate.is_set()
                assert received == []
                assert engine._event_dispatch_task.done()
            await wait_until(lambda: socket.closed, timeout=3)
            assert session._send_task.done() and session._recv_task.done()
            assert session._failure_diagnostics is captured.failure_diagnostics
            return {
                "operation": operation,
                "cancel": cancel,
                "parsed_chars": len(expected),
                "stale_degraded_deliveries": 0,
                "closed": socket.closed,
                "failure_reason": captured.failure_reason,
            }

        gate.set()
        if work is not None:
            await asyncio.wait_for(work, timeout=3)
        else:
            pending_turn = engine._turn
            await wait_until(lambda: pending_turn.terminal_ready.done(), timeout=3)
            assert pending_turn.terminal_ready.result().text == expected
        if operation != "seal":
            end = ledger.observe_vad_event(
                SpeechEnd(identifier, trailing_silence_ms=224, reason="silence"),
                now_monotonic_s=100.2,
            )
            await asyncio.wait_for(engine.handle_owned_vad_event(end), timeout=3)
        await wait_until(
            lambda: any(isinstance(event, STTProviderTurnTerminal) for event in received),
            timeout=3,
        )
        await asyncio.wait_for(engine.wait_for_event_ingress_drain(), timeout=3)
        terminals = [event for event in received if isinstance(event, STTProviderTurnTerminal)]
        assert len(terminals) == 1
        terminal = terminals[0]
        assert (terminal.outcome, terminal.text, terminal.text_authority) == (
            "degraded",
            expected,
            "degraded",
        )
        assert terminal.failure_reason == expected_reason
        assert terminal.failure_retryable is True
        assert terminal.failure_diagnostics.fields["recognition_cause"] == expected_reason
        assert (
            terminal.failure_diagnostics.diagnostic_id == captured.failure_diagnostics.diagnostic_id
        )
        assert all(
            terminal.failure_diagnostics.fields.get(key) == value
            for key, value in captured.failure_diagnostics.fields.items()
        )
        assert "".join(run.text for run in terminal.final_language_runs) == expected
        assert {run.language for run in terminal.final_language_runs} == {"en", "ko"}
        assert "".join(run.text for run in terminal.final_speaker_runs) == expected
        assert {run.speaker_id for run in terminal.final_speaker_runs} == {"A", "B"}
        assert all(
            run.session_scope == session.speaker_session_scope
            for run in terminal.final_speaker_runs
        )
        assert all(run.attribution_state == "identified" for run in terminal.final_speaker_runs)
        assert terminal.final_speaker_runs[0].source_start_ms == 0
        assert terminal.final_speaker_runs[-1].source_end_ms == parsed_count * 10
        assert all(item.native_request_id == "bounded-tape" for item in terminal.provenance)
        consumption = STTContributionConsumptionLedger()
        consumed = "".join(
            consumption.consume(event)
            for event in received
            if isinstance(event, (STTProviderTurnUpdate, STTProviderTurnTerminal))
        )
        assert consumed == expected
        assert consumption.consume(terminal) == ""
        assert len(engine._terminal_turn_ids) == 1
        await wait_until(lambda: socket.closed, timeout=3)
        assert session._failure_diagnostics is captured.failure_diagnostics
        return {
            "operation": operation,
            "channel": channel,
            "parsed_tokens": parsed_count,
            "parsed_chars": len(expected),
            "terminal_chars": len(terminal.text),
            "consumed_chars": len(consumed),
            "consumed_sha256": hashlib.sha256(consumed.encode()).hexdigest(),
            "terminal_count": len(terminals),
            "failure_reason": terminal.failure_reason,
            "original_diagnostic_preserved": True,
            "languages": sorted({run.language for run in terminal.final_language_runs}),
            "speakers": sorted({run.speaker_id for run in terminal.final_speaker_runs}),
            "duplicate_terminal_consumption": "",
            "closed": socket.closed,
        }
    finally:
        if work is not None and not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
        if cancel != "close":
            gate.set()
            await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["send", "seal", "keepalive"])
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_transport_failure_under_both_queue_pressure_conserves_entire_parsed_prefix(
    operation,
    channel,
):
    await transport_failure_scenario(operation, channel=channel)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["abort", "close"])
async def test_cancellation_revokes_transport_failure_terminal_admission(cancel):
    await transport_failure_scenario("writer", cancel=cancel)


@pytest.mark.asyncio
async def test_scoped_write_failure_receipt_is_immediate_and_carries_original_cause():
    session, socket = _session()
    request = _request(1)
    await session.begin_turn(request)
    ingestion = asyncio.create_task(session._handle_message(message(tokens(600))))
    error = OSError("controlled receipt failure")

    async def fail_send(_payload):
        raise error

    try:
        await wait_until(lambda: session._event_projection.scoped_event_depth == 512)
        expected = "".join(item.text for item in session._scoped_tokens)
        socket.send = fail_send
        with pytest.raises(STTScopedWriteFailure) as raised:
            await asyncio.wait_for(
                session.send_turn_audio(
                    request.identity,
                    b"audio",
                    payload_sequence=1,
                    source_ranges=(),
                    context_only=False,
                ),
                timeout=1,
            )
        assert raised.value.__cause__ is error
        assert raised.value.terminal.text == expected
        assert raised.value.terminal.failure_reason == "soniox_write_failed"
        assert session._event_projection.scoped_event_depth == 512
        assert not session._send_task.done()
        await asyncio.wait_for(session.abort_turn(request.identity, reason="cancelled"), timeout=1)
        await asyncio.wait_for(session._send_task, timeout=1)
        events = [await _next(session) for _ in range(session._event_projection.scoped_event_depth)]
        assert not any(
            isinstance(event, STTProviderTurnTerminal) and event.outcome == "degraded"
            for event in events
        )
        assert (
            sum(
                isinstance(event, STTProviderTurnTerminal) and event.outcome == "cancelled"
                for event in events
            )
            == 1
        )
    finally:
        ingestion.cancel()
        await asyncio.gather(ingestion, return_exceptions=True)
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("obsolete", ["identity", "epoch", "abort", "replacement"])
async def test_scoped_write_failure_evidence_cannot_override_obsolete_authority(obsolete):
    received = []
    engine, sessions, sockets, ledger, _ = await engine_fixture(received.append)
    session = sessions[0]
    turn = engine._turn
    identity = turn.identity
    if obsolete == "identity":
        identity = replace(identity, provider_turn_id="not-current")
    elif obsolete == "epoch":
        identity = replace(identity, provider_epoch_id="not-current")
    evidence = STTProviderTurnTerminal(
        identity,
        "degraded",
        "must not enter current source",
        text_authority="degraded",
        failure_reason="soniox_write_failed",
        epoch_disposition="retire",
        failure_retryable=True,
    )
    release = asyncio.Event()
    entered = asyncio.Event()
    raised = asyncio.Event()
    original = OSError("controlled late write")

    async def fail():
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        raised.set()
        raise STTScopedWriteFailure(evidence, original)

    work = asyncio.create_task(engine._run_write(session, turn, "send", fail()))
    ending = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        if obsolete in {"abort", "replacement"}:
            await asyncio.wait_for(engine.abort(reason="provider_replaced"), timeout=2)
        release.set()
        assert await asyncio.wait_for(work, timeout=2) is False
        assert raised.is_set()
        if obsolete == "replacement":
            start, chunk, end = segment_events(ledger, start_sample=100, now=101.0)
            await engine.handle_owned_vad_event(start)
            await engine.handle_owned_vad_event(chunk)
            ending = asyncio.create_task(engine.handle_owned_vad_event(end))
            await wait_until(lambda: len(sessions) == 2 and sessions[1]._event_projection.sealed)
        assert turn.normalizer.stable_text == ""
        if obsolete in {"identity", "epoch"}:
            assert turn.terminal_ready.result().failure_reason == "provider_send_failed"
            assert turn.terminal_ready.result().text == ""
        if ending is not None:
            sockets[1].push(message([{"text": "legitimate replacement", "is_final": True}, FIN]))
            await asyncio.wait_for(ending, timeout=2)
            await engine.wait_for_event_ingress_drain()
            terminals = [event for event in received if isinstance(event, STTProviderTurnTerminal)]
            assert [(event.outcome, event.text) for event in terminals] == [
                ("cancelled", ""),
                ("final", "legitimate replacement"),
            ]
        assert not any(getattr(event, "text", "") == evidence.text for event in received)
    finally:
        release.set()
        await asyncio.gather(work, return_exceptions=True)
        if ending is not None and not ending.done():
            ending.cancel()
            await asyncio.gather(ending, return_exceptions=True)
        await engine.close()


async def fin_failure_race_scenario(winner, *, cancel=None):
    from unittest.mock import patch

    consume = asyncio.Event()
    fin_reached = asyncio.Event()
    allow_fin = asyncio.Event()
    fin_resolved = asyncio.Event()
    sessions, sockets, received, native_first = [], [], [], []
    original_events = soniox._SonioxSession.turn_events
    original_fin = soniox._SonioxSession._resolve_scoped_fin
    original_capacity = STTProviderEventBuffer.wait_for_capacity
    ending = successor_ending = None
    engine = None
    original_error = OSError("controlled final race keepalive failure")

    async def gated_events(session):
        if session is sessions[0]:
            await consume.wait()
        async for event in original_events(session):
            if session is sessions[0]:
                native_first.append(event)
            yield event

    async def gated_fin(session, data, *, fin_detail):
        if session is sessions[0]:
            fin_reached.set()
            if winner == "failure":
                await allow_fin.wait()
        await original_fin(session, data, fin_detail=fin_detail)
        if session is sessions[0]:
            fin_resolved.set()

    async def ordered_capacity(buffer, event, *, is_current):
        accepted = await original_capacity(buffer, event, is_current=is_current)
        if accepted and winner == "final" and sessions and event is sessions[0]._failure_terminal:
            await fin_resolved.wait()
        return accepted

    async def factory(_settings, epoch):
        session, socket = _soniox_session(epoch=epoch)
        sessions.append(session)
        sockets.append(socket)
        return session

    async def fail_send(_payload):
        raise original_error

    with (
        patch.object(soniox._SonioxSession, "turn_events", gated_events),
        patch.object(soniox._SonioxSession, "_resolve_scoped_fin", gated_fin),
        patch.object(STTProviderEventBuffer, "wait_for_capacity", ordered_capacity),
    ):
        try:
            engine = ScopedRecognitionEngine(session_factory=factory)
            engine.bind_event_sink(received.append)
            ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
            start, chunk, end = segment_events(ledger, start_sample=0, now=100.0)
            await engine.handle_owned_vad_event(start)
            await engine.handle_owned_vad_event(chunk)
            session, socket = sessions[0], sockets[0]
            consumer = engine._session_consumer
            ending = asyncio.create_task(engine.handle_owned_vad_event(end))
            await wait_until(lambda: session._event_projection.sealed)
            pieces = tokens(512)
            expected = "".join(item["text"] for item in pieces).strip()
            socket.push(message([*pieces, FIN]))
            await asyncio.wait_for(fin_reached.wait(), timeout=1)
            await wait_until(lambda: session._event_projection.scoped_event_depth == 512)
            if winner == "final":
                assert session._pending_finalize_requests == 0
                assert session._scoped_provenance[-1].barrier == "manual_finalize"
            socket.send = fail_send
            session.keepalive_interval_s = 0.01
            session._last_send_at = None
            session._keepalive_task = asyncio.create_task(session._keepalive_loop())
            await wait_until(lambda: session._failure_terminal is not None)
            assert not session._keepalive_task.done()
            assert session._event_projection.scoped_event_depth == 512
            captured = session._failure_terminal
            if cancel is not None:
                if cancel == "abort":
                    await asyncio.wait_for(
                        engine.abort(reason="stream_input_ownership_changed"), timeout=2
                    )
                    consume.set()
                else:
                    await asyncio.wait_for(engine.close(), timeout=8)
                allow_fin.set()
                await asyncio.wait_for(ending, timeout=2)
                await wait_until(lambda: socket.closed, timeout=3)
                await asyncio.gather(consumer, return_exceptions=True)
                async for event in original_events(session):
                    native_first.append(event)
                terminals = [
                    event for event in received if isinstance(event, STTProviderTurnTerminal)
                ]
                assert [(event.outcome, event.text) for event in terminals] == [("cancelled", "")]
                epochs = [
                    event for event in native_first if isinstance(event, STTProviderEpochEnded)
                ]
                assert len(epochs) == 1
                assert epochs[0].reason == (
                    "closed" if cancel == "close" else "stream_input_ownership_changed"
                )
                assert not any(
                    isinstance(event, STTProviderTurnTerminal)
                    and event.outcome in {"final", "degraded"}
                    for event in native_first
                )
                return {
                    "winner": winner,
                    "cancel": cancel,
                    "terminal": ["cancelled", ""],
                    "epoch_count": 1,
                    "epoch_reason": epochs[0].reason,
                    "cancellation_preserved": True,
                    "closed": socket.closed,
                }

            allow_fin.set()
            consume.set()
            await asyncio.wait_for(ending, timeout=3)
            await wait_until(
                lambda: any(isinstance(event, STTProviderEpochEnded) for event in received),
                timeout=3,
            )
            await engine.wait_for_event_ingress_drain()
            await wait_until(lambda: socket.closed, timeout=3)
            first_terminals = [
                event for event in received if isinstance(event, STTProviderTurnTerminal)
            ]
            assert len(first_terminals) == 1
            terminal = first_terminals[0]
            assert (terminal.outcome, terminal.text, terminal.text_authority) == (
                "final" if winner == "final" else "degraded",
                expected,
                "authoritative" if winner == "final" else "degraded",
            )
            assert terminal.epoch_disposition == ("reuse" if winner == "final" else "retire")
            epochs = [event for event in received if isinstance(event, STTProviderEpochEnded)]
            assert len(epochs) == 1
            assert epochs[0].provider_epoch_id == terminal.identity.provider_epoch_id
            assert epochs[0].reason == "soniox_keepalive_failed"
            assert epochs[0].orderly is False
            assert epochs[0].failure_retryable is True
            assert epochs[0].failure_diagnostics is captured.failure_diagnostics
            assert session._failure_cause is original_error
            assert session._event_projection._epoch_ended
            consumption = STTContributionConsumptionLedger()
            assert (
                "".join(
                    consumption.consume(event)
                    for event in received
                    if isinstance(event, (STTProviderTurnUpdate, STTProviderTurnTerminal))
                )
                == expected
            )
            assert consumption.consume(terminal) == ""
            start, chunk, end = segment_events(ledger, start_sample=100, now=101.0)
            await asyncio.wait_for(engine.handle_owned_vad_event(start), timeout=3)
            await engine.handle_owned_vad_event(chunk)
            assert len(sessions) == 2
            assert (
                sessions[1]._event_projection.provider_epoch_id
                != session._event_projection.provider_epoch_id
            )
            successor_ending = asyncio.create_task(engine.handle_owned_vad_event(end))
            await wait_until(lambda: sessions[1]._event_projection.sealed)
            sockets[1].push(message([{"text": "fresh successor", "is_final": True}, FIN]))
            await asyncio.wait_for(successor_ending, timeout=3)
            await engine.wait_for_event_ingress_drain()
            terminals = [event for event in received if isinstance(event, STTProviderTurnTerminal)]
            assert [(event.outcome, event.text) for event in terminals] == [
                ("final" if winner == "final" else "degraded", expected),
                ("final", "fresh successor"),
            ]
            assert consumption.consume(terminals[1]) == "fresh successor"
            assert consumption.consume(terminals[1]) == ""
            assert len(engine._terminal_turn_ids) == 2
            return {
                "winner": winner,
                "first_terminal": terminal.outcome,
                "first_authority": terminal.text_authority,
                "first_chars": len(terminal.text),
                "first_epoch_disposition": terminal.epoch_disposition,
                "epoch_count": len(epochs),
                "epoch_reason": epochs[0].reason,
                "original_diagnostics_preserved": True,
                "new_session_count": len(sessions),
                "successor": [terminals[1].outcome, terminals[1].text],
                "duplicate_consumption": "",
                "first_socket_closed": socket.closed,
            }
        finally:
            consume.set()
            allow_fin.set()
            fin_resolved.set()
            for task in (ending, successor_ending):
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            if engine is not None:
                await engine.close()
            assert all(socket.closed for socket in sockets)
            assert all(
                session._send_task.done() and session._recv_task.done() for session in sessions
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["final", "failure"])
async def test_fin_transport_failure_race_retires_epoch_and_opens_fresh_successor(winner):
    await fin_failure_race_scenario(winner)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["abort", "close"])
async def test_fin_transport_failure_race_preserves_generation_revocation(cancel):
    await fin_failure_race_scenario("final", cancel=cancel)
