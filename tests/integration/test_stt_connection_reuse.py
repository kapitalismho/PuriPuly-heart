from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import _recognition_watchdogs
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.core.audio.ownership import (
    CaptureStreamInput,
    OwnedStreamInput,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.runtime.peer_channel import _CaptureGeneration, _GenerationGuardedVadSink
from puripuly_heart.core.stt.backend import (
    STTProviderInputTerminal,
    STTProviderTurnTerminal,
    STTRecognitionUnitTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.stt.rolling import RollingProviderDefinition, RollingSTTBackend
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine, STTRecognitionWatchdogs
from puripuly_heart.providers.stt.deepgram import _CLOSE_STREAM, _DeepgramSDKSession
from tests.core.runtime.test_peer_capture_session import make_config, make_owner
from tests.core.test_stt_scoped_engine import ControlledMonotonicClock, ControlledScopedSession
from tests.providers.test_protocol_a_scoped_sessions import (
    _deepgram_result,
    _deepgram_session,
    _gemini_message,
    _gemini_session,
    _owned_deepgram_events,
    _request,
    _scribe_session,
    _soniox_session,
    _wait,
)


@dataclass
class _PreparedMemberBackend:
    open_member: Callable[[str], object]
    sessions: list[object] = field(default_factory=list)
    boundaries: list[object] = field(default_factory=list)
    opens: int = 0

    async def open_session(self, *, projection: STTSessionProjection):
        self.opens += 1
        opened = self.open_member(projection.provider_epoch_id or "")
        if asyncio.iscoroutine(opened):
            opened = await opened
        session, boundary = opened
        self.sessions.append(session)
        self.boundaries.append(boundary)
        return session


class _ControlledSleeper:
    def __init__(self) -> None:
        self.waits: list[tuple[float, asyncio.Future[None]]] = []

    async def __call__(self, delay_s: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self.waits.append((delay_s, future))
        await future

    async def release_next(self) -> float:
        await _wait(lambda: any(not future.done() for _delay, future in self.waits))
        delay, future = next(item for item in self.waits if not item[1].done())
        future.set_result(None)
        await asyncio.sleep(0)
        return delay


@dataclass
class _CurrentPeerGeneration:
    segment_ledger: PeerAudioSegmentLedger

    def is_current_generation(self, generation: int) -> bool:
        return generation == 1


async def _complete_native_turn(
    member: STTProviderName, session: object, boundary: object, text: str
) -> None:
    if member is STTProviderName.DEEPGRAM:
        session._build_transcript_event(_deepgram_result(text, from_finalize=True))
        await asyncio.sleep(0)
        return
    if member is STTProviderName.GEMINI_TRANSCRIBE:
        boundary.push(_gemini_message(final=text))
        return
    if member is STTProviderName.SONIOX:
        boundary.push(
            json.dumps(
                {
                    "tokens": [
                        *([{"text": text, "is_final": True, "language": "en"}] if text else []),
                        {"text": "<fin>", "is_final": True},
                    ]
                }
            )
        )
        return
    from elevenlabs.types import CommittedTranscriptPayload

    session._on_committed(CommittedTranscriptPayload(text=text))


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize(
    ("member", "rolling_route"),
    [
        (STTProviderName.SONIOX, False),
        (STTProviderName.ELEVENLABS_SCRIBE, False),
        (STTProviderName.DEEPGRAM, False),
        (STTProviderName.ELEVENLABS_SCRIBE, True),
        (STTProviderName.DEEPGRAM, True),
    ],
)
async def test_actual_adapter_reuses_one_epoch_for_final_and_empty(
    member: STTProviderName,
    channel: str,
    rolling_route: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deepgram_writes: list[tuple[_DeepgramSDKSession, object]] = []

    async def deepgram_write(session: _DeepgramSDKSession, payload: object) -> None:
        deepgram_writes.append((session, payload))

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", deepgram_write)

    async def open_member(epoch: str):
        if member is STTProviderName.DEEPGRAM:
            return _deepgram_session(epoch=epoch), None
        if member is STTProviderName.GEMINI_TRANSCRIBE:
            return await _gemini_session(epoch=epoch)
        if member is STTProviderName.SONIOX:
            return _soniox_session(epoch=epoch)
        return _scribe_session(epoch=epoch)

    prepared = _PreparedMemberBackend(open_member)
    rolling = (
        RollingSTTBackend(
            providers=(
                RollingProviderDefinition(
                    name=member,
                    build_backend=lambda: prepared,
                    is_configured=lambda: True,
                ),
            )
        )
        if rolling_route
        else None
    )
    emitted: list[object] = []

    async def open_session(_settings, provider_epoch_id: str):
        projection = STTSessionProjection("scoped", provider_epoch_id)
        if rolling is not None:
            return await rolling.open_session(projection=projection)
        return await prepared.open_session(projection=projection)

    engine = ScopedRecognitionEngine(
        channel=channel,
        session_factory=open_session,
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(
            write_timeout_s=0.5,
            final_timeout_s=0.5,
            drain_timeout_s=0.1,
        ),
    )
    settings = _request("rolling_free" if rolling_route else member.value).settings

    for order, text in ((1, "same"), (2, "")):
        start, chunks, end = _owned_deepgram_events(
            PeerAudioSegmentLedger(activation_generation=1, settings=settings),
            start_sample=order * 1000,
        )
        await engine.handle_owned_vad_event(start)
        for chunk in chunks:
            await engine.handle_owned_vad_event(chunk)
        end_task = asyncio.create_task(engine.handle_owned_vad_event(end))
        await _wait(
            lambda: bool(prepared.sessions) and prepared.sessions[0]._event_projection.sealed
        )
        await _complete_native_turn(
            member,
            prepared.sessions[0],
            prepared.boundaries[0],
            text,
        )
        await end_task

    terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
    assert [(terminal.outcome, terminal.text) for terminal in terminals] == [
        ("final", "same"),
        ("empty", ""),
    ]
    assert [terminal.epoch_disposition for terminal in terminals] == ["reuse", "reuse"]
    assert terminals[0].identity.provider_epoch_id == terminals[1].identity.provider_epoch_id
    assert terminals[0].identity.provider_turn_id != terminals[1].identity.provider_turn_id
    assert prepared.opens == 1
    assert len(prepared.sessions) == 1

    session = prepared.sessions[0]
    boundary = prepared.boundaries[0]
    if member is STTProviderName.DEEPGRAM:
        assert all(owner is session for owner, _payload in deepgram_writes)
        assert _CLOSE_STREAM not in [payload for _owner, payload in deepgram_writes]
    elif member is STTProviderName.SONIOX:
        finalize_count = sum(
            isinstance(payload, str) and payload and json.loads(payload).get("type") == "finalize"
            for payload in boundary.sent
        )
        assert finalize_count == 2
        assert boundary.closed is False
    else:
        assert boundary.commits == 2
        assert boundary.closed is False

    await engine.close()
    if boundary is not None:
        assert boundary.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("rolling_route", [False, True])
async def test_gemini_engine_reuses_stream_for_receipt_units_across_local_inputs(
    rolling_route: bool,
) -> None:
    prepared = _PreparedMemberBackend(lambda epoch: _gemini_session(epoch=epoch))
    rolling = (
        RollingSTTBackend(
            providers=(
                RollingProviderDefinition(
                    name=STTProviderName.GEMINI_TRANSCRIBE,
                    build_backend=lambda: prepared,
                    is_configured=lambda: True,
                ),
            )
        )
        if rolling_route
        else None
    )
    emitted: list[object] = []

    async def open_session(_settings, provider_epoch_id: str):
        projection = STTSessionProjection("scoped", provider_epoch_id)
        return await (
            rolling.open_session(projection=projection)
            if rolling is not None
            else prepared.open_session(projection=projection)
        )

    engine = ScopedRecognitionEngine(
        channel="self",
        session_factory=open_session,
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(
            write_timeout_s=0.5,
            final_timeout_s=0.05,
            drain_timeout_s=0.1,
        ),
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=_request("rolling_free" if rolling_route else "gemini_transcribe").settings,
    )
    try:
        start_a, chunks_a, end_a = _owned_deepgram_events(ledger, start_sample=1000)
        await engine.handle_owned_vad_event(start_a)
        for chunk in chunks_a:
            await engine.handle_owned_vad_event(chunk)
        wire = prepared.boundaries[0]
        wire.push(_gemini_message(final="early"))
        await _wait(lambda: any(isinstance(event, STTRecognitionUnitTerminal) for event in emitted))
        await engine.handle_owned_vad_event(end_a)
        await _wait(lambda: any(isinstance(event, STTProviderInputTerminal) for event in emitted))
        assert [
            event.outcome for event in emitted if isinstance(event, STTProviderInputTerminal)
        ] == ["submitted"]
        assert [call.get("audio_stream_end") for call in wire.sent].count(True) == 1

        start_b, chunks_b, end_b = _owned_deepgram_events(ledger, start_sample=1084)
        await engine.handle_owned_vad_event(start_b)
        for chunk in chunks_b:
            await engine.handle_owned_vad_event(chunk)
        assert [call.get("audio_stream_end") for call in wire.sent].count(True) == 1
        wire.push(_gemini_message(interim="not final", ack=True))
        await engine.handle_owned_vad_event(end_b)
        wire.push(_gemini_message(final="same"))
        wire.push(_gemini_message(final="same"))
        await _wait(
            lambda: sum(isinstance(event, STTRecognitionUnitTerminal) for event in emitted) == 3
        )
        finals = [event for event in emitted if isinstance(event, STTRecognitionUnitTerminal)]
        assert [(event.outcome, event.unit.text) for event in finals] == [
            ("final", "early"),
            ("final", "same"),
            ("final", "same"),
        ]
        assert [event.unit.identity.receipt_sequence for event in finals] == [1, 2, 3]
        assert len({event.unit.identity.unit_id for event in finals}) == 3
        assert len({event.unit.identity.stream for event in finals}) == 1
        assert [
            event.outcome for event in emitted if isinstance(event, STTProviderInputTerminal)
        ] == ["submitted", "submitted"]
        assert [next(iter(call)) for call in wire.sent].count("audio_stream_end") == 2
        assert prepared.opens == 1
        stream = finals[0].unit.identity.stream
        assert engine.is_current_recognition_stream(stream)
        last_capture = chunks_b[-1].event.chunk_capture[0]
        source_end = last_capture.normalized_end_sample
        assert source_end is not None
        next_capture = replace(
            last_capture,
            source_start_sample=source_end,
            source_end_sample=source_end + 4,
            normalized_start_sample=source_end,
            normalized_end_sample=source_end + 4,
            source_start_monotonic_s=source_end / 16000,
            source_end_monotonic_s=(source_end + 4) / 16000,
        )
        writes_before = len(wire.sent)
        await engine.handle_stream_input(
            OwnedStreamInput(
                event=CaptureStreamInput(np.ones(4, dtype=np.float32), (next_capture,)),
                ledger=ledger,
                settings=ledger.settings,
                activation_generation=0,
            )
        )
        assert len(wire.sent) == writes_before
        before_abort = len(finals)
        await engine.abort(reason="generation_retired")
        assert not engine.is_current_recognition_stream(stream)
        wire.push(_gemini_message(final="stale"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert (
            sum(isinstance(event, STTRecognitionUnitTerminal) for event in emitted) == before_abort
        )
    finally:
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("rolling_route", [False, True])
async def test_native_stream_reopens_for_new_capture_generation_with_same_settings(
    rolling_route: bool,
) -> None:
    prepared = _PreparedMemberBackend(lambda epoch: _gemini_session(epoch=epoch))
    backend = (
        RollingSTTBackend(
            providers=(
                RollingProviderDefinition(
                    name=STTProviderName.GEMINI_TRANSCRIBE,
                    build_backend=lambda: prepared,
                    is_configured=lambda: True,
                ),
            )
        )
        if rolling_route
        else prepared
    )
    emitted = []
    engine = ScopedRecognitionEngine(
        channel="self",
        session_factory=lambda _settings, epoch: backend.open_session(
            projection=STTSessionProjection("scoped", epoch)
        ),
        event_sink=emitted.append,
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=_request("rolling_free" if rolling_route else "gemini_transcribe").settings,
    )
    try:
        for generation in (1, 2):
            ledger.rebind(activation_generation=generation, settings=ledger.settings)
            start, chunks, end = _owned_deepgram_events(
                ledger, start_sample=1000 + (generation - 1) * 84
            )
            await engine.handle_owned_vad_event(start)
            for chunk in chunks:
                await engine.handle_owned_vad_event(chunk)
            await engine.handle_owned_vad_event(end)
            assert [
                event.outcome for event in emitted if isinstance(event, STTProviderInputTerminal)
            ] == ["submitted"] * generation
            prepared.boundaries[-1].push(_gemini_message(final=f"generation-{generation}"))
            await _wait(
                lambda: sum(isinstance(event, STTRecognitionUnitTerminal) for event in emitted)
                == generation
            )
        finals = [event for event in emitted if isinstance(event, STTRecognitionUnitTerminal)]
        assert [event.unit.text for event in finals] == ["generation-1", "generation-2"]
        assert [event.unit.identity.stream.activation_generation for event in finals] == [1, 2]
        assert len({event.unit.identity.stream.provider_epoch_id for event in finals}) == 2
        assert prepared.opens == 2
        assert prepared.boundaries[0].closed
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_gemini_input_release_does_not_wait_for_recognition_consumer() -> None:
    prepared = _PreparedMemberBackend(lambda epoch: _gemini_session(epoch=epoch))
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, epoch: prepared.open_session(
            projection=STTSessionProjection("scoped", epoch)
        ),
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=_request("gemini_transcribe").settings,
    )
    consuming = asyncio.Event()
    release = asyncio.Event()
    finals = []

    async def consume(event):
        if isinstance(event, STTRecognitionUnitTerminal):
            consuming.set()
            await release.wait()
            finals.append(event.unit.text)
        elif isinstance(event, STTProviderInputTerminal):
            ledger.terminalize(
                event.identity.segment.segment_id,
                outcome=event.outcome,
                now_monotonic_s=0.0,
            )

    engine.bind_event_sink(consume)
    try:
        start, chunks, end = _owned_deepgram_events(ledger, start_sample=1000)
        await engine.handle_owned_vad_event(start)
        for chunk in chunks:
            await engine.handle_owned_vad_event(chunk)
        prepared.boundaries[0].push(_gemini_message(final="accepted final"))
        await asyncio.wait_for(consuming.wait(), timeout=1)
        await engine.handle_owned_vad_event(end)
        assert [receipt.outcome for receipt in ledger.terminal_receipts] == ["submitted"]
        assert finals == []
        release.set()
        await engine.wait_for_event_ingress_drain()
        assert finals == ["accepted final"]
    finally:
        release.set()
        await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("rolling_route", [False, True])
async def test_gemini_silent_stream_retires_without_native_completion_and_reopens_on_demand(
    rolling_route: bool,
) -> None:
    prepared = _PreparedMemberBackend(lambda epoch: _gemini_session(epoch=epoch))
    backend = (
        RollingSTTBackend(
            providers=(
                RollingProviderDefinition(
                    name=STTProviderName.GEMINI_TRANSCRIBE,
                    build_backend=lambda: prepared,
                    is_configured=lambda: True,
                ),
            )
        )
        if rolling_route
        else prepared
    )
    clock = ControlledMonotonicClock()
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        session_factory=lambda _settings, epoch: backend.open_session(
            projection=STTSessionProjection("scoped", epoch)
        ),
        event_sink=emitted.append,
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=_request("rolling_free" if rolling_route else "gemini_transcribe").settings,
    )
    try:
        start, chunks, end = _owned_deepgram_events(ledger, start_sample=1000)
        await engine.observe_source_activity(speech_observed=True)
        await engine.handle_owned_vad_event(start)
        for chunk in chunks:
            await engine.handle_owned_vad_event(chunk)
        await engine.handle_owned_vad_event(end)
        await engine.observe_source_activity(speech_observed=False)
        await _wait(lambda: any(isinstance(event, STTProviderInputTerminal) for event in emitted))
        last_capture = chunks[-1].event.chunk_capture[0]
        source_end = last_capture.normalized_end_sample
        assert source_end is not None
        silent_capture = replace(
            last_capture,
            source_start_sample=source_end,
            source_end_sample=source_end + 4,
            normalized_start_sample=source_end,
            normalized_end_sample=source_end + 4,
            source_start_monotonic_s=source_end / 16000,
            source_end_monotonic_s=(source_end + 4) / 16000,
        )
        silence = OwnedStreamInput(
            CaptureStreamInput(np.zeros(4, dtype=np.float32), (silent_capture,)),
            ledger,
            ledger.settings,
            1,
        )
        await clock.advance_to(59.9)
        await engine.handle_stream_input(silence)
        assert not prepared.boundaries[0].closed
        await clock.advance_to(60.0)
        await _wait(lambda: prepared.boundaries[0].closed)
        await engine.handle_stream_input(silence)
        assert prepared.opens == 1
        assert not any(isinstance(event, STTRecognitionUnitTerminal) for event in emitted)
        next_start, next_chunks, next_end = _owned_deepgram_events(ledger, start_sample=2000)
        await engine.observe_source_activity(speech_observed=True)
        await engine.handle_owned_vad_event(next_start)
        for chunk in next_chunks:
            await engine.handle_owned_vad_event(chunk)
        await engine.handle_owned_vad_event(next_end)
        prepared.boundaries[0].push(_gemini_message(final="retired"))
        prepared.boundaries[1].push(_gemini_message(final="fresh"))
        await _wait(lambda: any(isinstance(event, STTRecognitionUnitTerminal) for event in emitted))
        assert prepared.opens == 2
        assert [
            event.unit.text for event in emitted if isinstance(event, STTRecognitionUnitTerminal)
        ] == ["fresh"]
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_common_idle_close_defers_real_soniox_route_until_source_quiet() -> None:
    now = 0.0
    sleeper = _ControlledSleeper()
    prepared = _PreparedMemberBackend(lambda epoch: _soniox_session(epoch=epoch))
    emitted: list[object] = []

    async def open_session(_settings, provider_epoch_id: str):
        return await prepared.open_session(
            projection=STTSessionProjection("scoped", provider_epoch_id)
        )

    engine = ScopedRecognitionEngine(
        channel="peer",
        session_factory=open_session,
        event_sink=emitted.append,
        monotonic_clock=lambda: now,
        sleep=sleeper,
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(
            write_timeout_s=0.5,
            final_timeout_s=0.5,
            drain_timeout_s=0.1,
            idle_timeout_s=60.0,
        ),
    )
    settings = _request("soniox").settings
    start, chunks, end = _owned_deepgram_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings),
        start_sample=1000,
    )
    await engine.handle_owned_vad_event(start)
    for chunk in chunks:
        await engine.handle_owned_vad_event(chunk)
    session = prepared.sessions[0]
    socket = prepared.boundaries[0]
    end_task = asyncio.create_task(engine.handle_owned_vad_event(end))
    await _wait(lambda: session._event_projection.sealed)

    now = 200.0
    await engine.observe_source_activity(speech_observed=True, observed_at_monotonic_s=now)
    await asyncio.sleep(0)
    assert socket.closed is False
    assert end_task.done() is False

    now = 300.0
    await engine.observe_source_activity(speech_observed=True, observed_at_monotonic_s=now)
    await _complete_native_turn(STTProviderName.SONIOX, session, socket, "protected")
    await end_task
    assert socket.closed is False
    assert prepared.opens == 1

    await engine.observe_source_activity(speech_observed=False, observed_at_monotonic_s=now)
    now = 359.0
    await sleeper.release_next()
    await asyncio.sleep(0)
    assert socket.closed is False
    assert prepared.opens == 1
    now = 360.0
    await sleeper.release_next()
    await _wait(lambda: socket.closed)
    assert prepared.opens == 1
    assert len([event for event in emitted if isinstance(event, STTProviderTurnTerminal)]) == 1
    await engine.close()


@pytest.mark.asyncio
async def test_peer_production_dispatch_blocks_queued_b_before_soniox_a_terminal() -> None:
    prepared = _PreparedMemberBackend(lambda epoch: _soniox_session(epoch=epoch))
    emitted: list[object] = []

    async def open_session(_settings, provider_epoch_id: str):
        return await prepared.open_session(
            projection=STTSessionProjection("scoped", provider_epoch_id)
        )

    engine = ScopedRecognitionEngine(
        channel="peer",
        session_factory=open_session,
        event_sink=emitted.append,
        watchdog_resolver=lambda _settings: STTRecognitionWatchdogs(
            write_timeout_s=0.5,
            final_timeout_s=0.5,
            drain_timeout_s=0.1,
        ),
    )
    ingress_ready = asyncio.Event()
    ingress_ready.set()
    settings = _request("soniox").settings
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    dispatch = _GenerationGuardedVadSink(
        sink=engine,
        runtime=_CurrentPeerGeneration(ledger),
        capture_generation=_CaptureGeneration(1),
        provider_ingress_ready=ingress_ready,
    )
    a_events = _owned_deepgram_events(ledger, start_sample=1000)
    b_events = _owned_deepgram_events(ledger, start_sample=2000)

    ordered_events = (
        a_events[0],
        *a_events[1],
        a_events[2],
        b_events[0],
        *b_events[1],
        b_events[2],
    )
    for event in ordered_events:
        await dispatch.handle_owned_vad_event(event)

    await _wait(lambda: bool(prepared.sessions) and prepared.sessions[0]._event_projection.sealed)
    session = prepared.sessions[0]
    socket = prepared.boundaries[0]
    a_identity = session._event_projection.active_identity
    assert a_identity is not None
    assert a_identity.segment == a_events[0].segment.identity
    assert sum(isinstance(payload, bytes) and len(payload) == 8 for payload in socket.sent) == 21

    await _complete_native_turn(STTProviderName.SONIOX, session, socket, "a")
    await _wait(
        lambda: (
            session._event_projection.sealed
            and session._event_projection.active_identity is not None
            and session._event_projection.active_identity.segment == b_events[0].segment.identity
        )
    )
    assert sum(isinstance(payload, bytes) and len(payload) == 8 for payload in socket.sent) == 42
    await _complete_native_turn(STTProviderName.SONIOX, session, socket, "b")
    await _wait(
        lambda: len([event for event in emitted if isinstance(event, STTProviderTurnTerminal)]) == 2
    )

    terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
    assert [terminal.text for terminal in terminals] == ["a", "b"]
    assert terminals[0].identity.provider_epoch_id == terminals[1].identity.provider_epoch_id
    assert prepared.opens == 1
    await dispatch.finish()
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("member", "second_turn_at", "expected_connections"),
    [
        (STTProviderName.GEMINI_TRANSCRIBE, 550.0, 2),
        (STTProviderName.DEEPGRAM, 550.0, 1),
        (STTProviderName.DEEPGRAM, 3550.0, 2),
        (STTProviderName.ELEVENLABS_SCRIBE, 3550.0, 2),
    ],
)
async def test_rolling_connection_lifetime_follows_selected_provider(
    member: STTProviderName,
    second_turn_at: float,
    expected_connections: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def deepgram_write(session, payload):
        return None

    monkeypatch.setattr(_DeepgramSDKSession, "_write_payload", deepgram_write)

    async def open_member(epoch):
        if member is STTProviderName.GEMINI_TRANSCRIBE:
            return await _gemini_session(epoch=epoch)
        if member is STTProviderName.ELEVENLABS_SCRIBE:
            return _scribe_session(epoch=epoch)
        return _deepgram_session(epoch=epoch), None

    prepared = _PreparedMemberBackend(open_member)
    rolling = RollingSTTBackend(
        providers=(
            RollingProviderDefinition(
                name=member,
                build_backend=lambda: prepared,
                is_configured=lambda: True,
            ),
        )
    )
    now = 0.0
    sleeper = _ControlledSleeper()
    emitted = []

    async def open_session(settings, epoch):
        return await rolling.open_session(projection=STTSessionProjection("scoped", epoch))

    engine = ScopedRecognitionEngine(
        channel="peer",
        session_factory=open_session,
        event_sink=emitted.append,
        monotonic_clock=lambda: now,
        sleep=sleeper,
        watchdog_resolver=lambda _: _recognition_watchdogs(
            SimpleNamespace(provider="rolling_free", drain_timeout_s=0.1)
        ),
    )
    ledger = PeerAudioSegmentLedger(
        activation_generation=1, settings=_request("rolling_free").settings
    )
    try:
        for index, (sample, instant) in enumerate(((1000, 0.0), (2000, second_turn_at)), start=1):
            now = instant
            await engine.observe_source_activity(speech_observed=True)
            start, chunks, end = _owned_deepgram_events(ledger, start_sample=sample)
            await engine.handle_owned_vad_event(start)
            for chunk in chunks:
                await engine.handle_owned_vad_event(chunk)
            if member is STTProviderName.GEMINI_TRANSCRIBE:
                await engine.handle_owned_vad_event(end)
                await _complete_native_turn(
                    member, prepared.sessions[-1], prepared.boundaries[-1], str(sample)
                )
                await _wait(
                    lambda: sum(isinstance(event, STTRecognitionUnitTerminal) for event in emitted)
                    == index
                )
            else:
                end_task = asyncio.create_task(engine.handle_owned_vad_event(end))
                await _wait(lambda: prepared.sessions[-1]._event_projection.sealed)
                await _complete_native_turn(
                    member, prepared.sessions[-1], prepared.boundaries[-1], str(sample)
                )
                await end_task
        if member is STTProviderName.GEMINI_TRANSCRIBE:
            submitted = [event for event in emitted if isinstance(event, STTProviderInputTerminal)]
            await _wait(
                lambda: sum(isinstance(event, STTRecognitionUnitTerminal) for event in emitted) == 2
            )
            finals = [event for event in emitted if isinstance(event, STTRecognitionUnitTerminal)]
            assert [event.outcome for event in submitted] == ["submitted", "submitted"]
            assert [event.unit.text for event in finals] == ["1000", "2000"]
            assert (
                finals[0].unit.identity.stream.provider_epoch_id
                == finals[1].unit.identity.stream.provider_epoch_id
            ) is (expected_connections == 1)
        else:
            terminals = [event for event in emitted if isinstance(event, STTProviderTurnTerminal)]
            assert [event.text for event in terminals] == ["1000", "2000"]
            assert (
                terminals[0].identity.provider_epoch_id == terminals[1].identity.provider_epoch_id
            ) is (expected_connections == 1)
        assert prepared.opens == expected_connections
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_peer_cap_terminal_waits_for_source_seal_and_preserves_event_dispatch() -> None:
    owner, *_ = make_owner()
    await owner.apply_intent(make_config(), enabled=True)
    await _wait(lambda: owner.segment_ledger is not None)
    ledger = owner.segment_ledger
    clock = ControlledMonotonicClock()
    sessions: list[ControlledScopedSession] = []
    admitted = []

    async def factory(_settings, _epoch):
        session = ControlledScopedSession()
        session.terminal_on_seal = ("final", "after cap")
        sessions.append(session)
        return session

    def receive(event):
        if isinstance(event, STTProviderTurnTerminal):
            admitted.extend(owner.admit_provider_terminal(event))

    engine = ScopedRecognitionEngine(
        channel="peer",
        session_factory=factory,
        watchdog_resolver=lambda _: STTRecognitionWatchdogs(
            max_session_age_s=8.0, drain_timeout_s=0.1
        ),
        monotonic_clock=clock.now,
        sleep=clock.sleep,
    )
    engine.bind_event_sink(receive)
    raw_ledger = PeerAudioSegmentLedger(
        activation_generation=1, settings=_request("soniox").settings
    )
    start, chunks, end = _owned_deepgram_events(raw_ledger, start_sample=1000)

    async def deliver(event):
        await engine.handle_owned_vad_event(
            ledger.observe_vad_event(event.event, now_monotonic_s=clock.now())
        )

    try:
        await engine.observe_source_activity(speech_observed=True)
        for event in (start, *chunks):
            await deliver(event)
        await clock.advance_to(8.0)
        await _wait(lambda: ("close",) in sessions[0].calls)
        assert ledger.snapshots[0].state == "open"
        assert admitted == []
        await deliver(end)
        await engine.wait_for_event_ingress_drain()
        assert len(admitted) == 1
        assert admitted[0][0].failure_reason == "provider_session_lifetime_exceeded"
        assert admitted[0][1].identity.segment == start.segment.identity
        start, chunks, end = _owned_deepgram_events(raw_ledger, start_sample=2000)
        starting = asyncio.create_task(deliver(start))
        await _wait(lambda: any(not future.done() for _, future in clock.sleepers))
        await clock.advance_to(9.0)
        await starting
        for event in (*chunks, end):
            await deliver(event)
        await engine.wait_for_event_ingress_drain()
        assert len(sessions) == 2
        assert [terminal.outcome for _, terminal in admitted] == ["failed", "final"]
        assert admitted[1][1].text == "after cap"
        assert (
            admitted[0][1].identity.provider_epoch_id != admitted[1][1].identity.provider_epoch_id
        )
    finally:
        await engine.close()
        await owner.close()
