from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Callable

import pytest

from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.runtime.peer_channel import _CaptureGeneration, _GenerationGuardedVadSink
from puripuly_heart.core.stt.backend import (
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.rolling import RollingProviderDefinition, RollingSTTBackend
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine, STTRecognitionWatchdogs
from puripuly_heart.providers.stt.deepgram import _CLOSE_STREAM, _DeepgramSDKSession
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


class _CurrentPeerGeneration:
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
        boundary.push(_gemini_message(ack=True))
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
        (STTProviderName.GEMINI_TRANSCRIBE, False),
        (STTProviderName.DEEPGRAM, False),
        (STTProviderName.ELEVENLABS_SCRIBE, True),
        (STTProviderName.GEMINI_TRANSCRIBE, True),
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

    monkeypatch.setattr(_DeepgramSDKSession, "_write_thread_payload", deepgram_write)

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
    elif member is STTProviderName.GEMINI_TRANSCRIBE:
        assert sum(item.get("activity_start") is not None for item in boundary.sent) == 2
        assert sum(item.get("activity_end") is not None for item in boundary.sent) == 2
        assert boundary.closed is False
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
async def test_rolling_gemini_preserves_interim_on_outer_final_timeout() -> None:
    async def open_gemini(epoch: str):
        return await _gemini_session(timeout=1.0, epoch=epoch)

    prepared = _PreparedMemberBackend(open_gemini)
    rolling = RollingSTTBackend(
        providers=(
            RollingProviderDefinition(
                name=STTProviderName.GEMINI_TRANSCRIBE,
                build_backend=lambda: prepared,
                is_configured=lambda: True,
            ),
        )
    )
    emitted: list[object] = []

    async def open_session(_settings, provider_epoch_id: str):
        return await rolling.open_session(
            projection=STTSessionProjection("scoped", provider_epoch_id)
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
    settings = _request("rolling_free").settings
    start, chunks, end = _owned_deepgram_events(
        PeerAudioSegmentLedger(activation_generation=1, settings=settings),
        start_sample=1000,
    )

    await engine.handle_owned_vad_event(start)
    for chunk in chunks:
        await engine.handle_owned_vad_event(chunk)
    end_task = asyncio.create_task(engine.handle_owned_vad_event(end))
    await _wait(lambda: prepared.sessions[0]._event_projection.sealed)
    prepared.boundaries[0].push(_gemini_message(interim="interim only"))
    await _wait(lambda: any(isinstance(event, STTProviderTurnUpdate) for event in emitted))
    await end_task

    terminal = next(event for event in emitted if isinstance(event, STTProviderTurnTerminal))
    assert (terminal.outcome, terminal.text, terminal.text_authority) == (
        "degraded",
        "interim only",
        "degraded",
    )
    assert terminal.failure_reason == "provider_final_timeout"
    assert terminal.epoch_disposition == "retire"
    assert prepared.opens == 1
    await engine.close()


@pytest.mark.asyncio
async def test_common_age_rotation_defers_real_soniox_route_until_source_quiet() -> None:
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
            healthy_reset_age_s=180.0,
            recent_speech_window_s=10.0,
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
    now = 309.0
    assert await sleeper.release_next() == pytest.approx(10.0)
    await asyncio.sleep(0)
    assert socket.closed is False
    assert prepared.opens == 1
    now = 310.0
    assert await sleeper.release_next() == pytest.approx(1.0)
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
    dispatch = _GenerationGuardedVadSink(
        sink=engine,
        runtime=_CurrentPeerGeneration(),
        capture_generation=_CaptureGeneration(1),
        provider_ingress_ready=ingress_ready,
    )
    settings = _request("soniox").settings
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings)
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
