from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.format import AudioCaptureDiscontinuity
from puripuly_heart.core.audio.ownership import (
    CaptureStreamInput,
    OwnedStreamInput,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.stt.backend import (
    STTIndependentRecognitionSession,
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.stt.recognition_units import recognition_transcript
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.domain.recognition import slice_recognition_origins
from puripuly_heart.providers.stt import qwen_audio as qwen
from tests.core.test_stt_scoped_engine import (
    segment_events,
    settings,
    span,
    stream_input,
    wait_until,
    watchdogs,
)
from tests.providers.test_qwen_audio import FakeWebSocket, scoped_request


class ScriptedSocket(FakeWebSocket):
    def __init__(self, *, finish=True, tail=()):
        super().__init__()
        self.task_ids = []
        self.finish_automatically = finish
        self.tail = list(tail)
        self.finish_count = 0

    async def send(self, value):
        await super().send(value)
        if isinstance(value, bytes):
            return
        packet = json.loads(value)
        task = packet["header"]["task_id"]
        if packet["header"]["action"] == "run-task":
            self.task_ids.append(task)
            await self.push({"header": {"event": "task-started", "task_id": task}})
        else:
            self.finish_count += 1
            for sentence in self.tail:
                await self.push(native(task, **sentence))
            self.tail.clear()
            if self.finish_automatically:
                await self.push({"header": {"event": "task-finished", "task_id": task}})


def native(task, sentence_id=1, text=" No. ", **fields):
    return {
        "header": {"event": "result-generated", "task_id": task},
        "payload": {
            "output": {
                "sentence": {
                    "sentence_id": sentence_id,
                    "text": text,
                    "sentence_end": True,
                    **fields,
                }
            }
        },
    }


async def open_peer(socket=None, **options):
    socket = socket or ScriptedSocket()

    async def connect(*_args, **_kwargs):
        return socket

    backend = qwen.QwenAudioStreamingSTTBackend(
        api_key="offline",
        channel="peer",
        language_hints=("en",),
        websocket_factory=connect,
        task_start_timeout_s=0.2,
        task_finish_timeout_s=options.pop("task_finish_timeout_s", 0.2),
        **options,
    )
    session = await backend.open_session(
        projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
    )
    return session, socket


async def admit_stream(session):
    request = scoped_request("qwen_audio")
    await session.begin_turn(request)
    return request


async def next_native(session):
    return await asyncio.wait_for(session.turn_events().__anext__(), 1)


def wire_pcm(socket):
    return b"".join(value for value in socket.sent if isinstance(value, bytes))


@pytest.mark.asyncio
async def test_peer_local_closure_does_not_wait_for_final_and_tail_is_conserved():
    session, socket = await open_peer(
        ScriptedSocket(
            tail=[{"sentence_id": 3, "text": " tail. "}, {"sentence_id": 3, "text": " tail. "}]
        )
    )
    first = await admit_stream(session)
    try:
        await session.send_turn_audio(
            first.identity,
            b"\x01\x00" * 8000,
            payload_sequence=1,
            source_ranges=(span(0, 0, 8000),),
            context_only=True,
        )
        await session.seal_turn(
            first.identity,
            sealed_content_ranges=(span(0, 0, 8000),),
            seal_reason="delivery_deadline",
            observed_trailing_silence_ms=0,
        )
        terminal = await next_native(session)
        assert isinstance(terminal, STTProviderInputTerminal) and terminal.outcome == "submitted"
        assert socket.finish_count == 0
        second = replace(
            first,
            identity=replace(
                first.identity,
                segment=replace(first.identity.segment, segment_order=2, segment_id=uuid4()),
                provider_turn_id="second",
            ),
        )
        await session.begin_turn(second)
        await session.send_turn_audio(
            second.identity,
            b"\x01\x00" * 4000 + b"\x02\x00" * 4000,
            payload_sequence=1,
            source_ranges=(span(1, 4000, 12000),),
            context_only=True,
        )
        assert wire_pcm(socket) == b"\x01\x00" * 8000 + b"\x02\x00" * 4000
        assert len(socket.task_ids) == 1
        for sid in (1, 2, 2):
            await socket.push(native(socket.task_ids[0], sentence_id=sid, text="  No.  "))
        units = [await next_native(session), await next_native(session)]
        assert all(isinstance(unit, STTRecognitionUnit) for unit in units)
        assert [unit.text for unit in units] == ["  No.  ", "  No.  "]
        assert units[0].identity != units[1].identity
        assert session.state is qwen.QwenAudioSessionState.TASK_ACTIVE
        await session.end_stream(reason="source_eof")
        tail = await next_native(session)
        assert tail.text == " tail. "
        assert tail.provenance.native_event_id == "3"
        assert session._event_projection.scoped_event_depth == 0
        assert socket.finish_count == 1 and socket.closed
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_peer_native_evidence_survives_source_mapping_origins_slicing_and_budget():
    session, socket = await open_peer()
    request = await admit_stream(session)
    try:
        capture = replace(
            span(0, 16000, 24000),
            discontinuity_before=AudioCaptureDiscontinuity("known_loss", 1.0, 57),
        )
        await session.send_turn_audio(
            request.identity,
            b"\0\0" * 8000,
            payload_sequence=1,
            source_ranges=(capture,),
            context_only=True,
        )
        await session.send_stream_audio(b"\1\0" * 8000, source_ranges=(span(1, 24000, 32000),))
        await socket.push(
            native(
                socket.task_ids[0],
                text=" No. ",
                begin_time=0,
                end_time=1003.5625,
                words=[
                    {
                        "text": " No",
                        "punctuation": ". ",
                        "begin_time": 0,
                        "end_time": 1003.5625,
                        "fixed": True,
                    }
                ],
            )
        )
        unit = await next_native(session)
        evidence = unit.provenance.transcription
        assert evidence.native_task_id == socket.task_ids[0]
        assert evidence.native_sentence_id == "1"
        assert evidence.end_time_ms == 1003.5625
        assert evidence.task_input_samples == 16000
        assert [
            (
                r.input_start_sample,
                r.input_end_sample,
                r.normalized_start_sample,
                r.normalized_end_sample,
                r.context_only,
            )
            for r in evidence.audio_coverage
        ] == [(0, 8000, 16000, 24000, True), (8000, 16000, 24000, 32000, False)]
        assert evidence.timed_words[0].punctuation == ". "
        assert evidence.audio_coverage[0].discontinuity_kind == "known_loss"
        assert evidence.audio_coverage[0].lost_source_samples == 57
        assert evidence.audio_coverage[0].discontinuity_observed_at_monotonic_s == 1.0
        transcript = recognition_transcript(unit, created_at=0, publication_order=1)
        origin = slice_recognition_origins(transcript.recognition_origins, 1, 4)[0]
        assert origin.native is evidence
        assert origin.text_start == 0 and origin.text_end == 3
        assert origin.native.end_time_ms * 16 - origin.native.task_input_samples == 57
        assert unit.retained_bytes >= len(unit.text.encode()) + 2 * 192 + 64
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_missing_timing_remains_usable_and_fixed_heartbeat_do_not_admit():
    session, socket = await open_peer()
    request = await admit_stream(session)
    try:
        await session.send_turn_audio(
            request.identity,
            b"\0\0" * 6,
            payload_sequence=1,
            source_ranges=(span(0, 0, 6),),
            context_only=False,
        )
        await socket.push(
            native(socket.task_ids[0], sentence_end=False, fixed=True, text="partial")
        )
        await socket.push(native(socket.task_ids[0], sentence_id=0, heartbeat=True))
        await socket.push(native(socket.task_ids[0], text="Yes. ", words=None))
        unit = await next_native(session)
        assert unit.text == "Yes. "
        assert unit.provenance.transcription.begin_time_ms is None
        assert unit.provenance.transcription.audio_coverage[0].normalized_end_sample == 6
        assert session._event_projection.scoped_event_depth == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_duplicate_native_identity_is_ignored_but_unseen_reversed_final_fails():
    session, socket = await open_peer()
    await admit_stream(session)
    try:
        await socket.push(native(socket.task_ids[0], sentence_id=2))
        first = await next_native(session)
        await socket.push(native(socket.task_ids[0], sentence_id=2, text="different duplicate"))
        await socket.push(native("retired-task", sentence_id=3, text="stale"))
        await socket.push(native(socket.task_ids[0], sentence_id=1, text="reversed"))
        terminal = await next_native(session)
        ended = await next_native(session)
        assert first.text == " No. "
        assert isinstance(terminal, STTProviderInputTerminal) and terminal.outcome == "failed"
        assert isinstance(ended, STTProviderEpochEnded) and not ended.orderly
        assert session.state is qwen.QwenAudioSessionState.FAILED
        await wait_until(lambda: socket.closed)
        assert socket.closed
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_no_final_rotation_resets_task_origin_and_native_namespace_without_replay():
    session, socket = await open_peer()
    await admit_stream(session)
    pcm = b"\x03\0" * (13 * 16000)
    try:
        await session.send_stream_audio(pcm, source_ranges=(span(0, 0, 13 * 16000),))
        assert len(socket.task_ids) == 2
        assert wire_pcm(socket) == pcm
        assert sum(len(v) for v in socket.sent if isinstance(v, bytes)) == len(pcm)
        await socket.push(native(socket.task_ids[0], text="stale old task"))
        await socket.push(native(socket.task_ids[1], text="No. ", begin_time=0, end_time=1000))
        unit = await next_native(session)
        assert unit.text == "No. "
        evidence = unit.provenance.transcription
        assert evidence.task_input_samples == 16000
        assert evidence.audio_coverage[0].input_start_sample == 0
        assert evidence.audio_coverage[0].normalized_start_sample == 192000
        assert socket.finish_count == 1
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_sentence_count_rotation_and_same_sentence_id_on_replacement():
    session, socket = await open_peer()
    await admit_stream(session)
    try:
        for sid in range(1, 257):
            await socket.push(native(socket.task_ids[0], sentence_id=sid, text="No. "))
            unit = await next_native(session)
            assert unit.text == "No. "
        old = socket.task_ids[0]
        await wait_until(lambda: len(socket.task_ids) == 2)
        await socket.push(native(socket.task_ids[1], sentence_id=1, text="No. "))
        second = await next_native(session)
        assert second.provenance.native_event_id == "1"
        assert second.provenance.native_task_id != old
        assert second.identity.receipt_sequence == 257
        assert socket.finish_count == 1
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["map", "metadata"])
async def test_mapping_pressure_rotates_before_exceeding_evidence_budget(monkeypatch, limit):
    if limit == "map":
        monkeypatch.setattr(qwen, "QWEN_PEER_MAX_MAP_RECORDS", 2)
    else:
        monkeypatch.setattr(qwen, "QWEN_PEER_MAX_METADATA_BYTES", 384)
    session, socket = await open_peer()
    await admit_stream(session)
    try:
        for i in range(3):
            capture = replace(
                span(i, i, i + 1),
                source_start_monotonic_s=float(i),
                source_end_monotonic_s=float(i) + 0.01,
            )
            await session.send_stream_audio(b"\x04\0", source_ranges=(capture,))
        assert wire_pcm(socket) == b"\x04\0" * 3
        assert len(socket.task_ids) == 2
        assert socket.finish_count == 1
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_task_pcm_cap_rotates_even_when_finals_keep_arriving():
    session, socket = await open_peer()
    await admit_stream(session)
    try:
        for i in range(6):
            await session.send_stream_audio(
                b"\1\0" * 160000, source_ranges=(span(i, i * 160000, (i + 1) * 160000),)
            )
            if i < 5:
                await socket.push(native(socket.task_ids[0], sentence_id=i + 1))
                await next_native(session)
        await wait_until(lambda: len(socket.task_ids) == 2)
        assert len(wire_pcm(socket)) == 60 * 16000 * 2
        assert socket.finish_count == 1
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_task_wall_age_rotates_and_peer_never_fabricates_keepalive(monkeypatch):
    monkeypatch.setattr(qwen, "QWEN_PEER_TASK_MAX_S", 0.03)
    session, socket = await open_peer(keepalive_interval_s=0.005)
    await admit_stream(session)
    try:
        await wait_until(lambda: len(socket.task_ids) >= 2)
        assert wire_pcm(socket) == b""
        assert socket.finish_count >= 1
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_missing_task_terminal_fails_drain_without_inventing_empty_recognition():
    session, socket = await open_peer(ScriptedSocket(finish=False), task_finish_timeout_s=0.01)
    await admit_stream(session)
    try:
        with pytest.raises(qwen.QwenAudioProtocolError):
            await session.end_stream(reason="source_eof")
        events = [await next_native(session), await next_native(session)]
        assert isinstance(events[0], STTProviderInputTerminal) and events[0].outcome == "failed"
        assert isinstance(events[1], STTProviderEpochEnded) and not events[1].orderly
        assert not any(isinstance(event, STTRecognitionUnit) for event in events)
        assert socket.closed
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_recognition_byte_pressure_fails_epoch_and_closes_transport():
    session, socket = await open_peer()
    await admit_stream(session)
    try:
        await socket.push(native(socket.task_ids[0], text="x" * (1024 * 1024 + 1)))
        await wait_until(lambda: socket.closed)
        events = []
        while session._event_projection.scoped_event_depth:
            events.append(await next_native(session))
        assert any(
            isinstance(event, STTProviderEpochEnded) and not event.orderly for event in events
        )
        assert not any(isinstance(event, STTRecognitionUnit) for event in events)
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidation", ["stopped", "configuration", "source_discontinuity"])
async def test_engine_invalidation_rejects_old_units_and_fresh_source_recovers(invalidation):
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("qwen_audio"))
    sockets, sessions, events = [], [], []

    async def factory(_settings, epoch):
        socket = ScriptedSocket()

        async def connect(*_args, **_kwargs):
            return socket

        session = await qwen.QwenAudioStreamingSTTBackend(
            api_key="offline", channel="peer", websocket_factory=connect
        ).open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch))
        sockets.append(socket)
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        factory, channel="peer", event_sink=events.append, watchdog_resolver=lambda _: watchdogs()
    )
    start, chunk, end = segment_events(ledger, start_sample=0, now=0)
    try:
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(chunk)
        await engine.handle_owned_vad_event(end)
        old_stream = sessions[0]._stream
        if invalidation == "source_discontinuity":
            await engine.handle_stream_input(
                OwnedStreamInput(
                    CaptureStreamInput(
                        np.empty(0, dtype=np.float32), (), boundary_reason="source_discontinuity"
                    ),
                    ledger,
                    ledger.settings,
                    3,
                )
            )
        else:
            await engine.abort(reason=invalidation)
        await wait_until(lambda: sockets[0].closed)
        await sockets[0].push(native(sockets[0].task_ids[0], text="stale"))
        next_start, _, _ = segment_events(ledger, start_sample=100, now=1)
        await engine.handle_owned_vad_event(next_start)
        await sockets[1].push(native(sockets[1].task_ids[0], text="fresh"))
        await wait_until(
            lambda: any(isinstance(event, STTRecognitionUnitTerminal) for event in events)
        )
        finals = [event.unit for event in events if isinstance(event, STTRecognitionUnitTerminal)]
        assert [unit.text for unit in finals] == ["fresh"]
        assert finals[0].identity.stream != old_stream
        assert not engine.is_current_recognition_stream(old_stream)
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_idle_retirement_drains_tail_before_revoking_current_authority():
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("qwen_audio"))
    socket = ScriptedSocket(tail=[{"sentence_id": 1, "text": "idle tail"}])
    events = []

    async def factory(_settings, epoch):
        async def connect(*_args, **_kwargs):
            return socket

        return await qwen.QwenAudioStreamingSTTBackend(
            api_key="offline", channel="peer", websocket_factory=connect
        ).open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch))

    engine = ScopedRecognitionEngine(
        factory,
        channel="peer",
        event_sink=events.append,
        watchdog_resolver=lambda _: watchdogs(idle_timeout_s=0.02),
    )
    start, chunk, end = segment_events(ledger, start_sample=0, now=0)
    try:
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(chunk)
        await engine.handle_owned_vad_event(end)
        await wait_until(lambda: socket.closed)
        await wait_until(
            lambda: any(isinstance(event, STTRecognitionUnitTerminal) for event in events)
        )
        assert [
            event.unit.text for event in events if isinstance(event, STTRecognitionUnitTerminal)
        ] == ["idle tail"]
        assert socket.finish_count == 1
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_mapped_final_does_not_inherit_later_source_vad_time():
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("qwen_audio"))
    socket = ScriptedSocket()
    events = []

    async def factory(_settings, epoch):
        async def connect(*_args, **_kwargs):
            return socket

        return await qwen.QwenAudioStreamingSTTBackend(
            api_key="offline", channel="peer", websocket_factory=connect
        ).open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch))

    engine = ScopedRecognitionEngine(
        factory, channel="peer", event_sink=events.append, watchdog_resolver=lambda _: watchdogs()
    )
    start, _, _ = segment_events(ledger, start_sample=0, now=0)
    try:
        await engine.handle_owned_vad_event(start)
        await engine.handle_stream_input(stream_input(ledger, 6, 16006, speech_observed=True))
        await socket.push(native(socket.task_ids[0], text="earlier", begin_time=0, end_time=0.375))
        await wait_until(
            lambda: any(isinstance(event, STTRecognitionUnitTerminal) for event in events)
        )
        unit = next(event.unit for event in events if isinstance(event, STTRecognitionUnitTerminal))
        assert unit.estimated_last_speech_at is None
        assert unit.provenance.transcription.end_time_ms == 0.375
    finally:
        await engine.close()


def test_peer_qwen_frozen_bounds():
    assert (
        qwen.QWEN_PEER_SENTENCE_SILENCE_MS,
        qwen.QWEN_PEER_NO_FINAL_S,
        qwen.QWEN_PEER_TASK_MAX_S,
    ) == (800, 12, 60)
    assert (
        qwen.QWEN_PEER_MAX_SENTENCES,
        qwen.QWEN_PEER_MAX_MAP_RECORDS,
        qwen.QWEN_PEER_MAX_METADATA_BYTES,
    ) == (256, 4096, 256 * 1024)


@pytest.mark.asyncio
async def test_outer_write_watchdog_bounds_rotation_and_recovery_never_replays_attempted_input():
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("qwen_audio"))
    sockets, events = [], []

    async def factory(_settings, epoch):
        socket = ScriptedSocket(finish=bool(sockets))

        async def connect(*_args, **_kwargs):
            return socket

        sockets.append(socket)
        return await qwen.QwenAudioStreamingSTTBackend(
            api_key="offline", channel="peer", websocket_factory=connect, task_finish_timeout_s=0.2
        ).open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch))

    engine = ScopedRecognitionEngine(
        factory,
        channel="peer",
        event_sink=events.append,
        watchdog_resolver=lambda _: watchdogs(write_timeout_s=0.02),
    )
    start, _, _ = segment_events(ledger, start_sample=0, now=0)
    try:
        await engine.handle_owned_vad_event(start)
        await engine.handle_stream_input(stream_input(ledger, 6, 192006, speech_observed=True))
        await wait_until(lambda: sockets[0].closed)
        assert sockets[0].finish_count == 1
        assert not any(isinstance(event, STTRecognitionUnitTerminal) for event in events)
        await engine.handle_stream_input(stream_input(ledger, 192006, 192010, speech_observed=True))
        assert len(sockets) == 2
        assert len(wire_pcm(sockets[1])) == 8
        await sockets[1].push(native(sockets[0].task_ids[0], text="late failed task"))
        await sockets[1].push(native(sockets[1].task_ids[0], text="recovered"))
        await wait_until(
            lambda: any(isinstance(event, STTRecognitionUnitTerminal) for event in events)
        )
        assert [
            event.unit.text for event in events if isinstance(event, STTRecognitionUnitTerminal)
        ] == ["recovered"]
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_idle_drain_retires_session_and_does_not_reconnect_on_quiet_frames():
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("qwen_audio"))
    sockets, events = [], []

    async def factory(_settings, epoch):
        socket = ScriptedSocket(tail=[{"sentence_id": 1, "text": "drained"}])

        async def connect(*_args, **_kwargs):
            return socket

        sockets.append(socket)
        return await qwen.QwenAudioStreamingSTTBackend(
            api_key="offline", channel="peer", websocket_factory=connect
        ).open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch))

    engine = ScopedRecognitionEngine(
        factory,
        channel="peer",
        event_sink=events.append,
        watchdog_resolver=lambda _: watchdogs(idle_timeout_s=0.02),
    )
    start, chunk, end = segment_events(ledger, start_sample=0, now=0)
    try:
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(chunk)
        await engine.handle_owned_vad_event(end)
        await wait_until(lambda: engine._session is None)
        assert [
            event.unit.text for event in events if isinstance(event, STTRecognitionUnitTerminal)
        ] == ["drained"]
        await engine.handle_stream_input(stream_input(ledger, 10, 14))
        assert len(sockets) == 1
        await engine.handle_stream_input(stream_input(ledger, 14, 18, speech_observed=True))
        assert len(sockets) == 2
        assert len(wire_pcm(sockets[1])) == 8
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_factory_binds_only_direct_peer_qwen_to_independent_recognition():
    from puripuly_heart.app.wiring.wiring_stt_factory import create_stt_backend_from_resolved_config
    from puripuly_heart.config.runtime_resolution import STTRuntimeIntent, resolve_stt_config
    from puripuly_heart.core.storage.secrets import InMemorySecretStore

    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "offline")
    for channel, threshold in (("self", 6000), ("peer", 800)):
        resolved = resolve_stt_config(
            STTRuntimeIntent(channel=channel, provider="qwen_audio", source_language="en")
        )
        backend = create_stt_backend_from_resolved_config(resolved, secrets=secrets)
        socket = ScriptedSocket()

        async def connect(*_args, **_kwargs):
            return socket

        backend.websocket_factory = connect
        session = await backend.open_session(
            projection=STTSessionProjection(mode="scoped", provider_epoch_id="epoch-1")
        )
        try:
            assert isinstance(session, STTIndependentRecognitionSession) is (channel == "peer")
            parameters = json.loads(socket.sent[0])["payload"]["parameters"]
            assert parameters["max_sentence_silence"] == threshold
        finally:
            if channel == "self":
                await session.abort_for_toggle_off()
            await session.close()


@pytest.mark.asyncio
async def test_peer_rejects_legacy_unowned_audio_and_per_turn_task_control():
    session, socket = await open_peer()
    try:
        with pytest.raises(RuntimeError, match="owned source"):
            await session.send_audio(b"\0\0")
        with pytest.raises(RuntimeError, match="seal_turn"):
            await session.on_speech_end()
        assert wire_pcm(socket) == b"" and socket.finish_count == 0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_peer_transport_write_failure_retires_without_empty_success():
    session, socket = await open_peer()
    request = await admit_stream(session)
    socket.fail_audio = True
    try:
        with pytest.raises(qwen.QwenAudioProtocolError):
            await session.send_turn_audio(
                request.identity,
                b"\0\0" * 8,
                payload_sequence=1,
                source_ranges=(span(0, 0, 8),),
                context_only=False,
            )
        await wait_until(lambda: socket.closed)
        terminal, ended = await next_native(session), await next_native(session)
        assert isinstance(terminal, STTProviderInputTerminal) and terminal.outcome == "failed"
        assert isinstance(ended, STTProviderEpochEnded) and not ended.orderly
        assert not wire_pcm(socket)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_final_received_inside_successful_write_uses_committed_capture_map():
    class ImmediateFinalSocket(ScriptedSocket):
        async def send(self, value):
            await super().send(value)
            if isinstance(value, bytes):
                await self.push(
                    native(self.task_ids[-1], text="immediate", begin_time=0, end_time=500)
                )
                await asyncio.sleep(0)

    session, socket = await open_peer(ImmediateFinalSocket())
    request = await admit_stream(session)
    try:
        await session.send_turn_audio(
            request.identity,
            b"\1\0" * 8000,
            payload_sequence=1,
            source_ranges=(span(0, 24000, 32000),),
            context_only=True,
        )
        unit = await next_native(session)
        native_evidence = unit.provenance.transcription
        assert native_evidence.task_input_samples == 8000
        assert [
            (item.normalized_start_sample, item.normalized_end_sample)
            for item in native_evidence.audio_coverage
        ] == [(24000, 32000)]
        assert wire_pcm(socket) == b"\1\0" * 8000
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["active", "sealed", "eof"])
async def test_connection_lifetime_drains_current_tail_even_at_expired_input_deadline(completion):
    now = [0.0]
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("qwen_audio"))
    socket = ScriptedSocket(tail=[{"sentence_id": 1, "text": "lifetime tail"}])
    events = []

    async def factory(_settings, epoch):
        async def connect(*_args, **_kwargs):
            return socket

        return await qwen.QwenAudioStreamingSTTBackend(
            api_key="offline", channel="peer", websocket_factory=connect
        ).open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch))

    engine = ScopedRecognitionEngine(
        factory,
        channel="peer",
        event_sink=events.append,
        monotonic_clock=lambda: now[0],
        watchdog_resolver=lambda _: watchdogs(max_session_age_s=0.04),
    )
    start, chunk, end = segment_events(ledger, start_sample=0, now=0)
    try:
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(chunk)
        if completion != "active":
            now[0] = 0.03
            await engine.handle_owned_vad_event(end)
            assert not socket.closed
        now[0] = 0.05
        if completion == "eof":
            await engine.handle_stream_input(
                OwnedStreamInput(
                    CaptureStreamInput(
                        np.empty(0, dtype=np.float32), (), boundary_reason="source_eof"
                    ),
                    ledger,
                    ledger.settings,
                    3,
                )
            )
        await wait_until(lambda: engine._session is None)
        assert [
            event.unit.text for event in events if isinstance(event, STTRecognitionUnitTerminal)
        ] == ["lifetime tail"]
        assert socket.finish_count == 1
        assert socket.closed
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_next_onset_near_connection_ceiling_drains_before_new_provider_epoch():
    now = [0.0]
    ledger = PeerAudioSegmentLedger(activation_generation=3, settings=settings("qwen_audio"))
    sockets, events = [], []

    async def factory(_settings, epoch):
        socket = ScriptedSocket(tail=[{"sentence_id": 1, "text": "old tail"}])

        async def connect(*_args, **_kwargs):
            return socket

        sockets.append(socket)
        return await qwen.QwenAudioStreamingSTTBackend(
            api_key="offline", channel="peer", websocket_factory=connect
        ).open_session(projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch))

    engine = ScopedRecognitionEngine(
        factory,
        channel="peer",
        event_sink=events.append,
        monotonic_clock=lambda: now[0],
        watchdog_resolver=lambda _: watchdogs(max_session_age_s=0.04),
    )
    start, chunk, end = segment_events(ledger, start_sample=0, now=0)
    try:
        await engine.handle_owned_vad_event(start)
        await engine.handle_owned_vad_event(chunk)
        now[0] = 0.03
        await engine.handle_owned_vad_event(end)
        successor, _, _ = segment_events(ledger, start_sample=100, now=1)
        await asyncio.wait_for(engine.handle_owned_vad_event(successor), 1)
        assert len(sockets) == 2 and sockets[0].closed
        finals = [event.unit for event in events if isinstance(event, STTRecognitionUnitTerminal)]
        assert [unit.text for unit in finals] == ["old tail"]
        assert finals[0].provenance.native_task_id == sockets[0].task_ids[0]
        assert len(wire_pcm(sockets[1])) == 12
    finally:
        await engine.close()
