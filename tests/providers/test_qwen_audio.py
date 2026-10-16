from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.format import AudioCaptureDiscontinuity
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.stt.backend import (
    STTContributionConsumptionLedger,
    STTProviderTurnIdentity,
    STTProviderTurnRequest,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTSessionProjection,
)
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.domain.recognition import slice_recognition_origins
from puripuly_heart.providers.stt import qwen_audio as qwen
from tests.core.test_stt_scoped_engine import settings, span, wait_until


class ScriptedSocket:
    def __init__(self, *, ready=True, finish=False):
        self.sent = []
        self.incoming = asyncio.Queue()
        self.task_ids = []
        self.closed = False
        self.auto_ready = ready
        self.auto_finish = finish
        self.fail_audio = False
        self.fail_control = False
        self.audio_gate = None

    async def send(self, value):
        if isinstance(value, bytes):
            if self.audio_gate is not None:
                await self.audio_gate.wait()
            if self.fail_audio:
                raise ConnectionError("scripted uncertain write")
        elif self.fail_control:
            raise ConnectionError("scripted control failure")
        self.sent.append(value)
        if isinstance(value, bytes):
            return
        packet = json.loads(value)
        task_id = packet["header"]["task_id"]
        if packet["header"]["action"] == "run-task":
            self.task_ids.append(task_id)
            if self.auto_ready:
                await self.push(control(task_id, "task-started"))
        elif self.auto_finish:
            await self.push(control(task_id, "task-finished"))

    async def recv(self):
        return await self.incoming.get()

    async def push(self, value):
        await self.incoming.put(value)

    async def close(self):
        self.closed = True
        await self.incoming.put(None)


class SocketFactory:
    def __init__(self, sockets=()):
        self.planned = list(sockets)
        self.sockets = []
        self.options = []

    async def __call__(self, *args, **kwargs):
        socket = self.planned.pop(0) if self.planned else ScriptedSocket()
        self.sockets.append(socket)
        self.options.append((args, kwargs))
        return socket


def control(task, event, **fields):
    return {"header": {"event": event, "task_id": task, **fields}}


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


def request(channel="peer", order=1):
    identity = STTProviderTurnIdentity(
        AudioSegmentIdentity(1, order, uuid4(), 1),
        "epoch-1",
        f"turn-{order}",
        ("qwen_audio",),
    )
    return STTProviderTurnRequest(identity, settings("qwen_audio"), channel)


async def open_session(channel="peer", factory=None, **options):
    factory = factory or SocketFactory()
    backend = qwen.QwenAudioStreamingSTTBackend(
        api_key="offline",
        channel=channel,
        language_hints=("en",),
        websocket_factory=factory,
        task_start_timeout_s=options.pop("task_start_timeout_s", 0.2),
        task_finish_timeout_s=options.pop("task_finish_timeout_s", 0.2),
        send_timeout_s=options.pop("send_timeout_s", 0.2),
        **options,
    )
    session = await backend.open_session(projection=STTSessionProjection("scoped", "epoch-1"))
    return session, factory


async def send(session, req, *, start=0, count=8000, sequence=1, context=False, capture=None):
    pcm = b"\x01\x00" * count
    await session.send_turn_audio(
        req.identity,
        pcm,
        payload_sequence=sequence,
        source_ranges=(capture or span(sequence, start, start + count),),
        context_only=context,
    )
    return pcm


async def seal(session, req):
    await session.seal_turn(
        req.identity,
        sealed_content_ranges=(),
        seal_reason="silence",
        observed_trailing_silence_ms=224,
    )


async def collect(session, target):
    async for event in session.turn_events():
        target.append(event)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_two_slots_are_demand_driven_ready_gated_and_reused_without_replay(channel):
    session, factory = await open_session(channel)
    events = []
    receiver = asyncio.create_task(collect(session, events))
    try:
        assert factory.sockets == []
        first, second, third = [request(channel, order) for order in (1, 2, 3)]
        await session.begin_turn(first)
        assert factory.sockets == []
        pcm_a = await send(session, first)
        await seal(session, first)
        await session.begin_turn(second)
        pcm_b = await send(session, second, start=8000)
        assert len(factory.sockets) == 2
        a, b = factory.sockets
        assert a.task_ids != b.task_ids
        assert any(isinstance(value, bytes) for value in b.sent)
        assert not any(isinstance(event, STTProviderTurnTerminal) for event in events)
        await seal(session, second)
        await a.push(native(a.task_ids[0], text="A"))
        await a.push(control(a.task_ids[0], "task-finished"))
        await wait_until(
            lambda: any(isinstance(event, STTProviderTurnTerminal) for event in events)
        )
        await session.begin_turn(third)
        pcm_c = await send(session, third, start=16000)
        await seal(session, third)
        assert len(factory.sockets) == 2
        assert len(a.task_ids) == 2 and len(b.task_ids) == 1
        assert b"".join(value for value in a.sent if isinstance(value, bytes)) == pcm_a + pcm_c
        assert b"".join(value for value in b.sent if isinstance(value, bytes)) == pcm_b
        for socket in (a, b):
            await socket.push(control(socket.task_ids[-1], "task-finished"))
        await wait_until(
            lambda: sum(isinstance(event, STTProviderTurnTerminal) for event in events) == 3
        )
        await asyncio.sleep(0.01)
        assert len(a.task_ids) + len(b.task_ids) == 3
        assert all(
            json.loads(socket.sent[0])["payload"]["parameters"]["max_sentence_silence"]
            == (800 if channel == "peer" else 6000)
            for socket in factory.sockets
        )
    finally:
        await session.close()
        await receiver
    assert all(socket.closed for socket in factory.sockets)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_readiness_prevents_binary_and_times_out_without_replay(channel):
    socket = ScriptedSocket(ready=False)
    session, _ = await open_session(channel, SocketFactory([socket]), task_start_timeout_s=0.03)
    req = request(channel)
    await session.begin_turn(req)
    writing = asyncio.create_task(send(session, req))
    await wait_until(lambda: bool(socket.task_ids))
    assert not any(isinstance(value, bytes) for value in socket.sent)
    await socket.push(control("wrong-task", "task-started"))
    await asyncio.sleep(0)
    assert not writing.done()
    with pytest.raises(TimeoutError):
        await writing
    event = await session.turn_events().__anext__()
    assert event.outcome == "failed" and event.failure_reason == "qwen_readiness_failed"
    assert socket.closed and not any(isinstance(value, bytes) for value in socket.sent)
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_finals_before_seal_repeats_and_drain_tail_are_conserved_exactly(channel):
    session, factory = await open_session(channel)
    req = request(channel)
    events = []
    receiver = asyncio.create_task(collect(session, events))
    try:
        await session.begin_turn(req)
        await send(session, req)
        socket = factory.sockets[0]
        task = socket.task_ids[0]
        await socket.push(native(task, 0, "heartbeat"))
        await socket.push(native(task, 1, "ignored", sentence_end=False, fixed=True))
        await socket.push(native("stale-task", 1, "stale"))
        for sid in (1, 1, 2):
            await socket.push(native(task, sid, "  No.  "))
        await wait_until(lambda: session._slots[0].task.last_sentence_id == 2)
        assert not any(isinstance(event, STTProviderTurnTerminal) for event in events)
        if channel == "peer":
            assert events == []
        else:
            await wait_until(lambda: len(events) == 2)
        await seal(session, req)
        await socket.push(native(task, 3, " tail. "))
        await socket.push(native(task, 3, " tail. "))
        await socket.push(control(task, "task-finished"))
        await wait_until(
            lambda: any(isinstance(event, STTProviderTurnTerminal) for event in events)
        )
        terminal = next(event for event in events if isinstance(event, STTProviderTurnTerminal))
        assert terminal.text == "  No.    No.   tail. "
        assert [terminal.text[o.text_start : o.text_end] for o in terminal.recognition_origins] == [
            "  No.  ",
            "  No.  ",
            " tail. ",
        ]
        normalizer = STTScopedTurnNormalizer(req.identity, preserve_text=True)
        ledger = STTContributionConsumptionLedger()
        consumed = []
        for event in events:
            normalized = (
                normalizer.apply_update(event)
                if isinstance(event, STTProviderTurnUpdate)
                else normalizer.apply_terminal(event)
            )
            if normalized is not None:
                consumed.append(ledger.consume(normalized))
        assert "".join(consumed) == terminal.text
        assert len({origin.identity for origin in terminal.recognition_origins}) == 3
    finally:
        await session.close()
        await receiver


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_native_task_clock_and_capture_context_gap_evidence_survive_slicing(channel):
    session, factory = await open_session(channel)
    events = []
    receiver = asyncio.create_task(collect(session, events))
    try:
        for order in (1, 2):
            req = request(channel, order)
            await session.begin_turn(req)
            capture = replace(
                span(order, order * 16000, order * 16000 + 8000),
                discontinuity_before=AudioCaptureDiscontinuity("known_loss", float(order), 57),
            )
            await send(session, req, capture=capture, context=True)
            await send(session, req, start=order * 16000 + 8000, sequence=2)
            socket = factory.sockets[order - 1]
            await socket.push(
                native(
                    socket.task_ids[0],
                    1,
                    " No. ",
                    begin_time=250,
                    end_time=1100,
                    words=[{"text": "No", "begin_time": 250, "end_time": 1000}],
                )
            )
            await seal(session, req)
            await socket.push(control(socket.task_ids[0], "task-finished"))
        await wait_until(
            lambda: sum(isinstance(event, STTProviderTurnTerminal) for event in events) == 2
        )
        finals = [event for event in events if isinstance(event, STTProviderTurnTerminal)]
        for order, terminal in enumerate(finals, 1):
            origin = terminal.recognition_origins[0]
            native_evidence = origin.native
            assert native_evidence.begin_time_ms == 250 and native_evidence.end_time_ms == 1100
            assert native_evidence.task_input_samples == 16000
            assert [
                (c.input_start_sample, c.input_end_sample, c.context_only)
                for c in native_evidence.audio_coverage
            ] == [(4000, 8000, True), (8000, 16000, False)]
            assert native_evidence.audio_coverage[0].normalized_start_sample == order * 16000 + 4000
            assert native_evidence.audio_coverage[-1].normalized_end_sample == order * 16000 + 16000
            sliced = slice_recognition_origins(terminal.recognition_origins, 1, 4)
            assert (
                sliced[0].native == native_evidence
                and sliced[0].text_start == 0
                and sliced[0].text_end == 3
            )
        assert (
            finals[0].recognition_origins[0].native.native_task_id
            != finals[1].recognition_origins[0].native.native_task_id
        )
    finally:
        await session.close()
        await receiver


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "write",
        "write_timeout",
        "control",
        "task",
        "socket",
        "out_of_order",
        "metadata",
        "map",
        "ids",
        "final_timeout",
    ],
)
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_failure_is_explicit_and_cannot_replay_uncertain_pcm(channel, fault, monkeypatch):
    session, factory = await open_session(channel, task_finish_timeout_s=0.03, send_timeout_s=0.03)
    req = request(channel)
    events = []
    receiver = asyncio.create_task(collect(session, events))
    try:
        await session.begin_turn(req)
        await send(session, req)
        socket = factory.sockets[0]
        task = socket.task_ids[0]
        if fault in {"write", "write_timeout", "map"}:
            socket.fail_audio = fault == "write"
            socket.audio_gate = asyncio.Event() if fault == "write_timeout" else None
            if fault == "map":
                monkeypatch.setattr(qwen, "QWEN_AUDIO_MAX_MAP_RECORDS", 1)
            with pytest.raises(Exception):
                await send(session, req, start=16000, sequence=2)
        elif fault == "control":
            socket.fail_control = True
            with pytest.raises(ConnectionError):
                await seal(session, req)
        elif fault == "task":
            await socket.push(
                control(task, "task-failed", error_code="400", error_message="vocabulary rejected")
            )
        elif fault == "socket":
            await socket.push(None)
        elif fault == "out_of_order":
            await socket.push(native(task, 2, "first"))
            await socket.push(native(task, 1, "second"))
        elif fault == "metadata":
            await socket.push(native(task, 1, "x" * qwen.QWEN_AUDIO_MAX_METADATA_BYTES))
        elif fault == "ids":
            monkeypatch.setattr(qwen, "QWEN_AUDIO_MAX_SENTENCES", 1)
            await socket.push(native(task, 1, "one"))
            await socket.push(native(task, 2, "two"))
        else:
            await seal(session, req)
        await wait_until(
            lambda: any(isinstance(event, STTProviderTurnTerminal) for event in events)
        )
        terminal = next(event for event in events if isinstance(event, STTProviderTurnTerminal))
        assert (
            terminal.outcome == "failed"
            and terminal.failure_reason
            and terminal.epoch_disposition == "retire"
        )
        await wait_until(lambda: socket.closed)
        assert len(factory.sockets) == 1 and len(socket.task_ids) == 1
        assert sum(len(value) for value in socket.sent if isinstance(value, bytes)) == 16000
    finally:
        await session.abort_for_toggle_off()
        await receiver


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_stop_drains_both_slots_but_abort_rejects_late_tails(channel):
    for abort in (False, True):
        session, factory = await open_session(channel)
        events = []
        receiver = asyncio.create_task(collect(session, events))
        for order in (1, 2):
            req = request(channel, order)
            await session.begin_turn(req)
            await send(session, req, start=order * 8000)
            await seal(session, req)
        stopping = asyncio.create_task(session.abort_for_toggle_off() if abort else session.stop())
        await asyncio.sleep(0)
        for socket in reversed(factory.sockets):
            await socket.push(native(socket.task_ids[0], text=" tail "))
            await socket.push(control(socket.task_ids[0], "task-finished"))
        await stopping
        await receiver
        terminals = [event for event in events if isinstance(event, STTProviderTurnTerminal)]
        assert [event.text for event in terminals] == ([] if abort else [" tail ", " tail "])
        assert all(socket.closed for socket in factory.sockets)


@pytest.mark.asyncio
async def test_legacy_real_input_uses_same_handover_and_explicit_verifier_is_separate():
    factory = SocketFactory()
    backend = qwen.QwenAudioStreamingSTTBackend(api_key="offline", websocket_factory=factory)
    session = await backend.open_session()
    assert factory.sockets == []
    await session.send_audio(b"\1\0" * 10)
    await session.on_speech_end()
    await session.send_audio(b"\2\0" * 10)
    await session.on_speech_end()
    assert len(factory.sockets) == 2
    a, b = factory.sockets
    await b.push(native(b.task_ids[0], text="B"))
    await b.push(control(b.task_ids[0], "task-finished"))
    await a.push(native(a.task_ids[0], text="A"))
    await a.push(control(a.task_ids[0], "task-finished"))
    reader = session.events()
    assert (await reader.__anext__()).text == "A"
    assert (await reader.__anext__()).text == "B"
    await session.close()
    probe_factory = SocketFactory([ScriptedSocket(finish=True)])
    assert await qwen.QwenAudioStreamingSTTBackend.verify_api_key(
        "offline", websocket_factory=probe_factory
    )
    assert len(probe_factory.sockets) == 1 and probe_factory.sockets[0].closed
    assert not any(isinstance(value, bytes) for value in probe_factory.sockets[0].sent)
    assert await qwen.QwenAudioStreamingSTTBackend.verify_api_key("") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_late_matching_readiness_makes_progress_but_retired_task_cannot_deliver(channel):
    socket = ScriptedSocket(ready=False)
    session, _ = await open_session(channel, SocketFactory([socket]), task_start_timeout_s=1)
    req = request(channel)
    await session.begin_turn(req)
    writing = asyncio.create_task(send(session, req))
    await wait_until(lambda: bool(socket.task_ids))
    assert not any(isinstance(value, bytes) for value in socket.sent)
    await socket.push(control(socket.task_ids[0], "task-started"))
    await writing
    assert sum(len(value) for value in socket.sent if isinstance(value, bytes)) == 16000
    await session.abort_for_toggle_off()
    await session.inject_event(native(socket.task_ids[0], text="late"))
    assert [event async for event in session.turn_events()] == []


@pytest.mark.asyncio
async def test_channels_hold_four_isolated_slots_and_abort_does_not_cross_channels():
    shared = SocketFactory()
    self_session, _ = await open_session("self", shared)
    peer_session, _ = await open_session("peer", shared)
    events = []
    receiver = asyncio.create_task(collect(peer_session, events))
    try:
        for order in (1, 2):
            for channel, session in (("self", self_session), ("peer", peer_session)):
                req = request(channel, order)
                await session.begin_turn(req)
                await send(session, req)
                await seal(session, req)
        assert len(shared.sockets) == 4
        assert len({socket.task_ids[0] for socket in shared.sockets}) == 4
        await self_session.abort_for_toggle_off()
        assert shared.sockets[0].closed and shared.sockets[2].closed
        assert not shared.sockets[1].closed and not shared.sockets[3].closed
        for socket in (shared.sockets[3], shared.sockets[1]):
            await socket.push(native(socket.task_ids[0], text="peer survives"))
            await socket.push(control(socket.task_ids[0], "task-finished"))
        await wait_until(
            lambda: sum(isinstance(event, STTProviderTurnTerminal) for event in events) == 2
        )
        assert [event.text for event in events] == ["peer survives"] * 2
    finally:
        await self_session.close()
        await peer_session.close()
        await receiver


@pytest.mark.asyncio
async def test_legacy_abort_closes_waiting_consumer_without_fabricated_empty_result():
    session = await qwen.QwenAudioStreamingSTTBackend(
        api_key="offline", websocket_factory=SocketFactory()
    ).open_session()
    reader = asyncio.create_task(session.events().__anext__())
    await session.send_audio(b"\1\0" * 10)
    await session.abort_for_toggle_off()
    with pytest.raises(StopAsyncIteration):
        await reader
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_reused_socket_resets_sentence_namespace_and_clock_and_rejects_predecessor_tail(
    channel,
):
    session, factory = await open_session(channel)
    events = []
    receiver = asyncio.create_task(collect(session, events))
    first_task = None
    try:
        for order in (1, 2, 3):
            req = request(channel, order)
            await session.begin_turn(req)
            await send(session, req, start=order * 8000)
            socket = factory.sockets[(order - 1) % 2]
            current = socket.task_ids[-1]
            if order == 1:
                first_task = current
            if order == 3:
                await socket.push(
                    native(first_task, 2, "obsolete predecessor", begin_time=500, end_time=1000)
                )
                await socket.push(control(first_task, "task-finished"))
            await socket.push(native(current, 1, " repeat ", begin_time=0, end_time=500))
            await seal(session, req)
            await socket.push(control(current, "task-finished"))
            await wait_until(
                lambda: sum(isinstance(event, STTProviderTurnTerminal) for event in events) == order
            )
        terminals = [event for event in events if isinstance(event, STTProviderTurnTerminal)]
        assert [terminal.text for terminal in terminals] == [" repeat "] * 3
        evidence = [terminal.recognition_origins[0].native for terminal in terminals]
        assert len({native_evidence.native_task_id for native_evidence in evidence}) == 3
        assert [native_evidence.native_sentence_id for native_evidence in evidence] == ["1"] * 3
        assert [native_evidence.task_input_samples for native_evidence in evidence] == [8000] * 3
        assert [
            native_evidence.audio_coverage[0].normalized_start_sample
            for native_evidence in evidence
        ] == [8000, 16000, 24000]
        assert len(factory.sockets) == 2
    finally:
        await session.close()
        await receiver
