from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.core.audio.ownership import AudioSegmentTerminalReceipt, PeerAudioSegmentLedger
from puripuly_heart.core.orchestrator.self_translation_channel import SelfTranslationChannelOwner
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.stt.backend import (
    STTProviderEpochEnded,
    STTProviderTurnTerminal,
    STTRecognitionUnitTerminal,
)
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
from puripuly_heart.core.vad.gating import SpeechStart
from tests.core.test_source_context_contract import (
    ControlledProvider,
    OutputTrace,
    finish,
    until,
)
from tests.core.test_stt_scoped_engine import (
    ControlledMonotonicClock,
    segment_events,
    settings,
    span,
    watchdogs,
)
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.providers.test_soniox_reuse import _session


class SourceWebSocket:
    def __init__(self, *, drain_text=None, hold_stop=False):
        self.sent = []
        self.received = asyncio.Queue()
        self.closed = False
        self.close_code = None
        self.drain_text = drain_text
        self.stop_entered = asyncio.Event()
        self.stop_release = asyncio.Event()
        if not hold_stop:
            self.stop_release.set()

    def push(self, value):
        self.received.put_nowait(json.dumps(value))

    async def send(self, payload):
        self.sent.append(payload)
        if payload == "":
            self.stop_entered.set()
            await self.stop_release.wait()
            tokens = (
                []
                if self.drain_text is None
                else [
                    {"text": self.drain_text, "is_final": True, "start_ms": -12.5, "end_ms": 3.25},
                ]
            )
            self.push({"tokens": tokens, "finished": True})
            self.received.put_nowait(None)

    async def recv(self):
        return await self.received.get()

    async def close(self):
        self.closed = True
        self.close_code = 1000
        self.received.put_nowait(None)


class SonioxSourcePipeline:
    def __init__(
        self, channel, *, low_latency=False, drain_text=None, hold_stop=False, max_age=None
    ):
        self.channel = channel
        self.clock = ControlledMonotonicClock(100)
        self.ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings("soniox"))
        self.provider = ControlledProvider()
        self.harness = compose_translation_test_harness(
            stt=None,
            llm=self.provider,
            osc=RecordingOscQueue(),
            clock=self.clock,
            source_language="en",
            target_language="ja",
            peer_source_language="en",
            peer_target_language="ja",
            integrated_context_enabled=True,
            translation_enabled=True,
            peer_translation_enabled=True,
            low_latency_mode=low_latency,
            low_latency_finalize_wait_ms=400,
            low_latency_spec_min_chars=1000,
        )
        self.harness.output_runtime.activate_peer_generation(1)
        self.trace = OutputTrace(self.harness.translation_turns.output)
        self.harness.translation_turns.output = self.trace
        self.sessions = []
        self.sockets = []
        self.events = []
        self.current_scope = None
        self.callbacks = TranslationChannelOwnerCallbacks(self.harness.stt_sessions)
        self.callbacks.bind_self(self.harness.self_owner)
        self.callbacks.bind_peer(self.harness.peer_owner)

        async def factory(_settings, epoch):
            native, _ = _session(epoch=epoch)
            socket = SourceWebSocket(
                drain_text=drain_text if not self.sessions else None,
                hold_stop=hold_stop if not self.sessions else False,
            )
            native._ws = socket
            native._recv_task = asyncio.create_task(native._recv_loop())
            self.sessions.append(native)
            self.sockets.append(socket)
            return StreamingSpeakerSession(native) if channel == "peer" else native

        async def receive(event):
            if channel == "self":
                await self.callbacks.self_event_handler(event)
            else:
                await self.callbacks.peer_event_handler(event)
            self.events.append(event)

        self.engine = ScopedRecognitionEngine(
            factory,
            channel=channel,
            event_sink=receive,
            watchdog_resolver=lambda _: watchdogs(
                final_timeout_s=0.5,
                drain_timeout_s=0.5,
                max_session_age_s=max_age,
            ),
            monotonic_clock=self.clock.now,
            sleep=self.clock.sleep,
        )

        async def owned(_channel, event):
            await self.engine.handle_owned_vad_event(event)

        async def commit(_channel):
            return None

        runtime = SimpleNamespace(
            is_current_recognition_stream=lambda _channel, stream: self.engine.is_current_recognition_stream(
                stream
            ),
            handle_owned_vad_event=owned,
            commit_handoff=commit,
        )
        self.harness.self_owner.local_asr_runtime = runtime
        self.harness.peer_owner.local_asr_runtime = runtime
        capture = SimpleNamespace(
            snapshot=SimpleNamespace(generation=1),
            is_current_recognition_stream=lambda stream: (
                stream.activation_generation,
                stream.capture_epoch,
                stream.settings_scope,
            )
            == self.current_scope,
            note_recognition_terminal=lambda event: None,
            provider_source_snapshot=self.source_snapshot,
            admit_provider_terminal=self.admit_terminal,
        )
        self.callbacks.bind_self_capture(capture)
        self.callbacks.bind_peer_capture(capture)

    def source_snapshot(self, identity):
        return next(
            (
                snapshot
                for snapshot in self.ledger.snapshots
                if snapshot.identity == identity.segment
            ),
            None,
        )

    def admit_terminal(self, terminal):
        snapshot = self.source_snapshot(terminal.identity)
        if snapshot is None or snapshot.state != "sealed":
            return ()
        receipt = AudioSegmentTerminalReceipt(
            terminal.identity.segment,
            terminal.outcome,
            snapshot,
            terminal.identity.segment.segment_order,
        )
        return ((receipt, terminal),)

    @property
    def history(self):
        runtime = self.harness.self_runtime if self.channel == "self" else self.harness.peer_runtime
        return runtime.translation_history

    @property
    def sources(self):
        return [item.source_child for item in self.trace.items]

    async def owned(self, event):
        self.current_scope = (
            event.segment.identity.activation_generation,
            event.segment.identity.capture_epoch,
            ScopedRecognitionEngine._settings_scope(event.segment.settings),
        )
        if self.channel == "self":
            await self.harness.self_owner.handle_vad_event(event)
        else:
            await self.harness.peer_owner.handle_peer_owned_vad_event(event)

    async def first_completion(self, *, text="first source", tail=None):
        start, _, end = segment_events(self.ledger, start_sample=0, now=100)
        await self.engine.observe_source_activity(speech_observed=True)
        await self.owned(start)
        ending = asyncio.create_task(self.owned(end))
        await until(lambda: self.sessions[0]._written_manual_target is not None)
        await self.engine.observe_source_activity(speech_observed=False)
        tokens = [{"text": text, "is_final": True}, {"text": "<fin>", "is_final": True}]
        if tail is not None:
            tokens.extend([{"text": tail, "is_final": True}, {"text": "<end>", "is_final": True}])
        self.sockets[0].push({"tokens": tokens})
        await ending
        await until(
            lambda: any(isinstance(event, STTProviderTurnTerminal) for event in self.events)
        )
        return start, end

    async def close(self):
        for socket in self.sockets:
            socket.stop_release.set()
        await self.engine.close()
        await self.harness.self_owner.close()
        await self.harness.translation_turns.close()
        await self.harness.output_runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("retirement", ["idle", "max_age"])
async def test_lawful_retiring_soniox_tail_reaches_source_and_context_once(channel, retirement):
    pipeline = SonioxSourcePipeline(
        channel,
        drain_text="drained tail",
        max_age=60 if retirement == "max_age" else None,
    )
    try:
        await pipeline.first_completion()
        await pipeline.clock.advance_to(160)
        await until(lambda: pipeline.sockets[0].closed)
        await until(
            lambda: any(isinstance(event, STTRecognitionUnitTerminal) for event in pipeline.events)
        )
        await finish(pipeline.harness)
        assert [source.raw_text for source in pipeline.sources] == ["first source", "drained tail"]
        assert [entry.text for entry in pipeline.history] == ["first source", "drained tail"]
        assert pipeline.provider.calls[0][2] == ""
        assert pipeline.provider.calls[1][2] == ""
        tail = next(
            event for event in pipeline.events if isinstance(event, STTRecognitionUnitTerminal)
        )
        await (
            pipeline.callbacks.self_event_handler(tail)
            if channel == "self"
            else pipeline.callbacks.peer_event_handler(tail)
        )
        await finish(pipeline.harness)
        assert len(pipeline.sources) == len(pipeline.provider.calls) == 2
        assert tail.unit.provenance.barrier == "drain"
        assert tail.unit.provenance.audio_clocks[0].qualified is False
        assert tail.unit.provenance.text_timing[0].raw_start == -12.5
        assert tail.unit.provenance.text_timing[0].raw_end == 3.25
        assert tail.unit.final_speaker_runs == ()
        assert pipeline.sockets[0].sent[-1] == ""
        assert (
            sum(isinstance(value, str) and value != "" for value in pipeline.sockets[0].sent) == 1
        )
    finally:
        await pipeline.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("fence", ["stop", "reset", "settings", "capture"])
async def test_retiring_soniox_tail_cannot_cross_cancel_or_scope_replacement(channel, fence):
    pipeline = SonioxSourcePipeline(channel, drain_text="forbidden late tail", hold_stop=True)
    pending = None
    try:
        await pipeline.first_completion()
        old_stream = pipeline.sessions[0]._stream_identity
        await pipeline.clock.advance_to(160)
        await asyncio.wait_for(pipeline.sockets[0].stop_entered.wait(), 1)
        if fence == "stop":
            await pipeline.engine.stop()
        elif fence == "reset":
            await pipeline.engine.abort(reason="manual_reset")
        else:
            changed = (
                settings("soniox", signature="changed")
                if fence == "settings"
                else settings("soniox")
            )
            pipeline.ledger = PeerAudioSegmentLedger(activation_generation=1, settings=changed)
            capture = span(20, 20, 24)
            if fence == "capture":
                capture = replace(capture, capture_epoch=2)
            start = pipeline.ledger.observe_vad_event(
                SpeechStart(
                    uuid4(),
                    np.empty(0, dtype=np.float32),
                    np.full(4, 0.1, dtype=np.float32),
                    chunk_capture=(capture,),
                ),
                now_monotonic_s=160,
            )
            pending = asyncio.create_task(pipeline.owned(start))
            await until(lambda: not pipeline.engine.is_current_recognition_stream(old_stream))
        assert not pipeline.engine.is_current_recognition_stream(old_stream)
        pipeline.sockets[0].stop_release.set()
        await until(lambda: pipeline.sockets[0].closed)
        if pending is not None:
            await pending
        await finish(pipeline.harness)
        assert [source.raw_text for source in pipeline.sources] == ["first source"]
        assert [entry.text for entry in pipeline.history] == ["first source"]
        assert pipeline.sessions[0]._receipt_sequence == 1
        assert not any(isinstance(event, STTRecognitionUnitTerminal) for event in pipeline.events)
    finally:
        for socket in pipeline.sockets:
            socket.stop_release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await pipeline.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("failure", ["503", "receive", "write"])
async def test_nonorderly_soniox_failure_preserves_received_final_tail_once(
    channel, active, failure
):
    pipeline = SonioxSourcePipeline(channel)
    try:
        await pipeline.first_completion()
        native = pipeline.sessions[0]
        end = None
        if active:
            start, _, end = segment_events(pipeline.ledger, start_sample=20, now=101)
            await pipeline.owned(start)
        native._handle_message(
            json.dumps(
                {
                    "tokens": [
                        {"text": "valid", "is_final": True, "start_ms": -20.5, "end_ms": -10},
                        {"text": " final tail", "is_final": True, "start_ms": -10, "end_ms": 3.25},
                    ]
                }
            )
        )
        assert len(pipeline.history) == 1
        if failure == "503":
            native._handle_message(
                json.dumps({"error_code": 503, "error_type": "service_unavailable"})
            )
        else:
            native._scoped_transport_failure(
                f"soniox_{failure}_failed", orderly=False, exception=OSError("local failure")
            )
        await until(
            lambda: any(isinstance(event, STTRecognitionUnitTerminal) for event in pipeline.events)
        )
        if end is not None:
            await pipeline.owned(end)
        await until(
            lambda: any(isinstance(event, STTProviderEpochEnded) for event in pipeline.events)
        )
        await finish(pipeline.harness)
        assert [source.raw_text for source in pipeline.sources] == [
            "first source",
            "valid final tail",
        ]
        assert [entry.text for entry in pipeline.history] == ["first source", "valid final tail"]
        assert pipeline.provider.calls[0][2] == ""
        assert '"first source"' in pipeline.provider.calls[1][2]
        tail = next(
            event for event in pipeline.events if isinstance(event, STTRecognitionUnitTerminal)
        )
        assert tail.unit.provenance.audio_clocks[0].qualified is False
        assert [
            (timing.raw_start, timing.raw_end) for timing in tail.unit.provenance.text_timing
        ] == [(-20.5, -10), (-10, 3.25)]
        native._scoped_transport_failure("soniox_receive_failed", orderly=False)
        native._handle_message(
            json.dumps({"tokens": [{"text": "duplicate tail", "is_final": True}], "finished": True})
        )
        await (
            pipeline.callbacks.self_event_handler(tail)
            if channel == "self"
            else pipeline.callbacks.peer_event_handler(tail)
        )
        await finish(pipeline.harness)
        assert native._scoped_tokens == []
        assert native._receipt_sequence == 1
        assert len(pipeline.sources) == len(pipeline.provider.calls) == 2
        if active:
            failed = [
                event
                for event in pipeline.events
                if isinstance(event, STTProviderTurnTerminal) and event.outcome == "failed"
            ]
            assert len(failed) == 1
            assert failed[0].text == ""
            assert failed[0].failure_retryable is True
    finally:
        await pipeline.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_soniox_returned_text_does_not_extend_source_idle_deadline(channel):
    pipeline = SonioxSourcePipeline(channel)
    try:
        await pipeline.first_completion()
        await pipeline.clock.advance_to(154)
        pipeline.sockets[0].push(
            {
                "tokens": [
                    {"text": "late returned source", "is_final": True},
                    {"text": "<end>", "is_final": True},
                ]
            }
        )
        await until(
            lambda: any(isinstance(event, STTRecognitionUnitTerminal) for event in pipeline.events)
        )
        assert pipeline.engine._lifetime_deadline_s == 160
        assert pipeline.engine._last_source_speech_at_s == 100
        assert pipeline.engine._last_native_speech_at_s is None
        await pipeline.clock.advance_to(160)
        await until(lambda: pipeline.sockets[0].closed)
        await finish(pipeline.harness)
        assert [source.raw_text for source in pipeline.sources] == [
            "first source",
            "late returned source",
        ]
        assert [entry.text for entry in pipeline.history] == [
            "first source",
            "late returned source",
        ]
    finally:
        await pipeline.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("single_message", [False, True])
async def test_self_soniox_connection_tail_commits_prior_grace_before_context_snapshot(
    single_message, monkeypatch
):
    grace_entered = asyncio.Event()
    grace_release = asyncio.Event()
    original = SelfTranslationChannelOwner._finalize_wait_timeout

    async def held_grace(self, *args):
        grace_entered.set()
        await grace_release.wait()
        await original(self, *args)

    monkeypatch.setattr(SelfTranslationChannelOwner, "_finalize_wait_timeout", held_grace)
    pipeline = SonioxSourcePipeline("self", low_latency=True)
    try:
        await pipeline.first_completion(tail="late tail" if single_message else None)
        if not single_message:
            await asyncio.wait_for(grace_entered.wait(), 1)
            assert pipeline.harness.self_owner.merge_buffer is not None
            assert pipeline.history == []
            pipeline.sockets[0].push(
                {
                    "tokens": [
                        {"text": "late tail", "is_final": True},
                        {"text": "<end>", "is_final": True},
                    ]
                }
            )
        await until(
            lambda: any(isinstance(event, STTRecognitionUnitTerminal) for event in pipeline.events)
        )
        await finish(pipeline.harness)
        assert pipeline.harness.self_owner.merge_buffer is None
        assert [source.raw_text for source in pipeline.sources] == ["first source", "late tail"]
        assert [entry.text for entry in pipeline.history] == ["first source", "late tail"]
        assert [call[0] for call in pipeline.provider.calls] == ["first source", "late tail"]
        assert pipeline.provider.calls[0][2] == ""
        assert '"first source"' in pipeline.provider.calls[1][2]
        assert '"late tail"' not in pipeline.provider.calls[0][2]
        assert all(
            source.channel == "self" and source.speaker is None for source in pipeline.sources
        )
        grace_release.set()
        await asyncio.sleep(0)
        await finish(pipeline.harness)
        assert len(pipeline.sources) == len(pipeline.provider.calls) == 2
    finally:
        grace_release.set()
        await pipeline.close()


@pytest.mark.asyncio
async def test_self_pending_grace_and_stale_connection_tail_remain_cancelled_after_reset(
    monkeypatch,
):
    from puripuly_heart.core.stt.backend import STTRecognitionUnit
    from puripuly_heart.domain.recognition import RecognitionUnitIdentity

    grace_entered = asyncio.Event()

    async def held_grace(self, *args):
        grace_entered.set()
        await asyncio.Future()

    monkeypatch.setattr(SelfTranslationChannelOwner, "_finalize_wait_timeout", held_grace)
    pipeline = SonioxSourcePipeline("self", low_latency=True)
    try:
        await pipeline.first_completion()
        await asyncio.wait_for(grace_entered.wait(), 1)
        stream = pipeline.sessions[0]._stream_identity
        assert pipeline.harness.self_owner.merge_buffer is not None
        assert pipeline.history == []
        await pipeline.engine.abort(reason="settings_replaced")
        await pipeline.harness.self_owner.reset_provider_channel()
        await pipeline.callbacks.self_event_handler(
            STTRecognitionUnitTerminal(
                STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 1), "stale tail"),
                "final",
            )
        )
        await finish(pipeline.harness)
        assert pipeline.harness.self_owner.merge_buffer is None
        assert pipeline.sources == pipeline.history == []
        assert all(call == ("first source", "ja", "") for call in pipeline.provider.calls)
    finally:
        await pipeline.close()
