from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.app.adapters.self_capture.self_capture_vad_sink import SelfCaptureVadSinkAdapter
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.core.audio.ownership import CaptureStreamInput
from puripuly_heart.core.diagnostics import serialize_diagnostics
from puripuly_heart.core.orchestrator.translation_channel_callbacks import (
    TranslationChannelOwnerCallbacks,
)
from puripuly_heart.core.runtime.stt_session_projection import SttSessionStateProjection
from puripuly_heart.core.stt.backend import (
    STTChannelAvailability,
    STTProviderEpochEnded,
    STTProviderInputTerminal,
    STTProviderTurnTerminal,
    STTRecognitionUnitTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.stt.rolling import RollingProviderDefinition, RollingSTTBackend
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.events import UIEventType
from puripuly_heart.providers.stt import custom as custom_module
from puripuly_heart.providers.stt.gemini_transcribe import _GeminiTranscribeLiveSession
from puripuly_heart.providers.stt.soniox import SonioxRealtimeSTTBackend
from tests.core.runtime.test_peer_capture_session import make_config, make_owner
from tests.core.runtime.test_self_capture_session import build_owner, config
from tests.core.test_peer_caption_lifecycle import ControlledBackend
from tests.core.test_stt_scoped_engine import span, wait_until, watchdogs
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.providers.test_custom_stt import (
    _backend,
    _FakeAsyncClient,
    _FakeResponse,
    _FakeWebSocket,
)
from tests.providers.test_gemini_transcribe_reuse import _go_away, _Live, _message


@asynccontextmanager
async def recovery_pipeline(channel, provider_id, session_factory):
    events, failures, delays, ui_events = [], [], [], []
    callbacks = TranslationChannelOwnerCallbacks(SttSessionStateProjection())
    translator = ControlledBackend()
    osc = RecordingOscQueue()
    capture = provider = None

    async def sleep(delay):
        if delay > 1.6:
            await asyncio.sleep(delay)
        else:
            delays.append(delay)

    async def failed(failure):
        failures.append(failure)
        handler = (
            provider.terminal_failure_handler
            if channel == "self"
            else provider.terminal_handlers[-1]
        )
        await handler(failure)

    async def on_event(event):
        events.append(event)
        handler = (
            callbacks.self_event_handler if channel == "self" else callbacks.peer_event_handler
        )
        await handler(event)

    engine = ScopedRecognitionEngine(
        session_factory=session_factory,
        channel=channel,
        sleep=sleep,
        terminal_failure_sink=failed,
        session_lifetime_enabled=False,
        watchdog_resolver=lambda _: watchdogs(
            connect_retry_base_s=0.8,
            connect_retry_max_s=1.6,
            write_timeout_s=1.0,
            final_timeout_s=1.0,
        ),
    )
    translation = compose_translation_test_harness(
        stt=engine if channel == "self" else None,
        peer_stt=engine if channel == "peer" else None,
        llm=translator,
        osc=osc,
        peer_translation_enabled=True,
    )
    callbacks.bind_self(translation.self_owner)
    callbacks.bind_peer(translation.peer_owner)
    if channel == "self":
        capture, _, provider, _, _, _ = build_owner(
            sink=SelfCaptureVadSinkAdapter(lambda: translation.self_owner)
        )
        capture_config = replace(config(), provider_id=provider_id)
        callbacks.bind_self_capture(capture)
    else:
        capture, _, _, provider, _, _ = make_owner(sink=engine)
        capture_config = make_config(provider_id=provider_id)
        callbacks.bind_peer_capture(capture)
        capture.bind_publication_generation_observer(
            activated=translation.output_runtime.activate_peer_generation,
            retired=translation.output_runtime.retire_peer_generation,
        )
    engine.accepted_settings_scope = (
        provider_id,
        capture_config.provider_signature,
        capture_config.runtime_signature,
    )
    try:
        await translation.start()
        engine.bind_event_sink(on_event)
        await capture.apply_intent(capture_config, enabled=True)
        await wait_until(lambda: capture.snapshot.effective_active)
        guarded = capture.guard_vad_sink()

        async def vad_event(event, order):
            if channel == "self":
                await guarded.handle_vad_event(event)
            else:
                owned = capture.segment_ledger.observe_vad_event(
                    event, now_monotonic_s=float(order)
                )
                await guarded.handle_owned_vad_event(owned)

        async def speak(order, *, seal=True):
            segment_id = uuid4()
            offset = order * 160
            samples = np.full(160, 0.1 + order / 10, dtype=np.float32)
            ranges = (span(order, offset, offset + 160),)
            await vad_event(
                SpeechStart(
                    segment_id,
                    pre_roll=np.empty(0, dtype=np.float32),
                    chunk=samples,
                    chunk_capture=ranges,
                ),
                order,
            )
            if provider_id in {"gemini_transcribe", "rolling_free"}:
                await guarded.handle_stream_input(
                    CaptureStreamInput(samples, ranges, speech_observed=True)
                )
            if seal:
                await vad_event(SpeechEnd(segment_id), order)
            return segment_id

        async def reactivate():
            nonlocal guarded
            await capture.apply_intent(capture_config, enabled=False)
            await engine.abort(reason="toggle_off")
            await capture.apply_intent(capture_config, enabled=True)
            await wait_until(lambda: capture.snapshot.effective_active)
            guarded = capture.guard_vad_sink()

        async def output_for(text):
            def completed():
                while not translation.ui_events.empty():
                    ui_events.append(translation.ui_events.get_nowait())
                matching = {call.utterance_id for call in translator.calls if text in call.text}
                if channel == "self":
                    return any(text in message.text for message in osc.messages)
                return any(
                    event.type is UIEventType.TRANSLATION_DONE and event.utterance_id in matching
                    for event in ui_events
                )

            await wait_until(completed)

        yield SimpleNamespace(
            capture=capture,
            engine=engine,
            events=events,
            failures=failures,
            delays=delays,
            translator=translator,
            osc=osc,
            speak=speak,
            output_for=output_for,
            reactivate=reactivate,
        )
    finally:
        await capture.close()
        await engine.close()
        await translation.stop()


@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_canonical_custom_offline_recovers_real_http_failure_through_output(channel):
    requests, sessions = [], []
    recovered = "Recovered canonical custom offline source."

    async def respond(_url, **options):
        requests.append(options)
        return (
            _FakeResponse(503, {"error": "controlled unavailable"})
            if len(requests) == 1
            else _FakeResponse(200, {"text": recovered})
        )

    backend = _backend(http_client_factory=lambda **_: _FakeAsyncClient(respond))

    async def factory(_settings, epoch):
        session = await backend.open_session(projection=STTSessionProjection("scoped", epoch))
        sessions.append(session)
        return session

    try:
        async with recovery_pipeline(channel, "custom_offline", factory) as env:
            await env.speak(0)
            await wait_until(lambda: len(env.events) >= 1)
            first = next(
                event for event in env.events if isinstance(event, STTProviderTurnTerminal)
            )
            assert first.outcome == "failed" and first.text == ""
            assert first.failure_retryable
            assert first.channel_availability is STTChannelAvailability.RECOVERING
            assert first.identity.settings_scope[0] == "custom_offline"
            assert env.capture.snapshot.desired_active and env.capture.snapshot.effective_active
            assert not env.translator.calls
            await env.speak(1)
            await env.output_for(recovered)
            assert len(requests) == len(sessions) == 2
            assert env.delays == [0.8]
            assert not env.failures
            assert env.capture.snapshot.desired_active and env.capture.snapshot.effective_active
            assert [call.text for call in env.translator.calls] == [recovered]
    finally:
        await backend.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("fault", ["commit_connection", "final_timeout"])
async def test_canonical_custom_realtime_native_transients_reach_fresh_completed_output(
    channel, fault, monkeypatch
):
    monkeypatch.setattr(custom_module, "_STREAM_FINAL_TIMEOUT_S", 0.02)
    sockets = []
    recovered = "Recovered canonical custom realtime source."

    class Socket(_FakeWebSocket):
        async def send(self, payload):
            await super().send(payload)
            event = json.loads(payload)
            if event["type"] != "input_audio_buffer.commit":
                return
            if len(sockets) == 1:
                if fault == "commit_connection":
                    raise ConnectionError("controlled commit transport detail must stay private")
                return
            await self.push(
                json.dumps({"type": "input_audio_buffer.committed", "item_id": "fresh"})
            )
            await self.push(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": "fresh",
                        "event_id": "fresh-final",
                        "transcript": recovered,
                    }
                )
            )

    async def connect(*_args, **_kwargs):
        socket = Socket([], hang=True)
        sockets.append(socket)
        return socket

    backend = _backend(mode="realtime", compatibility="openai_realtime", websocket_connect=connect)

    async def factory(_settings, epoch):
        return await backend.open_session(projection=STTSessionProjection("scoped", epoch))

    try:
        async with recovery_pipeline(channel, "custom_realtime", factory) as env:
            await env.speak(0)
            await wait_until(
                lambda: any(isinstance(event, STTProviderTurnTerminal) for event in env.events)
            )
            first = next(
                event for event in env.events if isinstance(event, STTProviderTurnTerminal)
            )
            assert first.failure_reason == (
                "commit_write_failed" if fault == "commit_connection" else "final_timeout"
            )
            assert first.failure_retryable
            assert first.channel_availability is STTChannelAvailability.RECOVERING
            if fault == "commit_connection":
                assert first.failure_diagnostics.category == "network"
                assert first.failure_diagnostics.fields["exception_class"] == "ConnectionError"
                assert "controlled commit transport detail" not in str(first.failure_diagnostics)
            assert env.capture.snapshot.desired_active and env.capture.snapshot.effective_active
            assert not env.translator.calls
            await env.speak(1)
            await env.output_for(recovered)
            assert sockets[0].closed
            assert len(sockets) == 2
            audio = [
                base64.b64decode(event["audio"])
                for socket in sockets
                for payload in socket.sent
                if (event := json.loads(payload))["type"] == "input_audio_buffer.append"
            ]
            assert len(audio) == 2 and len(audio[0]) == len(audio[1]) == 320
            assert audio[0] != audio[1]
            assert env.delays == [0.8]
            assert not env.failures
            assert env.capture.snapshot.desired_active and env.capture.snapshot.effective_active
    finally:
        await backend.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize("provider_id", ["gemini_transcribe", "rolling_free"])
@pytest.mark.parametrize("fault", ["receive_failure", "go_away", "go_away_late_final"])
async def test_independent_sealed_input_uses_only_stream_recovery_owner(
    channel, provider_id, fault
):
    sessions, lives = [], []
    recovered = "Fresh independent recognized output after retirement."
    seal_task = None

    class Live(_Live):
        def __init__(self):
            super().__init__()
            self.sealing = asyncio.Event()
            self.release_seal = asyncio.Event()

        async def send_realtime_input(self, **options):
            await super().send_realtime_input(**options)
            if options.get("audio_stream_end") and len(lives) == 1:
                self.sealing.set()
                await self.release_seal.wait()

    class Backend:
        async def open_session(self, *, projection):
            live = Live()
            session = _GeminiTranscribeLiveSession(
                api_key="controlled",
                language_codes=[],
                custom_vocabulary=[],
                model="gemini-3.5-transcribe-live",
                sample_rate_hz=16000,
                connect_timeout_s=1.0,
                drain_timeout_s=0.02,
                projection=projection,
            )
            session._live_session = live
            session._send_task = asyncio.create_task(session._send_loop())
            session._recv_task = asyncio.create_task(session._recv_loop())
            sessions.append(session)
            lives.append(live)
            return session

    native = Backend()
    backend = (
        native
        if provider_id == "gemini_transcribe"
        else RollingSTTBackend(
            providers=(
                RollingProviderDefinition(
                    name=STTProviderName.GEMINI_TRANSCRIBE,
                    build_backend=lambda: native,
                    is_configured=lambda: True,
                ),
            ),
            channel=channel,
        )
    )

    async def factory(_settings, epoch):
        return await backend.open_session(projection=STTSessionProjection("scoped", epoch))

    try:
        async with recovery_pipeline(channel, provider_id, factory) as env:
            seal_task = asyncio.create_task(env.speak(0))
            await wait_until(lambda: bool(lives))
            await asyncio.wait_for(lives[0].sealing.wait(), 1)
            assert env.engine._turn.local_sealed
            assert (
                env.engine._stream_recovery is not None and not env.engine._stream_recovery.pending
            )
            if fault == "receive_failure":
                lives[0].push(ConnectionError("controlled receive failure"))
            else:
                lives[0].push(_go_away(time_left="0.03s"))
                if fault == "go_away_late_final":
                    lives[0].push(
                        _message(
                            final="Eligible late final from original source.", include_final=True
                        )
                    )
                    await wait_until(
                        lambda: any(
                            isinstance(event, STTRecognitionUnitTerminal) for event in env.events
                        )
                    )
            await wait_until(
                lambda: any(isinstance(event, STTProviderInputTerminal) for event in env.events)
            )
            terminal = next(
                event for event in env.events if isinstance(event, STTProviderInputTerminal)
            )
            assert terminal.outcome == "failed" and terminal.independent_recognition
            assert terminal.channel_availability is STTChannelAvailability.RECOVERING
            assert env.engine._episode_failures == (1 if fault == "receive_failure" else 0)
            assert env.engine._stream_recovery is not None and env.engine._stream_recovery.pending
            assert not env.failures
            assert env.capture.snapshot.desired_active and env.capture.snapshot.effective_active
            lives[0].release_seal.set()
            await asyncio.wait_for(seal_task, 1)
            await wait_until(lambda: env.engine.cleanup_debt == 0)
            await wait_until(
                lambda: any(isinstance(event, STTProviderEpochEnded) for event in env.events)
            )
            assert env.engine._episode_failures == (1 if fault == "receive_failure" else 0)
            assert len(sessions) == 1
            await env.speak(1)
            await wait_until(
                lambda: any(
                    isinstance(event, STTProviderInputTerminal) and event.outcome == "submitted"
                    for event in env.events
                )
            )
            assert env.engine._episode_failures == (1 if fault == "receive_failure" else 0)
            assert len(sessions) == len(lives) == 2
            fresh_message = _message(final=recovered, include_final=True)
            assert fresh_message.server_content.input_transcription.text == recovered
            lives[1].push(fresh_message)
            await wait_until(
                lambda: any(
                    isinstance(event, STTRecognitionUnitTerminal) and event.unit.text == recovered
                    for event in env.events
                )
            )
            await env.output_for(recovered)
            units = [
                event.unit for event in env.events if isinstance(event, STTRecognitionUnitTerminal)
            ]
            assert units[-1].text == recovered
            assert units[-1].identity.stream.settings_scope[0] == provider_id
            if fault == "go_away_late_final":
                assert units[0].text == "Eligible late final from original source."
                assert (
                    units[0].identity.stream.provider_epoch_id
                    != units[-1].identity.stream.provider_epoch_id
                )
                assert (
                    units[0].identity.stream.activation_generation
                    == units[-1].identity.stream.activation_generation
                )
            assert env.engine._episode_failures == 0
            assert env.delays == ([0.8] if fault == "receive_failure" else [])
            assert not env.failures
            assert len(sessions) == 2
            audio = [
                item["audio"]["data"] for live in lives for item in live.sent if "audio" in item
            ]
            assert len(audio) == 2 and len(audio[0]) == len(audio[1]) == 320
            assert audio[0] != audio[1]
            assert env.capture.snapshot.desired_active and env.capture.snapshot.effective_active
    finally:
        for live in lives:
            live.release_seal.set()
        if seal_task is not None:
            await asyncio.gather(seal_task, return_exceptions=True)
        for session in sessions:
            await session.close()


@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_soniox_startup_cleanup_debt_fences_allocation_until_explicit_reactivation(
    channel, monkeypatch
):
    sockets = []
    cleanup_allowed = False
    send_failure = ConnectionResetError("private-soniox-startup-send")
    recovered = "Fresh Soniox recognition after confirmed cleanup."

    class WebSocket:
        def __init__(self, *, fail_start):
            self.fail_start = fail_start
            self.closed = False
            self.close_calls = 0
            self.sent = []
            self.incoming = asyncio.Queue()

        async def send(self, payload):
            self.sent.append(payload)
            if self.fail_start:
                raise send_failure
            if (
                isinstance(payload, str)
                and payload
                and json.loads(payload).get("type") == "finalize"
            ):
                self.incoming.put_nowait(
                    json.dumps(
                        {
                            "tokens": [
                                {"text": recovered, "is_final": True, "end_ms": 10},
                                {"text": "<fin>", "is_final": True},
                            ]
                        }
                    )
                )

        async def recv(self):
            return await self.incoming.get()

        async def close(self):
            self.close_calls += 1
            if self.fail_start and not cleanup_allowed:
                raise RuntimeError("private-soniox-startup-cleanup")
            self.closed = True

    async def connect(*_args, **_kwargs):
        socket = WebSocket(fail_start=not sockets)
        sockets.append(socket)
        return socket

    monkeypatch.setattr("puripuly_heart.core.network_clients.external_websocket_connect", connect)
    backend = SonioxRealtimeSTTBackend(api_key="controlled", language_hints=["en"])

    async def factory(_settings, epoch):
        return await backend.open_session(projection=STTSessionProjection("scoped", epoch))

    try:
        async with recovery_pipeline(channel, "soniox", factory) as env:
            await env.speak(0)
            await wait_until(lambda: len(env.failures) == 1)
            await wait_until(lambda: not env.capture.snapshot.effective_active)
            terminals = [
                event for event in env.events if isinstance(event, STTProviderTurnTerminal)
            ]
            assert len(terminals) == 1
            terminal = terminals[0]
            assert terminal.outcome == "failed" and terminal.text == ""
            assert terminal.channel_availability is STTChannelAvailability.UNAVAILABLE
            assert terminal.failure_diagnostics.fields["channel_availability"] == "unavailable"
            assert terminal.failure_diagnostics.fields["recovery_pending"] is False
            assert env.delays == [0.8]
            assert env.engine._episode_failures == 3
            assert env.engine._terminal_failure_notified
            assert not env.translator.calls
            assert len(sockets) == 1 and not sockets[0].closed
            assert sockets[0].close_calls >= 2
            assert len(backend._starting_sessions) == 1
            assert backend._starting_sessions[0]._ws is sockets[0]
            evidence = json.dumps(serialize_diagnostics(env.failures[0]))
            assert send_failure.failure_diagnostics.diagnostic_id in evidence
            assert "ConnectionResetError" in evidence
            assert "stt.provider_cleanup_failed" in evidence
            assert any(
                cause.code == "stt.provider_cleanup_failed"
                and cause.operation == "cleanup"
                and cause.diagnostic_id != send_failure.failure_diagnostics.diagnostic_id
                and cause.cause_id == send_failure.failure_diagnostics.diagnostic_id
                and cause.fields["cleanup_confirmed"] is False
                for cause in send_failure.failure_diagnostics.causes
            )
            assert "private-soniox-startup" not in evidence

            cleanup_allowed = True
            await backend.close()
            assert sockets[0].closed
            assert not backend._starting_sessions
            assert env.engine._terminal_failure_notified
            assert not env.capture.snapshot.effective_active
            assert len(sockets) == 1

            authority = env.engine._authority_generation
            await env.reactivate()
            assert env.engine._authority_generation > authority
            assert not env.engine._terminal_failure_notified
            await env.speak(1)
            await env.output_for(recovered)
            fresh = [
                event
                for event in env.events
                if isinstance(event, STTProviderTurnTerminal) and event.outcome == "final"
            ]
            assert len(fresh) == 1 and fresh[0].text == recovered
            assert (
                fresh[0].identity.segment.activation_generation
                > terminal.identity.segment.activation_generation
            )
            assert len(sockets) == 2 and not sockets[1].closed
            assert not any(isinstance(payload, bytes) for payload in sockets[0].sent)
            assert any(isinstance(payload, bytes) for payload in sockets[1].sent)
            assert [call.text for call in env.translator.calls] == [recovered]
            assert len(env.failures) == 1 and env.delays == [0.8]
            assert env.engine._episode_failures == 0
            assert env.capture.snapshot.effective_active
        assert all(socket.closed for socket in sockets)
    finally:
        cleanup_allowed = True
        await backend.close()
