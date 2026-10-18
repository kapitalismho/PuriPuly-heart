from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import SharedSTTProviderFactory
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_peer_stt_provider_request,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core import network_clients
from puripuly_heart.core.audio.ownership import PeerAudioSegmentLedger
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.core.stt.backend import STTProviderTurnTerminal
from puripuly_heart.core.stt.streaming_speaker import StreamingSpeakerSession
from tests.core.test_stt_scoped_engine import segment_events, settings, wait_until, watchdogs
from tests.providers.test_elevenlabs_scribe_reuse import (
    _ControlledConnection,
    _cwt,
    _request,
    _seal,
    _session,
)
from tests.providers.test_qwen_audio import control, native
from tests.providers.test_qwen_audio_recovery_boundaries import FaultTransport
from tests.providers.test_qwen_audio_runtime_quarantine import ReleasableSocket


@pytest.mark.asyncio
async def test_production_peer_wrapper_keeps_unclosed_qwen_quarantined_until_real_retry(
    monkeypatch, tmp_path
):
    unsafe = ReleasableSocket()
    transport = FaultTransport([unsafe])
    monkeypatch.setattr(network_clients, "external_websocket_connect", transport)
    configured = AppSettingsVNext()
    configured = replace(
        configured,
        intent=replace(
            configured.intent, peer_stt=replace(configured.intent.peer_stt, provider="qwen_audio")
        ),
    )
    request = build_peer_stt_provider_request(
        build_peer_capture_session_config_from_vnext(configured), gpu_device_id="auto", warmup=False
    )
    secrets = InMemorySecretStore()
    secrets.set("alibaba_api_key_beijing", "offline")
    factory = SharedSTTProviderFactory(
        secrets=secrets, clock=SystemClock(), reset_deadline_s=0.05, gpu_model_path=tmp_path
    )
    engine = await factory.create(request, gpu_runtime=None)
    engine.watchdog_resolver = lambda _: watchdogs(
        readiness_timeout_s=0.03, drain_timeout_s=0.05, final_timeout_s=0.2
    )
    events = []
    engine.event_sink = events.append
    ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=replace(
            settings("qwen_audio"),
            provider_signature=request.provider_signature,
            runtime_signature=request.runtime_signature,
        ),
    )
    try:
        first = segment_events(ledger, start_sample=0, now=0)
        await engine.handle_owned_vad_event(first[0])
        await engine.handle_owned_vad_event(first[2])
        await wait_until(
            lambda: engine._quarantined_session is not None and not engine._cleanup_tasks
        )
        wrapper = engine._quarantined_session
        assert isinstance(wrapper, StreamingSpeakerSession)
        before = unsafe.close_attempts
        assert not await engine._reclaim_quarantined_session(0.05)
        assert unsafe.close_attempts > before and not unsafe.closed
        assert engine.cleanup_debt == 1
        assert wrapper._consumer.done()
        unsafe.allow_close = True
        assert await engine._reclaim_quarantined_session(0.1)
        assert unsafe.closed and engine.cleanup_debt == 0
        for index in (1, 2):
            start, _, end = segment_events(ledger, start_sample=index * 100, now=index)
            await engine.handle_owned_vad_event(start)
            await engine.handle_owned_vad_event(end)
        fresh = transport.sockets[1:]
        assert len(fresh) == 2
        for socket, text in zip(reversed(fresh), ("second", "first"), strict=True):
            await socket.push(native(socket.task_ids[0], text=text))
            await socket.push(control(socket.task_ids[0], "task-finished"))
        await wait_until(
            lambda: sum(
                isinstance(event, STTProviderTurnTerminal) and event.outcome == "final"
                for event in events
            )
            == 2
        )
        assert [
            event.text
            for event in events
            if isinstance(event, STTProviderTurnTerminal) and event.outcome == "final"
        ] == ["first", "second"]
    finally:
        unsafe.allow_close = True
        await engine.close()
    assert all(socket.closed for socket in transport.sockets)


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["timestamp", "abort", "error", "stop", "close"])
async def test_scribe_cleanup_joins_cancelled_fallback_before_returning(monkeypatch, retirement):
    connection = _ControlledConnection()
    session, _ = await _session(connection)
    request = _request(1)
    entered = asyncio.Event()
    released = asyncio.Event()
    original = type(session)._plain_fallback

    async def gated_fallback(self):
        try:
            await original(self)
        finally:
            entered.set()
            await released.wait()

    monkeypatch.setattr(type(session), "_plain_fallback", gated_fallback)
    cleanup = None
    timer = None
    try:
        await session.begin_turn(request)
        await _seal(session, request)
        connection.emit(
            "committed_transcript", {"message_type": "committed_transcript", "text": "plain"}
        )
        timer = session._plain_fallback_task
        assert timer is not None
        await asyncio.sleep(0)

        async def retire():
            if retirement == "timestamp":
                connection.emit("committed_transcript_with_timestamps", _cwt("canonical", 0.1, 0.2))
            elif retirement == "abort":
                await session.abort_turn(request.identity, reason="cancelled")
            elif retirement == "error":
                session._on_error_event({"message_type": "error"})
            elif retirement == "stop":
                await session.stop()
            await session.close()

        cleanup = asyncio.create_task(retire())
        await entered.wait()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(cleanup), 0.03)
        assert not timer.done()
        released.set()
        await asyncio.wait_for(cleanup, 1)
        assert timer.done()
        assert session._plain_fallback_task is None
    finally:
        released.set()
        if cleanup is not None:
            await asyncio.gather(cleanup, return_exceptions=True)
        if timer is not None:
            await asyncio.gather(timer, return_exceptions=True)
        await session.close()
