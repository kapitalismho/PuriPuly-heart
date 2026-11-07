from __future__ import annotations

import json
from dataclasses import replace
from uuid import uuid4

import pytest

from puripuly_heart.app.services.recognition_probe import load_recognition_probe_audio
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.core.runtime.gpu_asr import SharedGpuASRRuntime
from puripuly_heart.core.stt.backend import (
    STTProviderInputTerminal,
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.stt.recognition_probe import recognize_probe_audio
from puripuly_heart.core.stt.rolling import RollingProviderDefinition, RollingSTTBackend
from puripuly_heart.core.stt.scoped_engine import STTRecognitionWatchdogs
from puripuly_heart.providers.stt import gemini_transcribe as gemini
from puripuly_heart.providers.stt.local_gpu import LocalGpuSTTBackend
from puripuly_heart.providers.stt.qwen_audio import QwenAudioStreamingSTTBackend
from tests.core.runtime.test_gpu_asr import FakeASRWorkerClient, FakeASRWorkerFactory
from tests.core.test_stt_scoped_engine import settings
from tests.providers.test_gemini_transcribe_backend import (
    _FakeLiveSession,
    _final,
    _RecordingFactory,
    _StubAsyncTransport,
    _StubClient,
    _StubSyncTransport,
)
from tests.providers.test_qwen_audio import ScriptedSocket, SocketFactory, control, native

WATCHDOGS = STTRecognitionWatchdogs(readiness_timeout_s=1, write_timeout_s=1, final_timeout_s=0.15)


class FinalSocket(ScriptedSocket):
    def __init__(self, text):
        super().__init__()
        self.text = text

    async def send(self, value):
        await super().send(value)
        if isinstance(value, str):
            packet = json.loads(value)
            if packet["header"]["action"] == "finish-task":
                if self.text is not None:
                    await self.push(native(packet["header"]["task_id"], text=self.text))
                await self.push(control(packet["header"]["task_id"], "task-finished"))


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize(
    "text,succeeded", [("recognized synthetic speech", True), ("", False), (None, False)]
)
async def test_streaming_task_requires_nonempty_final_and_closes_only_new_socket(
    channel, text, succeeded
):
    live_socket = ScriptedSocket()
    probe_socket = FinalSocket(text)
    sockets = SocketFactory([live_socket, probe_socket])
    backend = QwenAudioStreamingSTTBackend(
        "offline-controlled", channel=channel, websocket_factory=sockets
    )
    live = await backend.open_session(projection=STTSessionProjection("scoped", "live"))
    from tests.providers.test_qwen_audio import request, send

    live_request = request(channel)
    live_request = replace(
        live_request, identity=replace(live_request.identity, provider_epoch_id="live")
    )
    await live.begin_turn(live_request)
    await send(live, live_request, count=160)
    pcm = load_recognition_probe_audio(16000)
    retained = []

    async def factory(snapshot, epoch):
        session = await backend.open_session(projection=STTSessionProjection("scoped", epoch))
        return session, "qwen_audio"

    try:
        outcome = await recognize_probe_audio(
            factory, settings("qwen_audio"), channel, pcm, WATCHDOGS, retained
        )
        assert (outcome.result == "succeeded") is succeeded
        assert outcome.final_event == ("authoritative_turn_final" if succeeded else None)
        assert b"".join(value for value in probe_socket.sent if isinstance(value, bytes)) == pcm
        assert b"".join(value for value in live_socket.sent if isinstance(value, bytes)) != pcm
        assert probe_socket.closed
        assert not live_socket.closed
        assert retained == []
    finally:
        await live.close()


class FinalLiveSession(_FakeLiveSession):
    def __init__(self, text):
        super().__init__()
        self.text = text

    async def send_realtime_input(self, **kwargs):
        await super().send_realtime_input(**kwargs)
        if kwargs.get("audio_stream_end") and self.text is not None:
            self.push(_final(self.text))


@pytest.mark.parametrize("channel", ["self", "peer"])
@pytest.mark.parametrize(
    "text,succeeded", [("final provider unit", True), ("", False), (None, False)]
)
async def test_native_stream_waits_for_final_unit_not_input_submission(
    monkeypatch, channel, text, succeeded
):
    transport = FinalLiveSession(text)
    backend = gemini.GeminiTranscribeSTTBackend(
        "offline-controlled",
        live_connect_factory=_RecordingFactory(transport),
        drain_timeout_s=0.01,
    )
    config = gemini._build_live_config_sync(backend.language_codes, backend.custom_vocabulary)

    def prepare(api_key, language_codes, custom_vocabulary):
        return gemini._GeminiClientResources(
            client=_StubClient(),
            sync_transport=_StubSyncTransport(),
            async_transport=_StubAsyncTransport(),
            config=config,
        )

    monkeypatch.setattr(gemini, "_prepare_gemini_resources_sync", prepare)
    retained = []

    async def factory(snapshot, epoch):
        session = await backend.open_session(projection=STTSessionProjection("scoped", epoch))
        return session, "gemini_transcribe"

    pcm = load_recognition_probe_audio(16000)
    if succeeded:
        outcome = await recognize_probe_audio(
            factory, settings("gemini_transcribe"), channel, pcm, WATCHDOGS, retained
        )
        assert outcome.result == "succeeded" and outcome.final_event == "final_recognition_unit"
    else:
        with pytest.raises(TimeoutError):
            await recognize_probe_audio(
                factory, settings("gemini_transcribe"), channel, pcm, WATCHDOGS, retained
            )
    written = b"".join(item["audio"]["data"] for item in transport.sent if "audio" in item)
    assert written == pcm
    assert any(item.get("audio_stream_end") is True for item in transport.sent)
    assert transport.closed and retained == []


@pytest.mark.parametrize(
    "kind", ["input_only", "foreign_final", "degraded", "stale_final", "failed_text"]
)
async def test_unrelated_submitted_degraded_and_stale_results_cannot_establish_success(kind):
    class Session:
        async def begin_turn(self, request):
            self.request = request

        async def send_turn_audio(self, *args, **kwargs):
            pass

        async def seal_turn(self, *args, **kwargs):
            pass

        async def close(self):
            self.closed = True

        async def turn_events(self):
            identity = self.request.identity
            if kind == "input_only":
                yield STTProviderInputTerminal(identity, "submitted", "self")
            else:
                if kind == "foreign_final":
                    identity = replace(identity, provider_turn_id=uuid4().hex)
                yield STTProviderTurnTerminal(
                    identity,
                    "failed" if kind == "failed_text" else "final",
                    "not enough evidence",
                    text_authority="degraded" if kind == "degraded" else "authoritative",
                    is_current=(lambda: False) if kind == "stale_final" else None,
                )

    session = Session()

    async def factory(snapshot, epoch):
        return session, "deepgram"

    outcome = await recognize_probe_audio(
        factory, settings("deepgram"), "self", b"\x01\x00" * 160, WATCHDOGS, []
    )
    assert outcome.result != "succeeded" and outcome.final_event is None
    assert session.closed


async def test_rolling_probe_uses_one_eligible_member_without_fallback_or_exclusion_mutation():
    calls = []

    class FailingBackend:
        async def open_session(self, **kwargs):
            calls.append("first")
            raise ConnectionError("offline controlled failure")

    class OtherBackend:
        async def open_session(self, **kwargs):
            calls.append("fallback")
            raise AssertionError("probe must not fall through")

    rolling = RollingSTTBackend(
        (
            RollingProviderDefinition(
                STTProviderName.ELEVENLABS_SCRIBE, FailingBackend, lambda: True
            ),
            RollingProviderDefinition(STTProviderName.DEEPGRAM, OtherBackend, lambda: True),
        )
    )
    before = dict(rolling._states)
    with pytest.raises(ConnectionError):
        await rolling.open_probe_session(projection=STTSessionProjection("scoped", "probe"))
    assert calls == ["first"] and rolling._states == before
    assert rolling._last_failures == {}


@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_gpu_probe_reuses_resident_model_and_preserves_live_channel_lease(tmp_path, channel):
    client = FakeASRWorkerClient()
    process_factory = FakeASRWorkerFactory([client])
    runtime = SharedGpuASRRuntime(process_factory=process_factory)
    backend = LocalGpuSTTBackend(runtime, channel, tmp_path / "model.gguf", "test-model", "auto")
    live = await backend.open_session(projection=STTSessionProjection("scoped", "live"))

    async def factory(snapshot, epoch):
        session = await backend.open_session(projection=STTSessionProjection("scoped", epoch))
        return session, "local_qwen_gpu"

    try:
        outcome = await recognize_probe_audio(
            factory,
            settings("local_qwen_gpu"),
            channel,
            load_recognition_probe_audio(16000),
            WATCHDOGS,
            [],
        )
        assert outcome.result == "succeeded"
        assert outcome.final_event == "authoritative_turn_final"
        assert runtime.active_channels == frozenset({channel})
        assert len(client.activate_calls) == 1
        assert not client.is_closed
    finally:
        await live.close()
        await backend.close()
        await runtime.close()
