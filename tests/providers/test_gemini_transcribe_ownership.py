from __future__ import annotations

import asyncio
import threading
from uuid import uuid4

import pytest

from puripuly_heart.core.audio.ownership import (
    AudioSegmentSettingsSnapshot,
    PeerAudioSegmentLedger,
)
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.stt.backend import (
    STTProviderTurnTerminal,
    STTSessionProjection,
)
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.stt.scoped_engine import ScopedRecognitionEngine
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.domain.events import STTFinalEvent, STTSessionState, STTSessionStateEvent
from puripuly_heart.providers.stt import gemini_transcribe as gemini_module
from puripuly_heart.providers.stt.gemini_transcribe import (
    GeminiTranscribeSTTBackend,
    _build_live_config_sync,
)
from tests.helpers.fakes import samples


class _FakeLiveSession:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self._queue: asyncio.Queue = asyncio.Queue()

    async def send_realtime_input(self, **kwargs) -> None:
        self.sent.append(kwargs)

    async def receive(self):
        while True:
            item = await self._queue.get()
            if item is _DONE:
                return
            yield item

    def push(self, item) -> None:
        self._queue.put_nowait(item)

    async def close(self) -> None:
        self.closed = True


_DONE = object()


class _FakeLiveContext:
    def __init__(self, session: _FakeLiveSession) -> None:
        self._session = session
        self.exited = False

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        await self._session.close()
        return False


class _StubSyncTransport:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class _StubAsyncTransport:
    def __init__(self) -> None:
        self.closed = False
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1
        self.closed = True


class _StubClientAio:
    def __init__(self) -> None:
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _StubClient:
    def __init__(self) -> None:
        self.aio = _StubClientAio()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _stub_resources(language_codes, custom_vocabulary):
    return gemini_module._GeminiClientResources(
        client=_StubClient(),
        sync_transport=_StubSyncTransport(),
        async_transport=_StubAsyncTransport(),
        config=_build_live_config_sync(language_codes, custom_vocabulary),
    )


def _patch_prepare(monkeypatch, made, *, gate=None, entered=None, idents=None, gate_key=None):
    def fake_prepare(api_key, language_codes, custom_vocabulary):
        if idents is not None:
            idents.append(threading.get_ident())
        gated = gate_key is None or api_key == gate_key
        if entered is not None and gated:
            entered.set()
        if gate is not None and gated:
            assert gate.wait(timeout=10)
        resources = _stub_resources(language_codes, custom_vocabulary)
        made.append(resources)
        return resources

    monkeypatch.setattr(gemini_module, "_prepare_gemini_resources_sync", fake_prepare)


class _RecordingFactory:
    def __init__(self) -> None:
        self.live: _FakeLiveSession | None = None
        self.calls: list = []
        self.context: _FakeLiveContext | None = None

    def __call__(self, *, model: str, config: object):
        live = _FakeLiveSession()
        self.live = live
        self.calls.append((model, config))
        self.context = _FakeLiveContext(live)
        return self.context


def _backend(factory, *, api_key="dummy-offline-key") -> GeminiTranscribeSTTBackend:
    return GeminiTranscribeSTTBackend(
        api_key=api_key,
        language_codes=("ko-KR",),
        live_connect_factory=factory,
    )


def _controller(backend, *, channel="self"):
    return ManagedSTTProvider(
        backend=backend,
        channel=channel,
        sample_rate_hz=16000,
        clock=FakeClock(),
        reset_deadline_s=90.0,
        connect_attempts=1,
    )


def _final(text: str):
    from google.genai import types

    return types.LiveServerMessage(
        server_content=types.LiveServerContent(input_transcription=types.Transcription(text=text))
    )


def _activity_end_ack():
    from google.genai import types

    return types.LiveServerMessage(
        voice_activity=types.VoiceActivity(
            voice_activity_type=types.VoiceActivityType.ACTIVITY_END,
        )
    )


async def _wait_for_sent(live: _FakeLiveSession, key: str) -> None:
    async def ready() -> None:
        while not any(key in call for call in live.sent):
            await asyncio.sleep(0)

    await asyncio.wait_for(ready(), timeout=5)


async def _next_streaming(stream) -> None:
    async def ready() -> None:
        while True:
            event = await stream.__anext__()
            if isinstance(event, STTSessionStateEvent) and event.state == STTSessionState.STREAMING:
                return

    await asyncio.wait_for(ready(), timeout=5)


async def _next_final(stream, timeout=5.0):
    async def ready():
        while True:
            event = await stream.__anext__()
            if isinstance(event, STTFinalEvent):
                return event

    return await asyncio.wait_for(ready(), timeout=timeout)


def _assert_resources_released_once(resources) -> None:
    assert resources.client.aio.aclose_calls == 1
    assert resources.client.close_calls == 1
    assert resources.sync_transport.close_calls == 1
    assert resources.sync_transport.closed is True
    assert resources.async_transport.aclose_calls == 1
    assert resources.async_transport.closed is True


@pytest.mark.asyncio
async def test_peer_channel_completes_while_self_setup_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made_self: list = []
    idents: list[int] = []
    gate = threading.Event()
    entered = threading.Event()
    _patch_prepare(
        monkeypatch,
        made_self,
        gate=gate,
        entered=entered,
        idents=idents,
        gate_key="dummy-offline-key",
    )
    factory_self = _RecordingFactory()
    factory_peer = _RecordingFactory()
    backend_self = _backend(factory_self)
    backend_peer = _backend(factory_peer, api_key="dummy-offline-peer")
    self_stt = _controller(backend_self, channel="self")
    peer_stt = _controller(backend_peer, channel="peer")
    self_stream = self_stt.events()
    peer_stream = peer_stt.events()
    try:
        uid_self = uuid4()
        self_open = asyncio.create_task(
            self_stt.handle_vad_event(
                SpeechStart(uid_self, pre_roll=samples(0.0), chunk=samples(0.1))
            )
        )
        assert await asyncio.to_thread(entered.wait, timeout=5)
        uid_peer = uuid4()
        await peer_stt.handle_vad_event(
            SpeechStart(uid_peer, pre_roll=samples(0.0), chunk=samples(0.1))
        )
        await _next_streaming(peer_stream)
        await peer_stt.handle_vad_event(SpeechChunk(uid_peer, chunk=samples(0.1)))
        await peer_stt.handle_vad_event(SpeechEnd(uid_peer, trailing_silence_ms=100))
        assert factory_peer.live is not None
        await _wait_for_sent(factory_peer.live, "activity_end")
        factory_peer.live.push(_final("peer says hello"))
        factory_peer.live.push(_activity_end_ack())
        final_event = await _next_final(peer_stream)
        assert final_event.transcript.text == "peer says hello"
        assert factory_self.calls == []
        gate.set()
        await self_open
        await _next_streaming(self_stream)
    finally:
        await self_stt.close()
        await peer_stt.close()
    assert idents and idents[0] != threading.get_ident()
    assert len(factory_self.calls) == 1
    assert len(made_self) == 2
    for resources in made_self:
        _assert_resources_released_once(resources)


@pytest.mark.asyncio
async def test_toggle_off_rejects_late_final_and_replacement_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list = []
    _patch_prepare(monkeypatch, made)
    factory = _RecordingFactory()
    stt = _controller(_backend(factory))
    stream = stt.events()
    try:
        uid_retired = uuid4()
        await stt.handle_vad_event(
            SpeechStart(uid_retired, pre_roll=samples(0.0), chunk=samples(0.1))
        )
        await _next_streaming(stream)
        await stt.handle_vad_event(SpeechChunk(uid_retired, chunk=samples(0.1)))
        await stt.handle_vad_event(SpeechEnd(uid_retired, trailing_silence_ms=100))
        assert factory.live is not None
        retired_live = factory.live
        await _wait_for_sent(retired_live, "activity_end")
        await stt.abort_for_toggle_off()
        stream = stt.events()
        uid_fresh = uuid4()
        await stt.handle_vad_event(
            SpeechStart(uid_fresh, pre_roll=samples(0.0), chunk=samples(0.1))
        )
        await _next_streaming(stream)
        fresh_live = factory.live
        assert fresh_live is not None and fresh_live is not retired_live
        retired_live.push(_final("retired session late text"))
        retired_live.push(_activity_end_ack())
        await asyncio.sleep(0.2)
        await stt.handle_vad_event(SpeechChunk(uid_fresh, chunk=samples(0.1)))
        await stt.handle_vad_event(SpeechEnd(uid_fresh, trailing_silence_ms=100))
        assert factory.live is fresh_live
        await _wait_for_sent(fresh_live, "activity_end")
        fresh_live.push(_final("fresh replacement text"))
        fresh_live.push(_activity_end_ack())
        final_event = await _next_final(stream)
        assert final_event.utterance_id == uid_fresh
        assert final_event.transcript.text == "fresh replacement text"
    finally:
        await stt.close()


@pytest.mark.asyncio
async def test_abort_during_open_leaves_single_owned_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list = []
    gate = threading.Event()
    entered = threading.Event()
    _patch_prepare(monkeypatch, made, gate=gate, entered=entered)
    factory = _RecordingFactory()
    stt = _controller(_backend(factory))
    try:
        uid = uuid4()
        open_task = asyncio.create_task(
            stt.handle_vad_event(SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(0.1)))
        )
        assert await asyncio.to_thread(entered.wait, timeout=5)
        await stt.abort_for_toggle_off()
        gate.set()
        await open_task
        assert len(made) == 1
        assert len(factory.calls) == 1
    finally:
        await stt.close()
    _assert_resources_released_once(made[0])
    assert factory.live is not None
    assert factory.live.closed is True


@pytest.mark.asyncio
async def test_capture_tasks_progress_on_peer_while_self_setup_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from puripuly_heart.core.runtime.audio_vad_loop import run_audio_vad_loop
    from puripuly_heart.core.vad.gating import VadGating
    from tests.helpers.audio import FakeAudioSource, make_frames
    from tests.helpers.vad import SequenceVadEngine

    gate = threading.Event()
    entered = threading.Event()
    idents: list[int] = []
    made: list = []
    _patch_prepare(
        monkeypatch,
        made,
        gate=gate,
        entered=entered,
        idents=idents,
        gate_key="dummy-offline-key",
    )
    factory_self = _RecordingFactory()
    factory_peer = _RecordingFactory()
    self_stt = _controller(_backend(factory_self), channel="self")
    peer_backend = _backend(factory_peer, api_key="dummy-offline-peer")
    peer_terminals: list[STTProviderTurnTerminal] = []

    async def open_peer_session(_settings, epoch_id):
        return await peer_backend.open_session(
            projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch_id)
        )

    peer_stt = ScopedRecognitionEngine(
        session_factory=open_peer_session,
        event_sink=lambda event: (
            peer_terminals.append(event) if isinstance(event, STTProviderTurnTerminal) else None
        ),
    )
    peer_ledger = PeerAudioSegmentLedger(
        activation_generation=1,
        settings=AudioSegmentSettingsSnapshot(
            provider_id="gemini_transcribe",
            provider_signature=("gemini_transcribe", "en"),
            runtime_signature=("gemini_transcribe", "peer"),
            source_mode="manual",
            source_language="en",
            expected_languages=("en",),
            target_sample_rate_hz=16000,
            vad_speech_threshold=0.5,
            vad_hangover_ms=0,
            vad_pre_roll_ms=32,
        ),
    )

    class PeerOwnedSink:
        async def handle_vad_event(self, event) -> None:
            owned = peer_ledger.observe_vad_event(
                event,
                now_monotonic_s=asyncio.get_running_loop().time(),
            )
            await peer_stt.handle_owned_vad_event(owned)

    self_stream = self_stt.events()

    def build_capture_loop(sink):
        audio = np.concatenate(
            [
                np.zeros(512, dtype=np.float32),
                np.ones(512, dtype=np.float32),
                np.ones(512, dtype=np.float32),
                np.zeros(512, dtype=np.float32),
                np.zeros(512, dtype=np.float32),
            ]
        )
        source = FakeAudioSource(make_frames(audio, sample_rate_hz=16000, splits=[512] * 5))
        vad = VadGating(
            SequenceVadEngine(probs=[0.0, 0.9, 0.9, 0.0, 0.0]),
            sample_rate_hz=16000,
            ring_buffer_ms=32,
            hangover_ms=0,
        )
        return asyncio.create_task(
            run_audio_vad_loop(source=source, vad=vad, sink=sink, target_sample_rate_hz=16000)
        )

    loop_self = build_capture_loop(self_stt)
    loop_peer = build_capture_loop(PeerOwnedSink())
    try:
        assert await asyncio.to_thread(entered.wait, timeout=5)
        assert factory_self.calls == []
        async with asyncio.timeout(5):
            while factory_peer.live is None:
                await asyncio.sleep(0)
        await _wait_for_sent(factory_peer.live, "activity_end")
        factory_peer.live.push(_final("peer capture says hello"))
        factory_peer.live.push(_activity_end_ack())
        await asyncio.wait_for(asyncio.shield(loop_peer), timeout=5)
        assert len(peer_terminals) == 1
        assert peer_terminals[0].text == "peer capture says hello"
        assert peer_terminals[0].identity.segment == peer_ledger.snapshots[0].identity
        gate.set()
        await asyncio.wait_for(asyncio.shield(loop_self), timeout=5)
        assert factory_self.live is not None
        await _wait_for_sent(factory_self.live, "activity_end")
        factory_self.live.push(_final("self capture says hello"))
        factory_self.live.push(_activity_end_ack())
        self_final = await _next_final(self_stream)
        assert self_final.transcript.text == "self capture says hello"
    finally:
        gate.set()
        await self_stt.close()
        await peer_stt.close_backend()
    assert idents and idents[0] != threading.get_ident()
    assert len(made) == 2
    for resources in made:
        _assert_resources_released_once(resources)
