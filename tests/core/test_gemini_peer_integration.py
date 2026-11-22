from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from puripuly_heart.app.adapters.peer_capture.peer_capture_vad_sink import (
    PeerCaptureVadSinkAdapter,
)
from puripuly_heart.app.wiring.root import compose_peer_capture_session_owner
from puripuly_heart.app.wiring.wiring_composition import create_peer_capture_audio_loop_adapter
from puripuly_heart.app.wiring.wiring_local_asr_provider_runtime import (
    LocalASRProviderRuntimeFactory,
    SharedSTTProviderFactory,
)
from puripuly_heart.app.wiring.wiring_stt_factory import (
    build_peer_capture_session_config_from_vnext,
    build_peer_stt_provider_request,
)
from puripuly_heart.config.settings_vnext.schema import AppSettingsVNext
from puripuly_heart.core.audio.format import (
    AudioCaptureSpan,
    AudioFrameF32,
    float32_to_pcm16le_bytes,
)
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.core.stt.backend import (
    STTRecognitionPreviewDeadline,
    STTRecognitionUnit,
    STTRecognitionUnitTerminal,
)
from puripuly_heart.core.vad.gating import create_peer_vad_gating
from puripuly_heart.providers.stt import gemini_transcribe as gemini
from tests.core.runtime.test_local_asr_provider_runtime import FakeProvisioningPort
from tests.core.runtime.test_peer_capture_session import (
    FakeAdmission,
    FakeTargetResolver,
    wait_until,
)
from tests.core.test_peer_caption_lifecycle import ControlledBackend, idle
from tests.core.test_peer_caption_lifecycle import preview_timeline as preview_timeline
from tests.core.test_peer_overlay_policy import Timeline
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.scoped_provider_sessions import _gemini_message
from tests.helpers.translation_owners import (
    TranslationOwnersTestHarness,
    compose_translation_test_harness,
)
from tests.helpers.vad import SequenceVadEngine
from tests.providers.test_gemini_transcribe_backend import (
    _FakeLiveContext,
    _FakeLiveSession,
    _StubAsyncTransport,
    _StubClient,
    _StubSyncTransport,
)

_SAMPLE_RATE = 16_000
_FRAME_SAMPLES = 512
_SPEECH_FRAMES = 8
_PAUSE_FRAMES = 16
_INITIAL = "Det vi faktiskt observerade"
_GROWTH = "Det vi faktiskt observerade genom det stora fonstret"
_LATE = "Det vi faktiskt observerade genom det stora fonstret innan lampan slocknade"
_FINAL = "Det riktiga svaret kom senare."
_FRESH = "En ny mening fran samma ljudkalla."


class _ControlledAudio:
    terminal_reason = None

    def __init__(self, started_at: float) -> None:
        self._started_at = started_at
        self._sequence = 0
        self._frames: asyncio.Queue[AudioFrameF32 | None] = asyncio.Queue()

    def feed(self, amplitude: float, count: int) -> None:
        for _ in range(count):
            sequence = self._sequence
            self._sequence += 1
            start = sequence * _FRAME_SAMPLES
            end = start + _FRAME_SAMPLES
            self._frames.put_nowait(
                AudioFrameF32(
                    samples=np.full(_FRAME_SAMPLES, amplitude, dtype=np.float32),
                    sample_rate_hz=_SAMPLE_RATE,
                    capture=AudioCaptureSpan(
                        capture_epoch=1,
                        callback_sequence=sequence + 1,
                        source_sample_rate_hz=_SAMPLE_RATE,
                        source_start_sample=start,
                        source_end_sample=end,
                        source_start_monotonic_s=self._started_at + start / _SAMPLE_RATE,
                        source_end_monotonic_s=self._started_at + end / _SAMPLE_RATE,
                    ),
                )
            )

    async def frames(self):
        while (frame := await self._frames.get()) is not None:
            yield frame

    async def close(self) -> None:
        self._frames.put_nowait(None)


def _pcm(live: _FakeLiveSession) -> bytes:
    return b"".join(send["audio"]["data"] for send in live.sent if "audio" in send)


def _fences(live: _FakeLiveSession) -> int:
    return sum(send.get("audio_stream_end", False) for send in live.sent)


@dataclass
class _PeerRoute:
    provider_id: str
    timeline: Timeline
    audio: _ControlledAudio
    connections: list[_FakeLiveSession]
    harness: TranslationOwnersTestHarness
    backend: ControlledBackend
    overlay: list
    partials: list[STTRecognitionUnit]
    deadlines: list[STTRecognitionPreviewDeadline]
    finals: list[STTRecognitionUnitTerminal]

    async def settle(self) -> None:
        async with asyncio.timeout(3):
            await idle(self.harness)

    async def start_preview(self) -> STTRecognitionUnit:
        self.audio.feed(0.8, _SPEECH_FRAMES)
        await wait_until(
            lambda: self.connections
            and len(_pcm(self.connections[0])) == _SPEECH_FRAMES * _FRAME_SAMPLES * 2,
            timeout_s=3,
        )
        self.connections[0].push(_gemini_message(interim=_INITIAL))
        await wait_until(lambda: len(self.partials) == 1, timeout_s=3)
        unit = self.partials[0]
        stream = unit.identity.stream
        assert stream.settings_scope[0] == self.provider_id
        assert stream.recognizer_id == (
            "gemini_transcribe" if self.provider_id == "rolling_free" else None
        )
        await self.timeline.advance(unit.observed_at + 5.01 - self.timeline.clock.now())
        await self.settle()
        shown = self.overlay[-1]
        assert shown.type == "translation_stream_update"
        assert (shown.utterance_id, shown.source_text, shown.peer_caption_revision) == (
            unit.identity.unit_id,
            _INITIAL,
            1,
        )
        assert [call.text for call in self.backend.calls] == [_INITIAL]
        assert self.harness.peer_runtime.translation_history == []
        return unit


@pytest.fixture(params=["gemini_transcribe", "rolling_free"], ids=["direct", "rolling"])
async def gemini_peer(request, monkeypatch, preview_timeline):
    timeline = preview_timeline
    timeline.clock.advance(asyncio.get_running_loop().time())
    audio = _ControlledAudio(timeline.clock.now())
    connections = []

    def connect(*, model, config):
        live = _FakeLiveSession()
        connections.append(live)
        return _FakeLiveContext(live)

    def prepare(_api_key, language_codes, vocabulary):
        client = _StubClient()
        client.aio.live = SimpleNamespace(connect=connect)
        return gemini._GeminiClientResources(
            client=client,
            sync_transport=_StubSyncTransport(),
            async_transport=_StubAsyncTransport(),
            config=gemini._build_live_config_sync(language_codes, vocabulary),
        )

    monkeypatch.setattr(gemini, "_prepare_gemini_resources_sync", prepare)
    secrets = InMemorySecretStore()
    secrets.set("gemini_api_key", "controlled")
    runtime_factory = LocalASRProviderRuntimeFactory(
        SharedSTTProviderFactory(secrets, timeline.clock, 300.0, None),
        FakeProvisioningPort(),
        timeline.clock,
    )
    callbacks = None
    partials = []
    deadlines = []
    finals = []

    async def observe(event):
        await callbacks.peer_event_handler(event)
        if isinstance(event, STTRecognitionUnit):
            partials.append(event)
        elif isinstance(event, STTRecognitionPreviewDeadline):
            deadlines.append(event)
        elif isinstance(event, STTRecognitionUnitTerminal):
            finals.append(event)

    class RuntimeFactory:
        def create(self, harness_callbacks):
            nonlocal callbacks
            callbacks = harness_callbacks.peer_exception_handler.__self__
            return runtime_factory.create(replace(harness_callbacks, peer_event_handler=observe))

    overlay = []

    async def emit(event):
        overlay.append(event)

    backend = ControlledBackend()
    harness = compose_translation_test_harness(
        local_asr_provider_runtime_factory=RuntimeFactory(),
        llm=backend,
        osc=RecordingOscQueue(),
        clock=timeline.clock,
        overlay_sink=SimpleNamespace(emit=emit, active_self_overlay_metadata=lambda: None),
        peer_source_language="sv",
        target_language="ko",
        peer_translation_enabled=True,
    )
    settings = AppSettingsVNext()
    settings = replace(
        settings,
        intent=replace(
            settings.intent,
            peer_stt=replace(settings.intent.peer_stt, provider=request.param),
            stt=replace(settings.intent.stt, cloud_free_tier_providers=["gemini_transcribe"]),
            languages=replace(
                settings.intent.languages, peer_source_mode="manual", peer_source_language="sv"
            ),
        ),
    )
    config = build_peer_capture_session_config_from_vnext(settings)
    capture = compose_peer_capture_session_owner(
        provider_runtime=harness.local_asr_runtime,
        channel_reset=harness.peer_owner,
        admission=FakeAdmission(),
        target_resolver=FakeTargetResolver(),
        clock=timeline.clock,
        provider_request_factory=lambda config, warmup: build_peer_stt_provider_request(
            config, gpu_device_id=config.gpu_device_id, warmup=warmup
        ),
        source_factory=lambda *_: audio,
        vad_factory=lambda current: create_peer_vad_gating(
            SequenceVadEngine(probs=([0.9] * _SPEECH_FRAMES + [0.0] * _PAUSE_FRAMES) * 3),
            sample_rate_hz=current.target_sample_rate_hz,
            ring_buffer_ms=current.vad_pre_roll_ms,
            speech_threshold=current.vad_speech_threshold,
            hangover_ms=current.vad_hangover_ms,
        ),
        run_audio_loop=create_peer_capture_audio_loop_adapter(log_basic=lambda _: None),
        vad_sink=PeerCaptureVadSinkAdapter(lambda: harness.peer_owner),
    )
    assert callbacks is not None
    callbacks.bind_peer_capture(capture)
    capture.bind_publication_generation_observer(
        activated=harness.output_runtime.activate_peer_generation,
        retired=harness.output_runtime.retire_peer_generation,
    )
    try:
        await harness.start()
        started = await capture.apply_intent(config, enabled=True)
        assert started.effective_active
        assert started.effective_delivery_profile == "unsupported_language"
        yield _PeerRoute(
            provider_id=request.param,
            timeline=timeline,
            audio=audio,
            connections=connections,
            harness=harness,
            backend=backend,
            overlay=overlay,
            partials=partials,
            deadlines=deadlines,
            finals=finals,
        )
    finally:
        await capture.close()
        await harness.stop()


@pytest.mark.asyncio
async def test_gemini_peer_preview_deadline_preserves_visible_revision_until_native_final(gemini_peer):
    route = gemini_peer
    partial = await route.start_preview()
    live = route.connections[0]
    shown = list(route.overlay)

    sealed_at = route.timeline.clock.now()
    route.audio.feed(0.0, _PAUSE_FRAMES)
    await wait_until(lambda: len(route.deadlines) == 1, timeout_s=3)
    deadline = route.deadlines[0]
    assert _fences(live) == 1
    assert (deadline.stream, deadline.unit_id) == (partial.identity.stream, partial.identity.unit_id)
    assert deadline.expires_at_s - sealed_at == pytest.approx(0.6)

    await route.timeline.advance(0.3)
    live.push(_gemini_message(interim=_GROWTH))
    await wait_until(lambda: len(route.partials) == 2, timeout_s=3)
    growth = route.partials[-1]
    assert growth.text == _GROWTH and growth.identity.unit_id == deadline.unit_id

    await route.timeline.advance(3.75)
    await route.settle()
    assert route.overlay == shown
    assert [call.text for call in route.backend.calls] == [_INITIAL]
    assert route.harness.peer_runtime.translation_history == []
    assert route.finals == []
    assert not live.closed

    live.push(_gemini_message(interim=_LATE))
    live.push(_gemini_message(final=_FINAL))
    await wait_until(lambda: len(route.finals) == 1, timeout_s=3)
    await route.settle()
    terminal = route.finals[0]
    assert terminal.outcome == "final"
    assert terminal.unit.identity.stream == deadline.stream
    assert terminal.unit.identity.unit_id == deadline.unit_id
    assert terminal.unit.identity.receipt_sequence == growth.identity.receipt_sequence + 2
    assert [unit.text for unit in route.partials] == [_INITIAL, _GROWTH]
    assert [call.text for call in route.backend.calls] == [_INITIAL, _FINAL]
    assert [
        (entry.utterance_id, entry.text) for entry in route.harness.peer_runtime.translation_history
    ] == [(deadline.unit_id, _FINAL)]
    assert [
        (event.utterance_id, event.source_text, event.text, event.peer_caption_revision)
        for event in route.overlay
        if event.type == "translation_final"
    ] == [(deadline.unit_id, _FINAL, f"translated:{_FINAL}", 2)]
    assert len(route.connections) == 1 and not live.closed
    assert route.harness.osc.messages == []


@pytest.mark.asyncio
async def test_gemini_peer_repeated_clear_end_recovers_for_new_audio_without_replay(gemini_peer):
    route = gemini_peer
    partial = await route.start_preview()
    stalled = route.connections[0]
    route.audio.feed(0.0, _PAUSE_FRAMES)
    await wait_until(lambda: len(route.deadlines) == 1, timeout_s=3)
    await route.timeline.advance(0.61)
    await route.settle()
    assert _fences(stalled) == 1 and not stalled.closed

    route.audio.feed(0.7, _SPEECH_FRAMES)
    route.audio.feed(0.0, _PAUSE_FRAMES)
    await wait_until(lambda: stalled.closed, timeout_s=3)
    assert _fences(stalled) == 2
    assert len(route.connections) == len(route.deadlines) == 1
    assert route.finals == []
    assert route.harness.peer_runtime.translation_history == []

    route.audio.feed(0.5, _SPEECH_FRAMES)
    expected = float32_to_pcm16le_bytes(
        np.full(_SPEECH_FRAMES * _FRAME_SAMPLES, 0.5, dtype=np.float32)
    )
    await wait_until(
        lambda: len(route.connections) == 2 and len(_pcm(route.connections[1])) >= len(expected),
        timeout_s=3,
    )
    recovered = route.connections[1]
    assert _pcm(recovered) == expected
    recovered.push(_gemini_message(final=_FRESH))
    await wait_until(lambda: len(route.finals) == 1, timeout_s=3)
    await route.settle()
    fresh = route.finals[0].unit.identity
    assert fresh.unit_id != partial.identity.unit_id
    assert fresh.stream.provider_epoch_id != partial.identity.stream.provider_epoch_id
    assert fresh.stream.activation_generation == partial.identity.stream.activation_generation
    assert [call.text for call in route.backend.calls] == [_INITIAL, _FRESH]
    assert [
        (entry.utterance_id, entry.text) for entry in route.harness.peer_runtime.translation_history
    ] == [(fresh.unit_id, _FRESH)]
    assert [
        (event.utterance_id, event.source_text)
        for event in route.overlay
        if event.type == "translation_final" and event.source_text
    ] == [(fresh.unit_id, _FRESH)]
