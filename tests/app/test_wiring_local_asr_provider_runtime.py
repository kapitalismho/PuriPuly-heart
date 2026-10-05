from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import puripuly_heart.app.wiring_local_asr_provider_runtime as runtime_wiring
import pytest
from puripuly_heart.app.adapters.self_capture_provider import SelfCaptureProviderAdapter
from puripuly_heart.app.wiring_local_asr_provider_runtime import (
    LocalASRProviderRuntimeFactory,
    SharedSTTProviderFactory,
    _recognition_retention_profile,
)
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest

from puripuly_heart.app.ports.gpu_worker import GpuWorkerRequestError
from puripuly_heart.app.wiring.wiring_stt_factory import build_peer_stt_provider_request
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.resolved import (
    ResolvedCredentialRequirement,
    ResolvedSTTConfig,
)
from puripuly_heart.core.audio.format import AudioCaptureSpan
from puripuly_heart.core.audio.ownership import AudioSegmentSettingsSnapshot, PeerAudioSegmentLedger
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.peer_capture import (
    PeerCaptureLanguageFacts,
    PeerCaptureSessionConfig,
    PeerCaptureTargetIntent,
)
from puripuly_heart.core.runtime.gpu_asr import SharedGpuASRRuntime
from puripuly_heart.core.runtime.local_asr_provider_runtime import LocalASRProviderRuntimeOwner
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.self_capture import SelfCaptureSessionConfig
from puripuly_heart.core.stt.backend import PermanentSTTScopedSessionError, STTSessionProjection
from puripuly_heart.core.stt.custom import (
    CustomSTTConfigurationError,
    normalize_custom_stt_extra,
    validate_peer_custom_stt_configuration,
)
from puripuly_heart.core.stt.scoped_engine import (
    ScopedRecognitionEngine,
)
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.providers.stt.custom import _OfflineOpenAITranscriptionSession
from tests.core.runtime.test_gpu_asr import (
    DEVICE,
    FakeGpuWorkerClient,
    FakeGpuWorkerFactory,
)
from tests.core.runtime.test_local_asr_provider_runtime import (
    FakeProvisioningPort,
    _resolved_config,
)


class _RuntimeLogging:
    def __init__(self) -> None:
        self.basic: list[tuple[str, int]] = []
        self.detailed: list[str] = []

    def emit_basic(self, message: str, *, level: int = 20) -> None:
        self.basic.append((message, level))

    def emit_diagnostic(self, message: str, **_kwargs: object) -> bool:
        self.detailed.append(message)
        return True


async def test_managed_provider_factory_uses_scoped_projection_for_both_channels(
    monkeypatch,
) -> None:
    calls: list[tuple[ResolvedSTTConfig, dict[str, object]]] = []

    class LegacySession:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class Backend:
        def __init__(self) -> None:
            self.sessions: list[LegacySession] = []

        async def open_session(self, **_kwargs) -> LegacySession:
            session = LegacySession()
            self.sessions.append(session)
            return session

    backend = Backend()

    def create_backend(config: ResolvedSTTConfig, **kwargs):
        calls.append((config, kwargs))
        return backend

    monkeypatch.setattr(runtime_wiring, "create_stt_backend_from_resolved_config", create_backend)
    config = ResolvedSTTConfig(
        channel="peer",
        source_language="ja",
        provider="deepgram",
        model="nova-3",
        endpoint=None,
        region=None,
        credential=ResolvedCredentialRequirement(
            source="secret_store",
            required=True,
            reference="deepgram:stt",
        ),
        input_host_api=None,
        input_device=None,
        output_device="headphones",
        sample_rate_hz=16000,
        channels=1,
        ring_buffer_ms=500,
        drain_timeout_s=2.0,
        vad_speech_threshold=0.5,
        vad_hangover_ms=900,
        vad_pre_roll_ms=320,
        low_latency_enabled=True,
        low_latency_merge_gap_ms=600,
        low_latency_spec_retry_max=1,
        custom_vocabulary_enabled=False,
        custom_terms={},
        provider_options={},
    )
    options = LocalASRSessionOptions(source_language="ja")
    request = ProviderRuntimeBuildRequest(
        config=config,
        gpu_device_id="vk:2",
        model_id="nova-3",
        session_options=options,
        provider_signature=("deepgram", "nova-3"),
        runtime_signature=("ja", "headphones"),
        recognition_projection="scoped",
    )
    observer = object()
    runtime_logging = _RuntimeLogging()
    factory = SharedSTTProviderFactory(
        secrets=object(),
        clock=FakeClock(),
        reset_deadline_s=300.0,
        gpu_model_path=Path("gpu.gguf"),
        event_ingress_observer=observer,
        runtime_logging=runtime_logging,
    )
    gpu_runtime = object()

    peer_provider = await factory.create(request, gpu_runtime=gpu_runtime)
    self_config = replace(config, channel="self")
    self_provider = await factory.create(
        ProviderRuntimeBuildRequest(
            config=self_config,
            gpu_device_id="vk:2",
            model_id="nova-3",
            session_options=options,
            provider_signature=("deepgram", "nova-3"),
            runtime_signature=("ja", "microphone"),
            recognition_projection="scoped",
        ),
        gpu_runtime=gpu_runtime,
    )
    self_like_provider = await factory.create(
        ProviderRuntimeBuildRequest(
            config=self_config,
            gpu_device_id="vk:2",
            model_id="nova-3",
            session_options=options,
            provider_signature=("deepgram", "nova-3"),
            runtime_signature=("ja", "microphone"),
            recognition_projection="scoped",
        ),
        gpu_runtime=gpu_runtime,
    )

    assert peer_provider.diagnostic_sink is not None
    peer_provider.diagnostic_sink(
        SimpleNamespace(
            reason="language_run_conservation_fallback",
            identity=SimpleNamespace(
                provider_turn_id="private-turn-id",
                provider_epoch_id="private-epoch-id",
            ),
        )
    )
    peer_provider.diagnostic_sink(
        SimpleNamespace(
            reason="future_normalization_evidence",
            identity=SimpleNamespace(provider_turn_id="private-normal-turn-id"),
        )
    )
    degraded_message, degraded_level = runtime_logging.basic[-1]
    assert "reason=language_run_conservation_fallback" in degraded_message
    assert "degraded=true" in degraded_message
    assert degraded_level == 30
    assert "future_normalization_evidence" in runtime_logging.detailed[-1]
    assert "degraded=false" in runtime_logging.detailed[-1]
    rendered_diagnostics = repr((runtime_logging.basic, runtime_logging.detailed))
    assert "private-turn-id" not in rendered_diagnostics
    assert "private-epoch-id" not in rendered_diagnostics
    assert "private-normal-turn-id" not in rendered_diagnostics
    assert [call[0] for call in calls] == [config, self_config, self_config]
    basic_log_sink = calls[0][1].pop("basic_log_sink")
    assert callable(basic_log_sink)
    basic_log_sink("measured CPU attempt", 20)
    assert runtime_logging.basic[-1] == ("measured CPU attempt", 20)
    assert calls[0][1] == {
        "secrets": factory.secrets,
        "gpu_runtime": gpu_runtime,
        "gpu_model_path": Path("gpu.gguf"),
        "gpu_device_id": "vk:2",
    }
    assert isinstance(peer_provider, ScopedRecognitionEngine)
    assert peer_provider.scoped_settings_scope == (
        "deepgram",
        ("deepgram", "nova-3"),
        ("ja", "headphones"),
    )
    assert peer_provider.watchdog_resolver(None).final_timeout_s == 9.0
    assert isinstance(self_provider, ScopedRecognitionEngine)
    assert self_provider.channel == "self"
    assert self_provider.scoped_settings_scope == (
        "deepgram",
        ("deepgram", "nova-3"),
        ("ja", "microphone"),
    )
    assert isinstance(self_like_provider, ScopedRecognitionEngine)
    assert self_like_provider.channel == "self"
    assert self_like_provider.scoped_settings_scope == (
        "deepgram",
        ("deepgram", "nova-3"),
        ("ja", "microphone"),
    )

    with pytest.raises(PermanentSTTScopedSessionError, match="does not implement scoped"):
        await peer_provider.session_factory(None, "epoch")
    assert backend.sessions[0].closed is True

    custom_config = replace(
        config,
        provider=STTProviderName.CUSTOM_REALTIME.value,
        provider_options={
            "mode": "realtime",
            "extra": {"turn_detection": {"type": "server_vad"}},
        },
    )
    with pytest.raises(CustomSTTConfigurationError, match="turn_detection=null"):
        await factory.create(
            ProviderRuntimeBuildRequest(
                config=custom_config,
                provider_signature=("custom",),
                runtime_signature=("custom",),
            ),
            gpu_runtime=gpu_runtime,
        )
    assert [call[0] for call in calls] == [config, self_config, self_config]


def test_custom_turn_detection_is_metadata_only_until_peer_validation() -> None:
    requested = normalize_custom_stt_extra(
        {" TURN_DETECTION ": {"type": "server_vad"}},
        preserve_turn_detection=True,
    )

    assert requested == {"turn_detection": {"type": "server_vad"}}
    assert normalize_custom_stt_extra(requested) == {}
    assert normalize_custom_stt_extra({}) == {}
    with pytest.raises(CustomSTTConfigurationError, match="turn_detection=null"):
        validate_peer_custom_stt_configuration(mode="realtime", extra=requested)
    with pytest.raises(CustomSTTConfigurationError, match="duplicate"):
        normalize_custom_stt_extra(
            {"turn_detection": None, " TURN_DETECTION ": {"type": "server_vad"}},
            preserve_turn_detection=True,
        )

    with pytest.raises(CustomSTTConfigurationError, match="turn_detection=null"):
        validate_peer_custom_stt_configuration(
            mode="realtime",
            extra={
                "turn_detection": None,
                " TURN_DETECTION ": {"type": "server_vad"},
            },
        )


def test_peer_request_rejects_custom_realtime_server_turn_detection() -> None:
    custom_config = ResolvedSTTConfig(
        channel="peer",
        source_language="en",
        provider=STTProviderName.CUSTOM_REALTIME.value,
        model="custom-model",
        endpoint="wss://example.test/realtime",
        region=None,
        credential=ResolvedCredentialRequirement(
            source="secret_store",
            required=False,
            reference="custom:stt",
        ),
        input_host_api=None,
        input_device=None,
        output_device="headphones",
        sample_rate_hz=16000,
        channels=1,
        ring_buffer_ms=500,
        drain_timeout_s=1.5,
        vad_speech_threshold=0.5,
        vad_hangover_ms=800,
        vad_pre_roll_ms=320,
        low_latency_enabled=False,
        low_latency_merge_gap_ms=600,
        low_latency_spec_retry_max=1,
        custom_vocabulary_enabled=False,
        custom_terms={},
        provider_options={
            "mode": "realtime",
            "extra": {"turn_detection": {"type": "server_vad"}},
        },
    )
    capture = PeerCaptureSessionConfig(
        provider_id=custom_config.provider,
        provider_signature=("custom",),
        runtime_signature=("custom-runtime",),
        capture_signature=("desktop",),
        capture_target=PeerCaptureTargetIntent(kind="default_output_device"),
        language=PeerCaptureLanguageFacts(
            source_mode="manual",
            source_language="en",
        ),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.5,
        vad_hangover_ms=800,
        vad_pre_roll_ms=320,
        provider_context=custom_config,
    )

    with pytest.raises(CustomSTTConfigurationError, match="turn_detection=null"):
        build_peer_stt_provider_request(capture, gpu_device_id="auto")


def test_local_asr_factory_binds_stt_event_ingress_observer() -> None:
    inner = SharedSTTProviderFactory(
        secrets=object(),
        clock=FakeClock(),
        reset_deadline_s=300.0,
        gpu_model_path=Path("gpu.gguf"),
    )
    factory = LocalASRProviderRuntimeFactory(
        provider_factory=inner,
        provisioning=object(),
        clock=FakeClock(),
    )
    observer = object()

    factory.bind_stt_event_ingress_observer(observer)

    assert inner.event_ingress_observer is observer


def _retention_settings(provider_id: str) -> AudioSegmentSettingsSnapshot:
    return AudioSegmentSettingsSnapshot(
        provider_id=provider_id,
        provider_signature=(provider_id,),
        runtime_signature=(provider_id,),
        source_mode="desktop",
        source_language="en",
        expected_languages=("en",),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.4,
        vad_hangover_ms=800,
        vad_pre_roll_ms=500,
    )


def _retention_owned_events(
    settings: AudioSegmentSettingsSnapshot,
    *,
    content_samples: int,
    prefix_samples: int = 0,
) -> tuple[object, object]:
    segment_id = uuid4()
    total = prefix_samples + content_samples
    capture = AudioCaptureSpan(
        capture_epoch=1,
        callback_sequence=1,
        source_sample_rate_hz=16000,
        source_start_sample=prefix_samples,
        source_end_sample=total,
        source_start_monotonic_s=prefix_samples / 16000,
        source_end_monotonic_s=total / 16000,
        normalized_sample_rate_hz=16000,
        normalized_start_sample=prefix_samples,
        normalized_end_sample=total,
    )
    ledger = PeerAudioSegmentLedger(activation_generation=1, settings=settings)
    start = ledger.observe_vad_event(
        SpeechStart(
            segment_id,
            np.ones(prefix_samples, dtype=np.float32),
            np.ones(content_samples, dtype=np.float32),
            chunk_capture=(capture,),
        ),
        now_monotonic_s=0.0,
    )
    end = ledger.observe_vad_event(
        SpeechEnd(segment_id, trailing_silence_ms=0, reason="delivery_deadline"),
        now_monotonic_s=content_samples / 16000,
    )
    return start, end


class _TerminalBatchSession:
    def __init__(self) -> None:
        self.events = STTProviderEventBuffer()

    async def begin_turn(self, request) -> None:
        self.identity = request.identity

    async def send_turn_audio(self, identity, pcm16le, **_kwargs) -> None:
        assert identity == self.identity

    async def seal_turn(self, identity, **_kwargs) -> None:
        from puripuly_heart.core.stt.backend import STTProviderTurnTerminal

        self.events.put(
            STTProviderTurnTerminal(
                identity=identity,
                outcome="final",
                text="accepted",
                text_authority="authoritative",
            )
        )

    async def abort_turn(self, _identity, *, reason) -> None:
        _ = reason

    async def turn_events(self):
        async for event in self.events.events():
            yield event

    async def stop(self) -> None:
        return None

    async def close(self) -> None:
        self.events.close()


def test_self_retention_profile_uses_exact_selected_sample_ceiling() -> None:
    profile = _recognition_retention_profile(
        SimpleNamespace(
            channel="self",
            provider="local_qwen_gpu",
            provider_options={},
            sample_rate_hz=16_000,
        ),
        _retention_settings("local_qwen_gpu"),
    )

    assert profile.max_retained_samples == 2_880_000
    assert profile.max_retained_bytes == 11_520_000


@pytest.mark.parametrize(
    ("vad_pre_roll_ms", "expected_samples"),
    [(500, 1_040_000), (100, 1_008_000)],
)
def test_listen_retention_profile_covers_six_seconds_and_largest_prefix(
    vad_pre_roll_ms: int,
    expected_samples: int,
) -> None:
    settings = replace(
        _retention_settings("local_qwen_gpu"),
        vad_pre_roll_ms=vad_pre_roll_ms,
    )
    profile = _recognition_retention_profile(
        SimpleNamespace(
            channel="peer",
            provider="local_qwen_gpu",
            provider_options={},
            sample_rate_hz=16_000,
        ),
        settings,
    )

    assert profile.max_retained_samples == expected_samples
    assert profile.max_retained_bytes == expected_samples * 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("channel", "content_samples"),
    [
        ("peer", 6 * 16000),
        ("self", 7 * 16000),
    ],
)
async def test_production_retention_profiles_accept_listen_boundary_and_long_self_like(
    channel: str,
    content_samples: int,
) -> None:
    settings = _retention_settings("local_qwen_gpu")
    config = SimpleNamespace(
        channel=channel,
        provider="local_qwen_gpu",
        provider_options={},
        sample_rate_hz=16000,
    )
    session = _TerminalBatchSession()
    emitted: list[object] = []
    engine = ScopedRecognitionEngine(
        channel=channel,
        session_factory=lambda _settings, _epoch: asyncio.sleep(0, result=session),
        event_sink=emitted.append,
        retention_profile_resolver=lambda snapshot: _recognition_retention_profile(
            config,
            snapshot,
        ),
    )
    start, end = _retention_owned_events(
        settings,
        content_samples=content_samples,
        prefix_samples=4800 if channel == "peer" else 0,
    )

    await engine.handle_owned_vad_event(start)
    assert engine.retention_snapshot.retained_samples == content_samples + (
        4800 if channel == "peer" else 0
    )
    await engine.handle_owned_vad_event(end)

    terminal = emitted[-1]
    assert terminal.outcome == "final"
    assert terminal.text == "accepted"
    await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["custom", "custom_offline"])
async def test_production_profile_accounts_real_offline_custom_buffer_for_both_selectors(
    provider_id: str,
) -> None:
    settings = _retention_settings(provider_id)
    config = SimpleNamespace(
        channel="peer",
        provider=provider_id,
        provider_options={"mode": "offline"},
        sample_rate_hz=16000,
    )
    sessions: list[_OfflineOpenAITranscriptionSession] = []

    async def open_session(_settings, epoch_id):
        session = _OfflineOpenAITranscriptionSession(
            endpoint="https://example.invalid/v1/audio/transcriptions",
            model="m",
            api_key="",
            source_language="en",
            sample_rate_hz=16000,
            http_client_factory=lambda **_kwargs: SimpleNamespace(aclose=lambda: asyncio.sleep(0)),
            projection=STTSessionProjection(mode="scoped", provider_epoch_id=epoch_id),
        )
        await session.start()
        sessions.append(session)
        return session

    engine = ScopedRecognitionEngine(
        channel="peer",
        session_factory=open_session,
        retention_profile_resolver=lambda snapshot: _recognition_retention_profile(
            config,
            snapshot,
        ),
    )
    start, _end = _retention_owned_events(settings, content_samples=1600)

    await engine.handle_owned_vad_event(start)
    session = sessions[0]

    assert len(session._scoped_buffer) == 3200
    assert engine.retention_snapshot.retained_samples == 1600
    assert engine.retention_snapshot.retained_bytes == len(session._scoped_buffer)
    await engine.abort()
    await engine.close()


def test_custom_realtime_profile_releases_after_completed_write() -> None:
    profile = _recognition_retention_profile(
        SimpleNamespace(
            channel="peer",
            provider="custom",
            provider_options={"mode": "realtime"},
            sample_rate_hz=16000,
        ),
        _retention_settings("custom"),
    )

    assert profile.release_after_write is True
    assert profile.retained_bytes_per_sample == 2


def _gpu_build_request(channel: str, *, warmup: bool) -> ProviderRuntimeBuildRequest:
    return ProviderRuntimeBuildRequest(
        config=_resolved_config(channel, "local_qwen_gpu"),
        gpu_device_id=DEVICE.device_id,
        warmup=warmup,
        provider_signature=("local_qwen_gpu",),
        runtime_signature=("local_qwen_gpu",),
        recognition_projection="scoped",
    )


def _production_gpu_owner(
    tmp_path: Path,
    client: FakeGpuWorkerClient,
) -> tuple[LocalASRProviderRuntimeOwner, FakeGpuWorkerFactory, asyncio.Queue, asyncio.Queue]:
    worker_factory = FakeGpuWorkerFactory([FakeGpuWorkerClient(), client])
    self_events = asyncio.Queue()
    peer_events = asyncio.Queue()
    owner = LocalASRProviderRuntimeOwner(
        provider_factory=SharedSTTProviderFactory(
            secrets=object(),
            clock=FakeClock(),
            reset_deadline_s=300.0,
            gpu_model_path=tmp_path / "model.gguf",
        ),
        gpu_runtime_factory=lambda sink: SharedGpuASRRuntime(
            process_factory=worker_factory,
            diagnostic_sink=sink,
        ),
        provisioning=FakeProvisioningPort(),
        self_event_handler=self_events.put,
        peer_event_handler=peer_events.put,
    )
    return owner, worker_factory, self_events, peer_events


async def _recognize_gpu_speech(
    owner,
    channel,
    events,
    *,
    settings: AudioSegmentSettingsSnapshot | None = None,
) -> object:
    start, end = _retention_owned_events(
        settings or _retention_settings("local_qwen_gpu"),
        content_samples=1600,
    )
    await owner.handle_owned_vad_event(channel, start)
    await owner.handle_owned_vad_event(channel, end)
    return await asyncio.wait_for(events.get(), timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_active", [False, True])
async def test_production_self_ingress_prepares_gpu_before_first_speech(
    tmp_path: Path,
    peer_active: bool,
) -> None:
    client = FakeGpuWorkerClient()
    owner, workers, self_events, peer_events = _production_gpu_owner(tmp_path, client)
    adapter = SelfCaptureProviderAdapter(owner, None)
    config = SelfCaptureSessionConfig(
        provider_id="local_qwen_gpu",
        provider_signature=("local_qwen_gpu",),
        capture_signature=("capture",),
        runtime_signature=("local_qwen_gpu",),
        target_sample_rate_hz=16000,
        local_gpu=True,
    )
    try:
        await owner.start()
        if peer_active:
            peer_result = await owner.replace_provider(
                _gpu_build_request("peer", warmup=True),
                start=True,
            )
            assert peer_result.status == "applied"
            assert owner.snapshot.gpu.active_channels == frozenset({"peer"})
            assert owner.snapshot.gpu.model_resident
        else:
            assert owner.snapshot.gpu.active_channels == frozenset()
            assert not owner.snapshot.gpu.model_resident
        assert not adapter.is_ready(config)
        result = await owner.replace_provider(
            _gpu_build_request("self", warmup=True),
            start=False,
        )
        assert result.status == "applied"
        assert adapter.is_ready(config)
        await adapter.start_ingress()
        expected_channels = frozenset({"self", "peer"} if peer_active else {"self"})
        assert owner.snapshot.gpu.active_channels == expected_channels
        assert owner.snapshot.gpu.phase == "ready"
        assert owner.snapshot.gpu.model_resident
        assert owner.snapshot.gpu.worker_pid == client.pid
        assert workers.modes == ["discovery", "persistent"]
        assert len(client.activate_calls) == 1
        assert client.transcribe_calls == []
        assert self_events.empty()
        assert peer_events.empty()
        await adapter.warmup()
        assert len(client.activate_calls) == 1
        terminal = await _recognize_gpu_speech(owner, "self", self_events)
        assert terminal.outcome == "final"
        assert terminal.text == "self-1"
        assert len(client.activate_calls) == 1
        await owner.release_channel("self", mode="abort")
        if peer_active:
            assert owner.snapshot.gpu.active_channels == frozenset({"peer"})
            assert owner.snapshot.gpu.worker_pid == client.pid
            assert client.close_calls == 0
            peer_terminal = await _recognize_gpu_speech(owner, "peer", peer_events)
            assert peer_terminal.outcome == "final"
            assert peer_terminal.text == "peer-2"
        else:
            assert owner.snapshot.gpu.active_channels == frozenset()
            assert client.close_calls == 1
    finally:
        await owner.close()
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_production_peer_without_warmup_stays_lazy_until_requested(
    tmp_path: Path,
) -> None:
    client = FakeGpuWorkerClient()
    owner, workers, _self_events, _peer_events = _production_gpu_owner(tmp_path, client)
    try:
        await owner.start()
        result = await owner.replace_provider(
            _gpu_build_request("peer", warmup=False),
            start=True,
        )
        assert result.status == "applied"
        assert workers.modes == ["discovery"]
        assert owner.snapshot.gpu.active_channels == frozenset()
        await owner.warmup_channel("peer")
        assert workers.modes == ["discovery", "persistent"]
        assert owner.snapshot.gpu.active_channels == frozenset({"peer"})
        assert owner.snapshot.gpu.model_resident
        assert client.transcribe_calls == []
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_production_failed_gpu_preparation_discards_backend_resources(
    tmp_path: Path,
) -> None:
    client = FakeGpuWorkerClient(activation_error=GpuWorkerRequestError("model_missing"))
    owner, workers, self_events, _peer_events = _production_gpu_owner(tmp_path, client)
    try:
        await owner.start()
        result = await owner.replace_provider(
            _gpu_build_request("self", warmup=True),
            start=False,
        )
        assert result.status == "failed"
        assert result.failure_code == "model_missing"
        assert result.failure_stage == "provider_warmup"
        assert result.failure_type == "GpuASRManualRetryRequired"
        assert owner.current_provider("self") is None
        assert owner.snapshot.gpu.active_channels == frozenset()
        assert not owner.snapshot.gpu.model_resident
        assert workers.modes == ["discovery", "persistent"]
        assert client.close_calls == 1
        assert client.transcribe_calls == []
        assert self_events.empty()
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_production_cancelled_gpu_preparation_discards_backend_resources(
    tmp_path: Path,
) -> None:
    client = FakeGpuWorkerClient(activation_gate=asyncio.Event())
    owner, workers, self_events, _peer_events = _production_gpu_owner(tmp_path, client)
    await owner.start()
    building = asyncio.create_task(
        owner.replace_provider(_gpu_build_request("self", warmup=True), start=False)
    )
    try:
        await asyncio.wait_for(client.activation_started.wait(), timeout=1)
        building.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(building, timeout=1)
        assert owner.current_provider("self") is None
        assert owner.snapshot.gpu.active_channels == frozenset()
        assert not owner.snapshot.gpu.model_resident
        assert workers.modes == ["discovery", "persistent"]
        assert client.close_calls == 1
        assert client.transcribe_calls == []
        assert self_events.empty()
    finally:
        client.activation_gate.set()
        await asyncio.gather(building, return_exceptions=True)
        await owner.close()


@pytest.mark.asyncio
async def test_scoped_gpu_warmup_cannot_revive_closing_or_closed_engine(
    tmp_path: Path,
) -> None:
    client = FakeGpuWorkerClient(activation_gate=asyncio.Event())
    owner, _workers, self_events, _peer_events = _production_gpu_owner(tmp_path, client)
    await owner.start()
    result = await owner.replace_provider(
        _gpu_build_request("self", warmup=False),
        start=False,
    )
    assert result.status == "applied"
    engine = owner.current_provider("self")
    preparing = asyncio.create_task(engine.warmup())
    closing = None
    try:
        await asyncio.wait_for(client.activation_started.wait(), timeout=1)
        closing = asyncio.create_task(engine.close_backend())
        async with asyncio.timeout(1):
            while engine.is_live:
                await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="engine is closed"):
            await engine.warmup()
        client.activation_gate.set()
        with pytest.raises(RuntimeError, match="engine is closed"):
            await asyncio.wait_for(preparing, timeout=1)
        await asyncio.wait_for(closing, timeout=1)
        assert owner.snapshot.gpu.active_channels == frozenset()
        assert client.close_calls == 1
        assert client.transcribe_calls == []
        assert self_events.empty()
        with pytest.raises(RuntimeError, match="engine is closed"):
            await engine.warmup()
    finally:
        client.activation_gate.set()
        await asyncio.gather(
            preparing,
            *(() if closing is None else (closing,)),
            return_exceptions=True,
        )
        await owner.close()


@pytest.mark.asyncio
async def test_production_same_channel_replacement_survives_retired_backend_close(
    tmp_path: Path,
) -> None:
    client = FakeGpuWorkerClient()
    owner, workers, self_events, _peer_events = _production_gpu_owner(tmp_path, client)
    try:
        await owner.start()
        request = _gpu_build_request("self", warmup=True)
        assert (await owner.replace_provider(request, start=True)).status == "applied"
        retired = owner.current_provider("self")
        replacement = replace(request, runtime_signature=("replacement_input",))
        assert (await owner.replace_provider(replacement, start=True)).status == "applied"
        current = owner.current_provider("self")
        assert current is not retired
        assert owner.snapshot.gpu.active_channels == frozenset({"self"})
        assert owner.snapshot.gpu.model_resident
        assert client.close_calls == 0
        assert workers.modes == ["discovery", "persistent"]
        assert len(client.activate_calls) == 1
        terminal = await _recognize_gpu_speech(
            owner,
            "self",
            self_events,
            settings=replace(
                _retention_settings("local_qwen_gpu"),
                runtime_signature=replacement.runtime_signature,
            ),
        )
        assert terminal.outcome == "final"
        assert terminal.text == "self-1"
        assert not retired.is_live
        assert owner.snapshot.gpu.active_channels == frozenset({"self"})
        assert client.close_calls == 0
        await owner.release_channel("self", mode="abort")
        assert owner.snapshot.gpu.active_channels == frozenset()
        assert client.close_calls == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_production_cancelled_same_channel_candidate_preserves_live_preparation(
    tmp_path: Path,
) -> None:
    client = FakeGpuWorkerClient(activation_gate=asyncio.Event())
    owner, workers, self_events, _peer_events = _production_gpu_owner(tmp_path, client)
    await owner.start()
    request = _gpu_build_request("self", warmup=False)
    assert (await owner.replace_provider(request, start=True)).status == "applied"
    current = owner.current_provider("self")
    warming = asyncio.create_task(owner.warmup_channel("self"))
    candidate = None
    try:
        await asyncio.wait_for(client.activation_started.wait(), timeout=1)
        candidate = asyncio.create_task(
            owner.replace_provider(replace(request, warmup=True), start=True)
        )
        async with asyncio.timeout(1):
            while len(owner._gpu_runtime._active_channels.get("self", ())) != 2:
                await asyncio.sleep(0)
        candidate.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(candidate, timeout=1)
        assert owner.current_provider("self") is current
        assert current.is_live
        assert owner.snapshot.gpu.active_channels == frozenset({"self"})
        assert client.close_calls == 0
        assert not warming.done()
        client.activation_gate.set()
        await asyncio.wait_for(warming, timeout=1)
        assert owner.snapshot.gpu.model_resident
        assert workers.modes == ["discovery", "persistent"]
        terminal = await _recognize_gpu_speech(owner, "self", self_events)
        assert terminal.outcome == "final"
        assert terminal.text == "self-1"
    finally:
        client.activation_gate.set()
        await asyncio.gather(
            warming,
            *(() if candidate is None else (candidate,)),
            return_exceptions=True,
        )
        await owner.close()
    assert client.close_calls == 1
