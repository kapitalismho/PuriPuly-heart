from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import puripuly_heart.app.wiring_local_asr_provider_runtime as runtime_wiring
import pytest
from puripuly_heart.app.wiring_local_asr_provider_runtime import (
    LocalASRProviderRuntimeFactory,
    SharedSTTProviderFactory,
    _deferred_age_rotation_enabled,
    _recognition_retention_profile,
    _recognition_watchdogs,
)
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest

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
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.stt.backend import STTSessionProjection
from puripuly_heart.core.stt.custom import (
    CustomSTTConfigurationError,
    normalize_custom_stt_extra,
    validate_peer_custom_stt_configuration,
)
from puripuly_heart.core.stt.scoped_engine import (
    PermanentSTTScopedSessionError,
    ScopedRecognitionEngine,
)
from puripuly_heart.core.stt.scoped_event_buffer import STTProviderEventBuffer
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.providers.stt.custom import _OfflineOpenAITranscriptionSession


class _RuntimeLogging:
    def __init__(self) -> None:
        self.basic: list[tuple[str, int]] = []
        self.detailed: list[str] = []

    def emit_basic(self, message: str, *, level: int = 20) -> None:
        self.basic.append((message, level))

    def emit_detailed(self, message: str, **_kwargs: object) -> bool:
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
    assert calls[0][1] == {
        "secrets": factory.secrets,
        "diagnostics_enabled": None,
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


@pytest.mark.parametrize("provider", tuple(STTProviderName))
def test_scoped_watchdog_policy_covers_every_configured_selector(
    provider: STTProviderName,
) -> None:
    resolved = _recognition_watchdogs(SimpleNamespace(provider=provider.value, drain_timeout_s=2.0))
    local = {
        STTProviderName.LOCAL_CPU_AUTO,
        STTProviderName.LOCAL_PARAKEET_V3,
        STTProviderName.LOCAL_PARAKEET_JAPANESE,
        STTProviderName.LOCAL_QWEN,
        STTProviderName.LOCAL_QWEN_GPU,
    }
    expected_final = {
        STTProviderName.DEEPGRAM: 9.0,
        STTProviderName.GEMINI_TRANSCRIBE: 2.0,
        STTProviderName.QWEN_AUDIO: 5.0,
        STTProviderName.CUSTOM_OFFLINE: 50.0,
    }.get(provider, 30.0 if provider in local else 20.0)

    assert resolved.readiness_timeout_s == (60.0 if provider in local else 30.0)
    assert resolved.write_timeout_s == 5.0
    assert resolved.final_timeout_s == expected_final
    assert resolved.drain_timeout_s == 2.0


@pytest.mark.parametrize("provider", tuple(STTProviderName))
def test_deferred_age_rotation_eligibility_preserves_non_target_routes(
    provider: STTProviderName,
) -> None:
    target_routes = {
        STTProviderName.SONIOX,
        STTProviderName.DEEPGRAM,
        STTProviderName.GEMINI_TRANSCRIBE,
        STTProviderName.ELEVENLABS_SCRIBE,
        STTProviderName.ROLLING_FREE,
    }
    assert _deferred_age_rotation_enabled(provider.value) is (provider in target_routes)


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
