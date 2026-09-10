from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import puripuly_heart.app.wiring_local_asr_provider_runtime as runtime_wiring
import pytest
from puripuly_heart.app.wiring_local_asr_provider_runtime import (
    LocalASRProviderRuntimeFactory,
    ManagedSTTProviderFactory,
    _recognition_watchdogs,
)
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest

from puripuly_heart.app.wiring.wiring_stt_factory import build_peer_stt_provider_request
from puripuly_heart.config.provider_values import STTProviderName
from puripuly_heart.config.resolved import (
    ResolvedCredentialRequirement,
    ResolvedSTTConfig,
)
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.peer_capture import (
    PeerCaptureLanguageFacts,
    PeerCaptureSessionConfig,
    PeerCaptureTargetIntent,
)
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.stt.custom import (
    CustomSTTConfigurationError,
    normalize_custom_stt_extra,
    validate_peer_custom_stt_configuration,
)
from puripuly_heart.core.stt.scoped_engine import (
    PermanentSTTScopedSessionError,
    ScopedRecognitionEngine,
)


async def test_managed_provider_factory_cuts_peer_to_scoped_and_preserves_self(
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

        async def open_session(self) -> LegacySession:
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
    )
    observer = object()
    factory = ManagedSTTProviderFactory(
        secrets=object(),
        clock=FakeClock(),
        reset_deadline_s=300.0,
        gpu_model_path=Path("gpu.gguf"),
        event_ingress_observer=observer,
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
        ),
        gpu_runtime=gpu_runtime,
    )

    assert [call[0] for call in calls] == [config, self_config]
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
    assert isinstance(self_provider, ManagedSTTProvider)
    assert self_provider.backend is backend
    assert self_provider.channel == "self"
    assert self_provider.bridging_ms == 500
    assert self_provider._pending_session_options == options
    assert self_provider.event_ingress_observer is observer

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
    assert [call[0] for call in calls] == [config, self_config]


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


def test_local_asr_factory_binds_stt_event_ingress_observer() -> None:
    inner = ManagedSTTProviderFactory(
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
