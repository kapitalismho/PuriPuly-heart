from __future__ import annotations

from pathlib import Path

import puripuly_heart.app.wiring_local_asr_provider_runtime as runtime_wiring
from puripuly_heart.app.wiring_local_asr_provider_runtime import (
    LocalASRProviderRuntimeFactory,
    ManagedSTTProviderFactory,
)
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest

from puripuly_heart.config.resolved import (
    ResolvedCredentialRequirement,
    ResolvedSTTConfig,
)
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions


async def test_managed_provider_factory_builds_only_from_immutable_request(
    monkeypatch,
) -> None:
    calls: list[tuple[ResolvedSTTConfig, dict[str, object]]] = []
    backend = object()

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

    provider = await factory.create(request, gpu_runtime=gpu_runtime)

    assert [call[0] for call in calls] == [config]
    assert calls[0][1] == {
        "secrets": factory.secrets,
        "diagnostics_enabled": None,
        "gpu_runtime": gpu_runtime,
        "gpu_model_path": Path("gpu.gguf"),
        "gpu_device_id": "vk:2",
    }
    assert provider.backend is backend
    assert provider.channel == "peer"
    assert provider.bridging_ms == 320
    assert provider._pending_session_options == options
    assert provider.event_ingress_observer is observer


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
