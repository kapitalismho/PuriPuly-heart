from __future__ import annotations

from pathlib import Path

from puripuly_heart.app.adapters.peer_capture_vad import PeerCaptureVadAdapter
from puripuly_heart.app.wiring import create_peer_capture_vad_adapter
from puripuly_heart.core.peer_capture import (
    PeerCaptureLanguageFacts,
    PeerCaptureSessionConfig,
    PeerCaptureTargetIntent,
)
from puripuly_heart.core.vad.smart_turn import SmartTurnExperimentConfig


def _config() -> PeerCaptureSessionConfig:
    return PeerCaptureSessionConfig(
        provider_id="soniox",
        provider_signature=("provider",),
        runtime_signature=("runtime",),
        capture_signature=("capture",),
        capture_target=PeerCaptureTargetIntent(kind="default_output_device"),
        language=PeerCaptureLanguageFacts(
            source_mode="manual",
            source_language="en",
        ),
        target_sample_rate_hz=24000,
        vad_speech_threshold=0.72,
        vad_hangover_ms=950,
        vad_pre_roll_ms=420,
    )


def test_adapter_constructs_engine_and_exact_peer_gating_policy() -> None:
    model_path = Path("peer-vad.onnx")
    model_calls: list[bool] = []
    engine_calls: list[dict[str, object]] = []
    gating_calls: list[dict[str, object]] = []
    logs: list[str] = []
    detailed = [False]
    engine = object()
    vad = object()

    def model_path_resolver() -> Path:
        model_calls.append(True)
        return model_path

    def engine_factory(**kwargs: object) -> object:
        engine_calls.append(dict(kwargs))
        return engine

    def gating_factory(**kwargs: object) -> object:
        gating_calls.append(dict(kwargs))
        return vad

    adapter = PeerCaptureVadAdapter(
        model_path_resolver=model_path_resolver,
        engine_factory=engine_factory,
        gating_factory=gating_factory,
        log_detailed=logs.append,
        diagnostics_enabled=lambda: detailed[0],
    )

    result = adapter(_config())

    assert result is vad
    assert model_calls == [True]
    assert engine_calls == [{"model_path": model_path}]
    assert gating_calls == [
        {
            "engine": engine,
            "sample_rate_hz": 24000,
            "ring_buffer_ms": 420,
            "speech_threshold": 0.72,
            "hangover_ms": 950,
            "diagnostic_event_callback": gating_calls[0]["diagnostic_event_callback"],
            "diagnostics_enabled": gating_calls[0]["diagnostics_enabled"],
            "diagnostic_label": "peer",
        }
    ]
    callback = gating_calls[0]["diagnostic_event_callback"]
    assert callable(callback)
    callback("[AudioDiag][VAD][peer] probe")
    assert logs == ["[AudioDiag][VAD][peer] probe"]
    diagnostics_enabled = gating_calls[0]["diagnostics_enabled"]
    assert callable(diagnostics_enabled)
    assert diagnostics_enabled() is False
    detailed[0] = True
    assert diagnostics_enabled() is True


def test_wiring_factory_composes_internal_peer_vad_adapter() -> None:
    adapter = create_peer_capture_vad_adapter(
        log_detailed=lambda _message: None,
        diagnostics_enabled=lambda: False,
    )

    assert isinstance(adapter, PeerCaptureVadAdapter)
    assert adapter.model_path_resolver.__name__ == "ensure_silero_vad_onnx"
    assert adapter.engine_factory.__name__ == "SileroVadOnnx"
    assert adapter.gating_factory.__name__ == "create_peer_vad_gating"


def test_active_smart_turn_replaces_peer_vad_hangover_with_hard_boundary() -> None:
    gating_calls: list[dict[str, object]] = []
    adapter = PeerCaptureVadAdapter(
        model_path_resolver=lambda: Path("peer-vad.onnx"),
        engine_factory=lambda **_kwargs: object(),
        gating_factory=lambda **kwargs: gating_calls.append(dict(kwargs)) or object(),
        log_detailed=lambda _message: None,
        diagnostics_enabled=lambda: False,
        smart_turn_config_provider=lambda: SmartTurnExperimentConfig(stage="active"),
    )

    adapter(_config())

    assert gating_calls[0]["hangover_ms"] == 800
