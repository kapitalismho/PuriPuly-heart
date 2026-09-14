from __future__ import annotations

from pathlib import Path

from puripuly_heart.app.adapters.self_capture_vad import SelfCaptureVadAdapter

from puripuly_heart.config.resolved import vad_exit_threshold
from puripuly_heart.core.self_capture import SelfCaptureSessionConfig


def _config() -> SelfCaptureSessionConfig:
    return SelfCaptureSessionConfig(
        provider_id="soniox",
        provider_signature=("provider",),
        runtime_signature=("runtime",),
        capture_signature=("capture",),
        target_sample_rate_hz=24000,
        ring_buffer_ms=1800,
        vad_speech_threshold=0.67,
        vad_hangover_ms=875,
    )


def test_adapter_constructs_engine_and_exact_self_gating_policy() -> None:
    model_path = Path("self-vad.onnx")
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

    adapter = SelfCaptureVadAdapter(
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
            "ring_buffer_ms": 1800,
            "speech_threshold": 0.67,
            "continuation_threshold": vad_exit_threshold(0.67),
            "hangover_ms": 875,
            "diagnostic_event_callback": gating_calls[0]["diagnostic_event_callback"],
            "diagnostics_enabled": gating_calls[0]["diagnostics_enabled"],
            "diagnostic_label": "self",
        }
    ]
    callback = gating_calls[0]["diagnostic_event_callback"]
    assert callable(callback)
    callback("[AudioDiag][VAD][self] probe")
    assert logs == ["[AudioDiag][VAD][self] probe"]
    diagnostics_enabled = gating_calls[0]["diagnostics_enabled"]
    assert callable(diagnostics_enabled)
    assert diagnostics_enabled() is False
    detailed[0] = True
    assert diagnostics_enabled() is True
