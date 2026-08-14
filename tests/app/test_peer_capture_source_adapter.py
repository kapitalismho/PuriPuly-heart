from __future__ import annotations

from types import SimpleNamespace

import pytest
from puripuly_heart.app.adapters.peer_capture_source import PeerCaptureSourceAdapter

from puripuly_heart.app.wiring import create_peer_capture_source_adapter
from puripuly_heart.core.peer_capture import (
    PeerCaptureLanguageFacts,
    PeerCaptureResolvedTarget,
    PeerCaptureSessionConfig,
    PeerCaptureTargetIntent,
)


def _config(target: PeerCaptureTargetIntent) -> PeerCaptureSessionConfig:
    return PeerCaptureSessionConfig(
        provider_id="soniox",
        provider_signature=("provider",),
        runtime_signature=("runtime",),
        capture_signature=("capture",),
        capture_target=target,
        language=PeerCaptureLanguageFacts(
            source_mode="manual",
            source_language="en",
        ),
        target_sample_rate_hz=24000,
        vad_speech_threshold=0.6,
        vad_hangover_ms=900,
        vad_pre_roll_ms=500,
        output_device="Fallback Speakers",
    )


@pytest.mark.parametrize(
    ("target", "expected_device"),
    [
        (
            PeerCaptureTargetIntent(
                kind="named_output_device",
                device_name="Named Speakers",
            ),
            "Named Speakers",
        ),
        (
            PeerCaptureTargetIntent(kind="default_output_device"),
            "Fallback Speakers",
        ),
    ],
)
def test_adapter_constructs_desktop_pipeline_with_selected_device_and_diagnostics(
    target: PeerCaptureTargetIntent,
    expected_device: str,
) -> None:
    logs: list[str] = []
    loopback_calls: list[str] = []
    raw_source = SimpleNamespace(
        resolved_device_name="Resolved Speakers [Loopback]",
        resolved_device_index=7,
        resolved_channels=2,
        actual_sample_rate_hz=48000,
        used_default_fallback=True,
    )

    def detailed_enabled() -> bool:
        return True

    wrapped_source = object()

    adapter = PeerCaptureSourceAdapter(
        loopback_source_factory=lambda *, device_name: (
            loopback_calls.append(device_name) or raw_source
        ),
        process_source_factory=lambda **kwargs: pytest.fail(
            f"process source constructed: {kwargs}"
        ),
        process_watcher_factory=object,
        pipeline_factory=lambda **kwargs: kwargs,
        log_detailed=logs.append,
        wrap_source=lambda source: wrapped_source if source is raw_source else None,
        is_detailed_enabled=detailed_enabled,
    )

    pipeline = adapter(_config(target), PeerCaptureResolvedTarget(intent=target))

    assert loopback_calls == [expected_device]
    assert pipeline["source"] is wrapped_source
    assert pipeline["target_sample_rate_hz"] == 24000
    assert pipeline["is_detailed_enabled"] is detailed_enabled
    assert callable(pipeline["log_detailed"])
    assert "requested_device=" in logs[0]
    assert "resolved_device_name='Resolved Speakers [Loopback]'" in logs[0]
    assert "used_default_fallback=True" in logs[0]


def test_adapter_constructs_strict_process_pipeline_without_device_fallback() -> None:
    target = PeerCaptureTargetIntent(
        kind="process",
        process_kind="discord",
        discord_channel="canary",
    )
    identity = object()
    watcher = object()
    raw_source = object()
    process_calls: list[dict[str, object]] = []
    logs: list[str] = []

    def process_source_factory(**kwargs: object) -> object:
        process_calls.append(dict(kwargs))
        return raw_source

    adapter = PeerCaptureSourceAdapter(
        loopback_source_factory=lambda **kwargs: pytest.fail(
            f"device fallback constructed: {kwargs}"
        ),
        process_source_factory=process_source_factory,
        process_watcher_factory=lambda: watcher,
        pipeline_factory=lambda **kwargs: kwargs,
        log_detailed=logs.append,
        wrap_source=lambda source: ("wrapped", source),
        is_detailed_enabled=lambda: False,
    )

    pipeline = adapter(
        _config(target),
        PeerCaptureResolvedTarget(
            intent=target,
            capture_descriptor=SimpleNamespace(identity=identity),
        ),
    )

    assert process_calls == [{"identity": identity, "watcher": watcher}]
    assert pipeline["source"] == ("wrapped", raw_source)
    assert pipeline["target_sample_rate_hz"] == 24000
    assert logs == ["[AudioDiag][ProcessCapture][peer] target_kind=discord capture=process"]


def test_adapter_rejects_resolved_process_without_identity() -> None:
    target = PeerCaptureTargetIntent(
        kind="process",
        process_kind="vrchat",
        executable_identity=r"c:\vrchat\vrchat.exe",
    )
    adapter = PeerCaptureSourceAdapter(
        loopback_source_factory=lambda **kwargs: pytest.fail(
            f"device fallback constructed: {kwargs}"
        ),
        process_source_factory=lambda **kwargs: pytest.fail(
            f"process source constructed: {kwargs}"
        ),
        process_watcher_factory=object,
        pipeline_factory=lambda **kwargs: kwargs,
        log_detailed=lambda _message: None,
        wrap_source=lambda source: source,
        is_detailed_enabled=lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="resolved process capture requires a process identity",
    ):
        adapter(
            _config(target),
            PeerCaptureResolvedTarget(
                intent=target,
                capture_descriptor=SimpleNamespace(identity=None),
            ),
        )


def test_wiring_factory_composes_internal_peer_capture_source_adapter() -> None:
    adapter = create_peer_capture_source_adapter(
        log_detailed=lambda _message: None,
        wrap_source=lambda source: source,
        is_detailed_enabled=lambda: False,
    )

    assert isinstance(adapter, PeerCaptureSourceAdapter)
    assert adapter.loopback_source_factory.__name__ == "DesktopLoopbackAudioSource"
    assert adapter.process_source_factory.__name__ == "ProcessAudioCaptureSource"
    assert adapter.process_watcher_factory.__name__ == "PsutilProcessIdentityWatcher"
    assert adapter.pipeline_factory.__name__ == "DesktopPeerPipeline"
