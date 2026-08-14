from __future__ import annotations

import asyncio

import pytest
from puripuly_heart.app.adapters.peer_capture_admission import (
    PeerCaptureAdmissionAdapter,
)

from puripuly_heart.app.wiring import create_peer_capture_admission_adapter
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmissionStatus,
    PeerCaptureLanguageFacts,
    PeerCaptureSessionConfig,
    PeerCaptureTargetIntent,
)


def _config(*, local_provider: bool) -> PeerCaptureSessionConfig:
    return PeerCaptureSessionConfig(
        provider_id="local_qwen" if local_provider else "soniox",
        provider_signature=("provider",),
        runtime_signature=("runtime",),
        capture_signature=("capture",),
        capture_target=PeerCaptureTargetIntent(kind="default_output_device"),
        language=PeerCaptureLanguageFacts(
            source_mode="manual",
            source_language="en",
        ),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.6,
        vad_hangover_ms=900,
        vad_pre_roll_ms=500,
        local_provider=local_provider,
    )


@pytest.mark.asyncio
async def test_adapter_rejects_unavailable_runtime_without_readiness_call() -> None:
    readiness_calls: list[bool] = []

    async def ensure_local_ready() -> bool:
        readiness_calls.append(True)
        return True

    result = await PeerCaptureAdmissionAdapter(
        runtime_available=lambda: False,
        ensure_local_ready=ensure_local_ready,
    ).admit(_config(local_provider=True))

    assert result.status is PeerCaptureAdmissionStatus.REJECTED
    assert result.reason == "runtime_unavailable"
    assert result.retain_intent is False
    assert readiness_calls == []


@pytest.mark.asyncio
async def test_adapter_admits_non_local_provider_without_readiness_call() -> None:
    readiness_calls: list[bool] = []

    async def ensure_local_ready() -> bool:
        readiness_calls.append(True)
        return False

    result = await PeerCaptureAdmissionAdapter(
        runtime_available=lambda: True,
        ensure_local_ready=ensure_local_ready,
    ).admit(_config(local_provider=False))

    assert result.status is PeerCaptureAdmissionStatus.ADMITTED
    assert result.reason is None
    assert readiness_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ready", "expected_status", "expected_reason", "retain_intent"),
    [
        (True, PeerCaptureAdmissionStatus.ADMITTED, None, False),
        (False, PeerCaptureAdmissionStatus.PENDING, "provider_unavailable", True),
    ],
)
async def test_adapter_maps_local_readiness_to_admission(
    ready: bool,
    expected_status: PeerCaptureAdmissionStatus,
    expected_reason: str | None,
    retain_intent: bool,
) -> None:
    calls = 0

    async def ensure_local_ready() -> bool:
        nonlocal calls
        calls += 1
        return ready

    result = await PeerCaptureAdmissionAdapter(
        runtime_available=lambda: True,
        ensure_local_ready=ensure_local_ready,
    ).admit(_config(local_provider=True))

    assert result.status is expected_status
    assert result.reason == expected_reason
    assert result.retain_intent is retain_intent
    assert calls == 1


@pytest.mark.asyncio
async def test_adapter_propagates_readiness_exception_and_cancellation() -> None:
    async def fail_readiness() -> bool:
        raise RuntimeError("readiness failed")

    adapter = PeerCaptureAdmissionAdapter(
        runtime_available=lambda: True,
        ensure_local_ready=fail_readiness,
    )
    with pytest.raises(RuntimeError, match="readiness failed"):
        await adapter.admit(_config(local_provider=True))

    async def cancel_readiness() -> bool:
        raise asyncio.CancelledError

    adapter = PeerCaptureAdmissionAdapter(
        runtime_available=lambda: True,
        ensure_local_ready=cancel_readiness,
    )
    with pytest.raises(asyncio.CancelledError):
        await adapter.admit(_config(local_provider=True))


def test_wiring_factory_composes_internal_peer_admission_adapter() -> None:
    async def ensure_local_ready() -> bool:
        return True

    adapter = create_peer_capture_admission_adapter(
        runtime_available=lambda: True,
        ensure_local_ready=ensure_local_ready,
    )

    assert isinstance(adapter, PeerCaptureAdmissionAdapter)
    assert adapter.ensure_local_ready is ensure_local_ready
