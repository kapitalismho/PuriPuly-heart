from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from puripuly_heart.app.adapters.self_capture_admission import (
    SelfCaptureAdmissionAdapter,
)

from puripuly_heart.app.ports.self_capture_admission import (
    SelfCaptureAdmissionEffect,
    SelfCaptureAdmissionEffectType,
    SelfCaptureAdmissionState,
)
from puripuly_heart.app.wiring import create_self_capture_admission_adapter
from puripuly_heart.core.self_capture import (
    SelfCaptureAdmissionStatus,
    SelfCaptureSessionConfig,
)


def _config(*, local_cpu: bool = False, local_gpu: bool = False) -> SelfCaptureSessionConfig:
    return SelfCaptureSessionConfig(
        provider_id="local" if local_cpu or local_gpu else "soniox",
        provider_signature=("provider",),
        runtime_signature=("runtime",),
        capture_signature=("capture",),
        target_sample_rate_hz=16000,
        local_cpu=local_cpu,
        local_gpu=local_gpu,
    )


def _state(**changes: object) -> SelfCaptureAdmissionState:
    baseline = SelfCaptureAdmissionState(
        settings_available=True,
        runtime_available=True,
        gpu_status="ready",
        local_cpu_supported=True,
        local_runtime_status="ready",
        activation_generation=7,
    )
    return replace(baseline, **changes)


def _adapter(
    *,
    states: list[SelfCaptureAdmissionState],
    validation_result: bool = True,
    effects: list[SelfCaptureAdmissionEffect] | None = None,
) -> tuple[SelfCaptureAdmissionAdapter, list[bool]]:
    validation_calls: list[bool] = []
    effect_calls = effects if effects is not None else []

    async def validate_gpu_activation() -> bool:
        validation_calls.append(True)
        return validation_result

    return (
        SelfCaptureAdmissionAdapter(
            state_provider=lambda _config: states.pop(0) if len(states) > 1 else states[0],
            validate_gpu_activation=validate_gpu_activation,
            effect_sink=effect_calls.append,
        ),
        validation_calls,
    )


@pytest.mark.asyncio
async def test_adapter_rejects_missing_settings_before_provider_work() -> None:
    effects: list[SelfCaptureAdmissionEffect] = []
    adapter, validation_calls = _adapter(
        states=[_state(settings_available=False)],
        effects=effects,
    )

    result = await adapter.admit(_config(local_gpu=True))

    assert result.status is SelfCaptureAdmissionStatus.REJECTED
    assert result.reason == "runtime_unavailable"
    assert result.retain_intent is False
    assert validation_calls == []
    assert effects == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_available", "status", "reason"),
    [
        (True, SelfCaptureAdmissionStatus.ADMITTED, None),
        (False, SelfCaptureAdmissionStatus.REJECTED, "runtime_unavailable"),
    ],
)
async def test_adapter_maps_remote_provider_runtime_availability(
    runtime_available: bool,
    status: SelfCaptureAdmissionStatus,
    reason: str | None,
) -> None:
    adapter, validation_calls = _adapter(
        states=[_state(runtime_available=runtime_available)],
    )

    result = await adapter.admit(_config())

    assert result.status is status
    assert result.reason == reason
    assert validation_calls == []


@pytest.mark.asyncio
async def test_adapter_admits_valid_gpu_without_effects() -> None:
    effects: list[SelfCaptureAdmissionEffect] = []
    adapter, validation_calls = _adapter(
        states=[_state(gpu_status="validating")],
        effects=effects,
    )

    result = await adapter.admit(_config(local_gpu=True))

    assert result.status is SelfCaptureAdmissionStatus.ADMITTED
    assert result.reason is None
    assert validation_calls == [True]
    assert effects == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gpu_status",
    ["not_installed", "invalid", "install_failed", "installing"],
)
async def test_adapter_retains_pending_gpu_intent_from_refreshed_state(
    gpu_status: str,
) -> None:
    effects: list[SelfCaptureAdmissionEffect] = []
    adapter, validation_calls = _adapter(
        states=[
            _state(gpu_status="validating"),
            _state(gpu_status=gpu_status),
        ],
        validation_result=False,
        effects=effects,
    )

    result = await adapter.admit(_config(local_gpu=True))

    assert result.status is SelfCaptureAdmissionStatus.PENDING
    assert result.reason == gpu_status
    assert result.retain_intent is True
    assert validation_calls == [True]
    assert effects == [
        SelfCaptureAdmissionEffect(
            SelfCaptureAdmissionEffectType.RETAIN_GPU_PENDING_INTENT,
            status=gpu_status,
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gpu_status", "reason"),
    [
        ("unsupported", "unsupported"),
        ("activation_failed", "activation_failed"),
        (None, "gpu_unavailable"),
    ],
)
async def test_adapter_rejects_terminal_gpu_status_without_effect(
    gpu_status: str | None,
    reason: str,
) -> None:
    effects: list[SelfCaptureAdmissionEffect] = []
    adapter, _ = _adapter(
        states=[
            _state(gpu_status="validating"),
            _state(gpu_status=gpu_status),
        ],
        validation_result=False,
        effects=effects,
    )

    result = await adapter.admit(_config(local_gpu=True))

    assert result.status is SelfCaptureAdmissionStatus.REJECTED
    assert result.reason == reason
    assert result.retain_intent is False
    assert effects == []


@pytest.mark.asyncio
async def test_adapter_rejects_unsupported_local_cpu_with_explicit_effect() -> None:
    effects: list[SelfCaptureAdmissionEffect] = []
    adapter, _ = _adapter(
        states=[_state(local_cpu_supported=False)],
        effects=effects,
    )

    result = await adapter.admit(_config(local_cpu=True))

    assert result.status is SelfCaptureAdmissionStatus.REJECTED
    assert result.reason == "language_unsupported"
    assert effects == [
        SelfCaptureAdmissionEffect(
            SelfCaptureAdmissionEffectType.REJECT_UNSUPPORTED_LANGUAGE,
        )
    ]


@pytest.mark.asyncio
async def test_adapter_retains_downloading_intent_with_generation_effect() -> None:
    effects: list[SelfCaptureAdmissionEffect] = []
    adapter, _ = _adapter(
        states=[_state(local_runtime_status="downloading", activation_generation=19)],
        effects=effects,
    )

    result = await adapter.admit(_config(local_cpu=True))

    assert result.status is SelfCaptureAdmissionStatus.PENDING
    assert result.reason == "downloading"
    assert result.retain_intent is True
    assert effects == [
        SelfCaptureAdmissionEffect(
            SelfCaptureAdmissionEffectType.RETAIN_DOWNLOAD_PENDING_INTENT,
            status="downloading",
            activation_generation=19,
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_status", ["missing", "invalid", "download_failed"])
async def test_adapter_requests_local_repair_with_status_and_generation(
    runtime_status: str,
) -> None:
    effects: list[SelfCaptureAdmissionEffect] = []
    adapter, _ = _adapter(
        states=[_state(local_runtime_status=runtime_status, activation_generation=23)],
        effects=effects,
    )

    result = await adapter.admit(_config(local_cpu=True))

    assert result.status is SelfCaptureAdmissionStatus.PENDING
    assert result.reason == runtime_status
    assert result.retain_intent is True
    assert effects == [
        SelfCaptureAdmissionEffect(
            SelfCaptureAdmissionEffectType.REQUEST_LOCAL_REPAIR,
            status=runtime_status,
            activation_generation=23,
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime_available", "status"),
    [
        (True, SelfCaptureAdmissionStatus.ADMITTED),
        (False, SelfCaptureAdmissionStatus.REJECTED),
    ],
)
async def test_adapter_maps_ready_local_cpu_runtime_availability(
    runtime_available: bool,
    status: SelfCaptureAdmissionStatus,
) -> None:
    adapter, _ = _adapter(
        states=[
            _state(
                runtime_available=runtime_available,
                local_runtime_status="ready",
            )
        ],
    )

    result = await adapter.admit(_config(local_cpu=True))

    assert result.status is status
    assert result.reason == (None if runtime_available else "runtime_unavailable")


@pytest.mark.asyncio
async def test_adapter_propagates_gpu_validation_exception_and_cancellation() -> None:
    async def fail_validation() -> bool:
        raise RuntimeError("validation failed")

    adapter = SelfCaptureAdmissionAdapter(
        state_provider=lambda _config: _state(),
        validate_gpu_activation=fail_validation,
        effect_sink=lambda _effect: None,
    )
    with pytest.raises(RuntimeError, match="validation failed"):
        await adapter.admit(_config(local_gpu=True))

    async def cancel_validation() -> bool:
        raise asyncio.CancelledError

    adapter = SelfCaptureAdmissionAdapter(
        state_provider=lambda _config: _state(),
        validate_gpu_activation=cancel_validation,
        effect_sink=lambda _effect: None,
    )
    with pytest.raises(asyncio.CancelledError):
        await adapter.admit(_config(local_gpu=True))


@pytest.mark.asyncio
async def test_adapter_propagates_effect_failure() -> None:
    def fail_effect(_effect: SelfCaptureAdmissionEffect) -> None:
        raise RuntimeError("effect failed")

    adapter = SelfCaptureAdmissionAdapter(
        state_provider=lambda _config: _state(local_cpu_supported=False),
        validate_gpu_activation=lambda: asyncio.sleep(0, result=True),
        effect_sink=fail_effect,
    )

    with pytest.raises(RuntimeError, match="effect failed"):
        await adapter.admit(_config(local_cpu=True))


def test_wiring_factory_composes_internal_self_admission_adapter() -> None:
    async def validate_gpu_activation() -> bool:
        return True

    def state_provider(_config: SelfCaptureSessionConfig) -> SelfCaptureAdmissionState:
        return _state()

    def effect_sink(_effect: SelfCaptureAdmissionEffect) -> None:
        return None

    adapter = create_self_capture_admission_adapter(
        state_provider=state_provider,
        validate_gpu_activation=validate_gpu_activation,
        effect_sink=effect_sink,
    )

    assert isinstance(adapter, SelfCaptureAdmissionAdapter)
    assert adapter.state_provider is state_provider
    assert adapter.validate_gpu_activation is validate_gpu_activation
    assert adapter.effect_sink is effect_sink
