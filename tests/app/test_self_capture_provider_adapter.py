from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from puripuly_heart.app.adapters.self_capture_provider import SelfCaptureProviderAdapter
from puripuly_heart.core.local_asr_provider_runtime import ProviderRuntimeBuildRequest

from puripuly_heart.config.resolved import ResolvedSTTConfig
from puripuly_heart.core.self_capture import (
    SelfCaptureProviderMutationStatus,
    SelfCaptureSessionConfig,
)


def _config(*, provider_id: str = "deepgram", local_gpu: bool = False):
    return SelfCaptureSessionConfig(
        provider_id=provider_id,
        provider_signature=(provider_id,),
        capture_signature=("capture",),
        runtime_signature=(provider_id, "capture"),
        target_sample_rate_hz=16000,
        local_gpu=local_gpu,
    )


def _request(provider_id: str = "deepgram") -> ProviderRuntimeBuildRequest:
    config = cast(
        ResolvedSTTConfig,
        SimpleNamespace(channel="self", provider=provider_id),
    )
    return ProviderRuntimeBuildRequest(config=config)


def _snapshot(
    *,
    provider_id: str | None = "deepgram",
    has_resources: bool = True,
    gpu_phase: str = "inactive",
    active_channels: frozenset[str] = frozenset(),
):
    channel = SimpleNamespace(provider_id=provider_id, has_resources=has_resources)
    return SimpleNamespace(
        channel_for=lambda requested: channel,
        gpu=SimpleNamespace(phase=gpu_phase, active_channels=active_channels),
    )


class RecordingRuntime:
    def __init__(self) -> None:
        self.snapshot = _snapshot()
        self.calls: list[tuple[object, ...]] = []
        self.replace_result = SimpleNamespace(status="applied", failure_type=None)
        self.handoff_result = SimpleNamespace(status="failed", failure_type="provider_error")

    async def reset_provider_channel(self, channel):
        self.calls.append(("reset", channel))

    async def replace_provider(self, request, **kwargs):
        self.calls.append(("replace", request, kwargs))
        return self.replace_result

    async def handoff_provider(self, request, **kwargs):
        self.calls.append(("handoff", request, kwargs))
        return self.handoff_result

    async def cancel_handoff(self, channel):
        self.calls.append(("cancel", channel))
        return True

    async def start_channel(self, channel):
        self.calls.append(("start", channel))

    async def warmup_channel(self, channel):
        self.calls.append(("warmup", channel))

    async def reconfigure_channel(self, channel, options):
        self.calls.append(("reconfigure", channel, options))

    async def release_channel(self, channel, **kwargs):
        self.calls.append(("release", channel, kwargs))


def test_readiness_requires_exact_self_provider_with_resources() -> None:
    runtime = RecordingRuntime()
    adapter = SelfCaptureProviderAdapter(cast(object, runtime), runtime)
    config = _config()

    assert adapter.is_ready(config) is True

    runtime.snapshot = _snapshot(provider_id="soniox")
    assert adapter.is_ready(config) is False

    runtime.snapshot = _snapshot(has_resources=False)
    assert adapter.is_ready(config) is False


@pytest.mark.asyncio
async def test_mutations_forward_terminal_failure_owner_and_map_results() -> None:
    runtime = RecordingRuntime()
    adapter = SelfCaptureProviderAdapter(cast(object, runtime), runtime)
    request = _request()

    async def terminal_failure(exc: Exception) -> None:
        _ = exc

    replaced = await adapter.replace(
        request,
        start=False,
        on_terminal_failure=terminal_failure,
    )
    handed_off = await adapter.handoff(
        request,
        start=True,
        on_terminal_failure=terminal_failure,
    )

    assert replaced.status is SelfCaptureProviderMutationStatus.APPLIED
    assert handed_off.status is SelfCaptureProviderMutationStatus.FAILED
    assert handed_off.reason == "provider_error"
    assert runtime.calls == [
        ("reset", "self"),
        (
            "replace",
            request,
            {"start": False, "on_terminal_failure": terminal_failure},
        ),
        (
            "handoff",
            request,
            {"start": True, "on_terminal_failure": terminal_failure},
        ),
    ]


@pytest.mark.asyncio
async def test_start_ingress_validates_provider_and_gpu_activation() -> None:
    runtime = RecordingRuntime()
    adapter = SelfCaptureProviderAdapter(cast(object, runtime), runtime)
    gpu_config = _config(provider_id="local_qwen_gpu", local_gpu=True)
    adapter.is_ready(gpu_config)
    runtime.snapshot = _snapshot(
        provider_id="local_qwen_gpu",
        gpu_phase="ready",
        active_channels=frozenset({"self"}),
    )

    await adapter.start_ingress()

    assert runtime.calls == [("start", "self")]

    runtime.snapshot = _snapshot(provider_id="deepgram")
    with pytest.raises(RuntimeError, match="ingress did not become ready"):
        await adapter.start_ingress()

    runtime.snapshot = _snapshot(
        provider_id="local_qwen_gpu",
        gpu_phase="ready",
        active_channels=frozenset(),
    )
    with pytest.raises(RuntimeError, match="GPU provider ingress"):
        await adapter.start_ingress()


@pytest.mark.asyncio
async def test_release_routes_only_self_and_accepts_pre_hub_cancellation() -> None:
    runtime = RecordingRuntime()
    adapter = SelfCaptureProviderAdapter(cast(object, runtime), runtime)

    await adapter.release(mode="drain", release_backend_after=2.5)
    await adapter.release(mode="abort")
    await SelfCaptureProviderAdapter(None, None).release(mode="abort")

    assert runtime.calls == [
        (
            "release",
            "self",
            {"mode": "drain", "release_backend_after": 2.5},
        ),
        ("reset", "self"),
        (
            "release",
            "self",
            {"mode": "abort", "release_backend_after": None},
        ),
    ]
