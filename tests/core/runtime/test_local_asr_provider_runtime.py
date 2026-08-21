from __future__ import annotations

import asyncio
import contextlib
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from puripuly_heart.core.local_asr_provider_runtime import (
    ProviderRuntimeBuildRequest,
    ProviderRuntimeGpuRecoveryRequest,
    ProviderRuntimeRecoveryChannel,
)
from puripuly_heart.core.local_asr_provisioning import (
    LocalASRModelProvisioningState,
    LocalASRProvisioningSnapshot,
)
from puripuly_heart.core.local_stt_assets import LOCAL_QWEN_GPU_MODEL_ID

from puripuly_heart.config.resolved import (
    CREDENTIAL_SOURCE_NONE,
    ResolvedCredentialRequirement,
    ResolvedSTTConfig,
)
from puripuly_heart.core.gpu_worker import (
    GpuWorkerActivation,
    GpuWorkerDevice,
    GpuWorkerRequestError,
    GpuWorkerTranscription,
)
from puripuly_heart.core.runtime.gpu_asr import GpuASRDiagnostic
from puripuly_heart.core.runtime.local_asr_provider_runtime import (
    LocalASRProviderRuntimeOwner,
)
from puripuly_heart.core.runtime.local_asr_transition import LocalASRSessionOptions
from tests.helpers.lifecycle import assert_lifecycle_structure

GPU_DEVICE = GpuWorkerDevice(
    device_id="vulkan-index-0",
    registry_index=0,
    name="GPU 0",
    description="Physical Vulkan GPU",
    device_type="discrete",
    memory_total_bytes=8_000_000_000,
    memory_free_bytes=1,
)


def _resolved_config(
    channel: str,
    provider: str = "deepgram",
) -> ResolvedSTTConfig:
    return ResolvedSTTConfig(
        channel=channel,
        source_language="ko" if channel == "self" else "en",
        provider=provider,
        model=None,
        endpoint=None,
        region=None,
        credential=ResolvedCredentialRequirement(
            source=CREDENTIAL_SOURCE_NONE,
            required=False,
            reference=None,
        ),
        input_host_api=None,
        input_device=None,
        output_device=None,
        sample_rate_hz=16_000,
        channels=1,
        ring_buffer_ms=500,
        drain_timeout_s=2.0,
        vad_speech_threshold=0.5,
        vad_hangover_ms=500,
        vad_pre_roll_ms=500,
        low_latency_enabled=True,
        low_latency_merge_gap_ms=600,
        low_latency_spec_retry_max=10,
        custom_vocabulary_enabled=False,
        custom_terms={},
        provider_options={},
    )


class FakeProvisioningPort:
    def __init__(self, integrity: str = "ready", operation: str = "idle") -> None:
        self.integrity = integrity
        self.operation = operation
        self.inspect_gpu_calls: list[tuple[bool, bool]] = []
        self.closed = False

    @property
    def snapshot(self) -> LocalASRProvisioningSnapshot:
        return LocalASRProvisioningSnapshot(
            models=(
                LocalASRModelProvisioningState(
                    model_id=LOCAL_QWEN_GPU_MODEL_ID,
                    backend="gpu",
                    integrity=self.integrity,
                    operation=self.operation,
                ),
            ),
            required_cpu_model_ids=(),
            gpu_model_id=LOCAL_QWEN_GPU_MODEL_ID,
        )

    @property
    def diagnostics(self) -> tuple[object, ...]:
        return ()

    async def inspect_gpu(
        self,
        *,
        explicit_intent: bool,
        verify_checksums: bool = False,
    ) -> LocalASRProvisioningSnapshot:
        self.inspect_gpu_calls.append((explicit_intent, verify_checksums))
        return self.snapshot

    async def close(self) -> None:
        self.closed = True


class FakeGpuRuntime:
    def __init__(
        self,
        sink,
        *,
        discovery_gate: asyncio.Event | None = None,
        discovery_results: list[tuple[GpuWorkerDevice, ...] | BaseException] | None = None,
    ) -> None:
        self.sink = sink
        self.discovery_gate = discovery_gate
        self.discovery_results = discovery_results
        self.state = "idle"
        self.discovery_state = "idle"
        self.active_channels = frozenset()
        self.pending_count = 0
        self.worker_pid = None
        self.last_failure_code = None
        self.configured_device_id = None
        self.discovery_calls = 0
        self.activation_calls: list[tuple[str, str]] = []
        self.retry_calls = 0
        self.close_calls = 0

    async def emit(self, diagnostic: GpuASRDiagnostic) -> None:
        await self.sink(diagnostic)

    async def discover_devices(self) -> tuple[GpuWorkerDevice, ...]:
        self.discovery_calls += 1
        self.discovery_state = "pending"
        await self.emit(GpuASRDiagnostic(kind="discovery_pending", fields={}))
        if self.discovery_gate is not None:
            await self.discovery_gate.wait()
        if self.discovery_results is not None:
            if not self.discovery_results:
                raise AssertionError("unexpected extra GPU discovery")
            result = self.discovery_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            self.discovery_state = "ready"
            return result
        self.discovery_state = "ready"
        return (GPU_DEVICE,)

    async def activate_channel(
        self,
        channel: str,
        *,
        model_path: Path,
        model_id: str,
        device_id: str,
    ) -> GpuWorkerActivation:
        _ = model_path, model_id
        self.activation_calls.append((channel, device_id))
        self.active_channels = frozenset({*self.active_channels, channel})
        self.state = "ready"
        self.worker_pid = 4242
        self.configured_device_id = device_id
        await self.emit(
            GpuASRDiagnostic(
                kind="activation_ready",
                fields={
                    "model": LOCAL_QWEN_GPU_MODEL_ID,
                    "device": device_id,
                    "model_load_seconds": 1.0,
                    "warmup_seconds": 0.25,
                },
            )
        )
        return GpuWorkerActivation(
            device=GPU_DEVICE,
            model_load_seconds=1.0,
            warmup_seconds=0.25,
        )

    async def retry(self) -> GpuWorkerActivation:
        self.retry_calls += 1
        self.state = "ready"
        self.last_failure_code = None
        return GpuWorkerActivation(
            device=GPU_DEVICE,
            model_load_seconds=1.0,
            warmup_seconds=0.25,
        )

    async def submit(
        self,
        channel: str,
        samples_f32: np.ndarray,
        *,
        speech_end_at: float,
        language_hint: str | None = None,
    ) -> GpuWorkerTranscription:
        _ = channel, samples_f32, speech_end_at, language_hint
        return GpuWorkerTranscription(
            text="ok",
            detected_language="en",
            audio_seconds=1.0,
            decode_seconds=0.1,
            rtf=0.1,
        )

    async def deactivate_channel(self, channel: str) -> None:
        self.active_channels = frozenset(
            active_channel for active_channel in self.active_channels if active_channel != channel
        )
        if not self.active_channels:
            self.worker_pid = None
            self.state = "idle"

    async def close(self) -> None:
        self.close_calls += 1
        self.active_channels = frozenset()
        self.worker_pid = None
        self.state = "closed"


class FakeGpuRuntimeFactory:
    def __init__(
        self,
        *,
        discovery_gate: asyncio.Event | None = None,
        discovery_results: list[tuple[GpuWorkerDevice, ...] | BaseException] | None = None,
    ) -> None:
        self.discovery_gate = discovery_gate
        self.discovery_results = discovery_results
        self.instances: list[FakeGpuRuntime] = []

    def __call__(self, sink) -> FakeGpuRuntime:
        results = None if self.discovery_results is None else list(self.discovery_results)
        runtime = FakeGpuRuntime(
            sink,
            discovery_gate=self.discovery_gate,
            discovery_results=results,
        )
        self.instances.append(runtime)
        return runtime


class FakeProvider:
    def __init__(
        self,
        provider_id: str,
        *,
        gpu_runtime: FakeGpuRuntime,
        channel: str,
        gpu_device_id: str,
    ) -> None:
        self.provider_id = provider_id
        self.gpu_runtime = gpu_runtime
        self.channel = channel
        self.gpu_device_id = gpu_device_id
        self.is_at_utterance_boundary = True
        self.warmup_calls = 0
        self.close_calls = 0
        self.close_backend_calls = 0
        self.reconfigure_calls: list[LocalASRSessionOptions] = []
        self.vad_events: list[object] = []
        self.vad_gate: asyncio.Event | None = None
        self.events_closed = asyncio.Event()

    async def warmup(self) -> None:
        self.warmup_calls += 1
        if self.provider_id == "local_qwen_gpu":
            await self.gpu_runtime.activate_channel(
                self.channel,
                model_path=Path("model.gguf"),
                model_id=LOCAL_QWEN_GPU_MODEL_ID,
                device_id=self.gpu_device_id,
            )

    async def close(self) -> None:
        self.close_calls += 1
        self.events_closed.set()

    async def close_backend(self) -> None:
        self.close_backend_calls += 1
        if self.provider_id == "local_qwen_gpu":
            await self.gpu_runtime.deactivate_channel(self.channel)

    async def reconfigure_session_options(self, options: LocalASRSessionOptions) -> None:
        self.reconfigure_calls.append(options)

    async def handle_vad_event(self, event: object) -> None:
        self.vad_events.append(event)
        if self.vad_gate is not None:
            await self.vad_gate.wait()

    async def events(self):
        await self.events_closed.wait()
        if False:
            yield None


class FakeProviderFactory:
    def __init__(self) -> None:
        self.requests: list[ProviderRuntimeBuildRequest] = []
        self.providers: list[FakeProvider] = []
        self.fail_next = False

    async def create(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        gpu_runtime: FakeGpuRuntime,
        on_terminal_failure=None,
    ) -> FakeProvider:
        _ = on_terminal_failure
        self.requests.append(request)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("provider secret payload")
        provider = FakeProvider(
            request.provider_id,
            gpu_runtime=gpu_runtime,
            channel=request.channel,
            gpu_device_id=request.gpu_device_id,
        )
        self.providers.append(provider)
        return provider


def _owner(
    *,
    provisioning: FakeProvisioningPort | None = None,
    gpu_factory: FakeGpuRuntimeFactory | None = None,
    provider_factory: FakeProviderFactory | None = None,
    state_changed=None,
    diagnostic_sink=None,
) -> tuple[
    LocalASRProviderRuntimeOwner,
    FakeProvisioningPort,
    FakeGpuRuntimeFactory,
    FakeProviderFactory,
]:
    provisioning = provisioning or FakeProvisioningPort()
    gpu_factory = gpu_factory or FakeGpuRuntimeFactory()
    provider_factory = provider_factory or FakeProviderFactory()
    owner = LocalASRProviderRuntimeOwner(
        provider_factory=provider_factory,
        gpu_runtime_factory=gpu_factory,
        provisioning=provisioning,
        state_changed=state_changed,
        diagnostic_sink=diagnostic_sink,
    )
    return owner, provisioning, gpu_factory, provider_factory


async def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0)


def test_build_request_and_snapshot_are_immutable() -> None:
    request = ProviderRuntimeBuildRequest(config=_resolved_config("self"))
    owner, _provisioning, _gpu_factory, _provider_factory = _owner()
    snapshot = owner.snapshot

    with pytest.raises(FrozenInstanceError):
        request.warmup = True
    with pytest.raises(FrozenInstanceError):
        snapshot.closed = True
    with pytest.raises(ValueError, match="gpu_device_id"):
        ProviderRuntimeBuildRequest(
            config=_resolved_config("peer"),
            gpu_device_id=" ",
        )


@pytest.mark.asyncio
async def test_owner_constructs_replaces_and_closes_each_provider() -> None:
    owner, provisioning, gpu_factory, provider_factory = _owner()
    first = ProviderRuntimeBuildRequest(
        config=_resolved_config("self", "deepgram"),
        warmup=True,
    )
    second = ProviderRuntimeBuildRequest(
        config=_resolved_config("self", "soniox"),
        warmup=False,
    )

    first_result = await owner.replace_provider(first, start=False)
    second_result = await owner.replace_provider(second, start=True)

    assert first_result.status == "applied"
    assert second_result.status == "applied"
    assert second_result.previous_provider_id == "deepgram"
    assert provider_factory.requests == [first, second]
    assert provider_factory.providers[0].warmup_calls == 1
    assert provider_factory.providers[0].close_calls == 0
    assert provider_factory.providers[0].close_backend_calls == 1
    assert owner.snapshot.channel_for("self").provider_id == "soniox"
    assert owner.snapshot.channel_for("self").phase == "running"
    assert provisioning.inspect_gpu_calls == []
    assert gpu_factory.instances[0].discovery_calls == 0

    await owner.close()
    assert provider_factory.providers[1].close_calls == 0
    assert provider_factory.providers[1].close_backend_calls == 1


@pytest.mark.asyncio
async def test_generic_handoff_waits_for_channel_boundary_commit() -> None:
    owner, _provisioning, _gpu_factory, provider_factory = _owner()
    old_request = ProviderRuntimeBuildRequest(config=_resolved_config("self", "deepgram"))
    next_request = ProviderRuntimeBuildRequest(config=_resolved_config("self", "soniox"))
    await owner.replace_provider(old_request, start=False)
    old_provider = provider_factory.providers[0]
    old_provider.is_at_utterance_boundary = False

    handoff = asyncio.create_task(
        owner.handoff_provider(next_request, start=False),
    )
    await _wait_until(lambda: owner.snapshot.channel_for("self").pending_handoff)

    assert handoff.done() is False
    assert owner.snapshot.channel_for("self").provider_id == "deepgram"

    await owner.commit_handoff("self")
    result = await handoff

    assert result.status == "applied"
    assert result.previous_provider_id == "deepgram"
    assert owner.snapshot.channel_for("self").provider_id == "soniox"
    await _wait_until(lambda: old_provider.close_backend_calls == 1)

    await owner.close()


@pytest.mark.asyncio
async def test_owner_dispatches_warmup_reconfigure_and_vad_without_exposing_provider() -> None:
    owner, _provisioning, _gpu_factory, provider_factory = _owner()
    initial_options = LocalASRSessionOptions(
        source_language="ko",
        source_mode="manual",
        language_hint="ko",
    )
    request = ProviderRuntimeBuildRequest(
        config=_resolved_config("self", "local_qwen"),
        model_id="qwen-model",
        session_options=initial_options,
    )
    await owner.replace_provider(request, start=False)
    provider = provider_factory.providers[0]
    next_options = LocalASRSessionOptions(
        source_language="en",
        source_mode="manual",
        language_hint="en",
    )
    event = object()

    await owner.warmup_channel("self")
    await owner.reconfigure_channel("self", next_options)
    await owner.handle_vad_event("self", event)

    channel = owner.snapshot.channel_for("self")
    assert channel.provider_id == "local_qwen"
    assert channel.model_id == "qwen-model"
    assert channel.phase == "ready"
    assert provider.warmup_calls == 1
    assert provider.reconfigure_calls == [next_options]
    assert provider.vad_events == [event]
    assert not hasattr(owner.snapshot, "provider")

    await owner.close()


@pytest.mark.asyncio
async def test_inactive_gpu_selection_does_not_discover_provision_or_start_worker() -> None:
    owner, provisioning, gpu_factory, _provider_factory = _owner()

    snapshot = await owner.inspect_gpu_readiness(
        explicit_intent=False,
        device_id="auto",
    )

    assert snapshot.gpu.phase == "inactive"
    assert snapshot.gpu.worker_pid is None
    assert provisioning.inspect_gpu_calls == []
    assert gpu_factory.instances[0].discovery_calls == 0
    assert gpu_factory.instances[0].activation_calls == []

    await owner.close()


@pytest.mark.asyncio
async def test_gpu_readiness_uses_only_provisioning_port_and_low_vram_is_advisory() -> None:
    owner, provisioning, gpu_factory, provider_factory = _owner()

    snapshot = await owner.inspect_gpu_readiness(
        explicit_intent=True,
        device_id=GPU_DEVICE.device_id,
    )

    assert snapshot.gpu.phase == "available"
    assert snapshot.gpu.devices == (GPU_DEVICE,)
    assert snapshot.gpu.devices[0].memory_free_bytes == 1
    assert provisioning.inspect_gpu_calls == [(True, False)]
    assert gpu_factory.instances[0].activation_calls == []

    request = ProviderRuntimeBuildRequest(
        config=_resolved_config("peer", "local_qwen_gpu"),
        gpu_device_id=GPU_DEVICE.device_id,
        warmup=False,
    )
    result = await owner.replace_provider(request, start=False)

    assert result.status == "applied"
    assert provider_factory.providers[-1].gpu_runtime is gpu_factory.instances[0]
    assert gpu_factory.instances[0].activation_calls == []

    await owner.close()


@pytest.mark.asyncio
async def test_gpu_readiness_preserves_active_model_download_state() -> None:
    provisioning = FakeProvisioningPort(integrity="missing", operation="downloading")
    owner, _provisioning, gpu_factory, _provider_factory = _owner(provisioning=provisioning)

    snapshot = await owner.inspect_gpu_readiness(
        explicit_intent=True,
        device_id=GPU_DEVICE.device_id,
    )

    assert snapshot.gpu.phase == "installing"
    assert snapshot.gpu.failure_code == "downloading"
    assert gpu_factory.instances[0].activation_calls == []

    await owner.close()


@pytest.mark.asyncio
async def test_missing_gpu_asset_rejects_construction_without_fallback() -> None:
    provisioning = FakeProvisioningPort(integrity="missing")
    owner, _provisioning, gpu_factory, provider_factory = _owner(provisioning=provisioning)
    request = ProviderRuntimeBuildRequest(
        config=_resolved_config("self", "local_qwen_gpu"),
        warmup=True,
    )

    result = await owner.replace_provider(request, start=True)

    assert result.status == "failed"
    assert provider_factory.requests == []
    assert owner.snapshot.channel_for("self").provider_id is None
    assert owner.snapshot.gpu.phase == "not_installed"
    assert gpu_factory.instances[0].activation_calls == []

    await owner.close()


@pytest.mark.asyncio
async def test_failed_build_preserves_current_provider_and_excludes_raw_error_text() -> None:
    owner, _provisioning, _gpu_factory, provider_factory = _owner()
    current = ProviderRuntimeBuildRequest(config=_resolved_config("self", "deepgram"))
    replacement = ProviderRuntimeBuildRequest(config=_resolved_config("self", "soniox"))
    await owner.replace_provider(current, start=True)
    provider_factory.fail_next = True

    result = await owner.handoff_provider(replacement, start=True)

    assert result.status == "failed"
    assert owner.snapshot.channel_for("self").provider_id == "deepgram"
    assert owner.snapshot.channel_for("self").phase == "running"
    assert provider_factory.providers[0].close_backend_calls == 0
    assert "provider secret payload" not in repr(owner.diagnostics[-1])

    await owner.close()


@pytest.mark.asyncio
async def test_gpu_discovery_is_single_flight_and_exposes_pending_state() -> None:
    gate = asyncio.Event()
    gpu_factory = FakeGpuRuntimeFactory(discovery_gate=gate)
    observed_phases: list[str] = []

    def state_changed(snapshot) -> None:
        observed_phases.append(snapshot.gpu.phase)

    owner, _provisioning, _gpu_factory, _provider_factory = _owner(
        gpu_factory=gpu_factory,
        state_changed=state_changed,
    )
    first = asyncio.create_task(owner.discover_gpu())
    await _wait_until(lambda: gpu_factory.instances[0].discovery_calls == 1)
    second = asyncio.create_task(owner.discover_gpu())

    assert gpu_factory.instances[0].discovery_calls == 1
    assert "discovery_pending" in observed_phases

    gate.set()
    first_snapshot, second_snapshot = await asyncio.gather(first, second)

    assert first_snapshot.gpu.devices == (GPU_DEVICE,)
    assert second_snapshot.gpu.devices == (GPU_DEVICE,)
    assert gpu_factory.instances[0].discovery_calls == 1

    await owner.close()


@pytest.mark.asyncio
async def test_failed_gpu_discovery_is_not_cached_and_retries_on_next_intent() -> None:
    gpu_factory = FakeGpuRuntimeFactory(
        discovery_results=[
            RuntimeError("worker_process_exited"),
            (GPU_DEVICE,),
        ]
    )
    owner, _provisioning, _gpu_factory, _provider_factory = _owner(gpu_factory=gpu_factory)

    first = await owner.discover_gpu()
    second = await owner.discover_gpu()

    assert first.gpu.phase == "failed"
    assert first.gpu.failure_code == "RuntimeError"
    assert second.gpu.phase == "idle"
    assert second.gpu.devices == (GPU_DEVICE,)
    assert gpu_factory.instances[0].discovery_calls == 2

    await owner.close()


@pytest.mark.asyncio
async def test_empty_gpu_discovery_is_unsupported_and_cached() -> None:
    gpu_factory = FakeGpuRuntimeFactory(discovery_results=[()])
    owner, _provisioning, _gpu_factory, _provider_factory = _owner(gpu_factory=gpu_factory)

    first = await owner.discover_gpu()
    second = await owner.discover_gpu()

    assert first.gpu.phase == "unsupported"
    assert first.gpu.failure_code == "no_supported_gpu"
    assert second.gpu.phase == "unsupported"
    assert gpu_factory.instances[0].discovery_calls == 1

    await owner.close()


@pytest.mark.asyncio
async def test_unsupported_capability_is_completed_no_gpu_answer() -> None:
    gpu_factory = FakeGpuRuntimeFactory(
        discovery_results=[GpuWorkerRequestError("unsupported_capability")]
    )
    owner, _provisioning, _gpu_factory, _provider_factory = _owner(gpu_factory=gpu_factory)

    first = await owner.discover_gpu()
    second = await owner.discover_gpu()

    assert first.gpu.phase == "unsupported"
    assert first.gpu.failure_code == "no_supported_gpu"
    assert second.gpu.phase == "unsupported"
    assert gpu_factory.instances[0].discovery_calls == 1

    await owner.close()


@pytest.mark.asyncio
async def test_inspect_gpu_readiness_does_not_rewrite_failed_discovery_as_unsupported() -> None:
    gpu_factory = FakeGpuRuntimeFactory(
        discovery_results=[
            RuntimeError("worker_process_exited"),
            RuntimeError("worker_process_exited"),
        ]
    )
    owner, _provisioning, _gpu_factory, _provider_factory = _owner(gpu_factory=gpu_factory)

    await owner.discover_gpu()
    snapshot = await owner.inspect_gpu_readiness(
        explicit_intent=True,
        device_id="auto",
    )

    assert snapshot.gpu.phase == "failed"
    assert snapshot.gpu.failure_code == "RuntimeError"
    assert gpu_factory.instances[0].discovery_calls == 2

    await owner.close()


@pytest.mark.asyncio
async def test_inspect_gpu_readiness_retries_incomplete_discovery_failure() -> None:
    gpu_factory = FakeGpuRuntimeFactory(
        discovery_results=[
            RuntimeError("worker_process_exited"),
            (GPU_DEVICE,),
        ]
    )
    owner, _provisioning, _gpu_factory, _provider_factory = _owner(gpu_factory=gpu_factory)

    first = await owner.discover_gpu()
    snapshot = await owner.inspect_gpu_readiness(
        explicit_intent=True,
        device_id=GPU_DEVICE.device_id,
    )

    assert first.gpu.phase == "failed"
    assert snapshot.gpu.phase == "available"
    assert snapshot.gpu.devices == (GPU_DEVICE,)
    assert gpu_factory.instances[0].discovery_calls == 2

    await owner.close()


@pytest.mark.asyncio
async def test_manual_recovery_is_explicit_quiesced_and_never_builds_cpu_fallback() -> None:
    owner, _provisioning, gpu_factory, provider_factory = _owner()
    request = ProviderRuntimeBuildRequest(
        config=_resolved_config("self", "local_qwen_gpu"),
        gpu_device_id=GPU_DEVICE.device_id,
        warmup=True,
    )
    result = await owner.replace_provider(request, start=True)
    runtime = gpu_factory.instances[0]

    assert result.status == "applied"
    assert runtime.retry_calls == 0
    assert result.snapshot.gpu.phase == "ready"
    runtime.state = "failed"
    runtime.last_failure_code = "decode_failed"

    quiesced: list[tuple[str, ...]] = []

    async def quiesce(channels: tuple[str, ...]) -> None:
        assert owner.snapshot.gpu.active_channels == frozenset({"self"})
        quiesced.append(channels)

    snapshot = await owner.recover_gpu(
        ProviderRuntimeGpuRecoveryRequest(
            device_id=GPU_DEVICE.device_id,
            channels=(ProviderRuntimeRecoveryChannel(request=request, start=True),),
            reason="manual_retry",
        ),
        quiesce=quiesce,
    )

    assert quiesced == [("self",)]
    assert runtime.retry_calls == 0
    assert runtime.close_calls == 1
    assert len(gpu_factory.instances) == 2
    assert snapshot.gpu.phase == "ready"
    assert [request.provider_id for request in provider_factory.requests] == [
        "local_qwen_gpu",
        "local_qwen_gpu",
    ]

    await owner.close()


@pytest.mark.asyncio
async def test_active_gpu_device_change_requires_owner_quiescence() -> None:
    owner, _provisioning, gpu_factory, provider_factory = _owner()
    current = ProviderRuntimeBuildRequest(
        config=_resolved_config("self", "local_qwen_gpu"),
        gpu_device_id="auto",
        warmup=True,
    )
    await owner.replace_provider(current, start=True)
    requested = ProviderRuntimeBuildRequest(
        config=_resolved_config("peer", "local_qwen_gpu"),
        gpu_device_id=GPU_DEVICE.device_id,
        warmup=True,
    )

    result = await owner.replace_provider(requested, start=True)

    assert result.status == "failed"
    assert result.snapshot.gpu.failure_code == "device_change_requires_quiesce"
    assert owner.diagnostics[-1].failure_code == "device_change_requires_quiesce"
    assert owner.snapshot.channel_for("peer").provider_id is None
    assert len(provider_factory.requests) == 1
    assert gpu_factory.instances[0].active_channels == frozenset({"self"})

    await owner.close()


@pytest.mark.asyncio
async def test_manual_retry_creates_one_fresh_runtime_after_full_quiesce() -> None:
    owner, _provisioning, gpu_factory, provider_factory = _owner()
    request = ProviderRuntimeBuildRequest(
        config=_resolved_config("peer", "local_qwen_gpu"),
        gpu_device_id=GPU_DEVICE.device_id,
        warmup=True,
    )
    await owner.replace_provider(request, start=False)
    await owner.release_channel("peer", mode="abort")
    failed_runtime = gpu_factory.instances[0]
    failed_runtime.state = "failed"
    failed_runtime.last_failure_code = "worker_failed"

    snapshot = await owner.recover_gpu(
        ProviderRuntimeGpuRecoveryRequest(
            device_id=GPU_DEVICE.device_id,
            channels=(ProviderRuntimeRecoveryChannel(request=request, start=False),),
            reason="manual_retry",
        )
    )

    assert len(gpu_factory.instances) == 2
    assert failed_runtime.close_calls == 1
    assert gpu_factory.instances[1].active_channels == frozenset({"peer"})
    assert snapshot.gpu.model_resident is True
    assert [request.provider_id for request in provider_factory.requests] == [
        "local_qwen_gpu",
        "local_qwen_gpu",
    ]
    diagnostic_count = len(owner.diagnostics)
    await failed_runtime.emit(
        GpuASRDiagnostic(
            kind="worker_failed",
            fields={"failure": "late_old_runtime_failure"},
        )
    )
    assert owner.snapshot.gpu.phase == "ready"
    assert len(owner.diagnostics) == diagnostic_count

    await owner.close()


@pytest.mark.asyncio
async def test_gpu_recovery_unavailability_quiesces_and_releases_previous_runtime() -> None:
    owner, provisioning, gpu_factory, provider_factory = _owner()
    request = ProviderRuntimeBuildRequest(
        config=_resolved_config("self", "local_qwen_gpu"),
        gpu_device_id=GPU_DEVICE.device_id,
        warmup=True,
    )
    await owner.replace_provider(request, start=True)
    old_runtime = gpu_factory.instances[0]
    provisioning.integrity = "missing"
    quiesced: list[tuple[str, ...]] = []

    async def quiesce(channels: tuple[str, ...]) -> None:
        quiesced.append(channels)

    snapshot = await owner.recover_gpu(
        ProviderRuntimeGpuRecoveryRequest(
            device_id=GPU_DEVICE.device_id,
            channels=(ProviderRuntimeRecoveryChannel(request=request, start=True),),
            reason="settings_restart",
        ),
        quiesce=quiesce,
    )

    assert quiesced == [("self",)]
    assert snapshot.gpu.phase == "not_installed"
    assert snapshot.channel_for("self").provider_id is None
    assert old_runtime.close_calls == 1
    assert len(gpu_factory.instances) == 2
    assert gpu_factory.instances[1].active_channels == frozenset()
    assert len(provider_factory.requests) == 1

    await owner.close()


@pytest.mark.asyncio
async def test_gpu_recovery_build_failure_aborts_partial_resources_and_keeps_retryable_owner() -> (
    None
):
    owner, _provisioning, gpu_factory, provider_factory = _owner()
    request = ProviderRuntimeBuildRequest(
        config=_resolved_config("peer", "local_qwen_gpu"),
        gpu_device_id=GPU_DEVICE.device_id,
        warmup=True,
    )
    await owner.replace_provider(request, start=False)
    provider_factory.fail_next = True

    with pytest.raises(RuntimeError, match="GPU provider recovery failed for peer"):
        await owner.recover_gpu(
            ProviderRuntimeGpuRecoveryRequest(
                device_id=GPU_DEVICE.device_id,
                channels=(ProviderRuntimeRecoveryChannel(request=request, start=False),),
                reason="manual_retry",
            )
        )

    assert owner.snapshot.gpu.phase == "failed"
    assert owner.snapshot.channel_for("peer").provider_id is None
    assert all(runtime.active_channels == frozenset() for runtime in gpu_factory.instances)
    assert len(gpu_factory.instances) == 3

    snapshot = await owner.recover_gpu(
        ProviderRuntimeGpuRecoveryRequest(
            device_id=GPU_DEVICE.device_id,
            channels=(ProviderRuntimeRecoveryChannel(request=request, start=False),),
            reason="manual_retry",
        )
    )

    assert snapshot.gpu.phase == "ready"
    assert snapshot.channel_for("peer").has_resources

    await owner.close()


@pytest.mark.asyncio
async def test_gpu_recovery_commands_are_serialized_by_owner() -> None:
    owner, _provisioning, _gpu_factory, _provider_factory = _owner()
    request = ProviderRuntimeBuildRequest(
        config=_resolved_config("self", "local_qwen_gpu"),
        gpu_device_id=GPU_DEVICE.device_id,
        warmup=True,
    )
    recovery = ProviderRuntimeGpuRecoveryRequest(
        device_id=GPU_DEVICE.device_id,
        channels=(ProviderRuntimeRecoveryChannel(request=request, start=True),),
        reason="manual_retry",
    )
    first_entered = asyncio.Event()
    allow_first = asyncio.Event()
    order: list[str] = []

    async def first_quiesce(_channels: tuple[str, ...]) -> None:
        order.append("first")
        first_entered.set()
        await allow_first.wait()

    async def second_quiesce(_channels: tuple[str, ...]) -> None:
        order.append("second")

    first = asyncio.create_task(owner.recover_gpu(recovery, quiesce=first_quiesce))
    await first_entered.wait()
    second = asyncio.create_task(owner.recover_gpu(recovery, quiesce=second_quiesce))
    await asyncio.sleep(0)

    assert order == ["first"]

    allow_first.set()
    await asyncio.gather(first, second)

    assert order == ["first", "second"]

    await owner.close()


@pytest.mark.asyncio
async def test_gpu_diagnostics_copy_only_safe_lifecycle_fields() -> None:
    observed = []
    owner, _provisioning, gpu_factory, _provider_factory = _owner(diagnostic_sink=observed.append)
    runtime = gpu_factory.instances[0]

    await runtime.emit(
        GpuASRDiagnostic(
            kind="worker_failed",
            fields={
                "failure": "worker_exit",
                "exit_code": 7,
                "transcript": "private user speech",
                "auth_token": "secret token",
                "audio_path": "C:/Users/private/audio.wav",
            },
        )
    )

    diagnostic = owner.diagnostics[-1]
    assert diagnostic.event == "worker_failed"
    assert diagnostic.failure_code == "worker_exit"
    assert diagnostic.worker_exit_code == 7
    assert "private user speech" not in repr(diagnostic)
    assert "secret token" not in repr(diagnostic)
    assert "C:/Users/private/audio.wav" not in repr(diagnostic)
    assert observed[-1] == diagnostic

    await owner.close()


@pytest.mark.asyncio
async def test_state_and_diagnostic_sink_failures_cannot_break_owner_lifecycle() -> None:
    def failing_state_sink(_snapshot) -> None:
        raise RuntimeError("private state sink payload")

    async def failing_diagnostic_sink(_diagnostic) -> None:
        raise RuntimeError("private diagnostic sink payload")

    owner, _provisioning, _gpu_factory, _provider_factory = _owner(
        state_changed=failing_state_sink,
        diagnostic_sink=failing_diagnostic_sink,
    )
    request = ProviderRuntimeBuildRequest(config=_resolved_config("self", "deepgram"))

    result = await owner.replace_provider(request, start=True)
    await owner.close()

    assert result.status == "applied"
    assert owner.snapshot.closed is True
    events = [diagnostic.event for diagnostic in owner.diagnostics]
    assert "state_changed_sink" in events
    assert "diagnostic_sink" in events
    assert "private state sink payload" not in repr(owner.diagnostics)
    assert "private diagnostic sink payload" not in repr(owner.diagnostics)


@pytest.mark.asyncio
async def test_close_cancels_pending_handoff_and_awaits_all_owned_resources() -> None:
    owner, _provisioning, gpu_factory, provider_factory = _owner()
    old_request = ProviderRuntimeBuildRequest(config=_resolved_config("peer", "deepgram"))
    next_request = ProviderRuntimeBuildRequest(config=_resolved_config("peer", "soniox"))
    await owner.replace_provider(old_request, start=False)
    provider_factory.providers[0].is_at_utterance_boundary = False
    handoff = asyncio.create_task(owner.handoff_provider(next_request, start=False))
    await _wait_until(lambda: owner.snapshot.channel_for("peer").pending_handoff)

    await owner.close()
    with contextlib.suppress(asyncio.CancelledError):
        await handoff
    await owner.close()

    assert handoff.done()
    assert owner.snapshot.closed is True
    assert owner.snapshot.channel_for("peer").pending_handoff is False
    assert provider_factory.providers[0].close_backend_calls == 1
    assert provider_factory.providers[1].close_backend_calls == 1
    assert gpu_factory.instances[0].close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        await owner.discover_gpu()


@pytest.mark.asyncio
async def test_close_retains_failed_pending_candidate_cleanup_for_retry() -> None:
    owner, _provisioning, _gpu_factory, provider_factory = _owner()
    old_request = ProviderRuntimeBuildRequest(config=_resolved_config("peer", "deepgram"))
    next_request = ProviderRuntimeBuildRequest(config=_resolved_config("peer", "soniox"))
    await owner.replace_provider(old_request, start=False)
    provider_factory.providers[0].is_at_utterance_boundary = False
    handoff = asyncio.create_task(owner.handoff_provider(next_request, start=False))
    await _wait_until(lambda: owner.snapshot.channel_for("peer").pending_handoff)
    candidate = provider_factory.providers[1]
    close_allowed = False
    original_close_backend = candidate.close_backend

    async def failing_close_backend() -> None:
        if not close_allowed:
            candidate.close_backend_calls += 1
            raise RuntimeError("pending candidate close failed")
        await original_close_backend()

    candidate.close_backend = failing_close_backend

    with pytest.raises(ExceptionGroup, match="provider runtime close failed"):
        await owner.close()

    assert handoff.done()
    assert owner._close_complete is False
    assert owner._pending_candidates == {"peer": candidate}
    assert owner._pending_requests == {"peer": next_request}
    with pytest.raises(RuntimeError, match="closed"):
        await owner.start()

    close_allowed = True
    await owner.close()

    assert owner._close_complete is True
    assert owner._pending_candidates == {}
    assert owner._pending_requests == {}
    assert candidate.close_backend_calls >= 3


@pytest.mark.asyncio
async def test_close_cancels_in_flight_provider_execution() -> None:
    owner, _provisioning, _gpu_factory, provider_factory = _owner()
    request = ProviderRuntimeBuildRequest(config=_resolved_config("self", "deepgram"))
    await owner.replace_provider(request, start=True)
    provider = provider_factory.providers[0]
    provider.vad_gate = asyncio.Event()
    execution = asyncio.create_task(owner.handle_vad_event("self", object()))
    await _wait_until(lambda: bool(provider.vad_events))

    await owner.close()

    assert execution.cancelled()
    assert provider.close_backend_calls == 1


@pytest.mark.asyncio
async def test_close_cancels_owner_serialized_gpu_recovery_and_finishes_cleanup() -> None:
    owner, _provisioning, gpu_factory, provider_factory = _owner()
    request = ProviderRuntimeBuildRequest(
        config=_resolved_config("self", "local_qwen_gpu"),
        gpu_device_id=GPU_DEVICE.device_id,
        warmup=True,
    )
    await owner.replace_provider(request, start=True)
    quiesce_entered = asyncio.Event()

    async def quiesce(_channels: tuple[str, ...]) -> None:
        quiesce_entered.set()
        await asyncio.Event().wait()

    recovery = asyncio.create_task(
        owner.recover_gpu(
            ProviderRuntimeGpuRecoveryRequest(
                device_id=GPU_DEVICE.device_id,
                channels=(ProviderRuntimeRecoveryChannel(request=request, start=True),),
                reason="manual_retry",
            ),
            quiesce=quiesce,
        )
    )
    await quiesce_entered.wait()

    await owner.close()

    with pytest.raises(asyncio.CancelledError):
        await recovery
    assert owner.snapshot.closed is True
    assert provider_factory.providers[0].close_backend_calls == 1
    assert gpu_factory.instances[0].close_calls == 1


@pytest.mark.asyncio
async def test_close_failure_is_retryable_without_reopening_ingress() -> None:
    owner, _provisioning, gpu_factory, provider_factory = _owner()
    request = ProviderRuntimeBuildRequest(config=_resolved_config("self", "deepgram"))
    await owner.replace_provider(request, start=True)
    runtime = gpu_factory.instances[0]
    original_close = runtime.close
    close_attempts = 0

    async def flaky_close() -> None:
        nonlocal close_attempts
        close_attempts += 1
        if close_attempts == 1:
            raise RuntimeError("worker close failed")
        await original_close()

    runtime.close = flaky_close

    with pytest.raises(ExceptionGroup, match="provider runtime close failed"):
        await owner.close()

    assert owner.snapshot.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        await owner.start()

    await owner.close()

    assert close_attempts == 2
    assert runtime.close_calls == 1
    assert provider_factory.providers[0].close_backend_calls == 1


def test_owner_lifecycle_inventory_names_provider_and_gpu_resources() -> None:
    owner, _provisioning, _gpu_factory, _provider_factory = _owner()

    inventory = owner.lifecycle_owner_snapshot()

    assert_lifecycle_structure(inventory)
    assert inventory["owner"] == "LocalASRProviderRuntimeOwner"
    assert inventory["provider_handles"].keys() == {"self", "peer"}
