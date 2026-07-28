from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from puripuly_heart.app.services.gpu_provider_recovery import (
    GpuProviderRecoveryChannelPlan,
    GpuProviderRecoveryDiagnostic,
    GpuProviderRecoveryExecution,
    GpuProviderRecoveryOwner,
)
from puripuly_heart.config.resolved import (
    CREDENTIAL_SOURCE_NONE,
    ResolvedCredentialRequirement,
    ResolvedSTTConfig,
)
from puripuly_heart.core.local_asr_provider_runtime import (
    ProviderRuntimeBuildRequest,
    ProviderRuntimeChannel,
    ProviderRuntimeGpuRecoveryRequest,
    ProviderRuntimeTerminalFailureSink,
)


def _request(channel: ProviderRuntimeChannel) -> ProviderRuntimeBuildRequest:
    return ProviderRuntimeBuildRequest(
        config=ResolvedSTTConfig(
            channel=channel,
            source_language="ko" if channel == "self" else "en",
            provider="local_qwen_gpu",
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
        ),
        gpu_device_id="vk:0",
        warmup=True,
    )


def _snapshot(
    *,
    channels: frozenset[ProviderRuntimeChannel],
    retry_required: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        gpu=SimpleNamespace(
            active_channels=channels,
            retry_required=retry_required,
        )
    )


@dataclass
class RuntimeStub:
    snapshot: object
    events: list[str]
    failure: Exception | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    requests: list[ProviderRuntimeGpuRecoveryRequest] = field(default_factory=list)

    async def recover_gpu(self, request, *, quiesce):
        self.requests.append(request)
        self.events.append(f"recover:{request.reason}")
        self.started.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        await quiesce(tuple(item.request.channel for item in request.channels))
        return self.snapshot


def _plan(
    channel: ProviderRuntimeChannel,
    events: list[str],
    *,
    prepare_failure: Exception | None = None,
) -> GpuProviderRecoveryChannelPlan:
    async def failure_handler(_exc: Exception) -> None:
        return None

    def prepare() -> ProviderRuntimeTerminalFailureSink:
        events.append(f"prepare:{channel}")
        if prepare_failure is not None:
            raise prepare_failure
        return failure_handler

    def abort(handler: ProviderRuntimeTerminalFailureSink) -> bool:
        assert handler is failure_handler
        events.append(f"abort:{channel}")
        return True

    async def adopt(handler: ProviderRuntimeTerminalFailureSink) -> None:
        assert handler is failure_handler
        events.append(f"adopt:{channel}")

    return GpuProviderRecoveryChannelPlan(
        request=_request(channel),
        start=channel == "self",
        prepare=prepare,
        abort=abort,
        adopt=adopt,
    )


def _execution(
    runtime: RuntimeStub,
    events: list[str],
    *,
    channels: tuple[GpuProviderRecoveryChannelPlan, ...],
    reason: str = "manual_retry",
    skip_if_no_channels: bool = False,
) -> GpuProviderRecoveryExecution:
    async def quiesce(selected: tuple[ProviderRuntimeChannel, ...]) -> None:
        events.append(f"quiesce:{','.join(selected)}")

    async def on_applied(selected: frozenset[ProviderRuntimeChannel]) -> None:
        events.append(f"applied:{','.join(sorted(selected))}")

    return GpuProviderRecoveryExecution(
        runtime=runtime,
        device_id="vk:0",
        reason=reason,
        channels=channels,
        quiesce=quiesce,
        on_incomplete=lambda _snapshot: events.append("incomplete"),
        on_applied=on_applied,
        on_failure=lambda: events.append("failure"),
        skip_if_no_channels=skip_if_no_channels,
    )


@pytest.mark.asyncio
async def test_owner_runs_complete_recovery_and_adopts_peer_before_self() -> None:
    events: list[str] = []
    diagnostics: list[GpuProviderRecoveryDiagnostic] = []
    runtime = RuntimeStub(
        snapshot=_snapshot(channels=frozenset({"self", "peer"})),
        events=events,
    )
    runtime.release.set()
    owner = GpuProviderRecoveryOwner(diagnostic_sink=diagnostics.append)

    result = await owner.recover(
        lambda: _execution(
            runtime,
            events,
            channels=(
                _plan("self", events),
                _plan("peer", events),
            ),
        )
    )

    assert result.status == "applied"
    assert events == [
        "prepare:self",
        "prepare:peer",
        "recover:manual_retry",
        "quiesce:self,peer",
        "adopt:peer",
        "adopt:self",
        "applied:peer,self",
        "abort:self",
        "abort:peer",
    ]
    assert runtime.requests[0].device_id == "vk:0"
    assert all(item.on_terminal_failure is not None for item in runtime.requests[0].channels)
    assert diagnostics[-1].outcome == "applied"


@pytest.mark.asyncio
async def test_owner_reports_incomplete_without_adoption() -> None:
    events: list[str] = []
    runtime = RuntimeStub(
        snapshot=_snapshot(channels=frozenset(), retry_required=True),
        events=events,
    )
    runtime.release.set()

    result = await GpuProviderRecoveryOwner().recover(
        lambda: _execution(
            runtime,
            events,
            channels=(_plan("self", events),),
        )
    )

    assert result.status == "incomplete"
    assert "incomplete" in events
    assert not any(event.startswith("adopt:") for event in events)
    assert events[-1] == "abort:self"


@pytest.mark.asyncio
async def test_owner_contains_runtime_failure_and_cleans_prepared_handlers() -> None:
    events: list[str] = []
    diagnostics: list[GpuProviderRecoveryDiagnostic] = []
    runtime = RuntimeStub(
        snapshot=_snapshot(channels=frozenset({"self"})),
        events=events,
        failure=RuntimeError("private recovery detail"),
    )
    runtime.release.set()

    result = await GpuProviderRecoveryOwner(diagnostic_sink=diagnostics.append).recover(
        lambda: _execution(
            runtime,
            events,
            channels=(_plan("self", events),),
        )
    )

    assert result.status == "failed"
    assert events[-2:] == ["failure", "abort:self"]
    assert diagnostics[-1].failure_type == "RuntimeError"
    assert "private recovery detail" not in repr(diagnostics[-1])


@pytest.mark.asyncio
async def test_owner_propagates_cancellation_after_cleanup() -> None:
    events: list[str] = []
    diagnostics: list[GpuProviderRecoveryDiagnostic] = []
    runtime = RuntimeStub(
        snapshot=_snapshot(channels=frozenset({"self"})),
        events=events,
    )
    owner = GpuProviderRecoveryOwner(diagnostic_sink=diagnostics.append)
    task = asyncio.create_task(
        owner.recover(
            lambda: _execution(
                runtime,
                events,
                channels=(_plan("self", events),),
            )
        )
    )
    await runtime.started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert events[-1] == "abort:self"
    assert diagnostics[-1].outcome == "cancelled"


@pytest.mark.asyncio
async def test_owner_serializes_through_consumer_adoption() -> None:
    events: list[str] = []
    first_adoption_started = asyncio.Event()
    release_first_adoption = asyncio.Event()
    first_runtime = RuntimeStub(
        snapshot=_snapshot(channels=frozenset({"self"})),
        events=events,
    )
    second_runtime = RuntimeStub(
        snapshot=_snapshot(channels=frozenset({"self"})),
        events=events,
    )
    first_runtime.release.set()
    second_runtime.release.set()
    owner = GpuProviderRecoveryOwner()
    base_first_plan = _plan("self", events)

    async def delayed_adopt(handler: ProviderRuntimeTerminalFailureSink) -> None:
        await base_first_plan.adopt(handler)
        first_adoption_started.set()
        await release_first_adoption.wait()

    first_plan = GpuProviderRecoveryChannelPlan(
        request=base_first_plan.request,
        start=base_first_plan.start,
        prepare=base_first_plan.prepare,
        abort=base_first_plan.abort,
        adopt=delayed_adopt,
    )
    factory_events: list[str] = []

    def first_factory() -> GpuProviderRecoveryExecution:
        factory_events.append("first")
        return _execution(first_runtime, events, channels=(first_plan,))

    def second_factory() -> GpuProviderRecoveryExecution:
        factory_events.append("second")
        return _execution(
            second_runtime,
            events,
            channels=(_plan("self", events),),
            reason="settings_restart",
        )

    first_task = asyncio.create_task(owner.recover(first_factory))
    await first_adoption_started.wait()
    second_task = asyncio.create_task(owner.recover(second_factory))
    await asyncio.sleep(0)

    assert factory_events == ["first"]

    release_first_adoption.set()
    await asyncio.gather(first_task, second_task)
    assert factory_events == ["first", "second"]


@pytest.mark.asyncio
async def test_empty_manual_retry_skips_but_settings_restart_releases_runtime() -> None:
    events: list[str] = []
    runtime = RuntimeStub(snapshot=_snapshot(channels=frozenset()), events=events)
    runtime.release.set()
    owner = GpuProviderRecoveryOwner()

    skipped = await owner.recover(
        lambda: _execution(
            runtime,
            events,
            channels=(),
            skip_if_no_channels=True,
        )
    )
    applied = await owner.recover(
        lambda: _execution(
            runtime,
            events,
            channels=(),
            reason="settings_restart",
        )
    )

    assert skipped.status == "skipped"
    assert applied.status == "applied"
    assert len(runtime.requests) == 1
    assert runtime.requests[0].channels == ()


@pytest.mark.asyncio
async def test_prepare_failure_aborts_prior_channel_and_propagates() -> None:
    events: list[str] = []
    runtime = RuntimeStub(snapshot=_snapshot(channels=frozenset()), events=events)
    runtime.release.set()

    with pytest.raises(RuntimeError, match="prepare failed"):
        await GpuProviderRecoveryOwner().recover(
            lambda: _execution(
                runtime,
                events,
                channels=(
                    _plan("self", events),
                    _plan(
                        "peer",
                        events,
                        prepare_failure=RuntimeError("prepare failed"),
                    ),
                ),
            )
        )

    assert events == ["prepare:self", "prepare:peer", "abort:self"]
    assert runtime.requests == []


def test_owner_declares_recovery_lifecycle_policy() -> None:
    assert GpuProviderRecoveryOwner().lifecycle_owner_snapshot() == {
        "owner": "GpuProviderRecoveryOwner",
        "resource_fields": ("_lock",),
        "operation_policy": (
            "serialize preparation, runtime recovery, consumer adoption and cleanup"
        ),
        "cancellation_policy": "propagate cancellation after prepared callback cleanup",
        "shutdown_policy": "no background task or external resource is retained",
    }
