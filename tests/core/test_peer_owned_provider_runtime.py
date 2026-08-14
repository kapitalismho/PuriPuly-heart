from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeChannelSnapshot,
    ProviderRuntimeGpuSnapshot,
    ProviderRuntimeMutationResult,
)

from puripuly_heart.app.wiring import compose_peer_capture_session_owner
from puripuly_heart.config.resolved import (
    CREDENTIAL_SOURCE_SECRET_STORE,
    ResolvedCredentialRequirement,
    ResolvedSTTConfig,
)
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.peer_capture import (
    PeerCaptureAdmission,
    PeerCaptureAdmissionStatus,
    PeerCaptureLanguageFacts,
    PeerCaptureResolvedTarget,
    PeerCaptureSessionConfig,
    PeerCaptureSessionState,
    PeerCaptureTargetIntent,
    PeerCaptureTargetResolution,
    PeerCaptureTargetStatus,
)


@dataclass(slots=True)
class DummySource:
    close_calls: int = 0

    async def close(self) -> None:
        self.close_calls += 1


class Admission:
    async def admit(self, _config) -> PeerCaptureAdmission:
        return PeerCaptureAdmission(PeerCaptureAdmissionStatus.ADMITTED)


class Resolver:
    async def resolve(self, target) -> PeerCaptureTargetResolution:
        return PeerCaptureTargetResolution(
            PeerCaptureTargetStatus.RESOLVED,
            target=PeerCaptureResolvedTarget(intent=target),
        )


class Sink:
    async def handle_vad_event(self, _event) -> None:
        return None


class FakePeerRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.releases: list[tuple[str, str, float | None]] = []
        self.requests: list[ProviderRuntimeBuildRequest] = []
        self.start_calls = 0
        self._peer = ProviderRuntimeChannelSnapshot(
            channel="peer",
            provider_id=None,
            model_id=None,
            phase="inactive",
            generation=0,
            pending_handoff=False,
            has_resources=False,
        )

    @property
    def snapshot(self) -> LocalASRProviderRuntimeSnapshot:
        return LocalASRProviderRuntimeSnapshot(
            channels=(
                ProviderRuntimeChannelSnapshot(
                    channel="self",
                    provider_id=None,
                    model_id=None,
                    phase="inactive",
                    generation=0,
                    pending_handoff=False,
                    has_resources=False,
                ),
                self._peer,
            ),
            gpu=ProviderRuntimeGpuSnapshot(
                phase="inactive",
                devices=(),
                active_channels=frozenset(),
                pending_count=0,
                worker_pid=None,
                configured_device_id=None,
                model_resident=False,
                retry_required=False,
                failure_code=None,
            ),
        )

    async def replace_provider(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        start: bool,
        on_terminal_failure,
    ) -> ProviderRuntimeMutationResult:
        _ = on_terminal_failure
        self.requests.append(request)
        self._peer = replace(
            self._peer,
            provider_id=request.provider_id,
            model_id=request.model_id,
            phase="running" if start else "dormant",
            has_resources=True,
        )
        return ProviderRuntimeMutationResult(
            status="applied",
            request=request,
            previous_provider_id=None,
            snapshot=self.snapshot,
        )

    async def handoff_provider(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        start: bool,
        on_terminal_failure,
    ) -> ProviderRuntimeMutationResult:
        return await self.replace_provider(
            request,
            start=start,
            on_terminal_failure=on_terminal_failure,
        )

    async def cancel_handoff(self, channel: str) -> bool:
        _ = channel
        return False

    async def start_channel(self, channel: str) -> None:
        _ = channel
        self.start_calls += 1

    async def warmup_channel(self, channel: str) -> None:
        _ = channel

    async def reconfigure_channel(self, channel: str, options: object) -> None:
        _ = channel, options

    async def release_channel(
        self,
        channel: str,
        *,
        mode: str,
        release_backend_after: float | None = None,
    ) -> None:
        self.events.append(f"release:{channel}:{mode}")
        self.releases.append((channel, mode, release_backend_after))
        if mode == "abort":
            self._peer = replace(
                self._peer,
                provider_id=None,
                model_id=None,
                phase="inactive",
                has_resources=False,
            )


class ChannelReset:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.pending_turn = False
        self.channel_runtime_state = False
        self.logical_turn_state = False
        self.latency_state = False

    def mark_pending_state(self) -> None:
        self.pending_turn = True
        self.channel_runtime_state = True
        self.logical_turn_state = True
        self.latency_state = True

    async def reset_provider_channel(self, channel: str) -> None:
        self.pending_turn = False
        self.channel_runtime_state = False
        self.logical_turn_state = False
        self.latency_state = False
        self.events.append(f"reset:{channel}")

    @property
    def state_cleared(self) -> bool:
        return not any(
            (
                self.pending_turn,
                self.channel_runtime_state,
                self.logical_turn_state,
                self.latency_state,
            )
        )


def make_config(provider: str) -> PeerCaptureSessionConfig:
    backend = ResolvedSTTConfig(
        channel="peer",
        provider=provider,
        source_language="ko",
        model="nova-3" if provider == "soniox" else "local",
        endpoint=None,
        region=None,
        credential=ResolvedCredentialRequirement(
            source=CREDENTIAL_SOURCE_SECRET_STORE,
            required=provider == "soniox",
            reference="soniox:stt" if provider == "soniox" else None,
        ),
        input_host_api=None,
        input_device=None,
        output_device="",
        sample_rate_hz=16000,
        channels=1,
        ring_buffer_ms=500,
        drain_timeout_s=2.0,
        vad_speech_threshold=0.6,
        vad_hangover_ms=900,
        vad_pre_roll_ms=500,
        low_latency_enabled=True,
        low_latency_merge_gap_ms=600,
        low_latency_spec_retry_max=10,
        custom_vocabulary_enabled=False,
        custom_terms={},
        provider_options={},
    )
    target = PeerCaptureTargetIntent(kind="default_output_device")
    return PeerCaptureSessionConfig(
        provider_id=provider,
        provider_signature=(provider, backend.model),
        runtime_signature=(provider, backend.model, target),
        capture_signature=(target, 16000),
        capture_target=target,
        language=PeerCaptureLanguageFacts("manual", "ko"),
        target_sample_rate_hz=16000,
        vad_speech_threshold=0.6,
        vad_hangover_ms=900,
        vad_pre_roll_ms=500,
        provider_context=backend,
        model_id=backend.model,
        local_provider=provider == "local_qwen",
        release_backend_after=600.0 if provider == "local_qwen" else None,
    )


def make_runtime(
    provider_runtime: FakePeerRuntime,
    channel_reset: ChannelReset,
    sources: list[DummySource],
):
    async def run_loop(**_kwargs) -> None:
        await asyncio.Event().wait()

    return compose_peer_capture_session_owner(
        provider_runtime=provider_runtime,
        channel_reset=channel_reset,
        admission=Admission(),
        target_resolver=Resolver(),
        clock=FakeClock(),
        provider_request_factory=lambda config, warmup: ProviderRuntimeBuildRequest(
            config=config.provider_context,
            warmup=warmup,
            model_id=config.model_id,
        ),
        source_factory=lambda _config, _target: sources.append(DummySource()) or sources[-1],
        vad_factory=lambda _config: "peer-vad",
        run_audio_loop=run_loop,
        vad_sink=Sink(),
    )


async def test_peer_capture_owner_requests_provider_and_never_constructs_it() -> None:
    events: list[str] = []
    channel_reset = ChannelReset(events)
    provider_runtime = FakePeerRuntime(events)
    sources: list[DummySource] = []
    config = make_config("soniox")
    runtime = make_runtime(provider_runtime, channel_reset, sources)

    await runtime.apply_intent(config, enabled=True)
    assert runtime.state is PeerCaptureSessionState.RUNNING
    assert [request.config for request in provider_runtime.requests] == [config.provider_context]
    assert provider_runtime.start_calls == 1

    events.clear()
    channel_reset.mark_pending_state()
    await runtime.apply_intent(config, enabled=False, stop_mode="release")
    assert runtime.state is PeerCaptureSessionState.STOPPED
    assert sources[0].close_calls == 1
    assert channel_reset.state_cleared
    assert events == ["reset:peer", "release:peer:abort"]
    assert provider_runtime.releases == [("peer", "abort", None)]


async def test_peer_capture_owner_reuses_retained_local_qwen_provider() -> None:
    events: list[str] = []
    channel_reset = ChannelReset(events)
    provider_runtime = FakePeerRuntime(events)
    sources: list[DummySource] = []
    config = make_config("local_qwen")
    runtime = make_runtime(provider_runtime, channel_reset, sources)

    await runtime.apply_intent(config, enabled=True)
    events.clear()
    await runtime.apply_intent(config, enabled=False, stop_mode="retain")
    await runtime.apply_intent(config, enabled=True)

    assert len(provider_runtime.requests) == 1
    assert provider_runtime.start_calls == 2
    assert provider_runtime.releases == [("peer", "drain", 600.0)]
    assert "reset:peer" not in events
    assert events[0] == "release:peer:drain"
    assert len(sources) == 2
    assert sources[0].close_calls == 1
    assert runtime.state is PeerCaptureSessionState.RUNNING

    await runtime.close()
    assert sources[1].close_calls == 1
    assert provider_runtime.releases[-1] == ("peer", "abort", None)
