from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from puripuly_heart.app.wiring import compose_peer_capture_session_owner
from puripuly_heart.config.resolved import (
    CREDENTIAL_SOURCE_SECRET_STORE,
    ResolvedCredentialRequirement,
    ResolvedSTTConfig,
)
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeChannelSnapshot,
    ProviderRuntimeGpuSnapshot,
    ProviderRuntimeMutationResult,
)
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


class FakePeerOwner:
    def __init__(self) -> None:
        self.releases: list[tuple[str, str, float | None]] = []
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

    def attach(self, request: ProviderRuntimeBuildRequest, *, start: bool) -> None:
        self._peer = replace(
            self._peer,
            provider_id=request.provider_id,
            model_id=request.model_id,
            phase="running" if start else "dormant",
            has_resources=True,
        )

    async def release_channel(
        self,
        channel: str,
        *,
        mode: str,
        release_backend_after: float | None = None,
    ) -> None:
        self.releases.append((channel, mode, release_backend_after))
        if mode == "abort":
            self._peer = replace(
                self._peer,
                provider_id=None,
                model_id=None,
                phase="inactive",
                has_resources=False,
            )


class FakeOwnedPeerHub:
    def __init__(self) -> None:
        self.local_asr_provider_runtime = FakePeerOwner()
        self.requests: list[ProviderRuntimeBuildRequest] = []
        self.start_calls = 0

    async def replace_peer_stt_provider_request(
        self,
        request: ProviderRuntimeBuildRequest,
        *,
        start: bool,
        on_terminal_failure,
    ) -> ProviderRuntimeMutationResult:
        _ = on_terminal_failure
        self.requests.append(request)
        self.local_asr_provider_runtime.attach(request, start=start)
        return ProviderRuntimeMutationResult(
            status="applied",
            request=request,
            previous_provider_id=None,
            snapshot=self.local_asr_provider_runtime.snapshot,
        )

    async def start_peer_stt_provider_ingress(self) -> None:
        self.start_calls += 1

    async def warmup_stt_channel(self, _channel: str) -> None:
        return None


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


def make_runtime(hub: FakeOwnedPeerHub, sources: list[DummySource]):
    async def run_loop(**_kwargs) -> None:
        await asyncio.Event().wait()

    return compose_peer_capture_session_owner(
        hub=hub,
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
    hub = FakeOwnedPeerHub()
    sources: list[DummySource] = []
    config = make_config("soniox")
    runtime = make_runtime(hub, sources)

    await runtime.apply_intent(config, enabled=True)
    assert runtime.state is PeerCaptureSessionState.RUNNING
    assert [request.config for request in hub.requests] == [config.provider_context]
    assert hub.start_calls == 1

    await runtime.apply_intent(config, enabled=False, stop_mode="release")
    assert runtime.state is PeerCaptureSessionState.STOPPED
    assert sources[0].close_calls == 1
    assert hub.local_asr_provider_runtime.releases == [("peer", "abort", None)]


async def test_peer_capture_owner_reuses_retained_local_qwen_provider() -> None:
    hub = FakeOwnedPeerHub()
    sources: list[DummySource] = []
    config = make_config("local_qwen")
    runtime = make_runtime(hub, sources)

    await runtime.apply_intent(config, enabled=True)
    await runtime.apply_intent(config, enabled=False, stop_mode="retain")
    await runtime.apply_intent(config, enabled=True)

    assert len(hub.requests) == 1
    assert hub.start_calls == 2
    assert hub.local_asr_provider_runtime.releases == [("peer", "drain", 600.0)]
    assert len(sources) == 2
    assert sources[0].close_calls == 1
    assert runtime.state is PeerCaptureSessionState.RUNNING

    await runtime.close()
    assert sources[1].close_calls == 1
    assert hub.local_asr_provider_runtime.releases[-1] == ("peer", "abort", None)
