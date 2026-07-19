from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeChannelSnapshot,
    ProviderRuntimeGpuSnapshot,
    ProviderRuntimeMutationResult,
)
from puripuly_heart.core.runtime.peer_channel import (
    PeerChannelRuntime,
    PeerChannelRuntimeState,
)
from tests.core.test_peer_channel_runtime import (
    DummySource,
    fake_run_audio_loop,
    make_local_qwen_runtime_config,
    make_peer_runtime_config,
)


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

    async def handle_peer_vad_event(self, event: object) -> None:
        _ = event


async def test_peer_capture_owner_requests_provider_and_never_constructs_it() -> None:
    hub = FakeOwnedPeerHub()
    sources: list[DummySource] = []
    config = make_peer_runtime_config()

    def forbidden_stt_factory(*_args, **_kwargs):
        raise AssertionError("legacy Peer provider factory was called")

    runtime = PeerChannelRuntime(
        hub=hub,
        clock=FakeClock(),
        stt_factory=forbidden_stt_factory,
        provider_request_factory=lambda peer_config, warmup: ProviderRuntimeBuildRequest(
            config=peer_config.backend,
            warmup=warmup,
            model_id=peer_config.model_id or peer_config.backend.model,
        ),
        source_factory=lambda _config: sources.append(DummySource()) or sources[-1],
        vad_factory=lambda _config, _model_path: "peer-vad",
        vad_model_resolver=lambda: Path("vad.onnx"),
        run_audio_loop=fake_run_audio_loop,
    )

    await runtime.apply_policy(config=config, desired_active=True)

    assert runtime.state is PeerChannelRuntimeState.RUNNING
    assert [request.config for request in hub.requests] == [config.backend]
    assert hub.start_calls == 1

    await runtime.apply_policy(config=config, desired_active=False, stop_mode="release")

    assert runtime.state is PeerChannelRuntimeState.STOPPED
    assert sources[0].close_calls == 1
    assert hub.local_asr_provider_runtime.releases == [("peer", "abort", None)]


async def test_peer_capture_owner_reuses_retained_local_qwen_provider() -> None:
    hub = FakeOwnedPeerHub()
    sources: list[DummySource] = []
    config = make_local_qwen_runtime_config()

    runtime = PeerChannelRuntime(
        hub=hub,
        clock=FakeClock(),
        stt_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy Peer provider factory was called")
        ),
        provider_request_factory=lambda peer_config, warmup: ProviderRuntimeBuildRequest(
            config=peer_config.backend,
            warmup=warmup,
            model_id=peer_config.model_id or peer_config.backend.model,
        ),
        source_factory=lambda _config: sources.append(DummySource()) or sources[-1],
        vad_factory=lambda _config, _model_path: "peer-vad",
        vad_model_resolver=lambda: Path("vad.onnx"),
        run_audio_loop=fake_run_audio_loop,
        idle_release_seconds=600.0,
    )

    await runtime.apply_policy(config=config, desired_active=True)
    await runtime.apply_policy(config=config, desired_active=False, stop_mode="retain")
    await runtime.apply_policy(config=config, desired_active=True)

    assert len(hub.requests) == 1
    assert hub.start_calls == 2
    assert hub.local_asr_provider_runtime.releases == [("peer", "drain", 600.0)]
    assert len(sources) == 2
    assert sources[0].close_calls == 1
    assert runtime.state is PeerChannelRuntimeState.RUNNING

    await runtime.close()

    assert sources[1].close_calls == 1
    assert hub.local_asr_provider_runtime.releases[-1] == ("peer", "abort", None)
