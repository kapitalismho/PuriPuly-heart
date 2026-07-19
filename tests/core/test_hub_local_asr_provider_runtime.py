from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import uuid4

from puripuly_heart.config.resolved import (
    ResolvedCredentialRequirement,
    ResolvedSTTConfig,
)
from puripuly_heart.core.local_asr_provider_runtime import (
    LocalASRProviderRuntimeCallbacks,
    LocalASRProviderRuntimeSnapshot,
    ProviderRuntimeBuildRequest,
    ProviderRuntimeChannelSnapshot,
    ProviderRuntimeGpuSnapshot,
    ProviderRuntimeMutationResult,
)
from puripuly_heart.core.orchestrator.hub import ClientHub
from puripuly_heart.core.vad.gating import SpeechEnd


@dataclass
class FakeOscQueue:
    messages: list[object]

    def enqueue(self, message: object) -> None:
        self.messages.append(message)

    def send_typing(self, enabled: bool) -> None:
        _ = enabled

    def set_typing_reason(self, reason: str, active: bool) -> None:
        _ = reason, active

    def send_immediate(self, text: str) -> bool:
        _ = text
        return True

    def process_due(self) -> None:
        return


class FakeOwnedRuntime:
    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self.vad_events: list[tuple[str, object]] = []
        self.commits: list[str] = []
        self.replacements: list[ProviderRuntimeBuildRequest] = []
        self._channels = {
            channel: ProviderRuntimeChannelSnapshot(
                channel=channel,
                provider_id=None,
                model_id=None,
                phase="inactive",
                generation=0,
                pending_handoff=False,
                has_resources=False,
            )
            for channel in ("self", "peer")
        }

    @property
    def snapshot(self) -> LocalASRProviderRuntimeSnapshot:
        return LocalASRProviderRuntimeSnapshot(
            channels=(self._channels["self"], self._channels["peer"]),
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

    @property
    def diagnostics(self) -> tuple[object, ...]:
        return ()

    async def start(self) -> None:
        self.started += 1

    async def replace_provider(self, request, *, start, on_terminal_failure=None):
        _ = on_terminal_failure
        self.replacements.append(request)
        self._channels[request.channel] = replace(
            self._channels[request.channel],
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

    async def handle_vad_event(self, channel, event) -> None:
        self.vad_events.append((channel, event))

    async def commit_handoff(self, channel) -> None:
        self.commits.append(channel)

    async def close(self) -> None:
        self.closed += 1


class FakeOwnedRuntimeFactory:
    def __init__(self) -> None:
        self.runtime = FakeOwnedRuntime()
        self.callbacks: LocalASRProviderRuntimeCallbacks | None = None

    def create(self, callbacks: LocalASRProviderRuntimeCallbacks) -> FakeOwnedRuntime:
        self.callbacks = callbacks
        return self.runtime


def _request() -> ProviderRuntimeBuildRequest:
    return ProviderRuntimeBuildRequest(
        config=ResolvedSTTConfig(
            channel="self",
            source_language="ko",
            provider="deepgram",
            model="nova-3",
            endpoint=None,
            region=None,
            credential=ResolvedCredentialRequirement(
                source="none",
                required=False,
                reference=None,
            ),
            input_host_api=None,
            input_device=None,
            output_device=None,
            sample_rate_hz=16000,
            channels=1,
            ring_buffer_ms=500,
            drain_timeout_s=1.5,
            vad_speech_threshold=0.5,
            vad_hangover_ms=500,
            vad_pre_roll_ms=300,
            low_latency_enabled=True,
            low_latency_merge_gap_ms=600,
            low_latency_spec_retry_max=1,
            custom_vocabulary_enabled=False,
            custom_terms={},
            provider_options={},
        )
    )


async def test_hub_delegates_self_provider_execution_and_close_to_one_owner() -> None:
    factory = FakeOwnedRuntimeFactory()
    hub = ClientHub(
        stt=None,
        llm=None,
        osc=FakeOscQueue(messages=[]),
        local_asr_provider_runtime_factory=factory,
    )
    request = _request()

    result = await hub.replace_stt_provider_request(request, start=False)
    await hub.start()
    event = SpeechEnd(uuid4())
    await hub.handle_vad_event(event)
    await hub.stop()

    assert factory.callbacks is not None
    assert result.status == "applied"
    assert factory.runtime.replacements == [request]
    assert factory.runtime.started == 1
    assert factory.runtime.vad_events == [("self", event)]
    assert factory.runtime.commits == ["self"]
    assert factory.runtime.closed == 1
    assert set(hub.provider_runtime_handles) == {"llm"}
