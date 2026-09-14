from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.channel_runtime import ChannelRuntime
from puripuly_heart.core.orchestrator.configuration import (
    TranslationRuntimeConfig,
    TranslationRuntimeConfigurationOwner,
)
from puripuly_heart.core.orchestrator.context import ContextResolver
from puripuly_heart.core.orchestrator.translation_diagnostics import (
    TranslationLatencyDiagnosticsOwner,
)
from puripuly_heart.core.orchestrator.translation_output_projection import TranslationUiMessage
from puripuly_heart.core.orchestrator.translation_request import (
    DirectTranslationRequest,
    TranslationProcessRequest,
    TranslationRequestOwner,
)
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.translation_backend import LlmTranslationBackend, TranslationBackend
from puripuly_heart.core.vrchat_scene import SceneSnapshotProvider, VrchatSceneSnapshot
from puripuly_heart.domain.models import ChannelId, Translation


@dataclass
class RecordingPresentation:
    messages: list[TranslationUiMessage] = field(default_factory=list)

    @staticmethod
    def chatbox_is_eligible(channel: ChannelId) -> bool:
        return channel == "self"

    async def publish_ui(self, message: TranslationUiMessage) -> None:
        self.messages.append(message)


@dataclass
class RecordingProvider:
    response: str = "translated"
    scene_calls: list[int | None] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
    ) -> Translation:
        self.scene_calls.append(scene_participant_count)
        return Translation(utterance_id=utterance_id, text=self.response)

    async def close(self) -> None:
        return


@dataclass
class SceneFixture:
    owner: TranslationRequestOwner
    configuration: TranslationRuntimeConfigurationOwner
    clock: FakeClock


@dataclass
class StaticSceneProvider:
    current: VrchatSceneSnapshot

    def snapshot(self) -> VrchatSceneSnapshot:
        return self.current


@dataclass
class SequenceSceneProvider:
    values: list[VrchatSceneSnapshot]
    position: int = 0

    def snapshot(self) -> VrchatSceneSnapshot:
        index = min(self.position, len(self.values) - 1)
        self.position += 1
        return self.values[index]


def build_scene_owner(
    scene_provider: SceneSnapshotProvider | None = None,
) -> SceneFixture:
    clock = FakeClock(100.0)
    configuration = TranslationRuntimeConfigurationOwner(
        TranslationRuntimeConfig(
            source_language="ko",
            target_language="en",
            peer_source_language="en",
            peer_target_language="ja",
            system_prompt="${sourceName}|${targetName}",
            translation_enabled=True,
            peer_translation_enabled=True,
            integrated_context_enabled=True,
        )
    )
    self_runtime = ChannelRuntime(channel="self")
    peer_runtime = ChannelRuntime(channel="peer")
    context_resolver = ContextResolver(
        clock=clock,
        config_snapshot=configuration.snapshot,
    )
    diagnostics = TranslationLatencyDiagnosticsOwner(
        clock=clock,
        config_snapshot=configuration.snapshot,
    )
    llm: TranslationBackend = LlmTranslationBackend(RecordingProvider())
    provider_runtime = ProviderRuntimeHandle(name="llm", provider=llm)
    presentation = RecordingPresentation()
    owner = TranslationRequestOwner(
        config_snapshot=configuration.snapshot,
        self_runtime=self_runtime,
        peer_runtime=peer_runtime,
        context_resolver=context_resolver,
        provider_runtime=provider_runtime,
        diagnostics=diagnostics,
        presentation=presentation,
        clock=clock,
        scene_provider=scene_provider,
    )
    return SceneFixture(
        owner=owner,
        configuration=configuration,
        clock=clock,
    )


def make_process_request(
    fixture: SceneFixture,
    text: str = "request",
    *,
    channel: ChannelId = "self",
) -> TranslationProcessRequest:
    utterance_id = uuid4()
    return TranslationProcessRequest(
        parent_utterance_id=utterance_id,
        utterance_id=utterance_id,
        sequence=0,
        text=text,
        channel=channel,
        source="Peer" if channel == "peer" else "Mic",
        target_language=fixture.owner.target_language_for(channel),
        context_policy="integrated_preferred",
        config_snapshot=fixture.configuration.snapshot(),
        publication_generation=0 if channel == "peer" else None,
        source_order=1 if channel == "peer" else None,
    )


def test_prepare_without_scene_provider_defaults_to_unavailable() -> None:
    fixture = build_scene_owner()
    prepared = fixture.owner.prepare("hello", channel="self")
    assert prepared.scene_snapshot == VrchatSceneSnapshot()
    assert prepared.scene_snapshot.status == "unavailable"
    assert prepared.scene_snapshot.participant_count is None
    assert prepared.system_prompt == "Korean|English"
    assert prepared.context == ""


def test_prepare_attaches_ready_snapshot_without_changing_prompt_or_context() -> None:
    scene = StaticSceneProvider(VrchatSceneSnapshot(status="ready", participant_count=3))
    fixture = build_scene_owner(scene_provider=scene)
    baseline = build_scene_owner()
    prepared = fixture.owner.prepare("hello", channel="self")
    reference = baseline.owner.prepare("hello", channel="self")
    assert prepared.scene_snapshot.status == "ready"
    assert prepared.scene_snapshot.participant_count == 3
    assert prepared.system_prompt == reference.system_prompt
    assert prepared.context == reference.context


def test_prepare_reads_fresh_snapshot_per_call_and_keeps_history_immutable() -> None:
    scene = SequenceSceneProvider(
        [
            VrchatSceneSnapshot(status="ready", participant_count=2),
            VrchatSceneSnapshot(status="ready", participant_count=5),
        ]
    )
    fixture = build_scene_owner(scene_provider=scene)
    first = fixture.owner.prepare("first", channel="self")
    second = fixture.owner.prepare("second", channel="self")
    assert first.scene_snapshot.participant_count == 2
    assert second.scene_snapshot.participant_count == 5
    assert first.scene_snapshot.participant_count == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.scene_snapshot.participant_count = 9
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.context = "mutated"


def test_admit_batch_reads_latest_snapshot_per_segment() -> None:
    scene = SequenceSceneProvider(
        [
            VrchatSceneSnapshot(status="ready", participant_count=2),
            VrchatSceneSnapshot(status="ready", participant_count=4),
        ]
    )
    fixture = build_scene_owner(scene_provider=scene)
    first = make_process_request(fixture, text="first segment")
    second = make_process_request(fixture, text="second segment")
    admitted = fixture.owner.admit((first, second))
    assert admitted[first.utterance_id].scene_snapshot.participant_count == 2
    assert admitted[second.utterance_id].scene_snapshot.participant_count == 4


@pytest.mark.parametrize("status", ["unavailable", "syncing", "degraded"])
def test_nonready_snapshot_withholds_count_and_keeps_messages_unchanged(status: str) -> None:
    unavailable_snapshots: dict[str, VrchatSceneSnapshot] = {
        "unavailable": VrchatSceneSnapshot(status="unavailable"),
        "syncing": VrchatSceneSnapshot(status="syncing"),
        "degraded": VrchatSceneSnapshot(status="degraded"),
    }
    scene = StaticSceneProvider(unavailable_snapshots[status])
    fixture = build_scene_owner(scene_provider=scene)
    baseline = build_scene_owner()
    prepared = fixture.owner.prepare("hello", channel="self")
    reference = baseline.owner.prepare("hello", channel="self")
    assert prepared.scene_snapshot.status == status
    assert prepared.scene_snapshot.participant_count is None
    assert prepared.system_prompt == reference.system_prompt
    assert prepared.context == reference.context


@pytest.mark.asyncio
async def test_translate_result_unchanged_with_ready_scene() -> None:
    scene = StaticSceneProvider(VrchatSceneSnapshot(status="ready", participant_count=3))
    fixture = build_scene_owner(scene_provider=scene)
    baseline = build_scene_owner()
    result = await fixture.owner.translate(
        DirectTranslationRequest(utterance_id=uuid4(), text="hello")
    )
    reference = await baseline.owner.translate(
        DirectTranslationRequest(utterance_id=uuid4(), text="hello")
    )
    assert result.text == reference.text
    assert result.target_language == reference.target_language
    assert result.source_text == reference.source_text


@dataclass
class CapturingBackend(TranslationBackend):
    counts: list[int | None] = field(default_factory=list)

    async def translate(self, request) -> Translation:  # type: ignore[no-untyped-def]
        self.counts.append(request.scene_participant_count)
        return Translation(utterance_id=request.utterance_id, text="captured")

    async def close(self) -> None:
        return None


def build_capturing_owner(
    scene_provider: SceneSnapshotProvider | None = None,
) -> tuple[SceneFixture, CapturingBackend]:
    fixture = build_scene_owner(scene_provider=scene_provider)
    backend = CapturingBackend()
    fixture.owner.provider_runtime = ProviderRuntimeHandle(name="llm", provider=backend)
    return fixture, backend


@pytest.mark.asyncio
async def test_direct_translate_emits_ready_count_from_prepared_snapshot() -> None:
    scene = StaticSceneProvider(VrchatSceneSnapshot(status="ready", participant_count=2))
    fixture, backend = build_capturing_owner(scene_provider=scene)
    result = await fixture.owner.translate(
        DirectTranslationRequest(utterance_id=uuid4(), text="hello")
    )
    assert result.text == "captured"
    assert backend.counts == [2]


@pytest.mark.asyncio
async def test_direct_translate_omits_count_when_nonready() -> None:
    scene = StaticSceneProvider(VrchatSceneSnapshot(status="syncing"))
    fixture, backend = build_capturing_owner(scene_provider=scene)
    await fixture.owner.translate(DirectTranslationRequest(utterance_id=uuid4(), text="hello"))
    assert backend.counts == [None]


@pytest.mark.asyncio
async def test_process_emits_ready_count_for_self_and_peer() -> None:
    scene = StaticSceneProvider(VrchatSceneSnapshot(status="ready", participant_count=4))
    fixture, backend = build_capturing_owner(scene_provider=scene)
    for channel in ("self", "peer"):
        request = make_process_request(fixture, text=f"hello {channel}", channel=channel)  # type: ignore[arg-type]
        await fixture.owner.process(request)
    assert backend.counts == [4, 4]


@pytest.mark.asyncio
async def test_process_uses_prepared_snapshot_immutably() -> None:
    scene = SequenceSceneProvider(
        [
            VrchatSceneSnapshot(status="ready", participant_count=2),
            VrchatSceneSnapshot(status="ready", participant_count=5),
        ]
    )
    fixture, backend = build_capturing_owner(scene_provider=scene)
    request = make_process_request(fixture, text="immutable")
    prepared = fixture.owner.prepare(
        request.text,
        channel=request.channel,
        detected_language=request.detected_language,
        target_language=request.target_language,
        context_policy=request.context_policy,
        config_snapshot=request.config_snapshot,
        parent_utterance_id=request.parent_utterance_id,
        target_index=request.target_index,
    )
    assert prepared.scene_snapshot.participant_count == 2
    await fixture.owner.process(request, prepared=prepared)
    assert backend.counts == [2]
    fresh = fixture.owner.prepare(
        request.text,
        channel=request.channel,
        detected_language=request.detected_language,
        target_language=request.target_language,
        context_policy=request.context_policy,
        config_snapshot=request.config_snapshot,
        parent_utterance_id=request.parent_utterance_id,
        target_index=request.target_index,
    )
    assert fresh.scene_snapshot.participant_count == 5


@pytest.mark.asyncio
async def test_process_without_prepared_reads_latest_snapshot() -> None:
    scene = SequenceSceneProvider(
        [
            VrchatSceneSnapshot(status="ready", participant_count=2),
            VrchatSceneSnapshot(status="ready", participant_count=6),
        ]
    )
    fixture, backend = build_capturing_owner(scene_provider=scene)
    first = make_process_request(fixture, text="first")
    second = make_process_request(fixture, text="second")
    await fixture.owner.process(first)
    await fixture.owner.process(second)
    assert backend.counts == [2, 6]


@pytest.mark.asyncio
async def test_ready_count_renders_exact_scene_block_in_outgoing_message() -> None:
    from puripuly_heart.providers.llm.messages import build_translation_user_message

    scene = StaticSceneProvider(VrchatSceneSnapshot(status="ready", participant_count=2))
    fixture, backend = build_capturing_owner(scene_provider=scene)
    prepared = fixture.owner.prepare("hello", channel="self")
    count = (
        prepared.scene_snapshot.participant_count
        if prepared.scene_snapshot.status == "ready"
        else None
    )
    assert count == 2
    user_message = build_translation_user_message(
        text="hello", context=prepared.context, scene_participant_count=count
    )
    assert user_message.startswith("<scene>\nPeople: 2\n</scene>\n\n")
    assert user_message.endswith("<input>\nhello\n</input>")
    assert backend.counts == []


@pytest.mark.asyncio
async def test_absent_scene_message_is_byte_identical() -> None:
    from puripuly_heart.providers.llm.messages import build_translation_user_message

    scene = StaticSceneProvider(VrchatSceneSnapshot(status="unavailable"))
    fixture, _ = build_capturing_owner(scene_provider=scene)
    baseline = build_scene_owner()
    prepared = fixture.owner.prepare("hello", channel="self")
    reference = baseline.owner.prepare("hello", channel="self")
    assert prepared.system_prompt == reference.system_prompt
    assert prepared.context == reference.context
    with_scene_none = build_translation_user_message(
        text="hello", context=prepared.context, scene_participant_count=None
    )
    without_scene = build_translation_user_message(text="hello", context=prepared.context)
    assert with_scene_none == without_scene
    assert with_scene_none == "<input>\nhello\n</input>"


@pytest.mark.asyncio
async def test_semaphore_provider_forwards_scene_count() -> None:
    import asyncio

    from puripuly_heart.core.llm.provider import SemaphoreLLMProvider

    inner = RecordingProvider()
    provider = SemaphoreLLMProvider(inner=inner, semaphore=asyncio.Semaphore(1))
    await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="system",
        source_language="en",
        target_language="ko",
        context="",
        scene_participant_count=3,
    )
    assert inner.scene_calls == [3]


@pytest.mark.asyncio
async def test_fallback_racing_forwards_same_scene_count_to_attempts() -> None:
    from puripuly_heart.core.llm.fallback_racing import FallbackRacingLLMProvider

    primary = RecordingProvider(response="primary")
    fallback = RecordingProvider(response="fallback")
    provider = FallbackRacingLLMProvider(primary=primary, fallback=fallback)
    result = await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="system",
        source_language="en",
        target_language="ko",
        context="",
        scene_participant_count=2,
    )
    assert result.text in ("primary", "fallback")
    assert primary.scene_calls == [2]


@pytest.mark.asyncio
async def test_lazy_factory_provider_forwards_scene_count() -> None:
    from puripuly_heart.app.wiring.wiring_llm_factory import _LazyFactoryLLMProvider

    inner = RecordingProvider()
    provider = _LazyFactoryLLMProvider(factory=lambda: inner)
    await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="system",
        source_language="en",
        target_language="ko",
        context="",
        scene_participant_count=5,
    )
    assert inner.scene_calls == [5]


@pytest.mark.asyncio
async def test_managed_openrouter_provider_forwards_scene_count() -> None:
    from puripuly_heart.core.openrouter.managed_openrouter_release import (
        ManagedOpenRouterLLMProvider,
        ManagedOpenRouterReleaseBehavior,
        ManagedOpenRouterReleaseResult,
    )

    class FakeIssueService:
        async def ensure_key_for_llm_start(self):  # type: ignore[no-untyped-def]
            return ManagedOpenRouterReleaseResult(
                behavior=ManagedOpenRouterReleaseBehavior.READY,
                message_key="managed_release.ready",
                api_key="managed-key",
                local_key_available=True,
                pending_issue=False,
            )

    delegate = RecordingProvider()
    provider = ManagedOpenRouterLLMProvider(
        release_service=FakeIssueService(),
        delegate_factory=lambda api_key: delegate,
    )
    await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="system",
        source_language="en",
        target_language="ko",
        context="",
        scene_participant_count=2,
    )
    assert delegate.scene_calls == [2]


@pytest.mark.asyncio
async def test_all_llm_outer_providers_forward_scene_to_inner_client() -> None:
    from puripuly_heart.providers.llm.deepseek import DeepSeekLLMProvider
    from puripuly_heart.providers.llm.gemini import GeminiLLMProvider
    from puripuly_heart.providers.llm.local_openai import LocalOpenAICompatibleLLMProvider
    from puripuly_heart.providers.llm.openrouter import OpenRouterLLMProvider
    from puripuly_heart.providers.llm.qwen import QwenLLMProvider
    from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider

    @dataclass
    class CapturingInner:
        calls: list[int | None] = field(default_factory=list)

        async def translate(
            self,
            *,
            text: str,
            system_prompt: str,
            source_language: str,
            target_language: str,
            context: str = "",
            scene_participant_count: int | None = None,
        ) -> str:
            self.calls.append(scene_participant_count)
            return "ok"

        async def close(self) -> None:
            return None

    inners: list[CapturingInner] = []
    providers = [
        QwenLLMProvider(api_key="k", client=(inners.append(CapturingInner()) or inners[-1])),
        AsyncQwenLLMProvider(api_key="k", client=(inners.append(CapturingInner()) or inners[-1])),
        OpenRouterLLMProvider(api_key="k", client=(inners.append(CapturingInner()) or inners[-1])),
        DeepSeekLLMProvider(api_key="k", client=(inners.append(CapturingInner()) or inners[-1])),
        GeminiLLMProvider(api_key="k", client=(inners.append(CapturingInner()) or inners[-1])),
        LocalOpenAICompatibleLLMProvider(client=(inners.append(CapturingInner()) or inners[-1])),
    ]
    for provider in providers:
        result = await provider.translate(
            utterance_id=uuid4(),
            text="hello",
            system_prompt="system",
            source_language="en",
            target_language="ko",
            context="prior",
            scene_participant_count=2,
        )
        assert result.text == "ok"
    assert [inner.calls for inner in inners] == [[2]] * len(inners)


@pytest.mark.asyncio
async def test_managed_gemma_provider_embeds_scene_in_user_message() -> None:
    from puripuly_heart.core.local_translation.runtime import (
        ManagedGemmaMetrics,
        ManagedGemmaResponse,
    )
    from puripuly_heart.providers.llm.managed_gemma import ManagedGemmaLLMProvider

    @dataclass
    class CapturingRuntime:
        user_messages: list[str] = field(default_factory=list)

        async def translate(self, **kwargs):  # type: ignore[no-untyped-def]
            self.user_messages.append(kwargs["user_message"])
            return ManagedGemmaResponse(
                text="ok",
                metrics=ManagedGemmaMetrics(1, 1, 1, 1.0, 1.0, 1.0),
            )

        async def release(self) -> None:
            return None

    runtime = CapturingRuntime()
    provider = ManagedGemmaLLMProvider(runtime=runtime, backend="cpu")  # type: ignore[arg-type]
    await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="system",
        source_language="en",
        target_language="ko",
        context="prior",
        scene_participant_count=2,
    )
    assert runtime.user_messages == [
        "<scene>\nPeople: 2\n</scene>\n\n<context>\nprior\n</context>\n\n<input>\nhello\n</input>"
    ]
    await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="system",
        source_language="en",
        target_language="ko",
        context="prior",
        scene_participant_count=None,
    )
    assert runtime.user_messages[1] == "<context>\nprior\n</context>\n\n<input>\nhello\n</input>"


def test_gemma_prefix_identity_ignores_scene_people() -> None:
    from puripuly_heart.core.local_translation.assets import e4b_gemma_spec
    from puripuly_heart.core.local_translation.runtime import _prefix_identity

    spec = e4b_gemma_spec()
    first = _prefix_identity(
        spec=spec, system_prompt="system", source_language="en", target_language="ko"
    )
    second = _prefix_identity(
        spec=spec, system_prompt="system", source_language="en", target_language="ko"
    )
    assert first == second


@pytest.mark.asyncio
async def test_httpx_clients_embed_scene_in_request_body() -> None:
    from puripuly_heart.providers.llm.deepseek import HttpxDeepSeekClient
    from puripuly_heart.providers.llm.local_openai import HttpxLocalOpenAIClient
    from puripuly_heart.providers.llm.openrouter import HttpxOpenRouterClient
    from puripuly_heart.providers.llm.qwen_async import HttpxQwenClient

    bodies = [
        HttpxDeepSeekClient(api_key="k", model="m")._build_request_body(
            text="hello",
            system_prompt="system",
            source_language="en",
            target_language="ko",
            context="prior",
            scene_participant_count=2,
        ),
        HttpxOpenRouterClient(api_key="k", model="m")._build_request_body(
            text="hello",
            system_prompt="system",
            source_language="en",
            target_language="ko",
            context="prior",
            scene_participant_count=2,
        ),
        HttpxQwenClient(api_key="k", model="m")._build_request_body(
            text="hello",
            system_prompt="system",
            source_language="en",
            target_language="ko",
            context="prior",
            scene_participant_count=2,
        ),
        HttpxLocalOpenAIClient(model="m")._build_request_body(
            text="hello",
            system_prompt="system",
            source_language="en",
            target_language="ko",
            context="prior",
            scene_participant_count=2,
        ),
    ]
    for body in bodies:
        content = body["messages"][1]["content"]
        assert isinstance(content, str)
        assert content.startswith("<scene>\nPeople: 2\n</scene>\n\n")
        assert "<context>\nprior\n</context>" in content
        assert content.endswith("<input>\nhello\n</input>")
        assert body["messages"][0]["content"] == "system"
