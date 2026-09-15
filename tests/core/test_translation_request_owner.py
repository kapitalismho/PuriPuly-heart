from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.http_extensions import parse_http_extension
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
    StaleProviderCompletion,
    TranslationProcessRequest,
    TranslationRequestOwner,
)
from puripuly_heart.core.orchestrator.translation_turn import _final_transcript_segments
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.core.stt.backend import STTProviderTurnIdentity, STTProviderTurnTerminal
from puripuly_heart.core.stt.scoped_normalizer import STTScopedTurnNormalizer
from puripuly_heart.core.translation_backend import LlmTranslationBackend, TranslationBackend
from puripuly_heart.domain.models import ChannelId, FinalLanguageRun, Transcript, Translation
from puripuly_heart.providers.extensions.http_extension_backend import (
    HttpExtensionTranslationBackend,
)
from puripuly_heart.providers.llm.openrouter import (
    HttpxOpenRouterClient,
    OpenRouterLLMProvider,
)


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
    calls: list[dict[str, object]] = field(default_factory=list)
    failure: Exception | None = None

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
        max_output_tokens: int | None = None,
    ) -> Translation:
        self.calls.append(
            {
                "text": text,
                "system_prompt": system_prompt,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
                "max_output_tokens": max_output_tokens,
            }
        )
        if self.failure is not None:
            raise self.failure
        return Translation(utterance_id=utterance_id, text=self.response)

    async def close(self) -> None:
        return


@dataclass
class BlockingProvider(RecordingProvider):
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def translate(self, **kwargs) -> Translation:
        self.entered.set()
        await self.release.wait()
        return await super().translate(**kwargs)


@dataclass
class BlockingHttpClient:
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False

    async def post(self, _url: str, **_kwargs: object) -> SimpleNamespace:
        self.entered.set()
        await self.release.wait()
        return SimpleNamespace(status_code=200, text="Hola")

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class OwnerFixture:
    owner: TranslationRequestOwner
    configuration: TranslationRuntimeConfigurationOwner
    provider_runtime: ProviderRuntimeHandle
    self_runtime: ChannelRuntime
    peer_runtime: ChannelRuntime
    presentation: RecordingPresentation
    clock: FakeClock


def build_owner(provider: object | None = None) -> OwnerFixture:
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
    backend = (
        provider
        if provider is None or isinstance(provider, TranslationBackend)
        else LlmTranslationBackend(provider)
    )
    provider_runtime = ProviderRuntimeHandle(name="llm", provider=backend)
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
    )
    return OwnerFixture(
        owner=owner,
        configuration=configuration,
        provider_runtime=provider_runtime,
        self_runtime=self_runtime,
        peer_runtime=peer_runtime,
        presentation=presentation,
        clock=clock,
    )


def process_request(
    fixture: OwnerFixture,
    *,
    channel: ChannelId = "self",
    detected_language: str | None = None,
) -> TranslationProcessRequest:
    utterance_id = uuid4()
    return TranslationProcessRequest(
        parent_utterance_id=utterance_id,
        utterance_id=utterance_id,
        sequence=0,
        text="request",
        channel=channel,
        source="Peer" if channel == "peer" else "Mic",
        target_language=fixture.owner.target_language_for(channel),
        context_policy="integrated_preferred",
        detected_language=detected_language,
        config_snapshot=fixture.configuration.snapshot(),
        publication_generation=0 if channel == "peer" else None,
        source_order=1 if channel == "peer" else None,
    )


def request_from_normalized_peer_terminal(
    fixture: OwnerFixture,
    *,
    text: str,
    final_language_runs: tuple[FinalLanguageRun, ...],
) -> TranslationProcessRequest:
    utterance_id = uuid4()
    identity = STTProviderTurnIdentity(
        segment=AudioSegmentIdentity(1, 1, utterance_id, 1),
        provider_epoch_id="epoch",
        provider_turn_id="turn",
        settings_scope=("provider", fixture.configuration.snapshot().revision),
    )
    terminal = STTScopedTurnNormalizer(identity).apply_terminal(
        STTProviderTurnTerminal(
            identity=identity,
            outcome="final",
            text=text,
            final_language_runs=final_language_runs,
            text_authority="authoritative",
        )
    )
    (segment,) = _final_transcript_segments(
        Transcript(
            utterance_id=utterance_id,
            text=terminal.text,
            is_final=True,
            channel="peer",
            final_language_runs=terminal.final_language_runs,
            publication_generation=0,
            source_order=1,
        ),
        split_speakers=True,
    )
    return replace(
        process_request(
            fixture,
            channel="peer",
            detected_language=segment.language or None,
        ),
        parent_utterance_id=utterance_id,
        text=segment.text,
    )


def peer_requests(fixture: OwnerFixture) -> tuple[TranslationProcessRequest, ...]:
    parent_id = uuid4()
    return tuple(
        TranslationProcessRequest(
            parent_utterance_id=parent_id,
            utterance_id=uuid4(),
            sequence=sequence,
            text=text,
            channel="peer",
            source="Peer",
            target_language="ja",
            context_policy="integrated_preferred",
            config_snapshot=fixture.configuration.snapshot(),
            detected_language=language,
            speaker_id=speaker,
            speaker_session_scope="soniox-session",
            publication_generation=0,
            source_order=1,
            parent_output_count=3,
        )
        for sequence, (text, language, speaker) in enumerate(
            (("one ", "en", "1"), ("둘째 ", "ko", None), ("three", "en", "1"))
        )
    )


@pytest.mark.asyncio
async def test_peer_plain_text_responses_preserve_each_segments_translation() -> None:
    translations = {"one ": "一", "둘째 ": "二", "three": "三"}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user_message = body["messages"][1]["content"]
        source = user_message.split("<input>\n", 1)[1].split("\n</input>", 1)[0]
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": translations[source]}}]
            },
        )

    client = HttpxOpenRouterClient(api_key="test-key", model="test/model")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenRouterLLMProvider(api_key="test-key", client=client)
    fixture = build_owner(provider)
    requests = peer_requests(fixture)
    prepared = fixture.owner.admit_peer(requests)
    try:
        results = await asyncio.gather(
            *(
                fixture.owner.process(request, prepared=prepared[request.utterance_id])
                for request in requests
            )
        )
    finally:
        await provider.close()

    assert [
        (result.output.source_text, result.output.translation.text)
        for result in results
        if result.output is not None and result.output.translation is not None
    ] == [("one ", "一"), ("둘째 ", "二"), ("three", "三")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("peer_source_mode", "expected_source_language"),
    (("manual", "en"), ("auto", "auto")),
)
async def test_normalized_peer_terminal_without_language_metadata_translates(
    peer_source_mode: str,
    expected_source_language: str,
) -> None:
    provider = RecordingProvider(response="翻訳済み")
    fixture = build_owner(provider)
    fixture.configuration.replace(
        replace(
            fixture.configuration.snapshot().value,
            peer_source_mode=peer_source_mode,
        )
    )
    request = request_from_normalized_peer_terminal(
        fixture,
        text="hello",
        final_language_runs=(),
    )

    result = await fixture.owner.process(request)

    assert result.outcome == "translated"
    assert result.output is not None
    assert result.output.translation is not None
    assert result.output.translation.text == "翻訳済み"
    assert result.output.translation.source_language == expected_source_language
    assert provider.calls[0]["source_language"] == expected_source_language


@pytest.mark.asyncio
async def test_normalized_peer_terminal_repairs_corrupt_language_metadata_to_unspecified() -> None:
    provider = RecordingProvider(response="修復済み")
    fixture = build_owner(provider)
    request = request_from_normalized_peer_terminal(
        fixture,
        text="hello",
        final_language_runs=(FinalLanguageRun("wrong text", "en"),),
    )

    result = await fixture.owner.process(request)

    assert result.outcome == "translated"
    assert result.output is not None
    assert result.output.translation is not None
    assert result.output.translation.text == "修復済み"
    assert result.output.translation.source_language == "en"
    assert provider.calls[0]["source_language"] == "en"


@pytest.mark.asyncio
async def test_normalized_peer_terminal_preserves_unsupported_provider_language() -> None:
    provider = RecordingProvider()
    fixture = build_owner(provider)
    request = request_from_normalized_peer_terminal(
        fixture,
        text="hello",
        final_language_runs=(FinalLanguageRun("hello", "unsupported"),),
    )

    result = await fixture.owner.process(request)

    assert result.outcome == "source_only"
    assert result.output is not None
    assert result.output.failure_code == "unsupported_source_language"
    assert result.output.source_text == "hello"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_peer_unsupported_segments_remain_reference_without_blocking_translation() -> None:
    provider = RecordingProvider(response="翻訳")
    fixture = build_owner(provider)
    requests = tuple(
        replace(request, detected_language=language)
        for request, language in zip(
            peer_requests(fixture), ("unknown", "ko", "unsupported"), strict=True
        )
    )
    prepared = fixture.owner.admit_peer(requests)
    results = [
        await fixture.owner.process(request, prepared=prepared.get(request.utterance_id))
        for request in requests
    ]

    assert [result.outcome for result in results] == ["source_only", "translated", "source_only"]
    assert [result.output.source_text for result in results] == [
        request.text for request in requests
    ]
    assert [result.output.failure_code for result in results] == [
        "unsupported_source_language",
        None,
        "unsupported_source_language",
    ]
    assert results[1].output.translation.text == "翻訳"
    assert len(provider.calls) == 1
    assert all(request.text in provider.calls[0]["context"] for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [False, True])
async def test_ineligible_peer_parent_does_not_call_provider(disabled: bool) -> None:
    provider = RecordingProvider()
    fixture = build_owner(provider)
    if disabled:
        fixture.configuration.transform(
            lambda current: replace(current, peer_translation_enabled=False)
        )
    requests = tuple(
        replace(request, detected_language="en" if disabled else "unsupported")
        for request in peer_requests(fixture)
    )
    prepared = fixture.owner.admit_peer(requests)
    results = [
        await fixture.owner.process(request, prepared=prepared.get(request.utterance_id))
        for request in requests
    ]
    assert [result.outcome for result in results] == ["source_only"] * 3
    assert [result.output.source_text for result in results] == [
        request.text for request in requests
    ]
    assert provider.calls == []


def test_peer_admission_records_history_once_and_separates_current_reference() -> None:
    fixture = build_owner(RecordingProvider())
    fixture.peer_runtime.remember_context(
        "earlier", timestamp=99.0, source_language="", target_language="ja"
    )
    requests = peer_requests(fixture)
    prepared = fixture.owner.admit_peer(requests)
    for item in prepared.values():
        assert item.context.count("earlier") == 1
        assert all(
            item.context.count(json.dumps(request.text, ensure_ascii=False)) == 1
            for request in requests
        )
    assert [entry.text for entry in fixture.peer_runtime.translation_history] == [
        "earlier",
        "one",
        "둘째",
        "three",
    ]
    assert [prepared[request.utterance_id].source_language for request in requests] == [
        "en",
        "ko",
        "en",
    ]


@pytest.mark.asyncio
async def test_prepared_peer_segments_reject_replacement_before_and_during_execution() -> None:
    old_provider = BlockingProvider()
    fixture = build_owner(old_provider)
    requests = peer_requests(fixture)
    prepared = fixture.owner.admit_peer(requests)
    task = asyncio.create_task(
        fixture.owner.process(requests[0], prepared=prepared[requests[0].utterance_id])
    )
    await old_provider.entered.wait()
    replacement = RecordingProvider()
    await fixture.provider_runtime.replace_provider(LlmTranslationBackend(replacement), start=False)
    old_provider.release.set()
    results = [await task]
    results.extend(
        [
            await fixture.owner.process(request, prepared=prepared[request.utterance_id])
            for request in requests[1:]
        ]
    )
    assert [result.output.failure_code for result in results] == ["stale_provider_completion"] * 3
    assert replacement.calls == []
    assert fixture.presentation.messages == []


def test_clear_context_clears_both_channels() -> None:
    fixture = build_owner()
    fixture.self_runtime.remember_context("self context", timestamp=fixture.clock.now())
    fixture.peer_runtime.remember_context("peer context", timestamp=fixture.clock.now())

    fixture.owner.clear_context()

    assert fixture.self_runtime.translation_history == []
    assert fixture.peer_runtime.translation_history == []


def test_prepare_uses_detected_language_and_integrated_peer_context() -> None:
    fixture = build_owner(RecordingProvider())
    fixture.peer_runtime.remember_context(
        "previous peer text",
        timestamp=fixture.clock.now(),
        source_language="zh",
        target_language="ja",
    )

    prepared = fixture.owner.prepare(
        "你好",
        channel="peer",
        detected_language="zh",
    )

    assert prepared.source_language == "zh"
    assert prepared.target_language == "ja"
    assert prepared.system_prompt == "Chinese|Japanese"
    assert "previous peer text" in prepared.context
    assert prepared.applied_context_mode == "integrated"


def test_prepare_peer_auto_without_detected_language_uses_unspecified_source() -> None:
    fixture = build_owner(RecordingProvider())
    fixture.configuration.replace(
        replace(fixture.configuration.snapshot().value, peer_source_mode="auto")
    )

    prepared = fixture.owner.prepare("hello", channel="peer")

    assert prepared.source_language == "auto"
    assert prepared.system_prompt == "<input>|Japanese"


def test_prepare_peer_auto_blank_detected_language_uses_unspecified_source() -> None:
    fixture = build_owner(RecordingProvider())
    fixture.configuration.replace(
        replace(fixture.configuration.snapshot().value, peer_source_mode="auto")
    )

    prepared = fixture.owner.prepare("hello", channel="peer", detected_language="  ")

    assert prepared.source_language == "auto"
    assert prepared.system_prompt == "<input>|Japanese"


def test_prepare_uses_integrated_context_without_eligible_peer_entry() -> None:
    fixture = build_owner(RecordingProvider())
    fixture.self_runtime.remember_context(
        "previous self text",
        timestamp=fixture.clock.now(),
        source_language="ko",
        target_language="en",
    )

    prepared = fixture.owner.prepare("안녕")

    assert "previous self text" in prepared.context
    assert prepared.applied_context_mode == "integrated"


def test_parent_admission_freezes_target_specific_context_before_registering_current_turn() -> None:
    fixture = build_owner(RecordingProvider())

    fixture.self_runtime.remember_context(
        "previous English target",
        timestamp=fixture.clock.now(),
        source_language="ko",
        target_language="en",
    )
    fixture.self_runtime.remember_context(
        "previous Japanese target",
        timestamp=fixture.clock.now(),
        source_language="ko",
        target_language="ja",
    )
    parent_id = uuid4()
    english_id = uuid4()
    japanese_id = uuid4()
    snapshot = fixture.configuration.snapshot()
    requests = (
        TranslationProcessRequest(
            parent_utterance_id=parent_id,
            utterance_id=english_id,
            sequence=0,
            text="current turn",
            channel="self",
            source="Mic",
            target_language="en",
            context_policy="integrated_preferred",
            config_snapshot=snapshot,
            target_index=0,
        ),
        TranslationProcessRequest(
            parent_utterance_id=parent_id,
            utterance_id=japanese_id,
            sequence=1,
            text="current turn",
            channel="self",
            source="Mic",
            target_language="ja",
            context_policy="integrated_preferred",
            config_snapshot=snapshot,
            target_index=1,
        ),
    )

    admitted = fixture.owner.admit(requests)

    assert admitted[english_id].target_language == "en"
    assert admitted[english_id].system_prompt == "Korean|English"
    assert admitted[english_id].context == '- [self] "previous English target"'
    assert admitted[japanese_id].target_language == "ja"
    assert admitted[japanese_id].system_prompt == "Korean|Japanese"
    assert admitted[japanese_id].context == '- [self] "previous Japanese target"'
    assert [entry.target_language for entry in fixture.self_runtime.translation_history[-2:]] == [
        "en",
        "ja",
    ]


@pytest.mark.asyncio
async def test_direct_request_uses_captured_configuration_snapshot() -> None:
    provider = BlockingProvider()
    fixture = build_owner(provider)
    snapshot = fixture.configuration.snapshot()
    task = asyncio.create_task(
        fixture.owner.translate(
            DirectTranslationRequest(
                utterance_id=uuid4(),
                text="hello",
                config_snapshot=snapshot,
            )
        )
    )
    await provider.entered.wait()
    fixture.configuration.transform(lambda current: replace(current, target_language="ja"))
    provider.release.set()

    result = await task

    assert provider.calls[0]["target_language"] == "en"
    assert result.target_language == "en"


@pytest.mark.asyncio
async def test_direct_request_rejects_stale_provider_completion() -> None:
    old_provider = BlockingProvider()
    fixture = build_owner(old_provider)
    task = asyncio.create_task(
        fixture.owner.translate(DirectTranslationRequest(utterance_id=uuid4(), text="hello"))
    )
    await old_provider.entered.wait()
    await fixture.provider_runtime.replace_provider(
        LlmTranslationBackend(RecordingProvider()),
        start=False,
    )
    old_provider.release.set()

    with pytest.raises(StaleProviderCompletion):
        await task


@pytest.mark.asyncio
async def test_http_backend_rejects_completion_after_runtime_replacement() -> None:
    client = BlockingHttpClient()
    extension = parse_http_extension(
        {
            "schema_version": 1,
            "id": "demo",
            "name": "Demo",
            "url": "http://127.0.0.1:1/translate",
            "request": {"body": {"type": "none"}},
            "response": {"type": "text"},
        }
    )
    backend = HttpExtensionTranslationBackend(
        extension,
        InMemorySecretStore(),
        client_factory=lambda **_kwargs: client,
    )
    fixture = build_owner(backend)
    task = asyncio.create_task(
        fixture.owner.translate(DirectTranslationRequest(utterance_id=uuid4(), text="hello"))
    )
    await client.entered.wait()

    await fixture.provider_runtime.replace_provider(
        LlmTranslationBackend(RecordingProvider()),
        start=False,
    )
    client.release.set()

    with pytest.raises(StaleProviderCompletion):
        await task
    assert client.closed is True


@pytest.mark.asyncio
async def test_process_propagates_parent_turn_and_target_identity_to_output() -> None:
    fixture = build_owner(RecordingProvider())
    request = replace(
        process_request(fixture),
        target_index=1,
        turn_generation=2,
        turn_order=7,
    )

    result = await fixture.owner.process(request)

    assert result.outcome == "translated"
    assert result.output is not None
    assert result.output.parent_utterance_id == request.parent_utterance_id
    assert result.output.target_index == 1
    assert result.output.target_language == request.target_language
    assert result.output.turn_generation == 2
    assert result.output.turn_order == 7


@pytest.mark.asyncio
async def test_process_returns_source_only_when_provider_is_unavailable() -> None:
    fixture = build_owner()

    result = await fixture.owner.process(process_request(fixture, channel="peer"))

    assert result.outcome == "source_only"
    assert result.output is not None
    assert result.output.failure_code == "translation_unavailable"
    assert result.output.channel == "peer"
    assert fixture.presentation.messages == []


@pytest.mark.asyncio
async def test_process_contains_stale_provider_completion_without_output_error() -> None:
    old_provider = BlockingProvider()
    fixture = build_owner(old_provider)
    task = asyncio.create_task(fixture.owner.process(process_request(fixture)))
    await old_provider.entered.wait()
    await fixture.provider_runtime.replace_provider(
        LlmTranslationBackend(RecordingProvider()),
        start=False,
    )
    old_provider.release.set()

    result = await task

    assert result.outcome == "failed"
    assert result.output is not None
    assert result.output.failure_code == "stale_provider_completion"
    assert fixture.presentation.messages == []


@pytest.mark.asyncio
async def test_process_rejects_unsupported_self_language_with_safe_error() -> None:
    fixture = build_owner(RecordingProvider())

    result = await fixture.owner.process(process_request(fixture, detected_language="unsupported"))

    assert result.outcome == "failed"
    assert result.output is not None
    assert result.output.failure_code == "unsupported_source_language"
    assert len(fixture.presentation.messages) == 1
    assert fixture.presentation.messages[0].channel == "self"


@pytest.mark.asyncio
async def test_process_preserves_cancellation_after_provider_completion() -> None:
    fixture = build_owner(RecordingProvider())

    with pytest.raises(asyncio.CancelledError):
        await fixture.owner.process(
            process_request(fixture),
            cancellation_requested=lambda: True,
        )


@pytest.mark.asyncio
async def test_process_contains_provider_failure_and_publishes_safe_error() -> None:
    fixture = build_owner(RecordingProvider(failure=RuntimeError("secret detail")))

    result = await fixture.owner.process(process_request(fixture))

    assert result.outcome == "failed"
    assert result.output is not None
    assert result.output.failure_code == "provider_error"
    assert len(fixture.presentation.messages) == 1
    assert fixture.presentation.messages[0].runtime_log_handled is True
