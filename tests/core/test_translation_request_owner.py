from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

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
from puripuly_heart.core.runtime.provider_handle import ProviderRuntimeHandle
from puripuly_heart.core.storage.secrets import InMemorySecretStore
from puripuly_heart.domain.models import ChannelId, Translation
from puripuly_heart.providers.extensions.http_extension_backend import (
    HttpExtensionTranslationBackend,
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
    calls: list[dict[str, str]] = field(default_factory=list)
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
    ) -> Translation:
        self.calls.append(
            {
                "text": text,
                "system_prompt": system_prompt,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
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
    provider_runtime = ProviderRuntimeHandle(name="llm", provider=provider)
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
    )


def test_clear_context_clears_both_channels_and_emits_established_diagnostic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fixture = build_owner()
    fixture.self_runtime.remember_context("self context", timestamp=fixture.clock.now())
    fixture.peer_runtime.remember_context("peer context", timestamp=fixture.clock.now())

    with caplog.at_level(logging.INFO, logger="puripuly_heart.core.orchestrator.translation"):
        fixture.owner.clear_context()

    assert fixture.self_runtime.translation_history == []
    assert fixture.peer_runtime.translation_history == []
    assert "[Translation] Context history cleared" in caplog.messages


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


def test_prepare_falls_back_to_local_context_without_eligible_peer_entry() -> None:
    fixture = build_owner(RecordingProvider())
    fixture.self_runtime.remember_context(
        "previous self text",
        timestamp=fixture.clock.now(),
        source_language="ko",
        target_language="en",
    )

    prepared = fixture.owner.prepare("안녕")

    assert "previous self text" in prepared.context
    assert prepared.applied_context_mode == "local"


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
    await fixture.provider_runtime.replace_provider(RecordingProvider(), start=False)
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

    await fixture.provider_runtime.replace_provider(RecordingProvider(), start=False)
    client.release.set()

    with pytest.raises(StaleProviderCompletion):
        await task
    assert client.closed is True


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
    await fixture.provider_runtime.replace_provider(RecordingProvider(), start=False)
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
