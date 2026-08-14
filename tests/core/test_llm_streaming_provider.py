from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest
from puripuly_heart.core.managed_openrouter_release import ManagedOpenRouterLLMProvider

from puripuly_heart.app.wiring import _LazyFactoryLLMProvider
from puripuly_heart.core.llm import FallbackRacingLLMProvider
from puripuly_heart.core.llm.provider import LLMProvider, SemaphoreLLMProvider
from puripuly_heart.domain.models import Translation
from puripuly_heart.providers.llm.deepseek import (
    DeepSeekClient,
    DeepSeekLLMProvider,
    HttpxDeepSeekClient,
)
from puripuly_heart.providers.llm.gemini import (
    GeminiClient,
    GeminiLLMProvider,
    GoogleGenaiGeminiClient,
)
from puripuly_heart.providers.llm.local_openai import (
    HttpxLocalOpenAIClient,
    LocalOpenAIClient,
    LocalOpenAICompatibleLLMProvider,
)
from puripuly_heart.providers.llm.openrouter import (
    HttpxOpenRouterClient,
    OpenRouterClient,
    OpenRouterLLMProvider,
)
from puripuly_heart.providers.llm.qwen import (
    DashScopeQwenClient,
    QwenClient,
    QwenLLMProvider,
)
from puripuly_heart.providers.llm.qwen_async import (
    AsyncQwenClient,
    AsyncQwenLLMProvider,
    HttpxQwenClient,
)


@dataclass(slots=True)
class TranslateOnlyLLMProvider(LLMProvider):
    translated_text: str = "hello"
    calls: list[dict[str, object]] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "system_prompt": system_prompt,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
            }
        )
        return Translation(
            utterance_id=utterance_id,
            translated_text=self.translated_text,
            source_text=text,
            source_language=source_language,
            target_language=target_language,
        )

    async def close(self) -> None:
        return


@pytest.mark.parametrize(
    "provider_type",
    [
        LLMProvider,
        SemaphoreLLMProvider,
        FallbackRacingLLMProvider,
        ManagedOpenRouterLLMProvider,
        _LazyFactoryLLMProvider,
        GeminiLLMProvider,
        GoogleGenaiGeminiClient,
        OpenRouterLLMProvider,
        HttpxOpenRouterClient,
        QwenLLMProvider,
        DashScopeQwenClient,
        AsyncQwenLLMProvider,
        HttpxQwenClient,
        DeepSeekLLMProvider,
        HttpxDeepSeekClient,
        LocalOpenAICompatibleLLMProvider,
        HttpxLocalOpenAIClient,
    ],
)
def test_production_llm_classes_do_not_expose_stream_translate(provider_type: type) -> None:
    assert "stream_translate" not in provider_type.__dict__


@pytest.mark.parametrize(
    "client_contract",
    [
        GeminiClient,
        OpenRouterClient,
        QwenClient,
        AsyncQwenClient,
        DeepSeekClient,
        LocalOpenAIClient,
    ],
)
def test_llm_client_contracts_do_not_require_stream_translate(client_contract: type) -> None:
    assert "stream_translate" not in client_contract.__dict__


@pytest.mark.asyncio
async def test_semaphore_provider_preserves_translate_behavior_without_stream_contract() -> None:
    inner = TranslateOnlyLLMProvider(translated_text="translated")
    provider = SemaphoreLLMProvider(inner=inner, semaphore=asyncio.Semaphore(1))
    utterance_id = uuid4()

    result = await provider.translate(
        utterance_id=utterance_id,
        text="안녕",
        system_prompt="PROMPT",
        source_language="ko",
        target_language="en",
        context='- "previous"',
    )

    assert result.translated_text == "translated"
    assert inner.calls == [
        {
            "utterance_id": utterance_id,
            "text": "안녕",
            "system_prompt": "PROMPT",
            "source_language": "ko",
            "target_language": "en",
            "context": '- "previous"',
        }
    ]
