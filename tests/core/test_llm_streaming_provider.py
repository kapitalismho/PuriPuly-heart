from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.llm.provider import LLMProvider, SemaphoreLLMProvider
from puripuly_heart.domain.models import Translation


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
        scene_participant_count: int | None = None,
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
