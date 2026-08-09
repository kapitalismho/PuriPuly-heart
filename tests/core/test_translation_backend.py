from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.translation_backend import (
    LlmTranslationBackend,
    TranslationBackendRequest,
)
from puripuly_heart.domain.models import Translation


@dataclass
class RecordingProvider:
    calls: list[dict[str, object]] = field(default_factory=list)
    closed: bool = False

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
                "utterance_id": utterance_id,
                "text": text,
                "system_prompt": system_prompt,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
            }
        )
        return Translation(utterance_id=utterance_id, text="translated")

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_llm_adapter_maps_request_object_to_legacy_provider() -> None:
    provider = RecordingProvider()
    backend = LlmTranslationBackend(provider)
    utterance_id = uuid4()

    result = await backend.translate(
        TranslationBackendRequest(
            utterance_id=utterance_id,
            text="hello",
            system_prompt="system",
            source_language="en",
            target_language="ko",
            context="prior",
        )
    )

    assert result.text == "translated"
    assert provider.calls == [
        {
            "utterance_id": utterance_id,
            "text": "hello",
            "system_prompt": "system",
            "source_language": "en",
            "target_language": "ko",
            "context": "prior",
        }
    ]


@pytest.mark.asyncio
async def test_llm_adapter_owns_close_delegation() -> None:
    provider = RecordingProvider()
    backend = LlmTranslationBackend(provider)

    await backend.close()

    assert provider.closed is True
