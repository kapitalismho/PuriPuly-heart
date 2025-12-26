from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from puripuly_heart.providers.llm.qwen import QwenClient, QwenLLMProvider


@dataclass
class FakeQwenClient(QwenClient):
    last_call: dict[str, str] | None = None

    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> str:
        self.last_call = {
            "text": text,
            "system_prompt": system_prompt,
            "source_language": source_language,
            "target_language": target_language,
            "context": context,
        }
        return "TRANSLATED"


@pytest.mark.asyncio
async def test_qwen_provider_uses_injected_client():
    fake = FakeQwenClient()
    provider = QwenLLMProvider(api_key="k", client=fake)

    utterance_id = uuid4()
    out = await provider.translate(
        utterance_id=utterance_id,
        text="hello",
        system_prompt="PROMPT",
        source_language="ko-KR",
        target_language="en",
    )

    assert out.utterance_id == utterance_id
    assert out.text == "TRANSLATED"
    assert fake.last_call == {
        "text": "hello",
        "system_prompt": "PROMPT",
        "source_language": "ko-KR",
        "target_language": "en",
        "context": "",
    }
