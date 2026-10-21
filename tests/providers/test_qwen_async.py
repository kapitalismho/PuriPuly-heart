from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from puripuly_heart.providers.llm.qwen_async import (
    AsyncQwenClient,
    AsyncQwenLLMProvider,
)


@dataclass
class FakeAsyncQwenClient(AsyncQwenClient):
    last_call: dict[str, object] | None = None
    closed: bool = False

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
        self.last_call = {
            "text": text,
            "system_prompt": system_prompt,
            "source_language": source_language,
            "target_language": target_language,
            "context": context,
            "scene_participant_count": scene_participant_count,
        }
        return "TRANSLATED"

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_async_qwen_provider_uses_injected_client():
    fake = FakeAsyncQwenClient()
    provider = AsyncQwenLLMProvider(api_key="k", client=fake)

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
        "scene_participant_count": None,
    }


@pytest.mark.asyncio
async def test_async_qwen_provider_passes_context():
    fake = FakeAsyncQwenClient()
    provider = AsyncQwenLLMProvider(api_key="k", client=fake)

    await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="PROMPT",
        source_language="ko",
        target_language="en",
        context='- "안녕"',
    )

    assert fake.last_call is not None
    assert fake.last_call["system_prompt"] == "PROMPT"
    assert fake.last_call["context"] == '- "안녕"'


@pytest.mark.asyncio
async def test_async_qwen_provider_close_cleans_up():
    fake = FakeAsyncQwenClient()
    provider = AsyncQwenLLMProvider(api_key="k", client=fake)
    provider._internal_client = fake

    await provider.close()

    assert fake.closed is True
    assert provider._internal_client is None


@pytest.mark.asyncio
async def test_async_qwen_verify_api_key_uses_model_and_base_url(monkeypatch):
    seen: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            seen["url"] = url
            seen["json"] = kwargs["json"]
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    ok = await AsyncQwenLLMProvider.verify_api_key(
        "secret",
        base_url="https://example/compatible-mode/v1",
        model="qwen3.5-plus",
    )

    assert ok is True
    assert seen["url"] == "https://example/compatible-mode/v1/chat/completions"
    body = seen["json"]
    assert body["model"] == "qwen3.8-flash"
    assert body["enable_thinking"] is False
