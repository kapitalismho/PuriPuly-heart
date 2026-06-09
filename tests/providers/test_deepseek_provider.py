from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from puripuly_heart.providers.llm.deepseek import (
    DeepSeekClient,
    DeepSeekLLMProvider,
    HttpxDeepSeekClient,
)


@dataclass
class FakeDeepSeekClient(DeepSeekClient):
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
    ) -> str:
        self.last_call = {
            "text": text,
            "system_prompt": system_prompt,
            "source_language": source_language,
            "target_language": target_language,
            "context": context,
        }
        return "TRANSLATED"

    async def close(self) -> None:
        self.closed = True


class FakeResponse:
    def __init__(self, *, status_code: int = 200, data: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._data = data or {"choices": [{"message": {"content": "OK"}}]}
        self.text = text

    def json(self):
        return self._data


class FakeAsyncClient:
    def __init__(
        self,
        *,
        response_data: dict | None = None,
        response_status: int = 200,
        response_text: str = "",
    ):
        self.last_request: dict = {}
        self.requests: list[dict] = []
        self.closed = False
        self._response_data = response_data
        self._response_status = response_status
        self._response_text = response_text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def aclose(self):
        self.closed = True

    async def post(self, url, **kwargs):
        request = {"url": url, **kwargs}
        self.last_request = request
        self.requests.append(request)
        return FakeResponse(
            status_code=self._response_status,
            data=self._response_data,
            text=self._response_text,
        )


@pytest.mark.asyncio
async def test_deepseek_provider_uses_injected_client() -> None:
    fake = FakeDeepSeekClient()
    provider = DeepSeekLLMProvider(api_key="k", client=fake)

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


@pytest.mark.asyncio
async def test_deepseek_provider_close_cleans_up() -> None:
    fake = FakeDeepSeekClient()
    provider = DeepSeekLLMProvider(api_key="k", client=fake)
    provider._internal_client = fake

    await provider.close()

    assert fake.closed is True
    assert provider._internal_client is None


def test_deepseek_provider_does_not_configure_translation_max_tokens_cap() -> None:
    provider = DeepSeekLLMProvider(api_key="k")

    client = provider._get_client()

    assert isinstance(client, HttpxDeepSeekClient)
    assert not hasattr(client, "max_tokens")


def test_deepseek_provider_passes_v4_pro_model_to_internal_httpx_client() -> None:
    from puripuly_heart.config.settings import DeepSeekLLMModel

    deepseek_model = getattr(DeepSeekLLMModel, "DEEPSEEK_V4_PRO", None)

    assert deepseek_model is not None

    provider = DeepSeekLLMProvider(api_key="k", model=deepseek_model.value)

    client = provider._get_client()

    assert isinstance(client, HttpxDeepSeekClient)
    assert client.model == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_httpx_deepseek_client_builds_non_thinking_request(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxDeepSeekClient(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://example.deepseek",
    )
    result = await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
        context='- "previous"',
    )

    assert result == "OK"
    assert fake_client.last_request["url"] == "https://example.deepseek/chat/completions"
    headers = fake_client.last_request["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"

    body = fake_client.last_request["json"]
    assert body["model"] == "deepseek-v4-flash"
    assert "max_tokens" not in body
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert body["messages"][1]["role"] == "user"
    assert "<context>" in body["messages"][1]["content"]
    assert "</context>" in body["messages"][1]["content"]
    assert "<input>\nhello\n</input>" in body["messages"][1]["content"]
    assert "Input: hello" not in body["messages"][1]["content"]


@pytest.mark.asyncio
async def test_httpx_deepseek_client_translate_raises_on_length_finish_reason(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient(
        response_data={
            "choices": [
                {
                    "message": {"content": "partial"},
                    "finish_reason": "length",
                }
            ]
        }
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxDeepSeekClient(api_key="k", model="m", base_url="https://example")

    with pytest.raises(RuntimeError, match="truncated"):
        await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )


@pytest.mark.asyncio
async def test_deepseek_verify_api_key_uses_chat_completion_probe(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    ok = await DeepSeekLLMProvider.verify_api_key("secret")

    assert ok is True
    assert fake_client.last_request["url"] == "https://api.deepseek.com/chat/completions"
    assert fake_client.last_request["headers"]["Authorization"] == "Bearer secret"
    body = fake_client.last_request["json"]
    assert body == {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "ping"}],
        "thinking": {"type": "disabled"},
        "max_tokens": 1,
    }
