from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from puripuly_heart.providers.llm.cerebras import (
    CerebrasClient,
    CerebrasLLMProvider,
    HttpxCerebrasClient,
)


@dataclass
class FakeCerebrasClient(CerebrasClient):
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
        **_kwargs,
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
async def test_cerebras_provider_uses_injected_client() -> None:
    fake = FakeCerebrasClient()
    provider = CerebrasLLMProvider(api_key="k", client=fake)

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
async def test_cerebras_provider_close_cleans_up() -> None:
    fake = FakeCerebrasClient()
    provider = CerebrasLLMProvider(api_key="k", client=fake)
    provider._internal_client = fake

    await provider.close()

    assert fake.closed is True
    assert provider._internal_client is None


def test_cerebras_provider_defaults_to_gemma_4_31b_model() -> None:
    provider = CerebrasLLMProvider(api_key="k")

    client = provider._get_client()

    assert isinstance(client, HttpxCerebrasClient)
    assert client.model == "gemma-4-31b"
    assert client.base_url == "https://api.cerebras.ai/v1"


def test_cerebras_provider_passes_max_completion_tokens_to_internal_httpx_client() -> None:
    provider = CerebrasLLMProvider(api_key="k", max_completion_tokens=42)

    client = provider._get_client()

    assert isinstance(client, HttpxCerebrasClient)
    assert client.max_completion_tokens == 42


@pytest.mark.asyncio
async def test_httpx_cerebras_client_builds_reasoning_disabled_request(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxCerebrasClient(
        api_key="test-key",
        model="gemma-4-31b",
        base_url="https://example.cerebras",
    )
    result = await client.translate(
        text="hello",
        system_prompt="SYSTEM {source_language}->{target_language}",
        source_language="ko",
        target_language="en",
        context='- "previous"',
    )

    assert result == "OK"
    assert fake_client.last_request["url"] == "https://example.cerebras/chat/completions"
    headers = fake_client.last_request["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"

    body = fake_client.last_request["json"]
    assert body["model"] == "gemma-4-31b"
    assert body["stream"] is False
    assert body["reasoning_effort"] == "none"
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM ko->en"}
    assert body["messages"][1]["role"] == "user"
    user_content = body["messages"][1]["content"]
    assert "<context>" in user_content
    assert "</context>" in user_content
    assert "<input>\nhello\n</input>" in user_content
    assert "Input: hello" not in user_content


@pytest.mark.asyncio
async def test_httpx_cerebras_client_translate_raises_on_length_finish_reason(
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

    client = HttpxCerebrasClient(api_key="k", model="m", base_url="https://example")

    with pytest.raises(RuntimeError, match="truncated"):
        await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )


@pytest.mark.asyncio
async def test_httpx_cerebras_client_translate_raises_on_non_200(monkeypatch) -> None:
    fake_client = FakeAsyncClient(
        response_status=401,
        response_data={"error": {"message": "Invalid API key"}},
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxCerebrasClient(api_key="bad-key", model="gemma-4-31b", base_url="https://example")

    with pytest.raises(RuntimeError, match="Cerebras request failed"):
        await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )


@pytest.mark.asyncio
async def test_cerebras_verify_api_key_uses_chat_completion_probe(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _ProbeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "OK"}}]}

    class _ProbeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            seen["url"] = url
            seen["headers"] = kwargs["headers"]
            seen["json"] = kwargs["json"]
            return _ProbeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _ProbeClient)

    ok = await CerebrasLLMProvider.verify_api_key("secret")

    assert ok is True
    assert seen["url"] == "https://api.cerebras.ai/v1/chat/completions"
    body = seen["json"]
    assert body["model"] == "gemma-4-31b"
    assert body["reasoning_effort"] == "none"
    assert body["max_completion_tokens"] == 1
    assert body["stream"] is False
    headers = seen["headers"]
    assert headers["Authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_cerebras_verify_api_key_returns_false_on_empty_key() -> None:
    ok = await CerebrasLLMProvider.verify_api_key("")
    assert ok is False


@pytest.mark.asyncio
async def test_httpx_cerebras_client_close_acloses_internal_client(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxCerebrasClient(api_key="k", model="m", base_url="https://example")
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko",
        target_language="en",
    )

    await client.close()

    assert fake_client.closed is True
