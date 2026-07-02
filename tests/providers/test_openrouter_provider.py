from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

import pytest

from puripuly_heart.config.settings import OpenRouterProviderRouting, OpenRouterRoutingMode
from puripuly_heart.providers.llm.openrouter import (
    HttpxOpenRouterClient,
    OpenRouterClient,
    OpenRouterKeyMetadata,
    OpenRouterLLMProvider,
)


@dataclass
class FakeOpenRouterClient(OpenRouterClient):
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


class SpyRuntimeLogging:
    def __init__(self, *, detailed_return: bool = False) -> None:
        self.detailed_return = detailed_return
        self.detailed_messages: list[tuple[str, int]] = []
        self.basic_messages: list[tuple[str, int]] = []

    def emit_detailed(self, message: str, *, level: int = logging.INFO) -> bool:
        self.detailed_messages.append((message, level))
        return self.detailed_return

    def emit_basic(self, message: str, *, level: int = logging.INFO) -> None:
        self.basic_messages.append((message, level))


class FakeResponse:
    status_code = 200

    def __init__(self, data: dict | None = None):
        self._data = data or {"choices": [{"message": {"content": "OK"}}]}

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class FakeAsyncClient:
    def __init__(
        self,
        *,
        response_data: dict | None = None,
    ):
        self.last_request: dict = {}
        self.requests: list[dict] = []
        self.closed = False
        self._response_data = response_data

    async def aclose(self):
        self.closed = True

    async def post(self, url, **kwargs):
        request = {"url": url, **kwargs}
        self.last_request = request
        self.requests.append(request)
        return FakeResponse(self._response_data)


@pytest.mark.asyncio
async def test_openrouter_provider_uses_injected_client() -> None:
    fake = FakeOpenRouterClient()
    provider = OpenRouterLLMProvider(api_key="k", client=fake)

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
async def test_openrouter_provider_close_cleans_up() -> None:
    fake = FakeOpenRouterClient()
    provider = OpenRouterLLMProvider(api_key="k", client=fake)
    provider._internal_client = fake

    await provider.close()

    assert fake.closed is True
    assert provider._internal_client is None


def test_openrouter_provider_passes_max_tokens_to_internal_httpx_client() -> None:
    provider = OpenRouterLLMProvider(api_key="k", max_tokens=17)

    client = provider._get_client()

    assert isinstance(client, HttpxOpenRouterClient)
    assert client.max_tokens == 17


def test_openrouter_provider_passes_user_identifier_to_internal_httpx_client() -> None:
    provider = OpenRouterLLMProvider(api_key="k", user_identifier="user-123")

    client = provider._get_client()

    assert isinstance(client, HttpxOpenRouterClient)
    assert client.user_identifier == "user-123"


@pytest.mark.asyncio
async def test_openrouter_verify_api_key_uses_key_endpoint(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            seen["url"] = url
            seen["headers"] = kwargs["headers"]
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    ok = await OpenRouterLLMProvider.verify_api_key("secret")

    assert ok is True
    assert seen["url"] == "https://openrouter.ai/api/v1/key"
    assert seen["headers"]["Authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_openrouter_fetch_key_metadata_uses_key_endpoint(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "data": {
                    "limit": 0.07,
                    "limit_remaining": 0.05,
                    "usage": 0.02,
                }
            }

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            seen["url"] = url
            seen["headers"] = kwargs["headers"]
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    metadata = await OpenRouterLLMProvider.fetch_key_metadata("secret")

    assert metadata == OpenRouterKeyMetadata(limit_usd=0.07, remaining_usd=0.05, usage_usd=0.02)
    assert seen["url"] == "https://openrouter.ai/api/v1/key"
    assert seen["headers"]["Authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_httpx_openrouter_client_builds_reasoning_disabled_request_with_latency_sort(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="google/gemma-4-26b-a4b-it",
        base_url="https://example",
        user_identifier="  managed-user-123  ",
    )
    result = await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
        context='- "previous"',
    )

    assert result == "OK"
    assert fake_client.last_request["url"] == "https://example/chat/completions"
    headers = fake_client.last_request["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"

    body = fake_client.last_request["json"]
    assert body["model"] == "google/gemma-4-26b-a4b-it"
    assert body["max_tokens"] == 100
    assert body["reasoning"] == {"effort": "none"}
    assert body["user"] == "managed-user-123"
    assert body["provider"] == {
        "order": ["cloudflare", "parasail/bf16", "wafer/fp8"],
        "only": ["cloudflare", "parasail", "wafer"],
        "allow_fallbacks": True,
    }
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert body["messages"][1]["role"] == "user"
    assert "<context>" in body["messages"][1]["content"]
    assert "</context>" in body["messages"][1]["content"]
    assert "<input>\nhello\n</input>" in body["messages"][1]["content"]
    assert "Input: hello" not in body["messages"][1]["content"]


@pytest.mark.asyncio
async def test_httpx_openrouter_client_gemma_latency_routing_prefers_cloudflare_then_parasail_then_wafer(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="google/gemma-4-26b-a4b-it",
        base_url="https://example",
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
    )

    body = fake_client.last_request["json"]
    assert body["provider"] == {
        "order": ["cloudflare", "parasail/bf16", "wafer/fp8"],
        "only": ["cloudflare", "parasail", "wafer"],
        "allow_fallbacks": True,
    }


@pytest.mark.asyncio
async def test_httpx_openrouter_client_non_gemma_latency_routing_preserves_latency_sort(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="qwen/qwen3.5-flash-02-23",
        base_url="https://example",
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
    )

    body = fake_client.last_request["json"]
    assert body["provider"] == {
        "sort": "latency",
        "allow_fallbacks": True,
        "ignore": ["venice", "deepinfra", "google-vertex"],
    }


@pytest.mark.asyncio
async def test_httpx_openrouter_client_deepseek_default_routing_prefers_cloudflare_then_deepseek_then_parasail(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="deepseek/deepseek-v4-flash",
        base_url="https://example",
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="zh-CN",
    )

    body = fake_client.last_request["json"]
    assert body["provider"] == {
        "order": ["cloudflare", "deepseek", "parasail/fp8"],
        "only": ["cloudflare", "deepseek", "parasail"],
        "allow_fallbacks": True,
    }


@pytest.mark.asyncio
async def test_httpx_openrouter_client_deepseek_china_routing_prefers_deepseek_then_baidu_qianfan(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="deepseek/deepseek-v4-flash",
        base_url="https://example",
        routing_mode=OpenRouterRoutingMode.PARASAIL_FIRST,
        provider_routing=OpenRouterProviderRouting.DEEPSEEK_ONLY,
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="zh-CN",
    )

    body = fake_client.last_request["json"]
    assert body["provider"] == {
        "order": ["deepseek", "baidu/fp8"],
        "only": ["deepseek", "baidu"],
        "allow_fallbacks": True,
    }


@pytest.mark.asyncio
async def test_httpx_openrouter_client_google_gemini_latency_routing_uses_google_providers_only(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="google/gemini-3-flash-preview",
        base_url="https://example",
        provider_routing=OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY,
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
    )

    body = fake_client.last_request["json"]
    provider = body["provider"]
    assert provider["sort"] == "latency"
    assert provider["only"] == ["google-vertex", "google-ai-studio"]
    assert provider["allow_fallbacks"] is True
    assert provider["data_collection"] == "deny"
    assert "ignore" not in provider


@pytest.mark.asyncio
async def test_httpx_openrouter_client_google_gemini_latency_routing_does_not_ignore_vertex(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="google/gemini-3.1-flash-lite",
        base_url="https://example",
        provider_routing=OpenRouterProviderRouting.GOOGLE_GEMINI_LATENCY,
    )
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
    )

    body = fake_client.last_request["json"]
    provider = body["provider"]
    assert "google-vertex" not in provider.get("ignore", [])
    assert provider["only"] == ["google-vertex", "google-ai-studio"]


@pytest.mark.asyncio
async def test_httpx_openrouter_client_builds_ordered_request_for_parasail_first(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(
        api_key="test-key",
        model="google/gemma-4-26b-a4b-it",
        base_url="https://example",
        routing_mode=OpenRouterRoutingMode.PARASAIL_FIRST,
    )
    result = await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
    )

    assert result == "OK"
    body = fake_client.last_request["json"]
    assert body["provider"] == {"order": ["Parasail", "Novita"], "allow_fallbacks": True}


@pytest.mark.asyncio
async def test_httpx_openrouter_client_translate_raises_on_length_finish_reason(
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

    client = HttpxOpenRouterClient(api_key="k", model="m", base_url="https://example")

    with pytest.raises(RuntimeError, match="truncated"):
        await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )


@pytest.mark.asyncio
async def test_httpx_openrouter_client_logs_basic_translate_success_without_runtime_logging(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(api_key="k", model="m", base_url="https://example")

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.openrouter"):
        result = await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
            context='- "previous"',
        )

    assert result == "OK"
    assert caplog.messages == [
        "[Basic][LLM] OpenRouter request [translate][context=yes] ko -> en: 'hello'",
        "[Basic][LLM] OpenRouter response [translate]: 'OK'",
    ]


@pytest.mark.asyncio
async def test_httpx_openrouter_client_logs_basic_translate_failure_without_runtime_logging(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    class ErrorResponse(FakeResponse):
        status_code = 429

        def __init__(self):
            super().__init__({"error": {"message": "quota exceeded"}})

        def raise_for_status(self):
            raise RuntimeError("quota exceeded")

    class ErrorAsyncClient(FakeAsyncClient):
        async def post(self, url, **kwargs):
            request = {"url": url, **kwargs}
            self.last_request = request
            self.requests.append(request)
            return ErrorResponse()

    fake_client = ErrorAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxOpenRouterClient(api_key="k", model="m", base_url="https://example")

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.openrouter"):
        with pytest.raises(RuntimeError, match="quota exceeded"):
            await client.translate(
                text="hello",
                system_prompt="SYSTEM",
                source_language="ko",
                target_language="en",
            )

    assert caplog.messages == [
        "[Basic][LLM] OpenRouter request [translate][context=no] ko -> en: 'hello'",
        "[Basic][LLM] OpenRouter request failed [translate]: status=429 message=quota exceeded",
    ]


@pytest.mark.asyncio
async def test_httpx_openrouter_client_runtime_logging_logs_basic_translate_success(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    runtime_logging = SpyRuntimeLogging(detailed_return=False)

    client = HttpxOpenRouterClient(
        api_key="k",
        model="m",
        base_url="https://example",
        runtime_logging=runtime_logging,
    )

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.openrouter"):
        result = await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
            context='- "previous"',
        )

    assert result == "OK"
    assert runtime_logging.basic_messages == [
        (
            "[Basic][LLM] OpenRouter request [translate][context=yes] ko -> en: 'hello'",
            logging.INFO,
        ),
        ("[Basic][LLM] OpenRouter response [translate]: 'OK'", logging.INFO),
    ]
    assert runtime_logging.detailed_messages == []
    assert caplog.messages == []


@pytest.mark.asyncio
async def test_httpx_openrouter_client_runtime_logging_logs_basic_translate_failure(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    class ErrorResponse(FakeResponse):
        status_code = 429

        def __init__(self):
            super().__init__({"error": {"message": "quota exceeded"}})

    class ErrorAsyncClient(FakeAsyncClient):
        async def post(self, url, **kwargs):
            request = {"url": url, **kwargs}
            self.last_request = request
            self.requests.append(request)
            return ErrorResponse()

    fake_client = ErrorAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    runtime_logging = SpyRuntimeLogging(detailed_return=False)

    client = HttpxOpenRouterClient(
        api_key="k",
        model="m",
        base_url="https://example",
        runtime_logging=runtime_logging,
    )

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.openrouter"):
        with pytest.raises(RuntimeError, match="quota exceeded"):
            await client.translate(
                text="hello",
                system_prompt="SYSTEM",
                source_language="ko",
                target_language="en",
            )

    assert runtime_logging.detailed_messages == []
    assert runtime_logging.basic_messages == [
        (
            "[Basic][LLM] OpenRouter request [translate][context=no] ko -> en: 'hello'",
            logging.INFO,
        ),
        (
            "[Basic][LLM] OpenRouter request failed [translate]: status=429 message=quota exceeded",
            logging.ERROR,
        ),
    ]
    assert caplog.messages == []
