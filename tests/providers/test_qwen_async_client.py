from __future__ import annotations

import asyncio

import pytest

from puripuly_heart.providers.llm.error_details import LLMResponseError
from puripuly_heart.providers.llm.qwen_async import AsyncQwenLLMProvider, HttpxQwenClient


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
async def test_httpx_client_builds_correct_request(monkeypatch):
    fake_client = FakeAsyncClient()

    def fake_async_client(**kwargs):
        return fake_client

    monkeypatch.setattr("httpx.AsyncClient", fake_async_client)

    client = HttpxQwenClient(api_key="test-key", model="qwen3.5-flash", base_url="https://example")
    result = await client.translate(
        text="hello",
        system_prompt="Translate {source_language} to {target_language}",
        source_language="ko-KR",
        target_language="en",
        context='- "previous"',
    )

    assert result == "OK"

    # Check URL
    assert fake_client.last_request["url"] == "https://example/chat/completions"

    # Check headers
    headers = fake_client.last_request["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"

    # Check body
    body = fake_client.last_request["json"]
    assert body["model"] == "qwen3.5-flash"
    assert body["enable_thinking"] is False
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "Translate ko-KR to en"
    assert body["messages"][1]["role"] == "user"
    assert "<context>" in body["messages"][1]["content"]
    assert "</context>" in body["messages"][1]["content"]
    assert "<input>\nhello\n</input>" in body["messages"][1]["content"]
    assert "Input: hello" not in body["messages"][1]["content"]


@pytest.mark.asyncio
async def test_httpx_client_reuses_cached_async_client(monkeypatch):
    fake_client = FakeAsyncClient()
    created_clients: list[FakeAsyncClient] = []

    def fake_async_client(**_kwargs):
        created_clients.append(fake_client)
        return fake_client

    monkeypatch.setattr("httpx.AsyncClient", fake_async_client)

    client = HttpxQwenClient(api_key="k", model="m", base_url="https://example")
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko",
        target_language="en",
    )
    await client.translate(
        text="world",
        system_prompt="SYSTEM",
        source_language="ko",
        target_language="en",
    )

    assert created_clients == [fake_client]
    assert len(fake_client.requests) == 2
    assert fake_client.requests[0]["json"]["messages"][1]["content"] == "<input>\nhello\n</input>"
    assert fake_client.requests[1]["json"]["messages"][1]["content"] == "<input>\nworld\n</input>"


@pytest.mark.asyncio
async def test_httpx_client_close_recreates_async_client(monkeypatch):
    first_client = FakeAsyncClient()
    second_client = FakeAsyncClient()
    created_clients: list[FakeAsyncClient] = []

    def fake_async_client(**_kwargs):
        client = first_client if not created_clients else second_client
        created_clients.append(client)
        return client

    monkeypatch.setattr("httpx.AsyncClient", fake_async_client)

    client = HttpxQwenClient(api_key="k", model="m", base_url="https://example")
    await client.translate(
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko",
        target_language="en",
    )
    await client.close()
    await client.translate(
        text="again",
        system_prompt="SYSTEM",
        source_language="ko",
        target_language="en",
    )

    assert created_clients == [first_client, second_client]
    assert first_client.closed is True
    assert second_client.closed is False


@pytest.mark.asyncio
async def test_httpx_client_raises_on_empty_choices(monkeypatch):
    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda **kw: FakeAsyncClient(response_data={"choices": []}),
    )

    client = HttpxQwenClient(api_key="k", model="m", base_url="https://example")
    with pytest.raises(RuntimeError, match="did not contain choices"):
        await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )


@pytest.mark.asyncio
async def test_httpx_client_raises_on_empty_content(monkeypatch):
    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda **kw: FakeAsyncClient(response_data={"choices": [{"message": {}}]}),
    )

    client = HttpxQwenClient(api_key="k", model="m", base_url="https://example")
    with pytest.raises(RuntimeError, match="message content"):
        await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )


@pytest.mark.asyncio
async def test_httpx_client_handles_cancellation(monkeypatch):
    entered = asyncio.Event()

    class SlowFakeAsyncClient:
        async def aclose(self):
            return None

        async def post(self, url, **kwargs):
            entered.set()
            await asyncio.sleep(10)  # Long wait
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda **kw: SlowFakeAsyncClient())

    client = HttpxQwenClient(api_key="k", model="m", base_url="https://example")

    async def translate_task():
        return await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )

    task = asyncio.create_task(translate_task())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["qwen3.5-plus", "qwen3.8-flash"])
async def test_async_provider_translates_with_canonical_model(monkeypatch, model):
    from uuid import uuid4

    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    provider = AsyncQwenLLMProvider(
        api_key="synthetic-key",
        model=model,
        base_url="https://example/compatible-mode/v1",
    )
    utterance_id = uuid4()
    try:
        result = await provider.translate(
            utterance_id=utterance_id,
            text="hello",
            system_prompt="Translate {source_language} to {target_language}",
            source_language="ko",
            target_language="en",
            context='- "previous"',
        )
        assert result.utterance_id == utterance_id
        assert result.text == "OK"
        assert fake_client.last_request["url"] == (
            "https://example/compatible-mode/v1/chat/completions"
        )
        body = fake_client.last_request["json"]
        assert body["model"] == "qwen3.8-flash"
        assert body["enable_thinking"] is False
        assert body["messages"] == [
            {"role": "system", "content": "Translate ko to en"},
            {
                "role": "user",
                "content": '<context>\n- "previous"\n</context>\n\n<input>\nhello\n</input>',
            },
        ]
    finally:
        await provider.close()
    assert fake_client.closed


@pytest.mark.asyncio
async def test_async_client_rate_limit_preserves_safe_diagnostics(monkeypatch, caplog):
    private_body = "synthetic-provider-private-body"

    class RateLimitedResponse(FakeResponse):
        status_code = 429
        text = private_body

    class RateLimitedClient(FakeAsyncClient):
        async def post(self, url, **kwargs):
            return RateLimitedResponse({"error": {"message": private_body}})

    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: RateLimitedClient())
    client = HttpxQwenClient(api_key="synthetic-key", model="qwen3.8-flash")
    try:
        with pytest.raises(LLMResponseError) as caught:
            await client.translate(
                text="hello",
                system_prompt="Translate",
                source_language="en",
                target_language="ko",
            )
        assert caught.value.failure_diagnostics.status_code == 429
        assert caught.value.failure_diagnostics.category == "rate_limit"
        assert private_body not in str(caught.value)
        assert private_body not in str(caught.value.failure_diagnostics)
        assert private_body not in caplog.text
    finally:
        await client.close()
