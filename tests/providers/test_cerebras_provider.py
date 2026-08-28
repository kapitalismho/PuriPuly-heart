from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from puripuly_heart.providers.llm.cerebras import (
    CerebrasClient,
    CerebrasLLMProvider,
    HttpxCerebrasClient,
)
from puripuly_heart.providers.llm.error_details import PROVIDER_ERROR_DETAIL_MAX_LENGTH


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
        self._data = {"choices": [{"message": {"content": "OK"}}]} if data is None else data
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


class FakeRuntimeLogging:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit_basic(self, message: str, **_kwargs) -> None:
        self.messages.append(message)


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
    assert body["temperature"] == 0.6
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM ko->en"}
    assert body["messages"][1]["role"] == "user"
    assert "<input>\nhello\n</input>" in body["messages"][1]["content"]


@pytest.mark.asyncio
async def test_httpx_cerebras_client_translate_raises_on_empty_response(monkeypatch) -> None:
    fake_client = FakeAsyncClient(response_data={"choices": []})
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxCerebrasClient(api_key="k", model="m", base_url="https://example")

    with pytest.raises(RuntimeError, match="did not contain choices"):
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )


@pytest.mark.asyncio
async def test_httpx_cerebras_client_translate_raises_on_length_finish_reason(monkeypatch) -> None:
    fake_client = FakeAsyncClient(
        response_data={"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)

    client = HttpxCerebrasClient(api_key="k", model="m", base_url="https://example")

    with pytest.raises(RuntimeError, match="truncated"):
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )


@pytest.mark.asyncio
async def test_httpx_cerebras_client_translate_raises_safely_on_non_200(monkeypatch) -> None:
    secret = "bad-secret-token"
    provider_json_payload = "upstream rejected raw provider payload"
    provider_text_payload = "raw response body with diagnostic internals"
    fake_client = FakeAsyncClient(
        response_status=401,
        response_data={"error": {"message": provider_json_payload}},
        response_text=provider_text_payload,
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    runtime_logging = FakeRuntimeLogging()

    client = HttpxCerebrasClient(
        api_key=secret,
        model="gemma-4-31b",
        base_url="https://example",
        runtime_logging=runtime_logging,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    rendered_error = str(exc_info.value)
    rendered_logs = "\n".join(runtime_logging.messages)
    assert rendered_error == f"Cerebras request failed (status=401 message={provider_json_payload})"
    assert secret not in rendered_error
    assert provider_text_payload not in rendered_error
    assert secret not in rendered_logs
    assert f"message={provider_json_payload}" in rendered_logs
    assert provider_text_payload not in rendered_logs
    assert "status=401" in rendered_logs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_data", "response_text", "expected"),
    [
        ({"message": "top level detail"}, "ignored text", "top level detail"),
        ({"error": {"message": "nested detail"}}, "ignored text", "nested detail"),
        ({"error": "string detail"}, "ignored text", "string detail"),
        ({"detail": "not used"}, "plain text preview", "plain text preview"),
        ({}, "", "unknown error"),
    ],
)
async def test_httpx_cerebras_client_non_200_extracts_safe_detail_order(
    monkeypatch,
    response_data: dict,
    response_text: str,
    expected: str,
) -> None:
    fake_client = FakeAsyncClient(
        response_status=400,
        response_data=response_data,
        response_text=response_text,
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    runtime_logging = FakeRuntimeLogging()

    client = HttpxCerebrasClient(
        api_key="test-key",
        model="gemma-4-31b",
        base_url="https://example",
        runtime_logging=runtime_logging,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    rendered_error = str(exc_info.value)
    rendered_logs = "\n".join(runtime_logging.messages)
    assert rendered_error == f"Cerebras request failed (status=400 message={expected})"
    assert "status=400" in rendered_logs
    assert f"message={expected}" in rendered_logs
    assert "ignored text" not in rendered_error
    assert "ignored text" not in rendered_logs


@pytest.mark.asyncio
async def test_httpx_cerebras_client_non_200_detail_is_bounded_and_redacted(monkeypatch) -> None:
    secret = "cerebras-api-key-123"
    full_payload_tail = "TAIL_SHOULD_NOT_APPEAR"
    raw_text = (
        f"response body token={secret} Bearer {secret} "
        + "x" * (PROVIDER_ERROR_DETAIL_MAX_LENGTH + 40)
        + full_payload_tail
    )
    fake_client = FakeAsyncClient(response_status=502, response_data={}, response_text=raw_text)
    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: fake_client)
    runtime_logging = FakeRuntimeLogging()

    client = HttpxCerebrasClient(
        api_key=secret,
        model="gemma-4-31b",
        base_url="https://example",
        runtime_logging=runtime_logging,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="request body text",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )

    rendered = str(exc_info.value) + "\n" + "\n".join(runtime_logging.messages)
    assert secret not in rendered
    assert f"Bearer {secret}" not in rendered
    assert "token=[redacted]" in rendered
    assert "request body text" not in rendered
    assert full_payload_tail not in rendered
    detail = str(exc_info.value).split("message=", 1)[1].removesuffix(")")
    assert len(detail) <= PROVIDER_ERROR_DETAIL_MAX_LENGTH


@pytest.mark.asyncio
async def test_cerebras_verify_api_key_uses_chat_completion_probe(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _ProbeResponse:
        status_code = 200

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
    assert seen["headers"]["Authorization"] == "Bearer secret"


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
        text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
    )

    await client.close()

    assert fake_client.closed is True
