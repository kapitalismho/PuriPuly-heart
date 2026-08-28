from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.providers.llm.local_openai import (
    HttpxLocalOpenAIClient,
    LocalOpenAIClient,
    LocalOpenAICompatibleLLMProvider,
    LocalOpenAIReservedExtraBodyKeyError,
    LocalOpenAISensitiveExtraBodyKeyError,
)

APP_RESERVED_KEYS = (
    "model",
    "messages",
    "stream",
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "max_tokens",
)
SENSITIVE_KEYS = ("api_key", "authorization", "headers", "token", "secret", "password")


@dataclass
class FakeLocalClient(LocalOpenAIClient):
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
    def __init__(
        self, *, status_code: int = 200, data: dict | Exception | None = None, text: str = ""
    ):
        self.status_code = status_code
        self._data = data if data is not None else {"choices": [{"message": {"content": "OK"}}]}
        self.text = text

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


class FakeAsyncClient:
    def __init__(
        self,
        *,
        response_data: dict | Exception | None = None,
        response_status: int = 200,
        response_text: str = "",
        request_exception: Exception | None = None,
        **kwargs,
    ):
        self.init_kwargs = kwargs
        self.last_request: dict = {}
        self.closed = False
        self._response_data = response_data
        self._response_status = response_status
        self._response_text = response_text
        self._request_exception = request_exception

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def aclose(self):
        self.closed = True

    async def post(self, url, **kwargs):
        if self._request_exception is not None:
            raise self._request_exception
        self.last_request = {"url": url, **kwargs}
        return FakeResponse(
            status_code=self._response_status,
            data=self._response_data,
            text=self._response_text,
        )


@pytest.mark.asyncio
async def test_local_provider_uses_injected_client() -> None:
    fake = FakeLocalClient()
    provider = LocalOpenAICompatibleLLMProvider(client=fake)
    utterance_id = uuid4()

    out = await provider.translate(
        utterance_id=utterance_id,
        text="안녕",
        system_prompt="SYSTEM",
        source_language="ko-KR",
        target_language="en",
    )

    assert out.utterance_id == utterance_id
    assert out.text == "TRANSLATED"
    assert fake.last_call == {
        "text": "안녕",
        "system_prompt": "SYSTEM",
        "source_language": "ko-KR",
        "target_language": "en",
        "context": "",
    }


@pytest.mark.asyncio
async def test_httpx_local_client_builds_minimal_request_and_merges_extra_body(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(
        base_url="http://127.0.0.1:11434/v1/",
        model="llama3.1:8b",
        api_key="",
        extra_body={"think": False, "temperature": 0.1},
    )

    result = await client.translate(
        text="hello",
        system_prompt="SYSTEM {source_language}->{target_language}",
        source_language="ko",
        target_language="en",
        context='- "previous"',
    )

    assert result == "OK"
    assert fake_client.last_request["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert fake_client.last_request["headers"] == {"Content-Type": "application/json"}
    body = fake_client.last_request["json"]
    assert body["model"] == "llama3.1:8b"
    assert "max_tokens" not in body
    assert body["think"] is False
    assert body["temperature"] == 0.1
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM ko->en"}
    assert body["messages"][1]["role"] == "user"
    user_content = body["messages"][1]["content"]
    assert "context" in user_content
    assert "hello" in user_content
    assert "provider" not in body
    assert "reasoning" not in body
    assert "thinking" not in body
    assert "enable_thinking" not in body
    assert body["stream"] is False


@pytest.mark.asyncio
async def test_httpx_local_client_defaults_to_openai_reasoning_effort_none(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://127.0.0.1:11434/v1", model="gemma4:e2b")

    await client.translate(
        text="안녕",
        system_prompt="SYSTEM",
        source_language="ko",
        target_language="en",
    )

    body = fake_client.last_request["json"]
    assert body["reasoning_effort"] == "none"
    assert body["temperature"] == 0.6
    assert "think" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("key", APP_RESERVED_KEYS)
async def test_httpx_local_client_rejects_reserved_extra_body_keys(key: str) -> None:
    with pytest.raises(LocalOpenAIReservedExtraBodyKeyError, match=key):
        HttpxLocalOpenAIClient(
            base_url="http://127.0.0.1:11434/v1",
            model="llama3.1:8b",
            extra_body={key: True},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("key", SENSITIVE_KEYS)
async def test_httpx_local_client_rejects_sensitive_extra_body_keys(key: str) -> None:
    with pytest.raises(LocalOpenAISensitiveExtraBodyKeyError, match=key):
        HttpxLocalOpenAIClient(
            base_url="http://127.0.0.1:11434/v1",
            model="llama3.1:8b",
            extra_body={key: "do-not-persist"},
        )


def test_httpx_local_client_rejects_unsafe_extra_body_before_http_client_creation(
    monkeypatch,
) -> None:
    def factory(**kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("httpx.AsyncClient should not be created for invalid extra_body")

    monkeypatch.setattr("httpx.AsyncClient", factory)

    with pytest.raises(ValueError, match="JSON serializable"):
        HttpxLocalOpenAIClient(
            base_url="http://127.0.0.1:11434/v1",
            model="llama3.1:8b",
            extra_body={"callback": object()},
        )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "infinity", "negative_infinity"],
)
def test_httpx_local_client_rejects_non_standard_json_constants_extra_body(value: float) -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        HttpxLocalOpenAIClient(
            base_url="http://127.0.0.1:11434/v1",
            model="llama3.1:8b",
            extra_body={"temperature": value},
        )


@pytest.mark.asyncio
async def test_httpx_local_client_adds_auth_only_when_key_present(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(
        base_url="http://localhost:11434/v1", model="m", api_key="secret"
    )

    await client.translate(
        text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
    )

    assert fake_client.last_request["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
async def test_httpx_local_client_treats_whitespace_key_as_empty(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m", api_key="   ")

    await client.translate(
        text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
    )

    assert fake_client.last_request["headers"] == {"Content-Type": "application/json"}


@pytest.mark.asyncio
async def test_local_translate_redacts_trimmed_configured_api_key(monkeypatch) -> None:
    fake_client = FakeAsyncClient(
        response_status=500,
        response_data={"error": {"message": "plain leaked token local-secret"}},
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(
        base_url="http://localhost:11434/v1",
        model="m",
        api_key="  local-secret  ",
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert "local-secret" not in message
    assert "[redacted]" in message


def test_httpx_local_client_disables_env_proxy_and_redirects(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    seen_kwargs: dict[str, object] = {}

    def factory(**kwargs):
        seen_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setattr("httpx.AsyncClient", factory)
    client = HttpxLocalOpenAIClient(base_url="http://127.0.0.1:11434/v1", model="m")

    created = client._build_http_client()

    assert created is fake_client
    assert seen_kwargs["trust_env"] is False
    assert seen_kwargs["follow_redirects"] is False


@pytest.mark.asyncio
async def test_local_provider_close_cleans_up_internal_client(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    provider = LocalOpenAICompatibleLLMProvider()

    await provider.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko",
        target_language="en",
    )
    await provider.close()

    assert fake_client.closed is True
    assert provider._internal_client is None


@pytest.mark.asyncio
async def test_local_verify_connection_allows_empty_key(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)

    ok = await LocalOpenAICompatibleLLMProvider.verify_connection(
        base_url="http://localhost:11434/v1",
        model="llama3.1:8b",
        api_key="",
    )

    assert ok is True
    assert fake_client.last_request["json"]["messages"][0] == {"role": "system", "content": ""}
    assert "ping" in fake_client.last_request["json"]["messages"][1]["content"]
    assert fake_client.last_request["json"]["max_tokens"] == 1
    assert fake_client.last_request["json"]["reasoning_effort"] == "none"
    assert fake_client.last_request["json"]["temperature"] == 0.6


@pytest.mark.asyncio
async def test_local_translate_raises_on_non_200(monkeypatch) -> None:
    fake_client = FakeAsyncClient(
        response_status=404, response_data={"error": {"message": "model not found"}}
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="missing")

    with pytest.raises(RuntimeError, match="model not found"):
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )


@pytest.mark.asyncio
async def test_local_translate_sanitizes_server_error_text(monkeypatch) -> None:
    fake_client = FakeAsyncClient(
        response_status=500,
        response_data=ValueError("not json"),
        response_text="Authorization: Bearer secret http://127.0.0.1:11434/v1 body={...}",
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(
        base_url="http://localhost:11434/v1", model="m", api_key="secret"
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert "secret" not in message
    assert "127.0.0.1" not in message
    assert "Bearer [redacted]" in message


@pytest.mark.asyncio
async def test_local_translate_sanitizes_echoed_request_body_fragments(monkeypatch) -> None:
    fake_client = FakeAsyncClient(
        response_status=500,
        response_data=ValueError("not json"),
        response_text=(
            'body={"model":"llama3.1:8b","messages":[{"role":"user","content":"hello"}],"think":false} '
            'json={"messages":[{"role":"system","content":"SYSTEM"}],"think":false} '
            'messages=[{"role":"user","content":"hello"}] '
            'raw extra body {"think":false}'
        ),
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(
        base_url="http://localhost:11434/v1",
        model="llama3.1:8b",
        extra_body={"think": False},
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert "body=" not in message
    assert "json=" not in message
    assert "messages=" not in message
    assert '"think"' not in message
    assert "hello" not in message
    assert "[request-body-redacted]" in message


@pytest.mark.asyncio
async def test_local_translate_sanitizes_colon_form_request_body_fragments(monkeypatch) -> None:
    source_text = "SECRET_SOURCE_TEXT_DO_NOT_LEAK"
    fake_client = FakeAsyncClient(
        response_status=500,
        response_data=ValueError("not json"),
        response_text=(
            'body: {"model":"m","messages":[{"role":"user","content":"SECRET_SOURCE_TEXT_DO_NOT_LEAK"}],'
            '"think":false,"stream":false}'
        ),
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(
        base_url="http://localhost:11434/v1", model="m", extra_body={"think": False}
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text=source_text, system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert "body:" not in message
    assert "messages" not in message
    assert '"think"' not in message
    assert source_text not in message
    assert "[request-body-redacted]" in message


@pytest.mark.asyncio
async def test_local_translate_sanitizes_raw_json_request_body_fragments(monkeypatch) -> None:
    source_text = "RAW_JSON_SOURCE_TEXT_DO_NOT_LEAK"
    fake_client = FakeAsyncClient(
        response_status=500,
        response_data=ValueError("not json"),
        response_text=(
            '{"model":"m","messages":[{"role":"system","content":"SYSTEM"},'
            '{"role":"user","content":"RAW_JSON_SOURCE_TEXT_DO_NOT_LEAK"}],"think":false}'
        ),
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(
        base_url="http://localhost:11434/v1", model="m", extra_body={"think": False}
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text=source_text, system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert "messages" not in message
    assert '"think"' not in message
    assert source_text not in message
    assert "[request-body-redacted]" in message


@pytest.mark.asyncio
async def test_local_translate_sanitizes_plain_source_text_echo(monkeypatch) -> None:
    source_text = "PLAIN_SOURCE_TEXT_DO_NOT_LEAK"
    fake_client = FakeAsyncClient(
        response_status=500,
        response_data={"error": {"message": f"generation failed for input {source_text}"}},
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m")

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text=source_text, system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert source_text not in message
    assert "[source-text-redacted]" in message


@pytest.mark.asyncio
async def test_local_translate_redacts_plain_standalone_configured_model(monkeypatch) -> None:
    model_name = "llama3.1:8b"
    fake_client = FakeAsyncClient(
        response_status=404,
        response_data={"error": {"message": f"server could not load {model_name} for chat"}},
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model=model_name)

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert model_name not in message
    assert "server could not load [model-redacted] for chat" in message


@pytest.mark.asyncio
async def test_local_translate_redacts_configured_model_followed_by_period(monkeypatch) -> None:
    model_name = "llama3.1:8b"
    fake_client = FakeAsyncClient(
        response_status=404,
        response_data={"error": {"message": f"server could not load {model_name}."}},
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model=model_name)

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert model_name not in message
    assert "server could not load [model-redacted]." in message


@pytest.mark.asyncio
async def test_local_translate_redacts_configured_model_followed_by_trailing_colon(
    monkeypatch,
) -> None:
    model_name = "llama3.1:8b"
    fake_client = FakeAsyncClient(
        response_status=404,
        response_data={"error": {"message": f"server could not load {model_name}: not found"}},
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model=model_name)

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert model_name not in message
    assert "server could not load [model-redacted]: not found" in message


@pytest.mark.asyncio
async def test_local_translate_redacts_short_standalone_model_without_corrupting_words(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient(
        response_status=404,
        response_data={"error": {"message": "backend m failed with temporary mismatch"}},
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m")

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert "backend m failed" not in message
    assert "backend [model-redacted] failed" in message
    assert "temporary mismatch" in message


@pytest.mark.asyncio
async def test_local_translate_redacts_multikey_formatted_extra_body_json(monkeypatch) -> None:
    fake_client = FakeAsyncClient(
        response_status=500,
        response_data={
            "error": {
                "message": 'provider echoed extras { "seed" : 42, "think" : false } after failure'
            }
        },
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(
        base_url="http://localhost:11434/v1",
        model="m",
        extra_body={"think": False, "seed": 42},
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert '"seed" : 42' not in message
    assert '"think" : false' not in message
    assert "[extra-body-redacted]" in message


@pytest.mark.asyncio
async def test_local_translate_redacts_extra_body_key_value_fragments(monkeypatch) -> None:
    fake_client = FakeAsyncClient(
        response_status=500,
        response_data={
            "error": {"message": "provider rejected think: false; seed=42; num_ctx = 2048"}
        },
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(
        base_url="http://localhost:11434/v1",
        model="m",
        extra_body={"think": False, "seed": 42, "num_ctx": 2048},
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert "think: false" not in message
    assert "seed=42" not in message
    assert "num_ctx = 2048" not in message
    assert "[extra-body-redacted]" in message


@pytest.mark.asyncio
async def test_local_translate_redacts_nested_extra_body_jsonish_key_value_fragment(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient(
        response_status=500,
        response_data={
            "error": {
                "message": 'provider rejected thinking: { "type" : "disabled" } after failure'
            }
        },
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(
        base_url="http://localhost:11434/v1",
        model="m",
        extra_body={"thinking": {"type": "disabled"}},
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert 'thinking: { "type" : "disabled" }' not in message
    assert "[extra-body-redacted]" in message


@pytest.mark.asyncio
async def test_local_translate_redacts_nested_extra_body_python_repr_key_value_fragment(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient(
        response_status=500,
        response_data={
            "error": {"message": "provider rejected thinking={'type': 'disabled'} after failure"}
        },
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(
        base_url="http://localhost:11434/v1",
        model="m",
        extra_body={"thinking": {"type": "disabled"}},
    )

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert "thinking={'type': 'disabled'}" not in message
    assert "[extra-body-redacted]" in message


@pytest.mark.asyncio
async def test_local_translate_redacts_short_model_without_corrupting_ordinary_words(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient(
        response_status=404,
        response_data={"error": {"message": "model m failed with temporary mismatch"}},
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m")

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert "model m" not in message
    assert "[model-redacted]" in message
    assert "temporary mismatch" in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_message", "visible_fragment"),
    [
        ("model m:latest was not found", "model m:latest"),
        ("model=m.extra was not found", "model=m.extra"),
    ],
)
async def test_local_translate_does_not_partially_redact_short_model_inside_longer_contextual_tokens(
    monkeypatch, server_message: str, visible_fragment: str
) -> None:
    fake_client = FakeAsyncClient(
        response_status=404,
        response_data={"error": {"message": server_message}},
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m")

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert visible_fragment in message
    assert "[model-redacted]" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_message", "expected_fragment"),
    [
        ("model m: not found", "model [model-redacted]: not found"),
        ("backend m failed", "backend [model-redacted] failed"),
        ("model m.", "model [model-redacted]."),
        ("model m:", "model [model-redacted]:"),
    ],
)
async def test_local_translate_redacts_short_model_at_safe_boundaries(
    monkeypatch, server_message: str, expected_fragment: str
) -> None:
    fake_client = FakeAsyncClient(
        response_status=404,
        response_data={"error": {"message": server_message}},
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m")

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert server_message not in message
    assert expected_fragment in message


@pytest.mark.asyncio
@pytest.mark.parametrize("server_message", ["model m: not found", 'model "m": not found'])
async def test_local_translate_redacts_short_model_before_colon(
    monkeypatch, server_message: str
) -> None:
    fake_client = FakeAsyncClient(
        response_status=404,
        response_data={"error": {"message": server_message}},
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m")

    with pytest.raises(RuntimeError) as exc_info:
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )

    message = str(exc_info.value)
    assert "model m:" not in message
    assert 'model "m":' not in message
    assert "[model-redacted]" in message
    assert "not found" in message


@pytest.mark.asyncio
async def test_local_translate_raises_on_malformed_response(monkeypatch) -> None:
    fake_client = FakeAsyncClient(response_data={"choices": []})
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m")

    with pytest.raises(RuntimeError, match="choices"):
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )


@pytest.mark.asyncio
async def test_local_translate_accepts_content_part_arrays(monkeypatch) -> None:
    fake_client = FakeAsyncClient(
        response_data={
            "choices": [{"message": {"content": [{"type": "text", "text": "안녕하세요"}]}}]
        }
    )
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m")

    assert (
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="en", target_language="ko"
        )
        == "안녕하세요"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "choices"),
        ({"choices": [{}]}, "message"),
        ({"choices": [{"message": {}}]}, "content"),
        ({"choices": [{"message": {"content": ""}}]}, "empty"),
        (
            {"choices": [{"finish_reason": "length", "message": {"content": "truncated"}}]},
            "truncated",
        ),
    ],
)
async def test_local_translate_rejects_incomplete_or_truncated_payloads(
    monkeypatch, payload, message
) -> None:
    fake_client = FakeAsyncClient(response_data=payload)
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m")

    with pytest.raises(RuntimeError, match=message):
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )


@pytest.mark.asyncio
async def test_local_translate_rejects_invalid_json_response(monkeypatch) -> None:
    fake_client = FakeAsyncClient(response_data=ValueError("not json"))
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m")

    with pytest.raises(RuntimeError, match="JSON"):
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )


@pytest.mark.asyncio
async def test_local_translate_propagates_cancellation(monkeypatch) -> None:
    fake_client = FakeAsyncClient(request_exception=asyncio.CancelledError())
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)
    client = HttpxLocalOpenAIClient(base_url="http://localhost:11434/v1", model="m")

    with pytest.raises(asyncio.CancelledError):
        await client.translate(
            text="hello", system_prompt="SYSTEM", source_language="ko", target_language="en"
        )


@pytest.mark.asyncio
async def test_local_verify_connection_returns_false_for_http_and_request_failures(
    monkeypatch,
) -> None:
    fake_client = FakeAsyncClient(response_status=307, response_data={"error": "redirect"})
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake_client)

    assert (
        await LocalOpenAICompatibleLLMProvider.verify_connection(
            base_url="http://localhost:11434/v1", model="m"
        )
        is False
    )

    failing_client = FakeAsyncClient(request_exception=httpx.ConnectError("refused"))
    monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: failing_client)

    assert (
        await LocalOpenAICompatibleLLMProvider.verify_connection(
            base_url="http://localhost:11434/v1", model="m"
        )
        is False
    )


@pytest.mark.asyncio
async def test_local_verify_connection_disables_env_proxy_and_redirects(monkeypatch) -> None:
    fake_client = FakeAsyncClient()
    seen_kwargs: dict[str, object] = {}

    def factory(**kwargs):
        seen_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setattr("httpx.AsyncClient", factory)

    assert (
        await LocalOpenAICompatibleLLMProvider.verify_connection(
            base_url="http://localhost:11434/v1", model="m"
        )
        is True
    )
    assert seen_kwargs["trust_env"] is False
    assert seen_kwargs["follow_redirects"] is False


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://127.0.0.1:11434/v1",
        "http://user:pass@127.0.0.1:11434/v1",
        "http://@localhost:11434/v1",
        "http://127.0.0.1:11434/v1?x=1",
        "http://127.0.0.1:11434/v1#frag",
        "http://:11434/v1",
    ],
)
def test_httpx_local_client_rejects_invalid_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="base URL"):
        HttpxLocalOpenAIClient(base_url=base_url, model="m")
