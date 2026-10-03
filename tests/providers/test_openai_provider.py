from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

import httpx
import pytest

from puripuly_heart.config.runtime_resolution import OPENAI_MODEL_GPT_6_LUNA
from puripuly_heart.providers.llm.openai import (
    HttpxOpenAIClient,
    OpenAIClient,
    OpenAILLMProvider,
    OpenAIResponseError,
)


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        data: object = None,
        text: str = "",
        json_error: bool = False,
    ) -> None:
        self.status_code = status_code
        self._data = (
            {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}
            if data is None
            else data
        )
        self.text = text
        self.json_error = json_error

    def json(self) -> object:
        if self.json_error:
            raise ValueError("invalid response JSON")
        return self._data


class FakeAsyncClient:
    def __init__(
        self,
        *,
        response_data: object = None,
        response_status: int = 200,
        response_text: str = "",
        json_error: bool = False,
        cancel: bool = False,
    ) -> None:
        self.last_request: dict[str, object] | None = None
        self.requests: list[dict[str, object]] = []
        self.closed = False
        self._response_data = response_data
        self._response_status = response_status
        self._response_text = response_text
        self._json_error = json_error
        self._cancel = cancel

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        self.closed = True

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        request = {"url": url, **kwargs}
        self.last_request = request
        self.requests.append(request)
        if self._cancel:
            raise asyncio.CancelledError
        return FakeResponse(
            status_code=self._response_status,
            data=self._response_data,
            text=self._response_text,
            json_error=self._json_error,
        )


class FakeOpenAIClient(OpenAIClient):
    def __init__(self, *, cancel: bool = False) -> None:
        self.last_call: dict[str, object] | None = None
        self.closed = False
        self.cancel = cancel

    async def translate(
        self,
        *,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
        scene_participant_count: int | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        self.last_call = {
            "text": text,
            "system_prompt": system_prompt,
            "source_language": source_language,
            "target_language": target_language,
            "context": context,
            "scene_participant_count": scene_participant_count,
            "max_output_tokens": max_output_tokens,
        }
        if self.cancel:
            raise asyncio.CancelledError
        return "TRANSLATED"

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_openai_provider_forwards_translation_and_preserves_injected_client() -> None:
    fake = FakeOpenAIClient()
    provider = OpenAILLMProvider(api_key="not-for-repr", client=fake)
    utterance_id = uuid4()

    translation = await provider.translate(
        utterance_id=utterance_id,
        text="잠깐만 기다려 주세요.",
        system_prompt="Translate {source_language} into {target_language}.",
        source_language="Korean",
        target_language="English",
        context="At the station, the speaker asks a companion to wait.",
        scene_participant_count=2,
        max_output_tokens=37,
    )
    await provider.close()

    assert translation.utterance_id == utterance_id
    assert translation.text == "TRANSLATED"
    assert fake.last_call == {
        "text": "잠깐만 기다려 주세요.",
        "system_prompt": "Translate {source_language} into {target_language}.",
        "source_language": "Korean",
        "target_language": "English",
        "context": "At the station, the speaker asks a companion to wait.",
        "scene_participant_count": 2,
        "max_output_tokens": 37,
    }
    assert fake.closed is False
    assert "not-for-repr" not in repr(provider)


@pytest.mark.asyncio
async def test_httpx_openai_client_builds_reasoning_disabled_chat_completion_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_http = FakeAsyncClient(
        response_data={
            "choices": [
                {"message": {"content": "Please wait a little longer."}, "finish_reason": "stop"}
            ],
            "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        }
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: fake_http)
    client = HttpxOpenAIClient(
        api_key="openai-secret",
        model=OPENAI_MODEL_GPT_6_LUNA,
        base_url="https://example.openai/v1",
    )

    result = await client.translate(
        text="잠깐만 기다려 주세요.",
        system_prompt="Translate {source_language} to {target_language}.",
        source_language="Korean",
        target_language="English",
        context="At the station, the speaker asks a companion to wait.",
        scene_participant_count=2,
        max_output_tokens=37,
    )

    assert result == "Please wait a little longer."
    assert client.last_reasoning_tokens == 0
    assert fake_http.last_request is not None
    assert fake_http.last_request["url"] == "https://example.openai/v1/chat/completions"
    assert fake_http.last_request["headers"] == {
        "Authorization": "Bearer openai-secret",
        "Content-Type": "application/json",
    }
    body = fake_http.last_request["json"]
    assert body["model"] == "gpt-6-luna"
    assert body["reasoning_effort"] == "none"
    assert body["temperature"] == 0.6
    assert body["max_completion_tokens"] == 37
    assert "max_tokens" not in body
    assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert body["messages"] == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Translate Korean to English.",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {
            "role": "user",
            "content": (
                "<context>\nAt the station, the speaker asks a companion to wait.\n"
                "</context>\n\n<scene>\nPeople: 2\n</scene>\n\n"
                "<input>\n잠깐만 기다려 주세요.\n</input>"
            ),
        },
    ]
    await client.close()
    assert fake_http.closed is True


@pytest.mark.parametrize(
    ("model", "explicit_cache"),
    [
        (OPENAI_MODEL_GPT_6_LUNA, True),
        ("gpt-5.1", False),
        (f"{OPENAI_MODEL_GPT_6_LUNA}-preview", False),
    ],
)
def test_openai_cache_schema_preserves_custom_prompt_and_excludes_user_content(
    model: str, explicit_cache: bool
) -> None:
    client = HttpxOpenAIClient(api_key="test-key", model=model)
    custom_prompt = "Custom instructions\nKeep {literal} exactly as written."
    requests = [
        client._build_request_body(
            text=text,
            system_prompt=custom_prompt,
            source_language="Korean",
            target_language="English",
            context=context,
            scene_participant_count=participants,
        )
        for text, context, participants in (
            ("first input", "first context", 2),
            ("second input", "second context", 3),
        )
    ]

    expected_content: object = custom_prompt
    if explicit_cache:
        expected_content = [
            {
                "type": "text",
                "text": custom_prompt,
                "prompt_cache_breakpoint": {"mode": "explicit"},
            }
        ]
    for body in requests:
        assert body["messages"][0] == {"role": "system", "content": expected_content}
        assert isinstance(body["messages"][1]["content"], str)
        assert "max_completion_tokens" not in body
        if explicit_cache:
            assert body["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
        else:
            assert "prompt_cache_options" not in body
    assert requests[0]["messages"][0] == requests[1]["messages"][0]
    assert requests[0]["messages"][1] != requests[1]["messages"][1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "json_error", "message"),
    [
        ({"choices": []}, False, "did not contain choices"),
        ({"choices": [{"message": {"content": "  "}}]}, False, "empty message content"),
        (
            {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
            False,
            "truncated",
        ),
        ({"choices": [{"message": {"content": [{"type": "image"}]}}]}, False, "message content"),
        ({}, True, "not valid JSON"),
    ],
)
async def test_httpx_openai_client_rejects_incomplete_responses(
    monkeypatch: pytest.MonkeyPatch,
    data: object,
    json_error: bool,
    message: str,
) -> None:
    fake_http = FakeAsyncClient(response_data=data, json_error=json_error)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: fake_http)
    client = HttpxOpenAIClient(api_key="test-key")

    with pytest.raises(RuntimeError, match=message):
        await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )

    await client.close()


@pytest.mark.asyncio
async def test_httpx_openai_client_reports_status_without_upstream_body_or_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "openai-secret-material-123"
    fake_http = FakeAsyncClient(
        response_status=401,
        response_text=f"invalid key {secret} and private upstream detail",
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: fake_http)
    client = HttpxOpenAIClient(api_key=secret)

    with caplog.at_level(logging.ERROR), pytest.raises(OpenAIResponseError) as exc_info:
        await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )

    rendered = " ".join(caplog.messages) + str(exc_info.value)
    assert "status=401" in rendered
    assert secret not in rendered
    assert "private upstream detail" not in rendered
    await client.close()


@pytest.mark.asyncio
async def test_openai_provider_propagates_cancellation_and_only_closes_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = FakeOpenAIClient(cancel=True)
    provider = OpenAILLMProvider(api_key="key", client=injected)

    with pytest.raises(asyncio.CancelledError):
        await provider.translate(
            utterance_id=uuid4(),
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )
    await provider.close()
    assert injected.closed is False

    fake_http = FakeAsyncClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: fake_http)
    owned = OpenAILLMProvider(api_key="key")
    await owned.translate(
        utterance_id=uuid4(),
        text="hello",
        system_prompt="SYSTEM",
        source_language="ko",
        target_language="en",
    )
    await owned.close()
    assert fake_http.closed is True
    assert owned._internal_client is None


@pytest.mark.asyncio
async def test_httpx_openai_client_propagates_request_cancellation_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_http = FakeAsyncClient(cancel=True)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: fake_http)
    client = HttpxOpenAIClient(api_key="key")

    with pytest.raises(asyncio.CancelledError):
        await client.translate(
            text="hello",
            system_prompt="SYSTEM",
            source_language="ko",
            target_language="en",
        )

    assert len(fake_http.requests) == 1
    await client.close()


@pytest.mark.asyncio
async def test_openai_key_verification_probes_selected_luna_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_http = FakeAsyncClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: fake_http)

    assert await OpenAILLMProvider.verify_api_key("verify-secret", model=OPENAI_MODEL_GPT_6_LUNA)
    assert fake_http.last_request is not None
    assert fake_http.last_request["url"] == "https://api.openai.com/v1/chat/completions"
    assert fake_http.last_request["headers"]["Authorization"] == "Bearer verify-secret"
    assert fake_http.last_request["json"] == {
        "model": "gpt-6-luna",
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "reasoning_effort": "none",
        "temperature": 0.6,
        "max_completion_tokens": 1,
    }
    assert fake_http.closed is True
