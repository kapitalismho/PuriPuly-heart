from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

import pytest

from puripuly_heart.providers.llm.qwen_async import (
    AsyncQwenClient,
    AsyncQwenLLMProvider,
    HttpxQwenClient,
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


@pytest.mark.asyncio
async def test_async_qwen_warmup_uses_canonical_model(monkeypatch):
    seen: dict[str, str] = {}

    async def fake_verify(api_key: str, *, base_url: str, model: str) -> bool:
        seen["api_key"] = api_key
        seen["base_url"] = base_url
        seen["model"] = model
        return True

    monkeypatch.setattr(AsyncQwenLLMProvider, "verify_api_key", staticmethod(fake_verify))

    provider = AsyncQwenLLMProvider(
        api_key="secret",
        base_url="https://example/compatible-mode/v1",
        model="qwen3.8-flash",
    )
    await provider.warmup()

    assert seen == {
        "api_key": "secret",
        "base_url": "https://example/compatible-mode/v1",
        "model": "qwen3.8-flash",
    }


@pytest.mark.asyncio
async def test_httpx_qwen_client_logs_basic_request_failure(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    class FakeResponse:
        status_code = 429

        @staticmethod
        def json() -> dict[str, object]:
            return {"error": {"message": "quota exceeded"}}

        def raise_for_status(self) -> None:
            raise RuntimeError("quota exceeded")

    class FakeAsyncClient:
        async def post(self, _url: str, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: FakeAsyncClient())

    client = HttpxQwenClient(api_key="k", model="m", base_url="https://example")

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.qwen_async"):
        with pytest.raises(RuntimeError, match="quota exceeded"):
            await client.translate(
                text="hello",
                system_prompt="PROMPT",
                source_language="ko",
                target_language="en",
            )

    failure = caplog.messages[-1]
    assert "category=quota code=provider.quota" in failure
    assert "operation=translate status=429 provider=qwen" in failure
    assert "exception_type=RuntimeError" in failure
    assert "quota exceeded" not in "\n".join(caplog.messages)


@pytest.mark.asyncio
async def test_httpx_qwen_client_uses_runtime_logging_for_failure_breadcrumbs(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    class FakeResponse:
        status_code = 429

        @staticmethod
        def json() -> dict[str, object]:
            return {"error": {"message": "quota exceeded"}}

    class FakeAsyncClient:
        async def post(self, _url: str, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", lambda **_kwargs: FakeAsyncClient())
    runtime_logging = SpyRuntimeLogging(detailed_return=False)

    client = HttpxQwenClient(
        api_key="k",
        model="m",
        base_url="https://example",
        runtime_logging=runtime_logging,
    )

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.qwen_async"):
        with pytest.raises(RuntimeError, match="quota exceeded"):
            await client.translate(
                text="hello",
                system_prompt="PROMPT",
                source_language="ko",
                target_language="en",
            )

    assert runtime_logging.detailed_messages == []
    assert len(runtime_logging.basic_messages) == 1
    failure, level = runtime_logging.basic_messages[0]
    assert level == logging.ERROR
    assert "category=quota code=provider.quota" in failure
    assert "operation=translate status=429 provider=qwen" in failure
    assert "exception_type=RuntimeError" in failure
    assert "quota exceeded" not in failure
    assert caplog.messages == []
