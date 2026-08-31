from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from uuid import uuid4

import pytest

from puripuly_heart.providers.llm.qwen import (
    DashScopeQwenClient,
    QwenClient,
    QwenLLMProvider,
)


@dataclass
class FakeQwenClient(QwenClient):
    last_call: dict[str, object] | None = None

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

@pytest.mark.asyncio
async def test_qwen_verify_api_key_empty_key_returns_false_without_network(monkeypatch) -> None:
    def fail_if_constructed(*_args, **_kwargs):
        raise AssertionError("empty-key verification must not construct an HTTP client")

    monkeypatch.setattr("httpx.AsyncClient", fail_if_constructed)

    assert (
        await QwenLLMProvider.verify_api_key(
            "",
            base_url="https://example/api/v1",
            model="qwen3.8-flash",
        )
        is False
    )


def test_qwen_client_normalizes_language_codes() -> None:
    client = DashScopeQwenClient(api_key="k", model="m")
    assert client._normalize_language_code("") == "auto"
    assert client._normalize_language_code("auto") == "auto"
    assert client._normalize_language_code("zh-CN") == "zh"
    assert client._normalize_language_code("zh-Hant") == "zh_tw"
    assert client._normalize_language_code("ko-KR") == "ko"


@pytest.mark.asyncio
async def test_qwen_client_translates_with_options(monkeypatch):
    calls: dict[str, object] = {}

    class FakeResponse:
        output = {"choices": [{"message": {"content": "OK"}}]}

    class FakeGeneration:
        @staticmethod
        def call(**kwargs):
            calls.update(kwargs)
            return FakeResponse()

    class FakeDashScope:
        api_key = ""
        base_http_api_url = ""
        Generation = FakeGeneration

    monkeypatch.setitem(__import__("sys").modules, "dashscope", FakeDashScope)

    client = DashScopeQwenClient(api_key="k", model="m", base_url="https://example")
    result = await client.translate(
        text="hello",
        system_prompt="Translate naturally",
        source_language="ko-KR",
        target_language="en",
        context='- "이전 문장"',
    )

    assert result == "OK"
    messages = calls["messages"]
    assert messages[0]["role"] == "system"
    assert "Translate naturally" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "<context>" in messages[1]["content"]
    assert "</context>" in messages[1]["content"]
    assert "<input>\nhello\n</input>" in messages[1]["content"]
    assert "Input: hello" not in messages[1]["content"]


@pytest.mark.asyncio
async def test_qwen_client_raises_when_missing_content(monkeypatch):
    class FakeResponse:
        output = {"choices": [{"message": {}}]}

    class FakeGeneration:
        @staticmethod
        def call(**_kwargs):
            return FakeResponse()

    class FakeDashScope:
        api_key = ""
        base_http_api_url = ""
        Generation = FakeGeneration

    monkeypatch.setitem(__import__("sys").modules, "dashscope", FakeDashScope)

    client = DashScopeQwenClient(api_key="k", model="m")
    with pytest.raises(RuntimeError, match="message content"):
        await client.translate(
            text="hello",
            system_prompt="Translate",
            source_language="en",
            target_language="ko",
        )


@pytest.mark.asyncio
async def test_qwen_client_normalizes_legacy_plus_to_qwen38(monkeypatch):
    calls: dict[str, object] = {}

    class FakeHttpxResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict:
            return {"choices": [{"message": {"content": "OK35"}}]}

    def fake_httpx_post(url, **kwargs):
        calls["url"] = url
        calls.update(kwargs)
        return FakeHttpxResponse()

    class DummyGeneration:
        @staticmethod
        def call(**_kwargs):
            raise AssertionError("Generation.call must not be used for qwen3.8-flash")

    dummy = type(
        "DummyDashScope",
        (),
        {"api_key": "", "base_http_api_url": "", "Generation": DummyGeneration},
    )
    monkeypatch.setitem(sys.modules, "dashscope", dummy)
    monkeypatch.setattr("httpx.post", fake_httpx_post)

    client = DashScopeQwenClient(
        api_key="k", model="qwen3.5-plus", base_url="https://example/api/v1"
    )
    result = await client.translate(
        text="hello",
        system_prompt="Translate naturally",
        source_language="ko",
        target_language="en",
        context='- "이전 문장"',
    )

    assert result == "OK35"
    assert calls["url"] == "https://example/compatible-mode/v1/chat/completions"
    body = calls["json"]
    assert body["model"] == "qwen3.8-flash"
    assert body["enable_thinking"] is False
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert "<context>" in body["messages"][1]["content"]
    assert "</context>" in body["messages"][1]["content"]
    assert "<input>\nhello\n</input>" in body["messages"][1]["content"]
    assert "Input: hello" not in body["messages"][1]["content"]


@pytest.mark.asyncio
async def test_qwen_verify_api_key_handles_status_for_legacy_model(monkeypatch):
    calls: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

    class FakeGeneration:
        @staticmethod
        def call(**kwargs):
            calls.update(kwargs)
            return FakeResponse()

    class FakeDashScope:
        api_key = ""
        base_http_api_url = ""
        Generation = FakeGeneration

    monkeypatch.setitem(__import__("sys").modules, "dashscope", FakeDashScope)

    assert (
        await QwenLLMProvider.verify_api_key(
            "secret", base_url="https://example/api/v1", model="qwen-plus"
        )
        is True
    )
    assert calls["model"] == "qwen-plus"
    assert FakeDashScope.base_http_api_url == "https://example/api/v1"


@pytest.mark.asyncio
async def test_qwen_verify_api_key_uses_compatible_mode_for_qwen38(monkeypatch):
    calls: dict[str, object] = {}

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
            calls["url"] = url
            calls["json"] = kwargs["json"]
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    assert (
        await QwenLLMProvider.verify_api_key(
            "secret", base_url="https://example/api/v1", model="qwen3.8-flash"
        )
        is True
    )
    assert calls["url"] == "https://example/compatible-mode/v1/chat/completions"
    assert calls["json"]["model"] == "qwen3.8-flash"
    assert calls["json"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_qwen_warmup_uses_canonical_model(monkeypatch):
    seen: dict[str, str] = {}

    async def fake_verify(api_key: str, *, base_url: str, model: str) -> bool:
        seen["api_key"] = api_key
        seen["base_url"] = base_url
        seen["model"] = model
        return True

    monkeypatch.setattr(QwenLLMProvider, "verify_api_key", staticmethod(fake_verify))

    provider = QwenLLMProvider(
        api_key="secret",
        base_url="https://example/api/v1",
        model="qwen3.8-flash",
    )
    await provider.warmup()

    assert seen == {
        "api_key": "secret",
        "base_url": "https://example/api/v1",
        "model": "qwen3.8-flash",
    }


@pytest.mark.asyncio
async def test_qwen_client_logs_basic_request_and_response_for_qwen38(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    class FakeHttpxResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict[str, object]:
            return {"choices": [{"message": {"content": "OK35"}}]}

    def fake_httpx_post(_url, **_kwargs):
        return FakeHttpxResponse()

    class DummyGeneration:
        @staticmethod
        def call(**_kwargs):
            raise AssertionError("Generation.call must not be used for qwen3.8-flash")

    dummy = type(
        "DummyDashScope",
        (),
        {"api_key": "", "base_http_api_url": "", "Generation": DummyGeneration},
    )
    monkeypatch.setitem(sys.modules, "dashscope", dummy)
    monkeypatch.setattr("httpx.post", fake_httpx_post)

    client = DashScopeQwenClient(
        api_key="k", model="qwen3.8-flash", base_url="https://example/api/v1"
    )

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.qwen"):
        result = await client.translate(
            text="hello",
            system_prompt="PROMPT",
            source_language="ko",
            target_language="en",
            context='- "이전 문장"',
        )

    assert result == "OK35"
    assert "[Basic][LLM] Qwen request [translate][context=yes] ko -> en: 'hello'" in caplog.messages
    assert "[Basic][LLM] Qwen response [translate]: 'OK35'" in caplog.messages


@pytest.mark.asyncio
async def test_qwen_client_logs_basic_request_failure_for_qwen38(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    class FakeHttpxResponse:
        status_code = 429
        text = '{"error":{"message":"quota exceeded"}}'

        @staticmethod
        def json() -> dict[str, object]:
            return {"error": {"message": "quota exceeded"}}

    def fake_httpx_post(_url, **_kwargs):
        return FakeHttpxResponse()

    class DummyGeneration:
        @staticmethod
        def call(**_kwargs):
            raise AssertionError("Generation.call must not be used for qwen3.5 models")

    dummy = type(
        "DummyDashScope",
        (),
        {"api_key": "", "base_http_api_url": "", "Generation": DummyGeneration},
    )
    monkeypatch.setitem(sys.modules, "dashscope", dummy)
    monkeypatch.setattr("httpx.post", fake_httpx_post)

    client = DashScopeQwenClient(
        api_key="k", model="qwen3.8-flash", base_url="https://example/api/v1"
    )

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.qwen"):
        with pytest.raises(RuntimeError, match="quota exceeded"):
            await client.translate(
                text="hello",
                system_prompt="PROMPT",
                source_language="ko",
                target_language="en",
            )

    assert (
        "[Basic][LLM] Qwen request failed [translate]: category=quota code=provider.quota status=429"
        in caplog.messages
    )
    assert "quota exceeded" not in "\n".join(caplog.messages)


@pytest.mark.asyncio
async def test_qwen_client_uses_runtime_logging_for_basic_translate_payloads_for_qwen38(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    class FakeHttpxResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict[str, object]:
            return {"choices": [{"message": {"content": "OK35"}}]}

    def fake_httpx_post(_url, **_kwargs):
        return FakeHttpxResponse()

    class DummyGeneration:
        @staticmethod
        def call(**_kwargs):
            raise AssertionError("Generation.call must not be used for qwen3.5 models")

    dummy = type(
        "DummyDashScope",
        (),
        {"api_key": "", "base_http_api_url": "", "Generation": DummyGeneration},
    )
    monkeypatch.setitem(sys.modules, "dashscope", dummy)
    monkeypatch.setattr("httpx.post", fake_httpx_post)
    runtime_logging = SpyRuntimeLogging(detailed_return=False)

    client = DashScopeQwenClient(
        api_key="k",
        model="qwen3.8-flash",
        base_url="https://example/api/v1",
        runtime_logging=runtime_logging,
    )

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.qwen"):
        result = await client.translate(
            text="hello",
            system_prompt="PROMPT",
            source_language="ko",
            target_language="en",
            context='- "이전 문장"',
        )

    assert result == "OK35"
    assert runtime_logging.basic_messages == [
        ("[Basic][LLM] Qwen request [translate][context=yes] ko -> en: 'hello'", logging.INFO),
        ("[Basic][LLM] Qwen response [translate]: 'OK35'", logging.INFO),
    ]
    assert runtime_logging.detailed_messages == []
    assert caplog.messages == []


@pytest.mark.asyncio
async def test_qwen_client_uses_runtime_logging_for_failure_breadcrumbs_for_qwen38(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    class FakeHttpxResponse:
        status_code = 429
        text = '{"error":{"message":"quota exceeded"}}'

        @staticmethod
        def json() -> dict[str, object]:
            return {"error": {"message": "quota exceeded"}}

    def fake_httpx_post(_url, **_kwargs):
        return FakeHttpxResponse()

    class DummyGeneration:
        @staticmethod
        def call(**_kwargs):
            raise AssertionError("Generation.call must not be used for qwen3.8-flash")

    dummy = type(
        "DummyDashScope",
        (),
        {"api_key": "", "base_http_api_url": "", "Generation": DummyGeneration},
    )
    monkeypatch.setitem(sys.modules, "dashscope", dummy)
    monkeypatch.setattr("httpx.post", fake_httpx_post)
    runtime_logging = SpyRuntimeLogging(detailed_return=False)

    client = DashScopeQwenClient(
        api_key="k",
        model="qwen3.8-flash",
        base_url="https://example/api/v1",
        runtime_logging=runtime_logging,
    )

    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.qwen"):
        with pytest.raises(RuntimeError, match="quota exceeded"):
            await client.translate(
                text="hello",
                system_prompt="PROMPT",
                source_language="ko",
                target_language="en",
            )

    assert runtime_logging.detailed_messages == []
    assert runtime_logging.basic_messages == [
        ("[Basic][LLM] Qwen request [translate][context=no] ko -> en: 'hello'", logging.INFO),
        (
            "[Basic][LLM] Qwen request failed [translate]: category=quota code=provider.quota status=429",
            logging.ERROR,
        ),
    ]
    assert "quota exceeded" not in repr(runtime_logging.basic_messages)
    assert caplog.messages == []
