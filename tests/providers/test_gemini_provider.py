from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

from puripuly_heart.providers.llm.gemini import (
    GeminiClient,
    GeminiLLMProvider,
    GoogleGenaiGeminiClient,
)


@dataclass
class FakeGeminiClient(GeminiClient):
    last_call: dict[str, str] | None = None
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


@pytest.mark.asyncio
async def test_gemini_provider_uses_injected_client():
    fake = FakeGeminiClient()
    provider = GeminiLLMProvider(api_key="k", client=fake)

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


def _install_fake_google(monkeypatch, *, response_text: str | None) -> dict[str, object]:
    state: dict[str, object] = {}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeAutomaticFunctionCallingConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeThinkingLevel:
        MINIMAL = "minimal"
        LOW = "low"

    types_module = ModuleType("google.genai.types")
    types_module.GenerateContentConfig = FakeGenerateContentConfig
    types_module.ThinkingConfig = FakeThinkingConfig
    types_module.AutomaticFunctionCallingConfig = FakeAutomaticFunctionCallingConfig
    types_module.ThinkingLevel = FakeThinkingLevel

    class FakeModels:
        async def generate_content(self, **kwargs):
            state.update(kwargs)
            return SimpleNamespace(text=response_text)

    class FakeAio:
        def __init__(self):
            self.models = FakeModels()

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.aio = FakeAio()

    genai_module = ModuleType("google.genai")
    genai_module.Client = FakeClient
    genai_module.types = types_module

    google_module = ModuleType("google")
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)

    return state


def _install_fake_google_model_list(monkeypatch, *, names: list[str]) -> dict[str, object]:
    state: dict[str, object] = {"list_configs": []}

    class FakeModels:
        async def list(self, config=None):
            state["list_configs"].append(config)

            async def _items():
                for name in names:
                    yield SimpleNamespace(name=name)

            return _items()

    class FakeAio:
        def __init__(self):
            self.models = FakeModels()

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.aio = FakeAio()

    genai_module = ModuleType("google.genai")
    genai_module.Client = FakeClient

    google_module = ModuleType("google")
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    return state


def _install_fake_google_model_entries(
    monkeypatch,
    *,
    entries: list[object],
) -> dict[str, object]:
    state: dict[str, object] = {"list_configs": []}

    class FakeModels:
        async def list(self, config=None):
            state["list_configs"].append(config)

            async def _items():
                for entry in entries:
                    yield entry

            return _items()

    class FakeAio:
        def __init__(self):
            self.models = FakeModels()

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.aio = FakeAio()

    genai_module = ModuleType("google.genai")
    genai_module.Client = FakeClient

    google_module = ModuleType("google")
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    return state


@pytest.mark.asyncio
async def test_gemini_verify_api_key_checks_requested_model(monkeypatch):
    state = _install_fake_google_model_list(
        monkeypatch,
        names=["models/gemini-3.1-flash-lite"],
    )

    assert (
        await GeminiLLMProvider.verify_api_key(
            "secret",
            model="gemini-3.1-flash-lite",
        )
        is True
    )
    assert (
        await GeminiLLMProvider.verify_api_key(
            "secret",
            model="gemini-3.1-flash-lite-preview",
        )
        is False
    )
    assert state["list_configs"] == [{"page_size": 1000}, {"page_size": 1000}]


@pytest.mark.asyncio
async def test_gemini_verify_api_key_accepts_base_model_aliases(monkeypatch):
    state = _install_fake_google_model_entries(
        monkeypatch,
        entries=[
            SimpleNamespace(
                name="models/gemini-3.1-flash-lite-001",
                base_model_id="gemini-3.1-flash-lite",
            ),
            SimpleNamespace(
                name="models/gemini-3.7-flash-001",
                baseModelId="gemini-3.7-flash",
            ),
        ],
    )

    assert (
        await GeminiLLMProvider.verify_api_key(
            "secret",
            model="gemini-3.1-flash-lite",
        )
        is True
    )
    assert (
        await GeminiLLMProvider.verify_api_key(
            "secret",
            model="gemini-3.7-flash",
        )
        is True
    )
    assert state["list_configs"] == [{"page_size": 1000}, {"page_size": 1000}]


@pytest.mark.asyncio
async def test_gemini_provider_warmup_and_close_uses_client():
    fake = FakeGeminiClient()
    provider = GeminiLLMProvider(api_key="k", client=fake)

    await provider.warmup()

    assert fake.last_call is not None
    assert fake.last_call["text"] == "warmup"
    assert fake.last_call["system_prompt"] == "Reply with OK only."

    provider._internal_client = fake
    await provider.close()
    assert fake.closed is True
    assert provider._internal_client is None


@pytest.mark.asyncio
async def test_google_genai_client_formats_prompt_and_context(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    state = _install_fake_google(monkeypatch, response_text=" OK ")

    client = GoogleGenaiGeminiClient(api_key="k", model="m")
    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.gemini"):
        result = await client.translate(
            text="hello",
            system_prompt="Translate {source_language} to {target_language}.",
            source_language="ko",
            target_language="en",
            context="a -> b",
        )

    assert result == "OK"
    assert state["contents"] == "<context>\na -> b\n</context>\n\n<input>\nhello\n</input>"
    assert state["config"].system_instruction == "Translate ko to en."
    assert state["config"].thinking_config.thinking_level == "low"
    assert state["config"].automatic_function_calling.disable is True
    assert (
        "[Basic][LLM] Gemini request [translate][context=yes] ko -> en: 'hello'" in caplog.messages
    )
    assert "[Basic][LLM] Gemini response [translate]: 'OK'" in caplog.messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected_thinking_level"),
    [
        ("gemini-3.1-flash-lite", "minimal"),
        ("models/gemini-3.1-flash-lite", "minimal"),
        ("gemini-3.7-flash", "low"),
        ("m", "low"),
    ],
)
async def test_google_genai_client_thinking_level_follows_model(
    monkeypatch, model: str, expected_thinking_level: str
):
    state = _install_fake_google(monkeypatch, response_text=" OK ")

    client = GoogleGenaiGeminiClient(api_key="k", model=model)
    await client.translate(
        text="hello",
        system_prompt="PROMPT",
        source_language="en",
        target_language="ko",
    )

    assert state["config"].thinking_config.thinking_level == expected_thinking_level


@pytest.mark.asyncio
async def test_google_genai_client_raises_on_empty_response(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    _install_fake_google(monkeypatch, response_text=None)

    client = GoogleGenaiGeminiClient(api_key="k", model="m")
    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.gemini"):
        with pytest.raises(RuntimeError, match="Gemini response did not contain text"):
            await client.translate(
                text="hello",
                system_prompt="PROMPT",
                source_language="en",
                target_language="ko",
            )

    assert "[Basic][LLM] Gemini response missing text [translate]" in caplog.messages


@pytest.mark.asyncio
async def test_google_genai_client_uses_runtime_logging_for_basic_translate_payloads(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    state = _install_fake_google(monkeypatch, response_text=" OK ")
    runtime_logging = SpyRuntimeLogging(detailed_return=False)

    client = GoogleGenaiGeminiClient(api_key="k", model="m", runtime_logging=runtime_logging)
    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.gemini"):
        result = await client.translate(
            text="hello",
            system_prompt="Translate {source_language} to {target_language}.",
            source_language="ko",
            target_language="en",
            context="a -> b",
        )

    assert result == "OK"
    assert state["contents"] == "<context>\na -> b\n</context>\n\n<input>\nhello\n</input>"
    assert runtime_logging.basic_messages == [
        (
            "[Basic][LLM] Gemini request [translate][context=yes] ko -> en: 'hello'",
            logging.INFO,
        ),
        ("[Basic][LLM] Gemini response [translate]: 'OK'", logging.INFO),
    ]
    assert runtime_logging.detailed_messages == []
    assert caplog.messages == []


@pytest.mark.asyncio
async def test_google_genai_client_uses_runtime_logging_for_missing_text_warning(
    monkeypatch, caplog: pytest.LogCaptureFixture
):
    _install_fake_google(monkeypatch, response_text=None)
    runtime_logging = SpyRuntimeLogging(detailed_return=False)

    client = GoogleGenaiGeminiClient(api_key="k", model="m", runtime_logging=runtime_logging)
    with caplog.at_level(logging.INFO, logger="puripuly_heart.providers.llm.gemini"):
        with pytest.raises(RuntimeError, match="Gemini response did not contain text"):
            await client.translate(
                text="hello",
                system_prompt="PROMPT",
                source_language="en",
                target_language="ko",
            )

    assert runtime_logging.detailed_messages == []
    assert runtime_logging.basic_messages == [
        ("[Basic][LLM] Gemini request [translate][context=no] en -> ko: 'hello'", logging.INFO),
        ("[Basic][LLM] Gemini response missing text [translate]", logging.ERROR),
    ]
    assert caplog.messages == []
