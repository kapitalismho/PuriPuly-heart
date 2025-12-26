from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from ajebal_daera_translator.config.prompts import load_prompt_for_provider
from ajebal_daera_translator.core.orchestrator.hub import ClientHub
from ajebal_daera_translator.domain.models import Translation
from ajebal_daera_translator.providers.llm.qwen import DashScopeQwenClient


class FakeClock:
    def __init__(self, now_value: float = 0.0) -> None:
        self._now = now_value

    def now(self) -> float:
        return self._now


@dataclass
class FakeOscQueue:
    messages: list = None

    def __post_init__(self) -> None:
        if self.messages is None:
            self.messages = []

    def enqueue(self, msg) -> None:
        self.messages.append(msg)

    def send_typing(self, on: bool) -> None:
        _ = on

    def process_due(self) -> None:
        return


@dataclass
class FakeLLMProvider:
    last_prompt: str | None = None

    async def translate(
        self,
        *,
        utterance_id,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        _ = (text, source_language, target_language, context)
        self.last_prompt = system_prompt
        return Translation(utterance_id=utterance_id, text="ok")

    async def close(self) -> None:
        return


class DummyGeneration:
    last_call: dict | None = None

    @classmethod
    def call(cls, *_, **kwargs):
        cls.last_call = kwargs
        class Response:
            status_code = 200
            output = {"choices": [{"message": {"content": "OK"}}]}

        return Response()


def test_load_prompt_for_qwen_matches_file() -> None:
    prompt = load_prompt_for_provider("qwen")
    raw = Path("prompts/qwen.txt").read_text(encoding="utf-8").strip()
    assert prompt == raw
    assert prompt


@pytest.mark.asyncio
async def test_hub_substitutes_language_placeholders() -> None:
    fake_llm = FakeLLMProvider()
    hub = ClientHub(
        stt=None,
        llm=fake_llm,
        osc=FakeOscQueue(),
        clock=FakeClock(),
        source_language="ko",
        target_language="en",
        system_prompt="Translate ${sourceName} to ${targetName}.",
    )

    await hub._translate_and_enqueue(uuid4(), "hello")

    assert fake_llm.last_prompt is not None
    assert "${sourceName}" not in fake_llm.last_prompt
    assert "${targetName}" not in fake_llm.last_prompt
    assert "Korean" in fake_llm.last_prompt
    assert "English" in fake_llm.last_prompt


@pytest.mark.asyncio
async def test_qwen_client_builds_prompt_with_context(monkeypatch) -> None:
    dummy = SimpleNamespace(api_key=None, base_http_api_url=None, Generation=DummyGeneration)
    monkeypatch.setitem(sys.modules, "dashscope", dummy)

    client = DashScopeQwenClient(api_key="key", model="qwen-mt-flash")
    result = await client.translate(
        text="hello",
        system_prompt="PROMPT",
        source_language="ko",
        target_language="en",
        context="CTX",
    )

    assert result == "OK"
    assert DummyGeneration.last_call is not None
    messages = DummyGeneration.last_call.get("messages")
    assert messages == [{"role": "user", "content": "PROMPT\n\ncontext:\nCTX\n\nTranslate: hello"}]


@pytest.mark.asyncio
async def test_qwen_client_builds_prompt_without_context(monkeypatch) -> None:
    dummy = SimpleNamespace(api_key=None, base_http_api_url=None, Generation=DummyGeneration)
    monkeypatch.setitem(sys.modules, "dashscope", dummy)

    client = DashScopeQwenClient(api_key="key", model="qwen-mt-flash")
    result = await client.translate(
        text="hello",
        system_prompt="PROMPT",
        source_language="ko",
        target_language="en",
        context="",
    )

    assert result == "OK"
    assert DummyGeneration.last_call is not None
    messages = DummyGeneration.last_call.get("messages")
    assert messages == [{"role": "user", "content": "PROMPT\n\nhello"}]
