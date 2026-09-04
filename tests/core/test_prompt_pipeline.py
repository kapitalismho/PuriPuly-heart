from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.config.prompts import _reset_prompt_cache_for_tests
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.orchestrator.translation_request import DirectTranslationRequest
from puripuly_heart.domain.models import Translation
from tests.helpers.translation_owners import compose_translation_test_harness


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

    def set_typing_reason(self, reason: str, active: bool) -> None:
        _ = active

    def process_due(self) -> None:
        return


@dataclass
class FakeLLMProvider:
    last_prompt: str | None = None
    last_source_language: str | None = None
    last_context: str | None = None
    calls: list[dict[str, str]] = field(default_factory=list)

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
        _ = (text, target_language)
        self.last_prompt = system_prompt
        self.last_source_language = source_language
        self.last_context = context
        self.calls.append(
            {
                "text": text,
                "source_language": source_language,
                "context": context,
            }
        )
        return Translation(utterance_id=utterance_id, text="ok")

    async def close(self) -> None:
        return


@pytest.mark.asyncio
async def test_translation_substitutes_language_placeholders() -> None:
    fake_llm = FakeLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=fake_llm,
        osc=FakeOscQueue(),
        clock=FakeClock(),
        source_language="ko",
        target_language="en",
        system_prompt="Translate ${sourceName} to ${targetName}.",
    )

    await harness.process_translation(uuid4(), "hello")

    assert fake_llm.last_prompt is not None
    assert "${sourceName}" not in fake_llm.last_prompt
    assert "${sourceTextRef}" not in fake_llm.last_prompt
    assert "${targetName}" not in fake_llm.last_prompt
    assert "Korean" in fake_llm.last_prompt
    assert "English" in fake_llm.last_prompt


@pytest.mark.asyncio
async def test_translation_renders_dynamic_prompt_placeholders() -> None:
    fake_llm = FakeLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=fake_llm,
        osc=FakeOscQueue(),
        clock=FakeClock(),
        source_language="ko",
        target_language="en",
        system_prompt=(
            "${sourceName}|${targetName}|${inputChannel}|"
            "${targetLanguageRules}|${translationExamples}"
        ),
    )

    await harness.process_translation(uuid4(), "안녕")

    assert fake_llm.last_prompt is not None
    assert "Korean|English|self|" in fake_llm.last_prompt
    assert "Use contractions" in fake_llm.last_prompt
    assert "Context Use Example" in fake_llm.last_prompt
    assert "${inputChannel}" not in fake_llm.last_prompt
    assert "${targetLanguageRules}" not in fake_llm.last_prompt
    assert "${translationExamples}" not in fake_llm.last_prompt


def test_translation_renders_peer_runtime_dynamic_prompt_placeholders() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=FakeLLMProvider(),
        osc=FakeOscQueue(),
        clock=FakeClock(),
        source_language="ko",
        target_language="en",
        peer_translation_enabled=True,
        peer_source_language="en",
        peer_target_language="ja",
        system_prompt=(
            "${sourceName}|${targetName}|${inputChannel}|"
            "${targetLanguageRules}|${translationExamples}"
        ),
    )

    prompt, _, _ = harness.prepare_translation_request("hello", runtime=harness.peer_runtime)

    assert "English|Japanese|peer|" in prompt
    assert "Korean|English" not in prompt
    assert "タメ口" in prompt
    assert "Context Use Example" in prompt
    assert "${sourceName}" not in prompt
    assert "${targetName}" not in prompt
    assert "${inputChannel}" not in prompt
    assert "${targetLanguageRules}" not in prompt
    assert "${translationExamples}" not in prompt


def test_peer_auto_without_detected_language_uses_input_source_ref() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=FakeLLMProvider(),
        osc=FakeOscQueue(),
        clock=FakeClock(),
        peer_translation_enabled=True,
        peer_source_language="en",
        peer_target_language="ja",
        peer_source_mode="auto",
        system_prompt="Interpret ${sourceTextRef} to ${targetName}|${sourceName}",
    )

    prompt, _, _ = harness.prepare_translation_request("hello", runtime=harness.peer_runtime)

    assert prompt == "Interpret <input> to Japanese|<input>"
    assert "English" not in prompt


def test_peer_auto_with_detected_language_keeps_named_source_ref() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=FakeLLMProvider(),
        osc=FakeOscQueue(),
        clock=FakeClock(),
        peer_translation_enabled=True,
        peer_source_language="en",
        peer_target_language="ja",
        peer_source_mode="auto",
        system_prompt="Interpret ${sourceTextRef} to ${targetName}|${sourceName}",
    )

    prompt, _, _ = harness.prepare_translation_request(
        "你好",
        runtime=harness.peer_runtime,
        detected_language="zh",
    )

    assert prompt == "Interpret the Chinese text to Japanese|Chinese"


def test_peer_manual_without_detected_language_keeps_named_source_ref() -> None:
    harness = compose_translation_test_harness(
        stt=None,
        llm=FakeLLMProvider(),
        osc=FakeOscQueue(),
        clock=FakeClock(),
        peer_translation_enabled=True,
        peer_source_language="en",
        peer_target_language="ja",
        peer_source_mode="manual",
        system_prompt="Interpret ${sourceTextRef} to ${targetName}|${sourceName}",
    )

    prompt, _, _ = harness.prepare_translation_request("hello", runtime=harness.peer_runtime)

    assert prompt == "Interpret the English text to Japanese|English"


@pytest.mark.asyncio
async def test_detected_peer_language_drives_prompt_context_and_request() -> None:
    fake_llm = FakeLLMProvider()
    clock = FakeClock()
    harness = compose_translation_test_harness(
        stt=None,
        llm=fake_llm,
        osc=FakeOscQueue(),
        clock=clock,
        peer_translation_enabled=True,
        peer_source_language="en",
        peer_target_language="ja",
        system_prompt="${sourceName}|${targetName}",
    )
    harness.peer_runtime.remember_context(
        "previous Chinese run",
        timestamp=clock.now(),
        source_language="zh",
        target_language="ja",
    )

    await harness.translation_requests.translate(
        DirectTranslationRequest(
            utterance_id=uuid4(),
            text="你好",
            channel="peer",
            detected_language="zh",
        )
    )

    assert fake_llm.last_prompt == "Chinese|Japanese"
    assert fake_llm.last_source_language == "zh"
    assert fake_llm.last_context is not None
    assert "previous Chinese run" in fake_llm.last_context


@pytest.mark.asyncio
async def test_sequential_detected_peer_runs_reuse_normalized_language_context() -> None:
    fake_llm = FakeLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=fake_llm,
        osc=FakeOscQueue(),
        clock=FakeClock(),
        peer_translation_enabled=True,
        peer_source_language="en",
        peer_target_language="ja",
        system_prompt="${sourceName}|${targetName}",
    )

    await harness.process_translation(
        uuid4(),
        "first Chinese run",
        runtime=harness.peer_runtime,
        detected_language="zh",
    )
    await harness.process_translation(
        uuid4(),
        "second Chinese run",
        runtime=harness.peer_runtime,
        detected_language="zh",
    )

    assert [call["source_language"] for call in fake_llm.calls] == ["zh", "zh"]
    assert "first Chinese run" in fake_llm.calls[1]["context"]
    assert [entry.source_language for entry in harness.peer_runtime.translation_history] == [
        "zh",
        "zh",
    ]


@pytest.mark.asyncio
async def test_unmapped_detected_peer_language_uses_source_only_path() -> None:
    fake_llm = FakeLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=fake_llm,
        osc=FakeOscQueue(),
        clock=FakeClock(),
        peer_translation_enabled=True,
        peer_source_language="en",
        peer_target_language="ja",
    )

    await harness.process_translation(
        uuid4(),
        "unmapped",
        runtime=harness.peer_runtime,
        detected_language="xx",
    )

    assert fake_llm.last_prompt is None
    assert harness.peer_runtime.translation_history == []


@pytest.mark.asyncio
async def test_translation_renders_custom_prompt_without_dynamic_placeholders() -> None:
    fake_llm = FakeLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=fake_llm,
        osc=FakeOscQueue(),
        clock=FakeClock(),
        source_language="ja",
        target_language="ko",
        system_prompt="Custom ${sourceName} to ${targetName} prompt.",
    )

    await harness.process_translation(uuid4(), "こんにちは")

    assert fake_llm.last_prompt == "Custom Japanese to Korean prompt."


@pytest.mark.asyncio
async def test_translation_request_does_not_read_prompt_files_after_warmup(monkeypatch) -> None:
    _reset_prompt_cache_for_tests()
    fake_llm = FakeLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=fake_llm,
        osc=FakeOscQueue(),
        clock=FakeClock(),
        source_language="en",
        target_language="ja",
        system_prompt="${sourceName}|${targetName}|${targetLanguageRules}",
    )

    def fail_read(_path):
        raise AssertionError("prompt files must not be read during request assembly")

    monkeypatch.setattr("puripuly_heart.config.prompts._read_prompt_text", fail_read)

    await harness.process_translation(uuid4(), "hello")

    assert fake_llm.last_prompt is not None
    assert "English|Japanese" in fake_llm.last_prompt
    assert "タメ口" in fake_llm.last_prompt
