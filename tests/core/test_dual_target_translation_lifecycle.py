from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.domain.models import Translation
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass
class ConcurrentRecordingProvider:
    started: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        await self.started.put(
            {
                "utterance_id": utterance_id,
                "text": text,
                "system_prompt": system_prompt,
                "source_language": source_language,
                "target_language": target_language,
                "context": context,
            }
        )
        await self.release.wait()
        return Translation(
            utterance_id=utterance_id,
            text=f"translated-{target_language}",
            source_text=text,
            source_language=source_language,
            target_language=target_language,
        )

    async def close(self) -> None:
        return None


@dataclass
class TargetControlledProvider:
    started: asyncio.Queue[tuple[str, str]] = field(default_factory=asyncio.Queue)
    releases: dict[tuple[str, str], asyncio.Event] = field(default_factory=dict)

    async def translate(
        self,
        *,
        utterance_id: UUID,
        text: str,
        system_prompt: str,
        source_language: str,
        target_language: str,
        context: str = "",
    ) -> Translation:
        _ = system_prompt, context
        key = (text, target_language)
        release = self.releases.setdefault(key, asyncio.Event())
        await self.started.put(key)
        await release.wait()
        return Translation(
            utterance_id=utterance_id,
            text=f"translated-{target_language}",
            source_text=text,
            source_language=source_language,
            target_language=target_language,
        )

    async def close(self) -> None:
        return None


class RecordingOsc:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def enqueue(self, message: object) -> None:
        self.messages.append(message)

    def send_immediate(self, _text: str) -> bool:
        return True

    def send_typing(self, _is_typing: bool) -> None:
        return None

    def set_typing_reason(self, _reason: str, _active: bool) -> None:
        return None

    def clear_typing_reasons(self) -> None:
        return None

    def process_due(self) -> None:
        return None


@pytest.mark.asyncio
async def test_dual_target_turns_admit_in_order_and_execute_without_provider_barriers() -> None:
    provider = ConcurrentRecordingProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOsc(),
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )

    try:
        first_parent = await harness.self_owner.submit_text("first source")
        first_calls = [
            await asyncio.wait_for(provider.started.get(), timeout=1),
            await asyncio.wait_for(provider.started.get(), timeout=1),
        ]
        second_parent = await harness.self_owner.submit_text("second source")
        second_calls = [
            await asyncio.wait_for(provider.started.get(), timeout=1),
            await asyncio.wait_for(provider.started.get(), timeout=1),
        ]

        assert {call["target_language"] for call in first_calls} == {"zh-CN", "ja"}
        assert {call["target_language"] for call in second_calls} == {"zh-CN", "ja"}
        assert len({call["utterance_id"] for call in first_calls}) == 2
        assert len({call["utterance_id"] for call in second_calls}) == 2
        assert first_parent != second_parent
        assert all(call["context"] == "" for call in first_calls)
        assert all(call["context"] == '- [self] "first source"' for call in second_calls)
    finally:
        provider.release.set()
        await harness.translation_turns.wait_for_idle()
        await harness.translation_turns.close()


@pytest.mark.asyncio
async def test_failed_child_creation_does_not_retain_prepared_self_request() -> None:
    provider = ConcurrentRecordingProvider()
    provider.release.set()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOsc(),
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    original_child_created = harness.translation_turns.on_child_created

    async def fail_primary_creation(child) -> None:
        if child.channel == "self" and child.target_index == 0:
            raise RuntimeError("primary child creation failed")
        await original_child_created(child)

    harness.translation_turns.on_child_created = fail_primary_creation
    try:
        await harness.self_owner.submit_text("source")
        await harness.translation_turns.wait_for_idle()
        assert harness.self_owner._admitted_requests == {}
        assert harness.translation_turns.has_resources is False
    finally:
        await harness.translation_turns.close()


@pytest.mark.asyncio
async def test_end_to_end_secondary_first_publishes_progressive_parent_snapshots() -> None:
    provider = TargetControlledProvider()
    osc = RecordingOsc()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )

    try:
        parent_id = await harness.self_owner.submit_text("source text")
        started = {
            await asyncio.wait_for(provider.started.get(), timeout=1),
            await asyncio.wait_for(provider.started.get(), timeout=1),
        }
        assert started == {("source text", "zh-CN"), ("source text", "ja")}

        provider.releases[("source text", "ja")].set()
        for _ in range(100):
            if len(osc.messages) == 1:
                break
            await asyncio.sleep(0)
        assert [(message.utterance_id, message.text) for message in osc.messages] == [
            (parent_id, "translated-ja")
        ]

        provider.releases[("source text", "zh-CN")].set()
        await harness.translation_turns.wait_for_idle()
        assert [(message.utterance_id, message.text) for message in osc.messages] == [
            (parent_id, "translated-ja"),
            (parent_id, "translated-zh-CN\ntranslated-ja"),
        ]
        assert harness.output_projection.self_turn_aggregate_count == 0
    finally:
        for release in provider.releases.values():
            release.set()
        await harness.translation_turns.close()


@pytest.mark.asyncio
async def test_direct_self_translation_uses_dual_target_turn_lifecycle() -> None:
    provider = TargetControlledProvider()
    osc = RecordingOsc()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )
    parent_id = uuid4()
    task = asyncio.create_task(
        harness.self_owner.translate_and_enqueue(parent_id, "source text")
    )

    try:
        started = {
            await asyncio.wait_for(provider.started.get(), timeout=1),
            await asyncio.wait_for(provider.started.get(), timeout=1),
        }
        assert started == {("source text", "zh-CN"), ("source text", "ja")}

        provider.releases[("source text", "ja")].set()
        for _ in range(100):
            if len(osc.messages) == 1:
                break
            await asyncio.sleep(0)
        assert [(message.utterance_id, message.text) for message in osc.messages] == [
            (parent_id, "translated-ja")
        ]

        provider.releases[("source text", "zh-CN")].set()
        await asyncio.wait_for(task, timeout=1)
        assert [(message.utterance_id, message.text) for message in osc.messages] == [
            (parent_id, "translated-ja"),
            (parent_id, "translated-zh-CN\ntranslated-ja"),
        ]
    finally:
        for release in provider.releases.values():
            release.set()
        await asyncio.gather(task, return_exceptions=True)
        await harness.translation_turns.close()
