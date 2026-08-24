from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from puripuly_heart.core.clock import FakeClock
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
    failed_targets: set[str] = field(default_factory=set)

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
        if target_language in self.failed_targets:
            raise RuntimeError("target failed")
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


class RecordingRuntimeLogging:
    mode = "detailed"

    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit_basic(self, message: str, *, level: int = 20) -> None:
        _ = level
        self.messages.append(message)

    def emit_detailed(self, message: str, *, level: int = 20) -> bool:
        _ = level
        self.messages.append(message)
        return True

    def emit_detailed_lazy(self, build_message, *, level: int = 20) -> bool:
        _ = level
        self.messages.append(build_message())
        return True


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
async def test_dual_target_observability_distinguishes_parent_latency_milestones() -> None:
    provider = TargetControlledProvider()
    osc = RecordingOsc()
    clock = FakeClock(_now=10.0)
    runtime_logging = RecordingRuntimeLogging()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=osc,
        clock=clock,
        runtime_logging=runtime_logging,
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )

    try:
        parent_id = await harness.self_owner.submit_text("sensitive source text")
        started = {
            await asyncio.wait_for(provider.started.get(), timeout=1),
            await asyncio.wait_for(provider.started.get(), timeout=1),
        }
        assert started == {
            ("sensitive source text", "zh-CN"),
            ("sensitive source text", "ja"),
        }

        clock.advance(0.25)
        provider.releases[("sensitive source text", "ja")].set()
        for _ in range(100):
            if len(osc.messages) == 1:
                break
            await asyncio.sleep(0)

        clock.advance(0.25)
        provider.releases[("sensitive source text", "zh-CN")].set()
        await harness.translation_turns.wait_for_idle()

        admitted = next(
            message
            for message in runtime_logging.messages
            if "translation_turn_admitted" in message
        )
        target_starts = [
            message
            for message in runtime_logging.messages
            if "translation_target_started" in message
        ]
        target_completions = [
            message
            for message in runtime_logging.messages
            if "translation_target_completed" in message
        ]
        first_visible = next(
            message
            for message in runtime_logging.messages
            if "translation_first_result_published" in message
        )
        complete_visible = next(
            message
            for message in runtime_logging.messages
            if "translation_complete_result_published" in message
        )

        assert f"parent_utterance_id={parent_id}" in admitted
        assert "turn_order=0" in admitted
        assert "presentation_revision=0" in admitted
        assert len(target_starts) == 2
        assert any(
            "target_index=0" in message and "target_language=zh-CN" in message
            for message in target_starts
        )
        assert any(
            "target_index=1" in message and "target_language=ja" in message
            for message in target_starts
        )
        assert len(target_completions) == 2
        assert any("presentation_revision=1" in message for message in target_completions)
        assert any(
            "presentation_revision=2" in message
            and "all_targets_terminal_elapsed_ms=500" in message
            for message in target_completions
        )
        assert "target_indexes=(1,)" in first_visible
        assert "target_languages=('ja',)" in first_visible
        assert "first_success_elapsed_ms=250" in first_visible
        assert "all_targets_terminal_elapsed_ms=None" in first_visible
        assert "first_visible_elapsed_ms=250" in first_visible
        assert "complete_visible_elapsed_ms=None" in first_visible
        assert "target_indexes=(0, 1)" in complete_visible
        assert "target_languages=('zh-CN', 'ja')" in complete_visible
        assert "first_success_elapsed_ms=250" in complete_visible
        assert "all_targets_terminal_elapsed_ms=500" in complete_visible
        assert "first_visible_elapsed_ms=None" in complete_visible
        assert "complete_visible_elapsed_ms=500" in complete_visible
        combined = "\n".join(runtime_logging.messages)
        assert "sensitive source text" not in combined
        assert "translated-ja" not in combined
        assert "translated-zh-CN" not in combined
    finally:
        for release in provider.releases.values():
            release.set()
        await harness.translation_turns.close()


@pytest.mark.asyncio
async def test_dual_target_observability_records_all_terminal_time_after_failure() -> None:
    provider = TargetControlledProvider(failed_targets={"zh-CN"})
    clock = FakeClock(_now=20.0)
    runtime_logging = RecordingRuntimeLogging()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOsc(),
        clock=clock,
        runtime_logging=runtime_logging,
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )

    try:
        await harness.self_owner.submit_text("private failure source")
        await asyncio.wait_for(provider.started.get(), timeout=1)
        await asyncio.wait_for(provider.started.get(), timeout=1)
        clock.advance(0.25)
        provider.releases[("private failure source", "ja")].set()
        for _ in range(100):
            if any(
                "translation_first_result_published" in message
                for message in runtime_logging.messages
            ):
                break
            await asyncio.sleep(0)
        clock.advance(0.25)
        provider.releases[("private failure source", "zh-CN")].set()
        await harness.translation_turns.wait_for_idle()

        failed_completion = next(
            message
            for message in runtime_logging.messages
            if "translation_target_completed" in message and "outcome=failed" in message
        )
        assert "presentation_revision=1" in failed_completion
        assert "first_success_elapsed_ms=250" in failed_completion
        assert "all_targets_terminal_elapsed_ms=500" in failed_completion
        assert not any(
            "translation_complete_result_published" in message
            for message in runtime_logging.messages
        )
        assert "private failure source" not in "\n".join(runtime_logging.messages)
    finally:
        for release in provider.releases.values():
            release.set()
        await harness.translation_turns.close()


@pytest.mark.asyncio
async def test_dual_target_cancellation_records_each_outcome_and_all_terminal_time() -> None:
    provider = TargetControlledProvider()
    clock = FakeClock(_now=30.0)
    runtime_logging = RecordingRuntimeLogging()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=RecordingOsc(),
        clock=clock,
        runtime_logging=runtime_logging,
        source_language="en",
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
    )

    try:
        await harness.self_owner.submit_text("private cancellation source")
        await asyncio.wait_for(provider.started.get(), timeout=1)
        await asyncio.wait_for(provider.started.get(), timeout=1)
        clock.advance(0.5)

        await harness.translation_turns.cancel_pending(channel="self")

        target_completions = [
            message
            for message in runtime_logging.messages
            if "translation_target_completed" in message
        ]
        assert len(target_completions) == 2
        assert all("outcome=cancelled" in message for message in target_completions)
        assert (
            sum("all_targets_terminal_elapsed_ms=500" in message for message in target_completions)
            == 1
        )
        assert "private cancellation source" not in "\n".join(runtime_logging.messages)
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
    task = asyncio.create_task(harness.self_owner.translate_and_enqueue(parent_id, "source text"))

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
