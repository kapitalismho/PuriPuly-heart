from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.orchestrator.channel_runtime import (
    ChannelRuntime,
    ContextEntry,
    _MergeBuffer,
)
from puripuly_heart.domain.models import Transcript
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass
class FakeOscQueue:
    messages: list = field(default_factory=list)

    def enqueue(self, msg) -> None:  # noqa: ANN001
        self.messages.append(msg)

    def send_typing(self, on: bool) -> None:
        _ = on

    def set_typing_reason(self, reason: str, active: bool) -> None:
        _ = active

    def send_immediate(self, text: str) -> bool:
        _ = text
        return True

    def process_due(self) -> None:
        return None


def test_channel_runtime_keeps_merge_and_history_separate_per_channel() -> None:
    self_runtime = ChannelRuntime(channel="self")
    peer_runtime = ChannelRuntime(channel="peer")

    self_runtime.remember_context(
        "hello",
        timestamp=10.0,
        source_language="en",
        target_language="ko",
    )
    peer_runtime.remember_context(
        "world",
        timestamp=12.0,
        source_language="en",
        target_language="ko",
    )
    self_runtime.merge_buffer = _MergeBuffer(merge_id=uuid4())
    peer_runtime.merge_buffer = _MergeBuffer(merge_id=uuid4())
    self_runtime.merge_buffer.parts.append("self part")
    peer_runtime.merge_buffer.parts.append("peer part")

    assert [entry.text for entry in self_runtime.translation_history] == ["hello"]
    assert [entry.text for entry in peer_runtime.translation_history] == ["world"]
    assert self_runtime.merge_buffer.parts == ["self part"]
    assert peer_runtime.merge_buffer.parts == ["peer part"]


def test_translation_fixture_uses_fixed_self_and_peer_owner_runtimes() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=FakeOscQueue())

    assert harness.self_runtime.channel == "self"
    assert harness.peer_runtime.channel == "peer"
    assert harness.self_owner.runtime is harness.self_runtime
    assert harness.peer_owner.runtime is harness.peer_runtime
    assert harness.self_runtime.translation_history is harness.self_runtime.translation_history
    assert harness.self_runtime.translation_tasks is harness.self_runtime.translation_tasks


def test_self_runtime_state_is_visible_through_the_self_owner() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=FakeOscQueue())
    buffer = _MergeBuffer(merge_id=uuid4())

    harness.self_runtime.merge_buffer = buffer

    assert harness.self_owner.merge_buffer is buffer


@pytest.mark.asyncio
async def test_peer_transcript_stays_in_peer_runtime() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=FakeOscQueue())
    transcript = Transcript(utterance_id=uuid4(), text="peer text", is_final=True, channel="peer")

    await harness.dispatch_transcript(transcript, is_final=True, source="Peer")

    assert transcript.utterance_id not in harness.self_runtime.utterances
    assert transcript.utterance_id in harness.peer_runtime.utterances
    assert harness.peer_runtime.get_source(transcript.utterance_id) == "Peer"


@pytest.mark.asyncio
async def test_clear_live_translation_state_preserves_history_and_stt_task() -> None:
    runtime = ChannelRuntime(channel="self")
    merge_id_1 = uuid4()
    merge_id_2 = uuid4()
    preserved_id = uuid4()
    stt_task = asyncio.create_task(asyncio.sleep(60.0))
    translation_task = asyncio.create_task(asyncio.sleep(60.0))
    spec_task = asyncio.create_task(asyncio.sleep(60.0))
    finalize_wait_task = asyncio.create_task(asyncio.sleep(60.0))
    awaiting_vad_timeout_task = asyncio.create_task(asyncio.sleep(60.0))
    resume_end_timeout_task = asyncio.create_task(asyncio.sleep(60.0))
    all_tasks = [
        stt_task,
        translation_task,
        spec_task,
        finalize_wait_task,
        awaiting_vad_timeout_task,
        resume_end_timeout_task,
    ]

    runtime.stt_task = stt_task
    runtime.translation_tasks[merge_id_1] = translation_task
    runtime.get_or_create_bundle(merge_id_1)
    runtime.get_or_create_bundle(merge_id_2)
    runtime.get_or_create_bundle(preserved_id)
    runtime.utterance_sources[merge_id_1] = "Mic"
    runtime.utterance_sources[merge_id_2] = "Mic"
    runtime.utterance_sources[preserved_id] = "Mic"
    runtime.utterance_start_times[merge_id_1] = 1.0
    runtime.utterance_start_times[merge_id_2] = 2.0
    runtime.utterance_start_times[preserved_id] = 3.0
    runtime.speech_ended_ids.update({merge_id_1, merge_id_2, preserved_id})
    runtime.translation_history.append(
        ContextEntry(
            text="keep this history",
            source_language="ko",
            target_language="en",
            timestamp=1.0,
            channel="self",
        )
    )
    runtime.merge_buffer = _MergeBuffer(
        merge_id=uuid4(),
        utterance_ids=[merge_id_1, merge_id_2],
        spec_task=spec_task,
        finalize_wait_task=finalize_wait_task,
        awaiting_vad_timeout_task=awaiting_vad_timeout_task,
        resume_end_timeout_task=resume_end_timeout_task,
    )

    try:
        await runtime.clear_live_translation_state()

        assert runtime.translation_tasks == {}
        assert runtime.merge_buffer is None
        assert runtime.translation_history == [
            ContextEntry(
                text="keep this history",
                source_language="ko",
                target_language="en",
                timestamp=1.0,
                channel="self",
            )
        ]
        assert runtime.stt_task is stt_task
        assert merge_id_1 not in runtime.utterances
        assert merge_id_2 not in runtime.utterances
        assert merge_id_1 not in runtime.utterance_sources
        assert merge_id_2 not in runtime.utterance_sources
        assert merge_id_1 not in runtime.utterance_start_times
        assert merge_id_2 not in runtime.utterance_start_times
        assert merge_id_1 not in runtime.speech_ended_ids
        assert merge_id_2 not in runtime.speech_ended_ids
        assert preserved_id in runtime.utterances
        assert runtime.utterance_sources == {preserved_id: "Mic"}
        assert runtime.utterance_start_times == {preserved_id: 3.0}
        assert runtime.speech_ended_ids == {preserved_id}
        assert translation_task.cancelled() is True
        assert spec_task.cancelled() is True
        assert finalize_wait_task.cancelled() is True
        assert awaiting_vad_timeout_task.cancelled() is True
        assert resume_end_timeout_task.cancelled() is True
        assert stt_task.done() is False
    finally:
        for task in all_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_clear_live_translation_state_clears_standalone_translation_latency_bookkeeping() -> (
    None
):
    runtime = ChannelRuntime(channel="self")
    standalone_id = uuid4()
    stt_task = asyncio.create_task(asyncio.sleep(60.0))
    translation_task = asyncio.create_task(asyncio.sleep(60.0))
    entry = ContextEntry(
        text="keep history",
        source_language="ko",
        target_language="en",
        timestamp=1.0,
        channel="self",
    )

    runtime.stt_task = stt_task
    runtime.translation_tasks[standalone_id] = translation_task
    runtime.get_or_create_bundle(standalone_id)
    runtime.utterance_sources[standalone_id] = "Mic"
    runtime.utterance_start_times[standalone_id] = 1.0
    runtime.speech_ended_ids.add(standalone_id)
    runtime.translation_history.append(entry)

    try:
        await runtime.clear_live_translation_state()

        assert runtime.translation_tasks == {}
        assert runtime.merge_buffer is None
        assert standalone_id in runtime.utterances
        assert runtime.utterance_sources == {standalone_id: "Mic"}
        assert runtime.utterance_start_times == {}
        assert runtime.speech_ended_ids == set()
        assert runtime.translation_history == [entry]
        assert runtime.stt_task is stt_task
        assert translation_task.cancelled() is True
        assert stt_task.done() is False
    finally:
        for task in (stt_task, translation_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(stt_task, translation_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_reset_runtime_state_clears_both_channel_runtimes() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=FakeOscQueue())
    self_id = uuid4()
    peer_id = uuid4()
    self_task = asyncio.create_task(asyncio.sleep(60.0))
    peer_task = asyncio.create_task(asyncio.sleep(60.0))
    self_stt_task = asyncio.create_task(asyncio.sleep(60.0))
    peer_stt_task = asyncio.create_task(asyncio.sleep(60.0))

    harness.self_runtime.stt_task = self_stt_task
    harness.peer_runtime.stt_task = peer_stt_task

    harness.self_runtime.translation_tasks[self_id] = self_task
    harness.peer_runtime.translation_tasks[peer_id] = peer_task
    harness.self_runtime.get_or_create_bundle(self_id)
    harness.peer_runtime.get_or_create_bundle(peer_id)
    harness.self_runtime.utterance_sources[self_id] = "Mic"
    harness.peer_runtime.utterance_sources[peer_id] = "Peer"
    harness.self_runtime.utterance_start_times[self_id] = 1.0
    harness.peer_runtime.utterance_start_times[peer_id] = 2.0
    harness.self_runtime.speech_ended_ids.add(self_id)
    harness.peer_runtime.speech_ended_ids.add(peer_id)
    harness.self_runtime.merge_buffer = _MergeBuffer(merge_id=uuid4(), utterance_ids=[self_id])
    harness.peer_runtime.merge_buffer = _MergeBuffer(merge_id=uuid4(), utterance_ids=[peer_id])
    harness.self_runtime.translation_history.append(
        ContextEntry(
            text="self line",
            source_language="en",
            target_language="ko",
            timestamp=1.0,
            channel="self",
        )
    )
    harness.peer_runtime.translation_history.append(
        ContextEntry(
            text="peer line",
            source_language="en",
            target_language="ko",
            timestamp=1.0,
            channel="peer",
        )
    )

    try:
        await harness.reset_runtime_state()

        assert harness.self_runtime.translation_tasks == {}
        assert harness.peer_runtime.translation_tasks == {}
        assert harness.self_runtime.utterances == {}
        assert harness.peer_runtime.utterances == {}
        assert harness.self_runtime.utterance_sources == {}
        assert harness.peer_runtime.utterance_sources == {}
        assert harness.self_runtime.utterance_start_times == {}
        assert harness.peer_runtime.utterance_start_times == {}
        assert harness.self_runtime.speech_ended_ids == set()
        assert harness.peer_runtime.speech_ended_ids == set()
        assert harness.self_runtime.merge_buffer is None
        assert harness.peer_runtime.merge_buffer is None
        assert harness.self_runtime.translation_history == []
        assert harness.peer_runtime.translation_history == []
        assert harness.self_runtime.stt_task is None
        assert harness.peer_runtime.stt_task is None
        assert harness.self_runtime.stt_task is None
    finally:
        for task in (self_task, peer_task, self_stt_task, peer_stt_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            self_task,
            peer_task,
            self_stt_task,
            peer_stt_task,
            return_exceptions=True,
        )
