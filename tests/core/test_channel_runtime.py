from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.orchestrator.channel_runtime import (
    _LOW_LATENCY_COMMITTED_TOMBSTONE_LIMIT,
    ChannelRuntime,
    ContextEntry,
    _MergeBuffer,
)
from puripuly_heart.domain.events import STTFinalEvent
from puripuly_heart.domain.models import Transcript
from tests.helpers.translation_owners import (
    compose_translation_test_harness,
)


@dataclass
class FakeOscQueue:
    messages: list = field(default_factory=list)

    def enqueue(self, msg) -> None:  # noqa: ANN001
        self.messages.append(msg)

    def send_typing(self, on: bool) -> None:
        _ = on

    def set_typing_reason(self, reason: str, active: bool) -> None:
        _ = active

    def send_immediate(self, text: str) -> tuple[bool, None]:
        _ = text
        return True, None

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


def test_low_latency_committed_utterance_tombstones_are_bounded() -> None:
    runtime = ChannelRuntime(channel="self")
    utterance_ids = [uuid4() for _ in range(_LOW_LATENCY_COMMITTED_TOMBSTONE_LIMIT + 1)]

    for utterance_id in utterance_ids:
        runtime.remember_low_latency_committed_utterance(utterance_id)

    assert len(runtime.low_latency_committed_utterance_ids) == (
        _LOW_LATENCY_COMMITTED_TOMBSTONE_LIMIT
    )
    assert utterance_ids[0] not in runtime.low_latency_committed_utterance_ids
    assert utterance_ids[-1] in runtime.low_latency_committed_utterance_ids

    runtime.remember_low_latency_committed_utterance(utterance_ids[-1])

    assert len(runtime.low_latency_committed_utterance_ids) == (
        _LOW_LATENCY_COMMITTED_TOMBSTONE_LIMIT
    )


@pytest.mark.asyncio
async def test_peer_transcript_stays_in_peer_runtime() -> None:
    harness = compose_translation_test_harness(stt=None, llm=None, osc=FakeOscQueue())
    transcript = Transcript(utterance_id=uuid4(), text="peer text", is_final=True, channel="peer")

    await harness.dispatch_stt_event(STTFinalEvent(transcript.utterance_id, transcript))
    assert transcript.utterance_id not in harness.self_runtime.utterances
    assert len(harness.peer_runtime.utterances) == 1
    peer_bundle = next(iter(harness.peer_runtime.utterances.values()))
    assert peer_bundle.final is not None
    assert peer_bundle.final.text == transcript.text
    assert harness.peer_runtime.get_source(peer_bundle.utterance_id) == "Peer"


@pytest.mark.asyncio
async def test_clear_live_translation_state_preserves_history_and_stt_task() -> None:
    runtime = ChannelRuntime(channel="self")
    merge_id_1 = uuid4()
    merge_id_2 = uuid4()
    preserved_id = uuid4()
    stt_task = asyncio.create_task(asyncio.sleep(60.0))
    translation_task = asyncio.create_task(asyncio.sleep(60.0))
    finalize_wait_task = asyncio.create_task(asyncio.sleep(60.0))
    awaiting_vad_timeout_task = asyncio.create_task(asyncio.sleep(60.0))
    resume_end_timeout_task = asyncio.create_task(asyncio.sleep(60.0))
    all_tasks = [
        stt_task,
        translation_task,
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
@pytest.mark.parametrize("channel", ["self", "peer"])
async def test_channel_reset_cancels_translation_but_not_provider_owned_task(channel) -> None:
    runtime = ChannelRuntime(channel=channel)
    utterance_id = uuid4()
    translation_task = asyncio.create_task(asyncio.sleep(60))
    stt_task = asyncio.create_task(asyncio.sleep(60))
    runtime.stt_task = stt_task
    runtime.translation_tasks[utterance_id] = translation_task
    runtime.get_or_create_bundle(utterance_id)
    runtime.utterance_sources[utterance_id] = "synthetic source"
    runtime.utterance_start_times[utterance_id] = 1
    runtime.speech_ended_ids.add(utterance_id)
    runtime.remember_low_latency_committed_utterance(utterance_id)
    runtime.merge_buffer = _MergeBuffer(merge_id=uuid4(), utterance_ids=[utterance_id])
    runtime.remember_context(
        "synthetic history", timestamp=1, source_language="en", target_language="ko"
    )
    try:
        await runtime.reset_runtime_state()

        assert translation_task.cancelled()
        assert not stt_task.done()
        assert runtime.translation_tasks == {}
        assert runtime.stt_task is None
        assert runtime.utterances == {}
        assert runtime.utterance_sources == {}
        assert runtime.utterance_start_times == {}
        assert runtime.speech_ended_ids == set()
        assert runtime.low_latency_committed_utterance_ids == {}
        assert runtime.merge_buffer is None
        assert runtime.translation_history == []
    finally:
        for task in (translation_task, stt_task):
            task.cancel()
        await asyncio.gather(translation_task, stt_task, return_exceptions=True)
