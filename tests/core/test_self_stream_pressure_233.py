from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

import pytest

from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.vad.gating import SpeechEnd
from puripuly_heart.domain.events import UIEventType
from puripuly_heart.domain.models import OSCMessage, Translation
from tests.core.test_self_stream_owned_ingress_233 import (
    ControlledRecognitionProvider,
    OwnedIngress,
)
from tests.helpers.self_stream import Provider, Sender, begin, contribution, until
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass
class Clock:
    value: float = 10.0

    def now(self):
        return self.value


@dataclass
class FailingSender(Sender):
    attempts: int = 0

    def send_chatbox(self, text):
        self.attempts += 1
        if self.attempts == 1:
            raise OSError("synthetic local send failure")
        self.packets.append(text)


@dataclass
class PhysicalProvider(Provider):
    active: int = 0
    peak: int = 0
    ignore_cancellation: bool = False
    cancellation_observed: asyncio.Event = field(default_factory=asyncio.Event)

    async def translate(self, *, utterance_id, text, target_language, context="", **kwargs):
        self.calls.append({"text": text, "target": target_language, "context": context})
        self.active += 1
        self.peak = max(self.peak, self.active)
        release = self.releases.setdefault((text, target_language), asyncio.Event())
        try:
            try:
                await release.wait()
            except asyncio.CancelledError:
                self.cancellation_observed.set()
                self.cancelled.append((text, target_language))
                if not self.ignore_cancellation:
                    raise
                await release.wait()
            return Translation(utterance_id=utterance_id, text=f"{target_language}:{text}")
        finally:
            self.active -= 1


@dataclass
class FailsOnceProvider(Provider):
    failed_targets: tuple[str, ...] = ("en",)

    async def translate(self, *, utterance_id, text, target_language, context="", **kwargs):
        self.calls.append({"text": text, "target": target_language, "context": context})
        count = sum(call["target"] == target_language for call in self.calls)
        if target_language in self.failed_targets and count == 1:
            raise RuntimeError("synthetic first-operation failure")
        return Translation(utterance_id=utterance_id, text=f"{target_language}:{text}")


def snapshots(harness):
    return [
        event.payload for event in drain(harness) if event.type is UIEventType.SELF_TURN_UPDATED
    ]


def drain(harness):
    events = []
    while not harness.ui_events.empty():
        events.append(harness.ui_events.get_nowait())
    return events


async def owned_source(ingress, text, *, ended=True):
    identity, request = await ingress.start()
    if ended:
        await ingress.vad(SpeechEnd(identity))
    await ingress.terminal(request, "final", text)
    return identity


@pytest.mark.asyncio
@pytest.mark.parametrize("targets", [("en",), ("en", "ja")])
async def test_existing_generic_final_fallback_once_keeps_target_obligations(targets):
    provider = FailsOnceProvider(failed_targets=targets)
    sender = Sender()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=100,
        self_target_languages=targets,
        chatbox_include_source=True,
    )
    try:
        await begin(harness, "failure recovery", ended=True)
        await until(lambda: len(provider.calls) == len(targets))
        await asyncio.sleep(0.005)
        assert not any(event.type is UIEventType.ERROR for event in drain(harness))
        assert harness.self_runtime.translation_history == []
        assert sender.packets == []
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert [call["target"] for call in provider.calls].count("en") == 2
        if len(targets) == 2:
            assert [call["target"] for call in provider.calls].count("ja") == 2
            assert sender.packets[-1] == "en:failure recovery\nja:failure recovery"
        else:
            assert sender.packets == ["failure recovery (en:failure recovery)"]
        final = snapshots(harness)[-1]
        assert final.source_closed and final.status == "complete"
        assert all(target.status == "ready" for target in final.targets)
        assert [entry.text for entry in harness.self_runtime.translation_history] == [
            "failure recovery"
        ] * len(targets)
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_source_close_under_full_capacity_preserves_native_permits_and_bounded_admission():
    provider = PhysicalProvider()
    sender = Sender()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SemaphoreLLMProvider(provider, asyncio.Semaphore(2)),
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    texts = [f"source {index}" for index in range(12)]
    try:
        for text in texts:
            await begin(harness, text, ended=True)
        await until(lambda: len(provider.calls) == 2)
        observed = snapshots(harness)
        closed = {snapshot.source_text for snapshot in observed if snapshot.source_closed}
        assert closed == set(texts)
        assert provider.peak == 2 and provider.active == 2
        assert provider.cancelled == []
        assert any(
            snapshot.source_text == "source 3"
            and snapshot.status == "failed"
            and all(target.status == "source_only" for target in snapshot.targets)
            for snapshot in observed
        )
        for text in texts:
            provider.release(text)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 3)
        calls = [call["text"] for call in provider.calls]
        assert len(calls) == 11 and len(set(calls)) == 11
        assert set(calls) == set(texts) - {"source 3"}
        assert provider.peak == 2 and provider.active == 0
        assert provider.cancelled == []
    finally:
        for text in texts:
            provider.release(text)
        await harness.stop()


@pytest.mark.asyncio
async def test_waiting_expiry_progresses_without_reclaiming_live_speculative_work():
    provider = PhysicalProvider()
    sender = Sender()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    harness.translation_turns.self_speech_waiting_ttl_s = 0.02
    try:
        for index in range(4):
            await begin(harness, f"expiry {index}", ended=True)
        await until(lambda: len(provider.calls) == 3)
        await asyncio.sleep(0.04)
        observed = snapshots(harness)
        assert any(
            snapshot.source_text == "expiry 3" and snapshot.status == "expired"
            for snapshot in observed
        )
        assert [call["text"] for call in provider.calls] == ["expiry 0", "expiry 1", "expiry 2"]
        assert provider.active == 3 and provider.cancelled == []
        for index in range(4):
            provider.release(f"expiry {index}")
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert len(provider.calls) == 3
    finally:
        for index in range(4):
            provider.release(f"expiry {index}")
        await harness.stop()


@pytest.mark.asyncio
async def test_off_waits_for_physical_cleanup_without_minting_permit_and_preserves_manual():
    provider = PhysicalProvider(ignore_cancellation=True)
    sender = Sender()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SemaphoreLLMProvider(provider, asyncio.Semaphore(1)),
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        low_latency_mode=True,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    reset = None
    try:
        await begin(harness, "speech")
        await until(lambda: len(provider.calls) == 1)
        reset = asyncio.create_task(harness.self_owner.reset_provider_channel())
        await asyncio.wait_for(provider.cancellation_observed.wait(), 1)
        assert not reset.done() and provider.active == 1
        assert harness.self_runtime.translation_history == []
        provider.release("manual")
        await harness.self_owner.submit_text("manual")
        await asyncio.sleep(0.005)
        assert len(provider.calls) == 1
        assert any(snapshot.status == "cancelled" for snapshot in snapshots(harness))
        provider.release("speech")
        await asyncio.wait_for(reset, 2)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert [call["text"] for call in provider.calls] == ["speech", "manual"]
        assert provider.peak == 1 and provider.active == 0
        assert sender.packets == ["en:manual"]
        assert [entry.text for entry in harness.self_runtime.translation_history] == ["manual"]
    finally:
        provider.release("speech")
        provider.release("manual")
        if reset is not None:
            await asyncio.gather(reset, return_exceptions=True)
        await harness.stop()


@pytest.mark.asyncio
async def test_failed_identical_preview_gets_final_send_opportunity_without_provider_replay():
    provider = Provider()
    sender = FailingSender()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=10,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    try:
        identity = await begin(harness, "same payload")
        provider.release("same payload")
        await until(lambda: sender.attempts == 1)
        assert sender.packets == []
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert sender.attempts == 2 and sender.packets == ["en:same payload"]
        assert len(provider.calls) == 1
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_identical_final_preserves_live_pagination_cursor_and_preview_result_identity():
    provider = Provider()
    sender = Sender()
    clock = Clock()
    paginator = ChatboxPaginator(sender=sender, clock=clock, max_chars=12)
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=paginator,
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=10,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    text = "abcdefghijklmnopqrstuvwx"
    try:
        identity = await begin(harness, text)
        provider.release(text)
        await until(lambda: len(sender.packets) == 1)
        before = snapshots(harness)[-1]
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        final = snapshots(harness)[-1]
        assert final.status == "complete"
        assert final.targets[0].translation.update_id == before.targets[0].translation.update_id
        assert sender.packets == ["en:abcdefghi"]
        clock.value += 3
        paginator.process_due()
        assert sender.packets == ["en:abcdefghi", "jklmnopqrstu"]
        assert len(provider.calls) == 1
    finally:
        await harness.stop()


@pytest.mark.asyncio
async def test_queued_identical_final_is_sent_once_and_is_not_a_delivered_preview():
    provider = Provider()
    sender = Sender()
    clock = Clock()
    paginator = ChatboxPaginator(sender=sender, clock=clock, max_chars=12)
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=paginator,
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=10,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    try:
        paginator.enqueue(OSCMessage(uuid4(), "xxxxxxxxxxxxxxxxxxxxxxxx", created_at=clock.now()))
        identity = await begin(harness, "queued")
        provider.release("queued")
        await until(
            lambda: any(snapshot.targets[0].status == "ready" for snapshot in snapshots(harness))
        )
        assert sender.packets == ["xxxxxxxxxxxx"]
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert sender.packets == ["xxxxxxxxxxxx"]
        clock.value += 3
        paginator.process_due()
        await asyncio.sleep(0)
        assert sender.packets == ["xxxxxxxxxxxx", "xxxxxxxxxxxx", "en:queued"]
        assert len(provider.calls) == 1
    finally:
        await harness.stop()


@dataclass
class BlockedOverlay:
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    events: list[object] = field(default_factory=list)

    async def emit(self, event):
        self.started.set()
        await self.release.wait()
        self.events.append(event)

    def active_self_overlay_metadata(self):
        return None


@pytest.mark.asyncio
async def test_blocked_overlay_and_ui_do_not_block_source_close_or_chatbox_progress():
    provider = Provider()
    sender = Sender()
    overlay = BlockedOverlay()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        overlay_sink=overlay,
        ui_queue_maxsize=1,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=10,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    try:
        identity = await begin(harness, "independent")
        await asyncio.wait_for(overlay.started.wait(), 1)
        provider.release("independent")
        await until(lambda: sender.packets == ["en:independent"])
        assert harness.self_owner.merge_buffer is not None
        assert overlay.events == []
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert harness.self_owner.merge_buffer is None
        assert sender.packets == ["en:independent"]
        assert len(provider.calls) == 1
        first = await harness.ui_events.get()
        assert first.payload.source_closed is False
        async with asyncio.timeout(2):
            while True:
                event = await harness.ui_events.get()
                if (
                    event.type is UIEventType.SELF_TURN_UPDATED
                    and event.payload.status == "complete"
                ):
                    assert event.payload.source_closed
                    break
        overlay.release.set()
        await until(lambda: not harness.output_runtime.has_active_overlay_deliveries)
    finally:
        overlay.release.set()
        await harness.stop()


@pytest.mark.asyncio
async def test_draft_preparation_uses_real_pre_admission_context_and_remembers_source_once():
    provider = Provider()
    sender = Sender()
    harness = compose_translation_test_harness(
        stt=None,
        llm=provider,
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=10,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    try:
        provider.release("accepted seed")
        await harness.self_owner.submit_text("accepted seed")
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        identity = await begin(harness, "draft source")
        await until(lambda: any(call["text"] == "draft source" for call in provider.calls))
        draft_call = next(call for call in provider.calls if call["text"] == "draft source")
        assert "accepted seed" in draft_call["context"]
        assert "draft source" not in draft_call["context"]
        assert [entry.text for entry in harness.self_runtime.translation_history] == [
            "accepted seed"
        ]
        provider.release("intervening manual")
        await harness.self_owner.submit_text("intervening manual")
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
        await until(lambda: harness.self_owner.merge_buffer is None)
        assert [entry.text for entry in harness.self_runtime.translation_history] == [
            "accepted seed",
            "intervening manual",
            "draft source",
        ]
        provider.release("draft source")
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert [call["text"] for call in provider.calls].count("draft source") == 1
        assert "intervening manual" not in draft_call["context"]
    finally:
        provider.release("draft source")
        await harness.stop()


@pytest.mark.asyncio
async def test_physically_ready_speculation_retires_without_minting_canonical_slot_or_ninth_waiter():
    provider = PhysicalProvider()
    sender = Sender()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SemaphoreLLMProvider(provider, asyncio.Semaphore(3)),
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    texts = [f"ready pressure {index}" for index in range(12)]
    try:
        for text in texts[:11]:
            await begin(harness, text, ended=True)
        await until(lambda: len(provider.calls) == 3)
        initial = snapshots(harness)
        ready_turn = next(
            snapshot.utterance_id for snapshot in initial if snapshot.source_text == texts[2]
        )
        provider.release(texts[2])
        await asyncio.wait_for(harness.translation_turns.wait_for_parent(ready_turn), 1)
        assert provider.active == 2
        await begin(harness, texts[11], ended=True)
        await until(lambda: len(provider.calls) == 4)
        assert {call["text"] for call in provider.calls} == {
            texts[0],
            texts[1],
            texts[2],
            texts[11],
        }
        assert provider.active == 3 and provider.peak == 3
        assert not any(snapshot.status in {"failed", "expired"} for snapshot in snapshots(harness))
        for text in texts:
            provider.release(text)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 3)
        calls = [call["text"] for call in provider.calls]
        assert len(calls) == 12 and set(calls) == set(texts)
        assert provider.peak == 3 and provider.active == 0
        assert provider.cancelled == []
    finally:
        for text in texts:
            provider.release(text)
        await harness.stop()


@dataclass
class FailureAfterReleaseProvider(PhysicalProvider):
    failure_text: str = "failed occupancy"

    async def translate(self, *, text, **kwargs):
        result = await super().translate(text=text, **kwargs)
        if text == self.failure_text and sum(call["text"] == text for call in self.calls) == 1:
            raise RuntimeError("synthetic physically completed speculative failure")
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_after_deadline", [False, True])
async def test_completed_failed_speculation_uses_original_wait_deadline_and_releases_speculation(
    failure_after_deadline,
):
    provider = FailureAfterReleaseProvider()
    sender = Sender()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SemaphoreLLMProvider(provider, asyncio.Semaphore(3)),
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    harness.translation_turns.self_speech_waiting_ttl_s = 0.1
    texts = ["canonical one", "canonical two", provider.failure_text, "ordinary waiter", "recovery"]
    try:
        for text in texts[:4]:
            await begin(harness, text, ended=True)
        await until(lambda: len(provider.calls) == 3)
        initial = snapshots(harness)
        failed_id = next(
            item.utterance_id for item in initial if item.source_text == provider.failure_text
        )
        if failure_after_deadline:
            await asyncio.sleep(0.15)
            assert harness.translation_turns.is_parent_active(failed_id)
            assert provider.active == 3
        provider.release(provider.failure_text)
        await asyncio.wait_for(
            harness.translation_turns.wait_for_parent(failed_id),
            0.05 if failure_after_deadline else 0.3,
        )
        terminal = [item for item in snapshots(harness) if item.utterance_id == failed_id][-1]
        assert terminal.status == "expired"
        assert all(target.status == "source_only" for target in terminal.targets)
        assert provider.active == 2 and provider.peak == 3
        assert [call["text"] for call in provider.calls].count(provider.failure_text) == 1
        await begin(harness, "recovery", ended=True)
        await until(lambda: any(call["text"] == "recovery" for call in provider.calls))
        provider.release("recovery")
        await until(
            lambda: any(
                item.source_text == "recovery" and item.status == "complete"
                for item in snapshots(harness)
            )
        )
        assert provider.active == 2 and provider.peak == 3
        for text in texts:
            provider.release(text)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert provider.cancelled == []
        assert [call["text"] for call in provider.calls].count(provider.failure_text) == 1
    finally:
        for text in texts:
            provider.release(text)
        await harness.stop()


@pytest.mark.asyncio
async def test_completed_failure_rejoins_eight_waiter_budget_using_original_overflow_order():
    provider = FailureAfterReleaseProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SemaphoreLLMProvider(provider, asyncio.Semaphore(3)),
        osc=ChatboxPaginator(sender=Sender(), clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en",),
    )
    texts = ["canonical one", "canonical two", provider.failure_text] + [
        f"bounded waiter {index}" for index in range(8)
    ]
    try:
        for text in texts:
            await begin(harness, text, ended=True)
        await until(lambda: len(provider.calls) == 3)
        failed_id = next(
            item.utterance_id
            for item in snapshots(harness)
            if item.source_text == provider.failure_text
        )
        provider.release(provider.failure_text)
        await asyncio.wait_for(harness.translation_turns.wait_for_parent(failed_id), 1)
        terminal = [item for item in snapshots(harness) if item.utterance_id == failed_id][-1]
        assert terminal.status == "failed"
        assert all(target.status == "source_only" for target in terminal.targets)
        assert provider.active == 2 and provider.peak == 3
        for text in texts:
            provider.release(text)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        calls = [call["text"] for call in provider.calls]
        assert len(calls) == 11 and set(calls) == set(texts)
        assert calls.count(provider.failure_text) == 1
        assert provider.cancelled == []
    finally:
        for text in texts:
            provider.release(text)
        await harness.stop()


@pytest.mark.asyncio
async def test_completed_failure_frees_speculation_then_existing_final_fallback_runs_once():
    provider = FailureAfterReleaseProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SemaphoreLLMProvider(provider, asyncio.Semaphore(3)),
        osc=ChatboxPaginator(sender=Sender(), clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en",),
    )
    texts = ["canonical one", "canonical two", provider.failure_text, "fresh speculation"]
    try:
        for text in texts[:3]:
            await begin(harness, text, ended=True)
        await until(lambda: len(provider.calls) == 3)
        failed_id = next(
            item.utterance_id
            for item in snapshots(harness)
            if item.source_text == provider.failure_text
        )
        provider.release(provider.failure_text)
        await until(lambda: provider.active == 2)
        await begin(harness, texts[3], ended=True)
        await until(lambda: len(provider.calls) == 4)
        provider.release(texts[3])
        await until(lambda: provider.active == 2)
        provider.release(texts[0])
        await asyncio.wait_for(harness.translation_turns.wait_for_parent(failed_id), 1)
        calls = [call["text"] for call in provider.calls]
        assert calls.count(provider.failure_text) == 2
        assert calls.count(texts[3]) == 1
        assert provider.peak == 3
        assert [entry.text for entry in harness.self_runtime.translation_history].count(
            provider.failure_text
        ) == 1
        provider.release(texts[1])
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert provider.cancelled == []
    finally:
        for text in texts:
            provider.release(text)
        await harness.stop()


@pytest.mark.asyncio
async def test_retiring_native_cleanup_retains_live_exception_until_failed_replacement_settles():
    provider = FailureAfterReleaseProvider(ignore_cancellation=True)
    sender = Sender()
    harness = compose_translation_test_harness(
        stt=None,
        llm=SemaphoreLLMProvider(provider, asyncio.Semaphore(3)),
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    harness.translation_turns.self_speech_waiting_ttl_s = 0.1
    texts = [
        "canonical one",
        "canonical two",
        "obsolete running",
        provider.failure_text,
        "waiter",
        "cleanup successor",
    ]
    try:
        for text in texts[:2]:
            await begin(harness, text, ended=True)
        identity = await begin(harness, texts[2])
        await until(lambda: len(provider.calls) == 3)
        await contribution(harness, identity, provider.failure_text)
        await provider.cancellation_observed.wait()
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
        await begin(harness, texts[4], ended=True)
        await asyncio.sleep(0.15)
        observed = snapshots(harness)
        failed_id = next(
            item.utterance_id for item in observed if item.source_text == provider.failure_text
        )
        assert harness.translation_turns.is_parent_active(failed_id)
        assert any(item.source_text == "waiter" and item.status == "expired" for item in observed)
        assert provider.active == 3 and len(provider.calls) == 3
        successor = await begin(harness, texts[5])
        successor_id = harness.self_owner.merge_buffer.merge_id
        assert len(provider.calls) == 3
        provider.release(texts[5])
        provider.release(texts[2])
        await until(lambda: len(provider.calls) == 4)
        provider.release(provider.failure_text)
        await asyncio.wait_for(harness.translation_turns.wait_for_parent(failed_id), 0.05)
        await until(lambda: any(call["text"] == texts[5] for call in provider.calls))
        assert harness.self_owner.merge_buffer is not None
        assert [entry.text for entry in harness.self_runtime.translation_history].count(
            texts[5]
        ) == 0
        await harness.self_owner.handle_vad_event(SpeechEnd(successor))
        await asyncio.wait_for(harness.translation_turns.wait_for_parent(successor_id), 1)
        assert [call["text"] for call in provider.calls].count(texts[5]) == 1
        assert [entry.text for entry in harness.self_runtime.translation_history].count(
            texts[5]
        ) == 1
        assert provider.active == 2 and provider.peak == 3
        assert [call["text"] for call in provider.calls].count(provider.failure_text) == 1
        assert provider.cancelled == [(texts[2], "en")]
        assert not any(texts[2] in packet for packet in sender.packets)
        for text in texts:
            provider.release(text)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
    finally:
        for text in texts:
            provider.release(text)
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_existing_open_draft_wakes_after_speculative_physical_completion(failed):
    provider = FailureAfterReleaseProvider(
        failure_text="speculative predecessor" if failed else "unused failure"
    )
    semaphore = asyncio.Semaphore(3)
    harness = compose_translation_test_harness(
        stt=ControlledRecognitionProvider(),
        llm=SemaphoreLLMProvider(provider, semaphore),
        osc=ChatboxPaginator(sender=Sender(), clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en",),
    )
    harness.translation_turns.self_speech_waiting_ttl_s = 1
    ingress = OwnedIngress(harness)
    texts = ["canonical one", "canonical two", "speculative predecessor", "open successor"]
    try:
        for text in texts[:3]:
            await owned_source(ingress, text)
        await until(lambda: len(provider.calls) == 3)
        successor = await owned_source(ingress, texts[3], ended=False)
        successor_id = harness.self_owner.merge_buffer.merge_id
        assert len(provider.calls) == 3 and provider.active == 3
        provider.release(texts[2])
        await until(lambda: len(provider.calls) == 4)
        assert harness.self_owner.merge_buffer is not None
        assert provider.active == 3 and provider.peak == 3 and semaphore._value == 0
        assert texts[3] not in [entry.text for entry in harness.self_runtime.translation_history]
        provider.release(texts[3])
        await until(
            lambda: any(
                item.utterance_id == successor_id
                and not item.source_closed
                and item.targets[0].status == "ready"
                for item in snapshots(harness)
            )
        )
        await ingress.vad(SpeechEnd(successor))
        await asyncio.wait_for(harness.translation_turns.wait_for_parent(successor_id), 1)
        terminal = [item for item in snapshots(harness) if item.utterance_id == successor_id][-1]
        assert terminal.source_closed and terminal.status == "complete"
        assert terminal.targets[0].status == "ready"
        assert [call["text"] for call in provider.calls].count(texts[3]) == 1
        assert [entry.text for entry in harness.self_runtime.translation_history].count(
            texts[3]
        ) == 1
        assert provider.active == 2 and semaphore._value == 1
        for text in texts:
            provider.release(text)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 2)
        assert [call["text"] for call in provider.calls].count(texts[2]) == (2 if failed else 1)
        assert [entry.text for entry in harness.self_runtime.translation_history].count(
            texts[2]
        ) == 1
        assert provider.cancelled == [] and provider.peak == 3 and provider.active == 0
    finally:
        for text in texts:
            provider.release(text)
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("eligible_before_release", [False, True])
async def test_open_draft_wakeup_preserves_only_existing_secondary_eligibility(
    eligible_before_release,
):
    provider = PhysicalProvider()
    semaphore = asyncio.Semaphore(6)
    harness = compose_translation_test_harness(
        stt=ControlledRecognitionProvider(),
        llm=SemaphoreLLMProvider(provider, semaphore),
        osc=ChatboxPaginator(sender=Sender(), clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en", "ja"),
    )
    harness.translation_turns.self_speech_waiting_ttl_s = 0.1
    ingress = OwnedIngress(harness)
    texts = ["canonical one", "canonical two", "speculative predecessor", "open successor"]
    try:
        for text in texts[:3]:
            await owned_source(ingress, text)
        await until(lambda: len(provider.calls) == 6)
        if eligible_before_release:
            harness.replace_configuration(low_latency_finalize_wait_ms=200)
        successor = await owned_source(ingress, texts[3], ended=eligible_before_release)
        successor_id = harness.self_owner.merge_buffer.merge_id
        assert len(provider.calls) == 6
        for target in ("en", "ja"):
            provider.release(texts[2], target)
        await until(
            lambda: sum(call["text"] == texts[3] for call in provider.calls)
            == (2 if eligible_before_release else 1)
        )
        assert harness.self_owner.merge_buffer is not None
        assert texts[3] not in [entry.text for entry in harness.self_runtime.translation_history]
        if not eligible_before_release:
            await asyncio.sleep(0.02)
            assert [call["target"] for call in provider.calls if call["text"] == texts[3]] == ["en"]
            await ingress.vad(SpeechEnd(successor))
            await until(lambda: len(provider.calls) == 8)
        for target in ("en", "ja"):
            provider.release(texts[3], target)
        await asyncio.wait_for(harness.translation_turns.wait_for_parent(successor_id), 1)
        terminal = [item for item in snapshots(harness) if item.utterance_id == successor_id][-1]
        assert terminal.source_closed and terminal.status == "complete"
        assert all(target.status == "ready" for target in terminal.targets)
        assert [call["target"] for call in provider.calls if call["text"] == texts[3]] == [
            "en",
            "ja",
        ]
        assert [entry.text for entry in harness.self_runtime.translation_history].count(
            texts[3]
        ) == 2
        assert provider.active == 4 and semaphore._value == 2 and provider.peak == 6
        assert provider.cancelled == []
    finally:
        for text in texts:
            for target in ("en", "ja"):
                provider.release(text, target)
        await harness.stop()


@pytest.mark.asyncio
async def test_freed_speculation_does_not_bypass_sealed_waiter_expiry():
    provider = PhysicalProvider()
    semaphore = asyncio.Semaphore(3)
    harness = compose_translation_test_harness(
        stt=ControlledRecognitionProvider(),
        llm=SemaphoreLLMProvider(provider, semaphore),
        osc=ChatboxPaginator(sender=Sender(), clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en",),
    )
    harness.translation_turns.self_speech_waiting_ttl_s = 0.1
    ingress = OwnedIngress(harness)
    texts = ["canonical one", "canonical two", "speculative predecessor", "sealed waiter"]
    try:
        for text in texts:
            await owned_source(ingress, text)
        await until(lambda: len(provider.calls) == 3)
        observed = snapshots(harness)
        waiter_id = next(item.utterance_id for item in observed if item.source_text == texts[3])
        predecessor_id = next(
            item.utterance_id for item in observed if item.source_text == texts[2]
        )
        provider.release(texts[2])
        await asyncio.wait_for(harness.translation_turns.wait_for_parent(predecessor_id), 1)
        await asyncio.wait_for(harness.translation_turns.wait_for_parent(waiter_id), 1)
        terminal = [item for item in snapshots(harness) if item.utterance_id == waiter_id][-1]
        assert terminal.status == "expired" and terminal.source_closed
        assert all(target.status == "source_only" for target in terminal.targets)
        assert [call["text"] for call in provider.calls] == texts[:3]
        assert provider.active == 2 and provider.peak == 3 and semaphore._value == 1
        assert [entry.text for entry in harness.self_runtime.translation_history].count(
            texts[3]
        ) == 1
    finally:
        for text in texts:
            provider.release(text)
        await harness.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_close", [False, True])
async def test_speculation_cleanup_never_wakes_cancelled_open_draft(generation_close):
    provider = PhysicalProvider(ignore_cancellation=True)
    semaphore = asyncio.Semaphore(3)
    sender = Sender()
    harness = compose_translation_test_harness(
        stt=ControlledRecognitionProvider(),
        llm=SemaphoreLLMProvider(provider, semaphore),
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        self_target_languages=("en",),
        chatbox_include_source=False,
    )
    ingress = OwnedIngress(harness)
    texts = ["canonical one", "canonical two", "speculative predecessor", "open successor"]
    reset = None
    try:
        for text in texts[:3]:
            await owned_source(ingress, text)
        await until(lambda: len(provider.calls) == 3)
        await owned_source(ingress, texts[3], ended=False)
        reset = asyncio.create_task(
            harness.translation_turns.close_channel_ingress("self")
            if generation_close
            else harness.self_owner.reset_provider_channel()
        )
        await until(lambda: len(provider.cancelled) == 3)
        assert not reset.done()
        assert provider.active == 3 and semaphore._value == 0
        assert [call["text"] for call in provider.calls] == texts[:3]
        for text in texts:
            provider.release(text)
        await asyncio.wait_for(reset, 1)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 1)
        terminal = [item for item in snapshots(harness) if item.source_text == texts[3]][-1]
        assert terminal.status == "cancelled"
        assert all(target.status == "cancelled" for target in terminal.targets)
        assert [call["text"] for call in provider.calls] == texts[:3]
        assert provider.active == 0 and provider.peak == 3 and semaphore._value == 3
        assert texts[3] not in [entry.text for entry in harness.self_runtime.translation_history]
        assert harness.translation_turns.channel_ingress_open("self") is not generation_close
        assert sender.packets == []
    finally:
        for text in texts:
            provider.release(text)
        if reset is not None:
            await asyncio.gather(reset, return_exceptions=True)
        await harness.stop()
