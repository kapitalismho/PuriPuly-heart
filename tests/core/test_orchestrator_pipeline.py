from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.llm.provider import SemaphoreLLMProvider
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.stt.controller import ManagedSTTProvider
from puripuly_heart.core.vad.gating import SpeechChunk, SpeechEnd, SpeechStart
from puripuly_heart.domain.models import Translation
from tests.helpers.fakes import FakeSender, SpeechAwareFakeBackend, samples
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass(slots=True)
class FakeLLM:
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
        _ = (system_prompt, source_language, target_language)
        self.calls.append({"text": text, "context": context})
        await asyncio.sleep(0.01)
        return Translation(utterance_id=utterance_id, text="TRANSLATED")

    async def close(self) -> None:
        pass


async def test_translation_fixture_uses_integrated_context_when_enabled_and_safe():
    clock = FakeClock(_now=112.0)
    sender = FakeSender()
    osc = ChatboxPaginator(sender=sender, clock=clock)
    inner = FakeLLM()
    llm = SemaphoreLLMProvider(inner=inner, semaphore=asyncio.Semaphore(1))
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=osc,
        clock=clock,
        integrated_context_enabled=True,
        peer_translation_enabled=True,
    )
    harness.replace_configuration(source_language="en")
    harness.replace_configuration(target_language="ko")
    harness.self_runtime.remember_context(
        "I am ready",
        timestamp=100.0,
        source_language="en",
        target_language="ko",
    )
    harness.peer_runtime.remember_context(
        "hello from peer",
        timestamp=105.0,
        source_language="en",
        target_language="ko",
    )

    await harness.self_owner.submit_text("world")
    await asyncio.gather(*harness.self_runtime.translation_tasks.values(), return_exceptions=True)

    assert inner.calls[0]["context"] == ('- [self] "I am ready"\n- [peer] "hello from peer"')


async def test_orchestrator_e2e_pipeline():
    clock = FakeClock()
    sender = FakeSender()
    osc = ChatboxPaginator(sender=sender, clock=clock)

    stt = ManagedSTTProvider(
        backend=SpeechAwareFakeBackend(),
        sample_rate_hz=16000,
        clock=clock,
        reset_deadline_s=90.0,
    )

    llm = SemaphoreLLMProvider(inner=FakeLLM(), semaphore=asyncio.Semaphore(1))
    harness = compose_translation_test_harness(stt=stt, llm=llm, osc=osc, clock=clock)
    await harness.start(auto_flush_osc=False)

    uid = __import__("uuid").uuid4()
    await harness.self_owner.handle_vad_event(
        SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0))
    )
    await harness.self_owner.handle_vad_event(SpeechChunk(uid, chunk=samples(0.0)))
    await harness.self_owner.handle_vad_event(SpeechEnd(uid))

    # Wait for translation and OSC send
    for _ in range(50):
        if "FINAL (TRANSLATED)" in sender.sent:
            break
        await asyncio.sleep(0.01)

    assert "FINAL (TRANSLATED)" in sender.sent
    await harness.stop()


async def test_stt_connected_sends_promo_message():
    """버튼 클릭 시 'PuriPuly ON!' 메시지 전송."""
    clock = FakeClock()
    sender = FakeSender()
    osc = ChatboxPaginator(sender=sender, clock=clock)
    stt = ManagedSTTProvider(
        backend=SpeechAwareFakeBackend(),
        sample_rate_hz=16000,
        clock=clock,
        reset_deadline_s=90.0,
    )
    llm = SemaphoreLLMProvider(inner=FakeLLM(), semaphore=asyncio.Semaphore(1))
    harness = compose_translation_test_harness(stt=stt, llm=llm, osc=osc, clock=clock)
    await harness.start(auto_flush_osc=False)

    # 버튼 클릭 시뮬레이션
    harness.self_owner.mark_promo_eligible()

    # STT 연결 트리거
    uid = __import__("uuid").uuid4()
    await harness.self_owner.handle_vad_event(
        SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0))
    )
    await asyncio.sleep(0.05)

    assert "PuriPuly ON!" in sender.sent
    await harness.stop()


async def test_stt_promo_respects_interval():
    """5분 내 버튼 다시 눌러도 메시지 안 보냄."""
    clock = FakeClock()
    sender = FakeSender()
    osc = ChatboxPaginator(sender=sender, clock=clock)
    stt = ManagedSTTProvider(
        backend=SpeechAwareFakeBackend(),
        sample_rate_hz=16000,
        clock=clock,
        reset_deadline_s=90.0,
    )
    llm = SemaphoreLLMProvider(inner=FakeLLM(), semaphore=asyncio.Semaphore(1))
    harness = compose_translation_test_harness(stt=stt, llm=llm, osc=osc, clock=clock)
    await harness.start(auto_flush_osc=False)

    # 첫 번째 버튼 클릭
    harness.self_owner.mark_promo_eligible()
    uid = __import__("uuid").uuid4()
    await harness.self_owner.handle_vad_event(
        SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0))
    )
    await asyncio.sleep(0.05)

    initial_count = sender.sent.count("PuriPuly ON!")
    assert initial_count == 1

    # STT provider replacement keeps translation/output owners alive while preserving promo state.
    clock.advance(240.0)

    stt2 = ManagedSTTProvider(
        backend=SpeechAwareFakeBackend(),
        sample_rate_hz=16000,
        clock=clock,
        reset_deadline_s=90.0,
    )
    await harness.local_asr_runtime.replace_prebuilt_provider("self", stt2, start=True)

    # 두 번째 버튼 클릭 (5분 내)
    harness.self_owner.mark_promo_eligible()
    uid2 = __import__("uuid").uuid4()
    await harness.self_owner.handle_vad_event(
        SpeechStart(uid2, pre_roll=samples(0.0), chunk=samples(1.0))
    )
    await asyncio.sleep(0.05)

    # 5분 미만이므로 메시지 추가 안 됨
    assert sender.sent.count("PuriPuly ON!") == 1
    await harness.stop()


async def test_stt_promo_sends_after_interval():
    """5분 후 버튼 클릭 시 메시지 다시 보냄."""
    clock = FakeClock()
    sender = FakeSender()
    osc = ChatboxPaginator(sender=sender, clock=clock)
    stt = ManagedSTTProvider(
        backend=SpeechAwareFakeBackend(),
        sample_rate_hz=16000,
        clock=clock,
        reset_deadline_s=90.0,
    )
    llm = SemaphoreLLMProvider(inner=FakeLLM(), semaphore=asyncio.Semaphore(1))
    harness = compose_translation_test_harness(stt=stt, llm=llm, osc=osc, clock=clock)
    await harness.start(auto_flush_osc=False)

    # 첫 번째 버튼 클릭
    harness.self_owner.mark_promo_eligible()
    uid = __import__("uuid").uuid4()
    await harness.self_owner.handle_vad_event(
        SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0))
    )
    await asyncio.sleep(0.05)

    assert sender.sent.count("PuriPuly ON!") == 1

    # STT provider replacement keeps translation/output owners alive while preserving promo state.
    clock.advance(301.0)

    stt2 = ManagedSTTProvider(
        backend=SpeechAwareFakeBackend(),
        sample_rate_hz=16000,
        clock=clock,
        reset_deadline_s=90.0,
    )
    await harness.local_asr_runtime.replace_prebuilt_provider("self", stt2, start=True)

    # 두 번째 버튼 클릭 (5분 후)
    harness.self_owner.mark_promo_eligible()
    uid2 = __import__("uuid").uuid4()
    await harness.self_owner.handle_vad_event(
        SpeechStart(uid2, pre_roll=samples(0.0), chunk=samples(1.0))
    )
    await asyncio.sleep(0.05)

    # 5분 지났으므로 메시지 다시 전송됨
    assert sender.sent.count("PuriPuly ON!") == 2
    await harness.stop()


async def test_stt_promo_skipped_on_session_reset():
    """세션 자동 리셋 시에는 메시지 안 나감."""
    clock = FakeClock()
    sender = FakeSender()
    osc = ChatboxPaginator(sender=sender, clock=clock)
    stt = ManagedSTTProvider(
        backend=SpeechAwareFakeBackend(),
        sample_rate_hz=16000,
        clock=clock,
        reset_deadline_s=90.0,
    )
    llm = SemaphoreLLMProvider(inner=FakeLLM(), semaphore=asyncio.Semaphore(1))
    harness = compose_translation_test_harness(stt=stt, llm=llm, osc=osc, clock=clock)
    await harness.start(auto_flush_osc=False)

    # 버튼 클릭 없이 STT 연결 (세션 자동 리셋 시뮬레이션)
    uid = __import__("uuid").uuid4()
    await harness.self_owner.handle_vad_event(
        SpeechStart(uid, pre_roll=samples(0.0), chunk=samples(1.0))
    )
    await asyncio.sleep(0.05)

    # mark_promo_eligible() 호출 없이는 메시지 안 나감
    assert "PuriPuly ON!" not in sender.sent
    await harness.stop()
