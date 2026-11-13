from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.events import STTFinalEvent, UIEventType
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.self_stream import Sender
from tests.helpers.translation_owners import compose_translation_test_harness


async def eventually(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


class MixedProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.hold = asyncio.Event()

    async def translate(self, *, utterance_id, text, target_language, context="", **kwargs):
        self.calls.append((text, target_language))
        if text == "B":
            if target_language == "ja":
                return Translation(
                    utterance_id=utterance_id,
                    text="translated ja",
                    source_text=text,
                    target_language=target_language,
                )
            raise RuntimeError("synthetic zh failure")
        await self.hold.wait()
        return Translation(
            utterance_id=utterance_id,
            text=f"translated {target_language}",
            source_text=text,
            target_language=target_language,
        )

    def release(self) -> None:
        self.hold.set()

    async def close(self):
        return


async def speak(harness, text: str) -> None:
    uid = uuid4()
    await harness.self_owner.handle_vad_event(
        SpeechStart(
            uid, pre_roll=np.empty(0, dtype=np.float32), chunk=np.ones(512, dtype=np.float32)
        )
    )
    await harness.self_owner.handle_stt_event(
        STTFinalEvent(uid, Transcript(uid, text, is_final=True))
    )
    await harness.self_owner.handle_vad_event(SpeechEnd(uid))


def drain_self_turns(harness) -> list:
    items = []
    while not harness.ui_events.empty():
        event = harness.ui_events.get_nowait()
        if event.type is UIEventType.SELF_TURN_UPDATED:
            items.append(event.payload)
    return items


async def run_mixed_ttl_scenario(harness, provider, sender) -> tuple[list, list]:
    await speak(harness, "A1")
    await speak(harness, "A2")
    await eventually(lambda: sum(t == "A1" for t, _ in provider.calls) >= 1)
    await eventually(lambda: sum(t == "A2" for t, _ in provider.calls) >= 1)
    await speak(harness, "B")
    await eventually(lambda: ("B", "ja") in provider.calls and ("B", "zh-CN") in provider.calls)
    await asyncio.sleep(0.5)
    provider.release()
    await eventually(lambda: len(sender.packets) >= 1)
    await asyncio.sleep(0.3)
    final = [
        item
        for item in drain_self_turns(harness)
        if item.source_text == "B" and item.status != "active"
    ]
    return final, list(sender.packets)


def make_harness(provider, sender, **overrides):
    values = dict(
        stt=None,
        llm=provider,
        osc=ChatboxPaginator(sender=sender, clock=SystemClock()),
        overlay_sink=OverlayPresenter(calibration=OverlayCalibration()),
        low_latency_mode=True,
        low_latency_finalize_wait_ms=0,
        target_language="zh-CN",
        self_target_languages=("zh-CN", "ja"),
        chatbox_include_source=False,
    )
    values.update(overrides)
    harness = compose_translation_test_harness(**values)
    harness.translation_turns.self_speech_waiting_ttl_s = 0.1
    return harness


@pytest.mark.asyncio
async def test_waiting_ttl_keeps_successful_mixed_draft_translation() -> None:
    provider, sender = MixedProvider(), Sender()
    harness = make_harness(provider, sender)
    try:
        final, packets = await run_mixed_ttl_scenario(harness, provider, sender)
        assert final, f"no terminal snapshot for B; calls={provider.calls!r}"
        last = final[-1]
        state = (
            f"snapshot status={last.status!r} "
            f"targets={[(t.target_language, t.status) for t in last.targets]!r} "
            f"packets={packets!r}"
        )
        assert any(
            t.target_language == "ja" and t.status == "ready" for t in last.targets
        ), f"successful ja translation discarded at TTL expiry: {state}"
        assert any(
            p == "translated ja" for p in packets
        ), f"successful ja translation never reached chatbox: {state}"
        assert not any(
            p.strip() == "B" for p in packets
        ), f"source text exposed despite successful translation: {state}"
    finally:
        provider.release()
        await harness.translation_turns.close()
        await harness.stop()


@pytest.mark.asyncio
async def test_waiting_ttl_records_successful_mixed_draft_translation_in_history() -> None:
    class RecordingRuntimeLogging:
        def __init__(self) -> None:
            self.observations: list[dict] = []

        def emit_basic(self, message: str, *, level: int = 20) -> None:
            return None

        def emit_diagnostic(self, message: str, *, level: int = 20) -> bool:
            return True

        def emit_diagnostic_lazy(self, build_message, *, level: int = 20) -> bool:
            return True

        def record_conversation_observation(self, **fields) -> None:
            self.observations.append(fields)

    logging_sink = RecordingRuntimeLogging()
    provider, sender = MixedProvider(), Sender()
    harness = make_harness(provider, sender, runtime_logging=logging_sink)
    try:
        final, _packets = await run_mixed_ttl_scenario(harness, provider, sender)
        assert final, f"no terminal snapshot for B; calls={provider.calls!r}"
        b_parent = str(final[-1].utterance_id)
        translated = [
            item
            for item in logging_sink.observations
            if item["utterance_id"] == b_parent
            and item["metadata"].get("disposition") == "translated"
        ]
        assert translated, (
            "successful ja translation not recorded as translated conversation "
            f"observation for B: "
            f"{[(i['utterance_id'], i['metadata'].get('disposition')) for i in logging_sink.observations]!r}"
        )
    finally:
        provider.release()
        await harness.translation_turns.close()
        await harness.stop()


@pytest.mark.asyncio
async def test_waiting_ttl_fully_failed_draft_still_degrades_to_source_only() -> None:
    class FailingProvider:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.hold = asyncio.Event()

        async def translate(self, *, utterance_id, text, target_language, context="", **kwargs):
            self.calls.append((text, target_language))
            if text == "B":
                raise RuntimeError("synthetic failure")
            await self.hold.wait()
            return Translation(
                utterance_id=utterance_id,
                text=f"translated {target_language}",
                source_text=text,
                target_language=target_language,
            )

        def release(self) -> None:
            self.hold.set()

        async def close(self):
            return

    provider, sender = FailingProvider(), Sender()
    harness = make_harness(provider, sender)
    try:
        await speak(harness, "A1")
        await speak(harness, "A2")
        await eventually(lambda: sum(t == "A1" for t, _ in provider.calls) >= 1)
        await eventually(lambda: sum(t == "A2" for t, _ in provider.calls) >= 1)
        await speak(harness, "B")
        await eventually(lambda: len([c for c in provider.calls if c[0] == "B"]) >= 2)
        await asyncio.sleep(0.5)
        provider.release()
        await eventually(lambda: any("B" in p for p in sender.packets))
        await asyncio.sleep(0.3)
        final = [
            item
            for item in drain_self_turns(harness)
            if item.source_text == "B" and item.status != "active"
        ]
        assert final
        last = final[-1]
        assert all(t.status == "source_only" for t in last.targets), (
            f"expected source_only degradation, got "
            f"{[(t.target_language, t.status) for t in last.targets]!r}"
        )
        assert any(
            p.strip() == "B" for p in sender.packets
        ), f"source fallback missing for fully-failed waiter: {sender.packets!r}"
    finally:
        provider.release()
        await harness.translation_turns.close()
        await harness.stop()
