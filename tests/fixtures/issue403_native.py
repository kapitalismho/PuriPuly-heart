from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from uuid import UUID

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.orchestrator.translation_turn import _translation_turn_child_id
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.overlay.sink import OverlayEventAdapter, OverlayPublicationScope
from puripuly_heart.core.runtime import OutputRuntime
from tests.core.runtime.test_output_runtime import RecordingChatbox
from tests.core.test_peer_overlay_policy import Surface, Timeline

CORPUS = {
    "short": "오늘은 역 앞에서 만나요.",
    "L3": "오늘은 역 앞에서 만나서 공원까지 함께 걸어가기로 했어요. 비가 오면 산책은 취소하고 근처 도서관에서 책을 읽다가 점심을 먹으려고 해요.",
    "N1": "오늘은 역 앞에서 만나서 공원까지 함께 걸어가기로 했어요. 비가 오면 산책은 취소하고 근처 도서관에서 책을 읽다가 점심을 먹으려고 해요.",
    "N2": "We can meet at the station and walk to the park together. If it starts raining, we will go to the library instead and have lunch nearby.",
    "N3": "오늘 meeting starts at 10:30 and we will review the deployment plan together. 이후에는 테스트 결과와 다음 작업 일정을 확인합니다.",
    "G3": "가" * 43 + "。" + "나" * 65,
    "G5": "At the next meeting we will discuss the new project with Dr. Kim and review the revised delivery schedule before assigning the remaining tasks.",
}


class NativePresenter(OverlayPresenter):
    def __init__(self, **kwargs):
        self.applied_events = []
        super().__init__(**kwargs)

    async def emit(self, event):
        receipt = await super().emit(event)
        if receipt.outcome == "applied":
            self.applied_events.append(event)
        return receipt


class NativeFixture:
    def __init__(self, identity: int):
        self.timeline = Timeline()
        self.surface = Surface()
        self.presenter = NativePresenter(
            calibration=OverlayCalibration(),
            clock=self.timeline.clock,
            sleep=self.timeline.sleep,
            bridge=self.surface,
            native_retry_enabled=True,
        )
        self.output = OutputRuntime(
            chatbox=RecordingChatbox(),
            clock=self.timeline.clock,
            overlay_sink=self.presenter,
        )
        self.output.activate_peer_generation(1)
        self.adapter = OverlayEventAdapter(clock=self.timeline.clock)
        self.unit = UUID(int=identity)
        self.rows = []

    async def at(self, seconds: float):
        await self.timeline.advance(seconds - self.timeline.clock.now())

    def record(self, label: str, *, text: str | None = None, revision: int | None = None):
        self.rows.append(
            {
                "label": label,
                "at_ms": round(self.timeline.clock.now() * 1000),
                "input": text,
                "caption_revision": revision,
                "snapshot": self.presenter.snapshot().to_dict(),
                "expirations": dict(self.surface.expirations),
                "fragments": [
                    {
                        "id": entry.block_id,
                        "index": entry.peer_fragment_index,
                        "registered_at": entry.peer_registered_at,
                        "visible_since": entry.visible_since,
                        "last_meaningful_visible_at": entry.last_meaningful_visible_at,
                        "pending": entry.peer_pending,
                        "departed": entry.peer_departed,
                        "protected_until": entry.peer_protected_until,
                    }
                    for entry in self.presenter._entries.values()
                    if entry.channel == "peer"
                ],
            }
        )

    async def peer(
        self,
        label: str,
        text: str,
        revision: int,
        *,
        final: bool = True,
        output_scope: OverlayPublicationScope | None = None,
        source_order: int = 1,
        target_language: str | None = None,
    ):
        ready_at_ms = round(self.timeline.clock.now() * 1000)
        decision_start = len(self.output.routing_decisions)
        build = self.adapter.translation_final if final else self.adapter.translation_stream_update
        event = build(
            utterance_id=self.unit,
            channel="peer",
            text=text,
            source_text="Synthetic source",
            source_language="en",
            target_language=target_language
            or ("ko" if any(ord(char) > 127 for char in text) else "en"),
            applied_context_mode=None,
            peer_caption_revision=revision,
            output_scope=output_scope,
        )
        await self.output.publish_overlay_event(
            event, publication_generation=1, source_order=source_order
        )
        await self.output.wait_for_peer_output_idle()
        await self.timeline.advance(0)
        self.record(label, text=text, revision=revision)
        self.rows[-1]["content_ready_at_ms"] = ready_at_ms
        self.rows[-1]["logical_observed_at_ms"] = round(self.timeline.clock.now() * 1000)
        self.rows[-1]["output_decisions"] = [
            decision.reason for decision in self.output.routing_decisions[decision_start:]
        ]
        return event

    async def own(self, label: str):
        await self.presenter.emit(
            self.adapter.self_active_update(
                text="Later Self scene",
                secondary_text="Synthetic Self translation",
                utterance_id=UUID(int=999),
                occupant_key="self:synthetic",
                source_language="en",
                target_language="en",
                caption_source_closed=False,
                caption_revision=1,
            )
        )
        self.record(label)

    async def remove(self):
        replacement = self.adapter.translation_final(
            utterance_id=UUID(int=49999),
            channel="peer",
            text="Replacement synthetic caption",
            source_text="Replacement synthetic source",
            source_language="en",
            target_language="en",
            applied_context_mode=None,
            peer_caption_revision=1,
            peer_superseded_unit_ids=(self.unit,),
        )
        await self.output.publish_overlay_event(
            replacement, publication_generation=1, source_order=2
        )
        await self.output.wait_for_peer_output_idle()
        await self.timeline.advance(0)
        self.record("removed")

    async def close(self):
        await self.output.close()
        await self.presenter.close()


async def l2():
    fixture = NativeFixture(40302)
    try:
        await fixture.peer("final", "Original synthetic final", 1)
        await fixture.at(4.635)
        await fixture.peer("repair-one", "First corrected synthetic final", 2)
        await fixture.at(9.875)
        await fixture.peer("repair-two", "Second corrected synthetic final", 3)
        await fixture.at(10.5)
        await fixture.peer("duplicate", "Second corrected synthetic final", 3)
        await fixture.at(10.6)
        await fixture.peer("stale", "Obsolete synthetic final", 2)
        await fixture.at(10.7)
        await fixture.own("later-self")
        await fixture.at(10.9)
        await fixture.remove()
        await fixture.at(10.95)
        await fixture.peer("late-retired", "Late synthetic correction", 4)
        return fixture.rows
    finally:
        await fixture.close()


async def previews():
    fixture = NativeFixture(40303)
    try:
        await fixture.peer("preview-one", "Synthetic preview", 1, final=False)
        await fixture.at(0.65)
        await fixture.peer("preview-two", "Changed synthetic preview", 2, final=False)
        await fixture.at(1.3)
        await fixture.peer("first-final", "Completed synthetic final", 3)
        await fixture.at(1.95)
        await fixture.peer("duplicate", "Completed synthetic final", 3)
        await fixture.at(2.1)
        await fixture.remove()
        await fixture.at(2.15)
        await fixture.peer("late-retired", "Late synthetic final", 4)
        return fixture.rows
    finally:
        await fixture.close()


async def fragments():
    fixture = NativeFixture(40304)
    try:
        await fixture.peer("two-fragments", "가" * 64, 1)
        await fixture.at(0.2)
        await fixture.own("promotion")
        await fixture.at(0.8)
        await fixture.peer("corrected", "나" * 64, 2)
        await fixture.at(1.5)
        await fixture.peer("growth", "다" * 128, 3)
        await fixture.at(1.6)
        fixture.record("continuation")
        await fixture.at(1.7)
        await fixture.peer("shrink", "라" * 24, 4)
        await fixture.at(1.9)
        await fixture.remove()
        await fixture.at(1.95)
        await fixture.peer("late-retired", "마" * 24, 5)
        return fixture.rows
    finally:
        await fixture.close()


async def local():
    fixture = NativeFixture(40305)
    parent = UUID(int=50305)
    fixture.unit = _translation_turn_child_id(
        parent,
        turn_kind="peer",
        run_index=0,
        target_index=0,
        run_language="en",
        target_language="ko",
        primary_uses_parent_identity=False,
    )
    scope = OverlayPublicationScope(
        turn_kind="peer", parent_utterance_id=parent, turn_generation=1, turn_order=1
    )
    try:
        admitted = await fixture.output.admit_translation_parent(
            parent_id=str(parent),
            channel="peer",
            origin="peer",
            turn_generation=1,
            turn_order=1,
            retained_payloads=("Synthetic source",),
            destination_targets={"overlay": frozenset({0})},
        )
        assert admitted == frozenset({"overlay"})
        await fixture.own("self")
        await fixture.peer("local-final", "가" * 56 + "나" * 20, 0, output_scope=scope)
        closed = fixture.adapter.utterance_closed(
            utterance_id=fixture.unit, channel="peer", is_final=True, output_scope=scope
        )
        await fixture.output.publish_overlay_event(closed, publication_generation=1, source_order=1)
        await fixture.output.wait_for_peer_output_idle()
        fixture.record("local-closed")
        await fixture.at(0.9)
        fixture.record("queued")
        await fixture.at(1.0)
        fixture.record("local-continuation")
        await fixture.at(3.1)
        fixture.record("no-old-page-timer")
        for row in fixture.rows:
            row["capture_parent_id"] = str(parent)
            row["translation_child_id"] = str(fixture.unit)
        return fixture.rows
    finally:
        await fixture.close()


async def ready_order():
    fixture = NativeFixture(40306)
    captures = {name: UUID(int=50306 + index) for index, name in enumerate(("B", "A"))}

    async def ordinary(name, source_order):
        parent = captures[name]
        fixture.unit = _translation_turn_child_id(
            parent,
            turn_kind="peer",
            run_index=0,
            target_index=0,
            run_language="en",
            target_language="ko",
            primary_uses_parent_identity=False,
        )
        scope = OverlayPublicationScope(
            turn_kind="peer",
            parent_utterance_id=parent,
            turn_generation=1,
            turn_order=source_order,
        )
        assert await fixture.output.admit_translation_parent(
            parent_id=str(parent),
            channel="peer",
            origin="peer",
            turn_generation=1,
            turn_order=source_order,
            retained_payloads=("Synthetic source",),
            destination_targets={"overlay": frozenset({0})},
        ) == frozenset({"overlay"})
        event = await fixture.peer(
            f"ready-{name}",
            f"Ordinary local {name}",
            0,
            output_scope=scope,
            source_order=source_order,
            target_language="ko",
        )
        accepted_event = next(
            accepted
            for accepted in fixture.presenter.applied_events
            if accepted.event_id == event.event_id
        )
        fixture.rows[-1]["accepted_display_order"] = accepted_event.turn_order
        fixture.rows[-1]["source_order"] = source_order
        fixture.rows[-1]["capture_parent_id"] = str(parent)
        fixture.rows[-1]["translation_child_id"] = str(fixture.unit)
        closed = fixture.adapter.utterance_closed(
            utterance_id=fixture.unit, channel="peer", is_final=True, output_scope=scope
        )
        await fixture.output.publish_overlay_event(
            closed, publication_generation=1, source_order=source_order
        )
        await fixture.output.wait_for_peer_output_idle()
        fixture.record(f"closed-{name}")
        return accepted_event

    try:
        old_b = await ordinary("B", 2)
        await fixture.at(8.01)
        fixture.record("expired-B")
        await fixture.at(8.1)
        await ordinary("A", 1)
        await fixture.at(8.2)
        receipt = await fixture.presenter.emit(old_b)
        fixture.record("historical-B-receipt")
        fixture.rows[-1]["historical_receipt"] = receipt.outcome
        fixture.rows[-1]["historical_receipt_scene_revision"] = receipt.scene_revision
        await fixture.at(8.21)
        receipt = await fixture.presenter.emit(
            replace(old_b, event_id="issue403-retired-B-duplicate")
        )
        fixture.record("retired-B-duplicate")
        fixture.rows[-1]["retired_receipt"] = receipt.outcome
        return fixture.rows
    finally:
        await fixture.close()


async def corpus():
    result = {}
    for index, (name, text) in enumerate(CORPUS.items()):
        fixture = NativeFixture(40400 + index)
        try:
            if name == "L3":
                await fixture.peer("preview", "짧은 합성 미리보기", 1, final=False)
                await fixture.at(0.65)
            await fixture.peer("final", text, 2 if name == "L3" else 1)
            await fixture.at(fixture.timeline.clock.now() + 1.05)
            fixture.record("continuation")
            result[name] = fixture.rows
        finally:
            await fixture.close()
    return result


async def main():
    mode = sys.argv[1]
    cases = {
        "l2": l2,
        "previews": previews,
        "fragments": fragments,
        "local": local,
        "ready-order": ready_order,
        "corpus": corpus,
    }
    print(json.dumps(await cases[mode](), ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
