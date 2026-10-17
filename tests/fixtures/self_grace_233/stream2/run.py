from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from puripuly_heart.core.clock import SystemClock
from puripuly_heart.core.osc.chatbox_paginator import ChatboxPaginator
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart
from puripuly_heart.domain.events import STTFinalEvent, UIEventType
from puripuly_heart.domain.models import Transcript, Translation
from tests.helpers.translation_owners import compose_translation_test_harness


@dataclass
class Provider:
    calls: list[dict] = field(default_factory=list)
    releases: dict[tuple[str, str], asyncio.Event] = field(default_factory=dict)
    cancelled: list[tuple[str, str]] = field(default_factory=list)

    async def translate(self, *, utterance_id, text, target_language, context="", **kwargs):
        self.calls.append({"text": text, "target": target_language, "context": context})
        key = (text, target_language)
        try:
            await self.releases.setdefault(key, asyncio.Event()).wait()
        except asyncio.CancelledError:
            self.cancelled.append(key)
            raise
        return Translation(utterance_id=utterance_id, text=f"{target_language}:{text}")

    def release(self, text, target="en"):
        self.releases.setdefault((text, target), asyncio.Event()).set()

    async def close(self):
        return None


@dataclass
class Sender:
    packets: list[str] = field(default_factory=list)

    def send_chatbox(self, text):
        self.packets.append(text)

    def send_typing(self, value):
        return None


@dataclass
class DisabledOverlay:
    attempted_events: int = 0

    async def emit(self, event):
        self.attempted_events += 1

    def active_self_overlay_metadata(self):
        return None


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


async def contribution(harness, identity, text):
    await harness.dispatch_stt_event(
        STTFinalEvent(identity, Transcript(utterance_id=identity, text=text, is_final=True))
    )


async def begin(harness, text, *, ended=False):
    identity = uuid4()
    await harness.self_owner.handle_vad_event(
        SpeechStart(
            identity, pre_roll=np.empty(0, dtype=np.float32), chunk=np.ones(512, dtype=np.float32)
        )
    )
    if ended:
        await harness.self_owner.handle_vad_event(SpeechEnd(identity))
    await contribution(harness, identity, text)
    return identity


def collect(harness):
    snapshots = []
    while not harness.ui_events.empty():
        event = harness.ui_events.get_nowait()
        if event.type is UIEventType.SELF_TURN_UPDATED:
            snapshot = event.payload
            snapshots.append(
                {
                    "turn": str(snapshot.utterance_id),
                    "order": snapshot.turn_order,
                    "revision": snapshot.revision,
                    "source_revision": snapshot.source_revision,
                    "source": snapshot.source_text,
                    "closed": snapshot.source_closed,
                    "status": snapshot.status,
                    "targets": [
                        {
                            "language": target.target_language,
                            "status": target.status,
                            "source_revision": target.source_revision,
                            "update_id": (
                                None if target.translation is None else target.translation.update_id
                            ),
                        }
                        for target in snapshot.targets
                    ],
                }
            )
    return snapshots


def compose(*, targets=("en",), overlay=None):
    provider, sender = Provider(), Sender()
    clock = SystemClock()
    harness = compose_translation_test_harness(
        llm=provider,
        stt=None,
        osc=ChatboxPaginator(sender=sender, clock=clock),
        clock=clock,
        low_latency_mode=True,
        low_latency_finalize_wait_ms=30,
        self_target_languages=targets,
        chatbox_include_source=False,
        overlay_sink=overlay,
    )
    return harness, provider, sender


async def scenario(kind, *, targets=("en",), disabled_overlay=False):
    overlay = DisabledOverlay() if disabled_overlay else None
    harness, provider, sender = compose(targets=targets, overlay=overlay)
    started = perf_counter()
    observations = {}
    try:
        identity = await begin(harness, "alpha", ended=kind == "dual")
        await until(lambda: bool(provider.calls))
        if kind == "dual":
            await until(lambda: len(provider.calls) == 2)
            provider.release("alpha", targets[1])
            await until(lambda: bool(sender.packets))
            assert sender.packets == [f"{targets[1]}:alpha"]
        elif kind == "late_predecessor":
            await harness.self_owner.handle_vad_event(SpeechEnd(identity))
            await until(lambda: harness.self_owner.merge_buffer is None)
            successor = await begin(harness, "bravo")
            await until(lambda: len(provider.calls) == 2)
            provider.release("bravo")
            await asyncio.sleep(0.005)
            assert sender.packets == []
            observations["successor_computed_while_predecessor_pending"] = True
            provider.release("alpha")
            await until(lambda: len(sender.packets) == 2)
            assert sender.packets == ["en:alpha", "en:bravo"]
            await harness.self_owner.handle_vad_event(SpeechEnd(successor))
        else:
            provider.release("alpha")
            await until(lambda: bool(sender.packets))
            assert harness.self_owner.merge_buffer is not None
            observations["preview_before_source_close"] = True
            if kind == "changed_final":
                await contribution(harness, identity, "alpha revised")
                await until(lambda: len(provider.calls) == 2)
            await harness.self_owner.handle_vad_event(SpeechEnd(identity))
        if kind != "late_predecessor":
            await until(lambda: harness.self_owner.merge_buffer is None)
            before = len(provider.calls)
            await asyncio.sleep(0.005)
            assert len(provider.calls) == before
            observations["no_reissue_on_source_close"] = True
            if kind == "changed_final":
                provider.release("alpha revised")
            elif kind == "dual":
                provider.release("alpha", targets[0])
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 3)
        if kind == "identical_final":
            assert sender.packets == ["en:alpha"]
        elif kind == "changed_final":
            assert sender.packets == ["en:alpha", "en:alpha revised"]
        elif kind == "dual":
            assert sender.packets == [
                f"{targets[1]}:alpha",
                f"{targets[0]}:alpha\n{targets[1]}:alpha",
            ]
        assert provider.cancelled == []
        snapshots = collect(harness)
        assert snapshots[-1]["status"] == "complete"
        closed_sources = list(
            dict.fromkeys(
                (snapshot["turn"], snapshot["source"])
                for snapshot in snapshots
                if snapshot["closed"]
            )
        )
        return {
            "case": kind,
            "status": "passed",
            "overlay": "disabled" if disabled_overlay else "absent",
            "observations": observations,
            "packets_at_local_sender": sender.packets,
            "invocations": provider.calls,
            "cancelled": provider.cancelled,
            "closed_sources": closed_sources,
            "snapshots": snapshots,
            "elapsed_s": round(perf_counter() - started, 6),
            "destination_limit": "controlled local sender; no UDP reception, VRChat, GUI or HMD claim",
        }
    finally:
        await harness.stop()


async def fixed_schedule(fast):
    harness, provider, sender = compose()
    try:

        async def ingress():
            await begin(harness, "alpha", ended=True)
            await asyncio.sleep(0.065)
            await begin(harness, "bravo", ended=True)

        async def results():
            await asyncio.sleep(0.005 if fast else 0.11)
            provider.release("alpha")
            await asyncio.sleep(0.11 if fast else 0.005)
            provider.release("bravo")

        await asyncio.gather(ingress(), results())
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 3)
        snapshots = collect(harness)
        closed = list(
            dict.fromkeys(snapshot["source"] for snapshot in snapshots if snapshot["closed"])
        )
        assert closed == ["alpha", "bravo"]
        assert sender.packets == ["en:alpha", "en:bravo"]
        assert len(provider.calls) == 2 and not provider.cancelled
        return {
            "llm_profile": "fast" if fast else "late",
            "closed_sources": closed,
            "invocations": provider.calls,
            "cancelled": provider.cancelled,
            "packets": sender.packets,
        }
    finally:
        await harness.stop()


async def run():
    cases = []
    for disabled in (False, True):
        for kind in ("identical_final", "changed_final", "late_predecessor"):
            cases.append(await scenario(kind, disabled_overlay=disabled))
        cases.append(await scenario("dual", targets=("en", "ja"), disabled_overlay=disabled))
    causal = [await fixed_schedule(True), await fixed_schedule(False)]
    assert causal[0]["closed_sources"] == causal[1]["closed_sources"]
    return {
        "authority": "SELF-STREAM-2",
        "historical_baseline": False,
        "case_count": len(cases),
        "cases": cases,
        "fixed_source_schedule": causal,
        "limits": [
            "synthetic owned text/VAD ingress",
            "controlled provider and local sender",
            "no audio recognition, external services, GUI, UDP, VRChat or HMD observation",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = asyncio.run(run())
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
