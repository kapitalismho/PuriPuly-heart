from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from puripuly_heart.config.overlay_calibration import OverlayCalibration
from puripuly_heart.core.audio.ownership import AudioSegmentIdentity
from puripuly_heart.core.clock import FakeClock
from puripuly_heart.core.overlay.bridge import OverlayBridge
from puripuly_heart.core.overlay.presenter import OverlayPresenter
from puripuly_heart.core.stt.backend import (
    STTProviderTurnIdentity,
    STTProviderTurnTerminal,
    STTProviderTurnUpdate,
    STTRecognitionUnit,
    STTTextContribution,
)
from puripuly_heart.core.vad.gating import SpeechEnd
from puripuly_heart.domain.events import UIEvent, UIEventType
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity
from tests.core.test_overlay_bridge import _BlockingSendConnection, _wait_until
from tests.core.test_self_translation_low_latency import BlockingLLMProvider
from tests.helpers.fakes import RecordingOscQueue
from tests.helpers.translation_owners import compose_translation_test_harness
from tests.ui.test_event_bridge import DummyApp, make_bridge


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["independent", "stable", "final_only"])
async def test_real_self_owners_start_provider_and_apply_source_before_ui_release(
    route: str,
) -> None:
    clock = FakeClock(_now=10.0)
    overlay = OverlayPresenter(calibration=OverlayCalibration(), clock=clock)
    llm = BlockingLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        clock=clock,
        overlay_sink=overlay,
        ui_queue_maxsize=1,
        low_latency_mode=True,
    )
    ingress = None
    ui_entered = asyncio.Event()
    ui_release = asyncio.Event()
    bridge = make_bridge(DummyApp(), event_queue=harness.ui_events)
    handle_event = bridge._handle_event

    async def blocked_handle_event(event: UIEvent) -> None:
        ui_entered.set()
        await ui_release.wait()
        await handle_event(event)

    bridge._handle_event = blocked_handle_event
    try:
        await harness.start()
        harness.output_runtime.start_ui_event_bridge(bridge)
        harness.ui_events.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
        await asyncio.wait_for(ui_entered.wait(), 1.0)
        harness.ui_events.put_nowait(UIEvent(UIEventType.SESSION_STATE_CHANGED))
        if route == "independent":
            stream = RecognitionStreamIdentity("self", 1, 1, "epoch", ("gemini_transcribe",))
            operation = harness.self_owner.handle_recognition_unit(
                STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 1), "source")
            )
        else:
            identity = STTProviderTurnIdentity(
                AudioSegmentIdentity(1, 1, uuid4(), 1), "epoch", "turn"
            )
            if route == "stable":
                event = STTProviderTurnUpdate(
                    identity,
                    1,
                    "stable",
                    "append",
                    "source",
                    contribution=STTTextContribution("source", 0, 6),
                )
            else:
                assert not overlay.snapshot().blocks
                event = STTProviderTurnTerminal(
                    identity,
                    "final",
                    text="source",
                    text_authority="authoritative",
                )
            operation = harness.self_owner.handle_stt_event(event)
        ingress = asyncio.create_task(operation)
        await asyncio.wait_for(llm.started.wait(), 1.0)
        await asyncio.wait_for(asyncio.shield(ingress), 1.0)
        blocks = overlay.snapshot().blocks
        assert [block.primary_text for block in blocks] == ["source"]
        assert [block.secondary_text for block in blocks] == [""]
        assert len(llm.calls) == 1
        assert harness.ui_events.full()
        assert not ui_release.is_set()
        print(
            f"OWNER_SMOKE route={route} source_application=true provider_calls=1 ui_released=false"
        )
        if route != "independent":
            terminal = STTProviderTurnTerminal(
                identity,
                "final",
                text="source",
                text_authority="authoritative",
                included_contributions=(
                    (STTTextContribution("source", 0, 6),) if route == "stable" else ()
                ),
            )
            await harness.self_owner.handle_stt_event(terminal)
            await harness.self_owner.handle_stt_event(terminal)
            await harness.self_owner.handle_vad_event(SpeechEnd(identity.segment.segment_id))
            buffer = harness.self_owner.merge_buffer
            assert buffer is not None
            assert buffer.finalize_wait_task is not None
            await asyncio.wait_for(asyncio.shield(buffer.finalize_wait_task), 1.0)
            assert harness.self_owner.merge_buffer is buffer
            assert len(llm.calls) == 1
        llm.release.set()

        async def wait_for_translation_application() -> None:
            while not any(block.secondary_text for block in overlay.snapshot().blocks):
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_translation_application(), 1.0)
        if route != "independent":
            await _wait_until(lambda: harness.self_owner.merge_buffer is None)
        await asyncio.wait_for(harness.translation_turns.wait_for_idle(), 1.0)
        translated = overlay.snapshot().blocks[0]
        original = blocks[0]
        assert (translated.id, translated.occupant_key, translated.appearance_seq) == (
            original.id,
            original.occupant_key,
            original.appearance_seq,
        )
        assert harness.ui_events.full()
        assert not ui_release.is_set()
        print(f"OWNER_SMOKE route={route} translation_application=true ui_released=false")
        assert len(llm.calls) == 1
    finally:
        llm.release.set()
        ui_release.set()
        if ingress is not None:
            ingress.cancel()
            await asyncio.gather(ingress, return_exceptions=True)
        await harness.stop()
        await overlay.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["independent", "stable", "final_only"])
async def test_real_self_translation_completes_while_bridge_transport_is_blocked(
    route: str,
) -> None:
    clock = FakeClock(_now=10.0)
    transport = _BlockingSendConnection()
    bridge = OverlayBridge(session_token="owner-smoke", clock=clock)
    bridge._authenticated_connections.add(transport)
    overlay = OverlayPresenter(bridge=bridge, calibration=OverlayCalibration(), clock=clock)
    llm = BlockingLLMProvider()
    harness = compose_translation_test_harness(
        stt=None,
        llm=llm,
        osc=RecordingOscQueue(),
        clock=clock,
        overlay_sink=overlay,
        ui_queue_maxsize=1,
        low_latency_mode=True,
    )
    ui = make_bridge(DummyApp(), event_queue=harness.ui_events)
    try:
        await harness.start()
        harness.output_runtime.start_ui_event_bridge(ui)
        await harness.output_runtime.wait_for_ui_event_bridge_started()
        if route == "independent":
            stream = RecognitionStreamIdentity("self", 1, 1, "epoch", ("gemini_transcribe",))
            await harness.self_owner.handle_recognition_unit(
                STTRecognitionUnit(RecognitionUnitIdentity(stream, uuid4(), 1), "source")
            )
        else:
            identity = STTProviderTurnIdentity(
                AudioSegmentIdentity(1, 1, uuid4(), 1), "epoch", "turn"
            )
            event = (
                STTProviderTurnUpdate(
                    identity,
                    1,
                    "stable",
                    "append",
                    "source",
                    contribution=STTTextContribution("source", 0, 6),
                )
                if route == "stable"
                else STTProviderTurnTerminal(
                    identity,
                    "final",
                    text="source",
                    text_authority="authoritative",
                )
            )
            await harness.self_owner.handle_stt_event(event)
        await asyncio.wait_for(llm.started.wait(), 1.0)
        await asyncio.wait_for(transport.send_started.wait(), 1.0)
        source_revision = overlay.snapshot().revision
        llm.release.set()
        await _wait_until(lambda: bool(overlay.snapshot().blocks[0].secondary_text))
        translated = overlay.snapshot()
        assert translated.revision > source_revision
        assert bridge._mailbox.current_scene.snapshot.revision == translated.revision
        assert bridge._mailbox.current_scene.snapshot.blocks[0].secondary_text
        assert not transport.sent_payloads
        assert not transport.release_send.is_set()
        assert len(llm.calls) == 1
        print(
            f"OWNER_SMOKE route={route} provider_completed=true translation_application=true bridge_released=false"
        )
        transport.release_send.set()
        await _wait_until(lambda: len(transport.sent_payloads) >= 2)
        revisions = [message["payload"]["revision"] for message in transport.sent_payloads]
        assert revisions == sorted(set(revisions))
        assert revisions[-1] == translated.revision
    finally:
        llm.release.set()
        transport.release_send.set()
        await harness.stop()
        await overlay.close()
        await bridge.stop()
