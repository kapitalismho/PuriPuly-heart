from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from uuid import UUID, uuid4

from puripuly_heart.core.orchestrator.translation_output_projection import TranslationUiMessage
from puripuly_heart.core.overlay.sink import OverlayApplicationReceipt, SelfTranscriptFinal
from puripuly_heart.core.stt.backend import STTRecognitionUnit
from puripuly_heart.core.stt.recognition_units import recognition_transcript
from puripuly_heart.core.vad.gating import SpeechEnd
from puripuly_heart.domain.events import SelfTurnPresentation, STTFinalEvent, UIEventType
from puripuly_heart.domain.models import FinalLanguageRun, OSCMessage, Transcript, Translation
from puripuly_heart.domain.recognition import RecognitionStreamIdentity, RecognitionUnitIdentity
from puripuly_heart.ui.presentation_adapter import FletUiPresentationAdapter
from tests.helpers.translation_owners import compose_translation_test_harness


async def eventually(predicate, timeout=3.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def independent_recognition(runs: tuple[tuple[str, str], ...]) -> tuple[STTRecognitionUnit, UUID]:
    language_runs = tuple(FinalLanguageRun(text, language) for text, language in runs)
    stream = RecognitionStreamIdentity("self", 1, 2, str(uuid4()), ("controlled-independent",))
    unit = STTRecognitionUnit(
        RecognitionUnitIdentity(stream, uuid4(), 1),
        "".join(run.text for run in language_runs),
        final_language_runs=language_runs,
    )
    return unit, recognition_transcript(unit, created_at=0.0).utterance_id


@dataclass
class ControlledProvider:
    holds: dict[tuple[str, str], asyncio.Event] = field(default_factory=dict)
    failures: set[tuple[str, str]] = field(default_factory=set)
    calls: list[dict] = field(default_factory=list)

    async def translate(
        self,
        *,
        utterance_id,
        text,
        source_language,
        target_language,
        system_prompt,
        context="",
        scene_participant_count=None,
    ):
        key = (text, target_language)
        self.calls.append(
            {
                "utterance_id": utterance_id,
                "text": text,
                "target_language": target_language,
            }
        )
        gate = self.holds.get(key)
        if gate is not None:
            await gate.wait()
        if key in self.failures:
            raise RuntimeError("synthetic controlled target failure")
        return Translation(
            utterance_id,
            f"{target_language}:{text}",
            source_text=text,
            source_language=source_language,
            target_language=target_language,
        )

    async def close(self):
        for gate in self.holds.values():
            gate.set()


@dataclass
class ControlledSourceOverlay:
    held_text: str
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def emit(self, event):
        if isinstance(event, SelfTranscriptFinal) and event.text == self.held_text:
            self.started.set()
            await self.release.wait()
        return OverlayApplicationReceipt(
            stage="application_accepted",
            outcome="applied",
            publication_id=event.event_id,
            scene_revision=None,
        )


@dataclass
class RecordingChatbox:
    messages: list = field(default_factory=list)

    def enqueue(self, message):
        self.messages.append(message)

    def send_immediate(self, text):
        return True, None

    def send_typing(self, value):
        pass

    def set_typing_reason(self, reason, active):
        pass

    def clear_typing_reasons(self):
        pass

    def process_due(self):
        pass


class OwnerPresentation:
    def __init__(
        self,
        dashboard,
        logs,
        runtime_logging,
        *,
        targets=("en",),
        finalize_wait_ms=0,
        source_overlay=None,
    ):
        self.source_overlay = source_overlay
        self.provider = ControlledProvider()
        self.chatbox = RecordingChatbox()
        self.harness = compose_translation_test_harness(
            stt=None,
            llm=self.provider,
            osc=self.chatbox,
            runtime_logging=runtime_logging,
            overlay_sink=source_overlay,
            low_latency_mode=True,
            low_latency_finalize_wait_ms=finalize_wait_ms,
            source_language="fr",
            target_language=targets[0],
            self_target_languages=targets,
            ui_queue_maxsize=4,
        )
        adapter = FletUiPresentationAdapter(
            SimpleNamespace(view_dashboard=dashboard, view_logs=logs)
        )
        adapter.attach_runtime_log_sink(runtime_logging)
        self.bridge = adapter.create_ui_event_bridge(
            event_queue=self.harness.ui_events,
            runtime_logging=runtime_logging,
        )
        self.delivery_gate = asyncio.Event()
        self.delivery_gate.set()
        self.snapshots: dict[UUID, SelfTurnPresentation] = {}
        self.consumer = None
        handle = self.bridge._handle_event

        async def consume(event):
            await self.delivery_gate.wait()
            await handle(event)
            if event.type == UIEventType.SELF_TURN_UPDATED:
                self.snapshots[event.payload.utterance_id] = event.payload

        self.bridge._handle_event = consume

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_exception):
        await self.close()

    async def start(self):
        await self.harness.start()
        self.consumer = asyncio.create_task(self.bridge.run())
        await self.bridge.wait_started()

    async def source(self, text, *, close=True):
        source_id = uuid4()
        await self.harness.self_owner.handle_stt_event(
            STTFinalEvent(source_id, Transcript(source_id, text, True))
        )
        if close:
            await self.harness.self_owner.handle_vad_event(SpeechEnd(source_id))
        return source_id

    async def close_source(self, source_id):
        await self.harness.self_owner.handle_vad_event(SpeechEnd(source_id))

    async def fill_ui_queue_for_source_timeout(self):
        await self.harness.ui_events.join()
        self.delivery_gate.clear()
        for index in range(5):
            utterance_id = uuid4()
            await self.harness.output_projection.publish_ui(
                TranslationUiMessage(
                    event_type=UIEventType.OSC_SENT,
                    utterance_id=utterance_id,
                    payload=OSCMessage(
                        utterance_id,
                        f"owned pressure history {index}",
                        self.harness.self_owner.clock.now(),
                    ),
                    source="You",
                    channel="self",
                    runtime_log_handled=True,
                )
            )
            if index == 0:
                await eventually(self.harness.ui_events.empty)
        assert self.harness.ui_events.qsize() == 4
        writer = self.harness.output_projection.ui_messages
        previous_timeout = writer.write_timeout_s
        writer.write_timeout_s = 0.02
        return previous_timeout

    async def idle(self):
        await asyncio.wait_for(self.harness.translation_turns.wait_for_idle(), timeout=3)

    async def close(self):
        if self.source_overlay is not None:
            self.source_overlay.release.set()
        for gate in self.provider.holds.values():
            gate.set()
        self.delivery_gate.set()
        try:
            await self.harness.stop()
        finally:
            self.bridge.close()
            if self.consumer is not None:
                self.consumer.cancel()
                await asyncio.gather(self.consumer, return_exceptions=True)
