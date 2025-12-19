from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID
from uuid import uuid4

from ajebal_daera_translator.core.clock import Clock, SystemClock
from ajebal_daera_translator.core.llm.provider import LLMProvider
from ajebal_daera_translator.core.osc.smart_queue import SmartOscQueue
from ajebal_daera_translator.core.vad.gating import VadEvent
from ajebal_daera_translator.domain.events import (
    STTErrorEvent,
    STTFinalEvent,
    STTPartialEvent,
    STTSessionStateEvent,
    UIEvent,
    UIEventType,
)
from ajebal_daera_translator.domain.models import OSCMessage, Transcript, Translation, UtteranceBundle


class STTProvider(Protocol):
    async def handle_vad_event(self, event: VadEvent) -> None: ...
    async def close(self) -> None: ...
    def events(self): ...


@dataclass(slots=True)
class ClientHub:
    stt: STTProvider | None
    llm: LLMProvider | None
    osc: SmartOscQueue
    clock: Clock = SystemClock()

    source_language: str = "ko-KR"
    target_language: str = "en"
    system_prompt: str = ""
    fallback_transcript_only: bool = False
    translation_enabled: bool = True

    ui_events: asyncio.Queue[UIEvent] = field(default_factory=asyncio.Queue)

    _utterances: dict[UUID, UtteranceBundle] = field(default_factory=dict)
    _translation_tasks: dict[UUID, asyncio.Task[None]] = field(default_factory=dict)
    _utterance_sources: dict[UUID, str] = field(default_factory=dict)
    _stt_task: asyncio.Task[None] | None = None
    _osc_flush_task: asyncio.Task[None] | None = None
    _running: bool = False

    async def start(self, *, auto_flush_osc: bool = False) -> None:
        if self._running:
            return
        self._running = True
        if self.stt is not None:
            self._stt_task = asyncio.create_task(self._run_stt_event_loop())
        if auto_flush_osc:
            self._osc_flush_task = asyncio.create_task(self._run_osc_flush_loop())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        if self._osc_flush_task:
            self._osc_flush_task.cancel()
            await asyncio.gather(self._osc_flush_task, return_exceptions=True)
            self._osc_flush_task = None

        if self._stt_task:
            self._stt_task.cancel()
            await asyncio.gather(self._stt_task, return_exceptions=True)
            self._stt_task = None

        for task in list(self._translation_tasks.values()):
            task.cancel()
        await asyncio.gather(*self._translation_tasks.values(), return_exceptions=True)
        self._translation_tasks.clear()

        if self.stt is not None:
            await self.stt.close()

    async def handle_vad_event(self, event: VadEvent) -> None:
        if self.stt is not None:
            await self.stt.handle_vad_event(event)

    async def submit_text(self, text: str, *, source: str = "You") -> UUID:
        text = text.strip()
        if not text:
            raise ValueError("text must be non-empty")

        utterance_id = uuid4()
        self._remember_source(utterance_id, source)

        transcript = Transcript(
            utterance_id=utterance_id,
            text=text,
            is_final=True,
            created_at=self.clock.now(),
        )
        await self._handle_transcript(transcript, is_final=True, source=source)

        if self.llm is None or not self.translation_enabled:
            await self._enqueue_osc(utterance_id, transcript_text=text, translation_text=None)
        else:
            await self._ensure_translation(transcript)

        return utterance_id

    def get_or_create_bundle(self, utterance_id: UUID) -> UtteranceBundle:
        bundle = self._utterances.get(utterance_id)
        if bundle is None:
            bundle = UtteranceBundle(utterance_id=utterance_id)
            self._utterances[utterance_id] = bundle
        return bundle

    async def _run_stt_event_loop(self) -> None:
        try:
            async for ev in self.stt.events():
                await self._handle_stt_event(ev)
        except asyncio.CancelledError:
            raise

    async def _handle_stt_event(self, event: object) -> None:
        if isinstance(event, STTSessionStateEvent):
            await self.ui_events.put(UIEvent(type=UIEventType.SESSION_STATE_CHANGED, payload=event.state))
            return

        if isinstance(event, STTErrorEvent):
            await self.ui_events.put(UIEvent(type=UIEventType.ERROR, payload=event.message, source="Mic"))
            return

        if isinstance(event, STTPartialEvent):
            await self._handle_transcript(event.transcript, is_final=False, source="Mic")
            return

        if isinstance(event, STTFinalEvent):
            await self._handle_transcript(event.transcript, is_final=True, source="Mic")
            if self.llm is None or not self.translation_enabled:
                await self._enqueue_osc(
                    event.transcript.utterance_id,
                    transcript_text=event.transcript.text,
                    translation_text=None,
                )
            else:
                await self._ensure_translation(event.transcript)
            return

    async def _handle_transcript(self, transcript: Transcript, *, is_final: bool, source: str | None) -> None:
        bundle = self.get_or_create_bundle(transcript.utterance_id)
        bundle.with_transcript(transcript)
        self._remember_source(transcript.utterance_id, source)
        await self.ui_events.put(
            UIEvent(
                type=UIEventType.TRANSCRIPT_FINAL if is_final else UIEventType.TRANSCRIPT_PARTIAL,
                utterance_id=transcript.utterance_id,
                payload=transcript,
                source=source,
            )
        )

    def _remember_source(self, utterance_id: UUID, source: str | None) -> None:
        if not source:
            return
        self._utterance_sources[utterance_id] = source

    def _get_source(self, utterance_id: UUID) -> str | None:
        return self._utterance_sources.get(utterance_id)

    async def _ensure_translation(self, transcript: Transcript) -> None:
        if self.llm is None:
            return
        utterance_id = transcript.utterance_id
        if utterance_id in self._translation_tasks:
            return
        task = asyncio.create_task(self._translate_and_enqueue(utterance_id, transcript.text))
        self._translation_tasks[utterance_id] = task
        task.add_done_callback(lambda _t: self._translation_tasks.pop(utterance_id, None))

    async def _translate_and_enqueue(self, utterance_id: UUID, text: str) -> None:
        if self.llm is None:
            return
        try:
            translation = await self.llm.translate(
                utterance_id=utterance_id,
                text=text,
                system_prompt=self.system_prompt,
                source_language=self.source_language,
                target_language=self.target_language,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.ui_events.put(
                UIEvent(
                    type=UIEventType.ERROR,
                    utterance_id=utterance_id,
                    payload=str(exc),
                    source=self._get_source(utterance_id),
                )
            )
            if self.fallback_transcript_only:
                await self._enqueue_osc(utterance_id, transcript_text=text, translation_text=None)
            return

        bundle = self.get_or_create_bundle(utterance_id)
        bundle.with_translation(translation)
        await self.ui_events.put(
            UIEvent(
                type=UIEventType.TRANSLATION_DONE,
                utterance_id=utterance_id,
                payload=translation,
                source=self._get_source(utterance_id),
            )
        )
        await self._enqueue_osc(utterance_id, transcript_text=text, translation_text=translation.text)

    async def _enqueue_osc(self, utterance_id: UUID, *, transcript_text: str, translation_text: str | None) -> None:
        if translation_text is None:
            merged = transcript_text
        else:
            merged = f"{transcript_text} ({translation_text})"

        msg = OSCMessage(utterance_id=utterance_id, text=merged, created_at=self.clock.now())
        self.osc.enqueue(msg)
        await self.ui_events.put(
            UIEvent(
                type=UIEventType.OSC_SENT,
                utterance_id=utterance_id,
                payload=msg,
                source=self._get_source(utterance_id),
            )
        )

    async def _run_osc_flush_loop(self) -> None:
        try:
            while True:
                self.osc.process_due()
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
