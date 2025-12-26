from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

from puripuly_heart.core.clock import Clock, SystemClock
from puripuly_heart.core.language import get_llm_language_name
from puripuly_heart.core.llm.provider import LLMProvider
from puripuly_heart.core.osc.smart_queue import SmartOscQueue
from puripuly_heart.core.vad.gating import SpeechEnd, SpeechStart, VadEvent
from puripuly_heart.domain.events import (
    STTErrorEvent,
    STTFinalEvent,
    STTPartialEvent,
    STTSessionStateEvent,
    UIEvent,
    UIEventType,
)
from puripuly_heart.domain.models import (
    OSCMessage,
    Transcript,
    UtteranceBundle,
)


@dataclass(frozen=True, slots=True)
class ContextEntry:
    """Represents a recent utterance for context memory."""

    text: str  # Original text
    timestamp: float  # When the translation was requested


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

    source_language: str = "ko"
    target_language: str = "en"
    system_prompt: str = ""
    fallback_transcript_only: bool = False
    translation_enabled: bool = True
    hangover_s: float = 1.1  # VAD hangover in seconds (for E2E latency calculation)

    # Context memory settings
    context_time_window_s: float = 20.0  # Only include entries within this time window
    context_max_entries: int = 3  # Maximum number of context entries to include

    ui_events: asyncio.Queue[UIEvent] = field(default_factory=asyncio.Queue)

    _utterances: dict[UUID, UtteranceBundle] = field(default_factory=dict)
    _translation_tasks: dict[UUID, asyncio.Task[None]] = field(default_factory=dict)
    _utterance_sources: dict[UUID, str] = field(default_factory=dict)
    _utterance_start_times: dict[UUID, float] = field(
        default_factory=dict
    )  # For E2E latency tracking
    _translation_history: list[ContextEntry] = field(default_factory=list)  # Context memory
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

        if self.llm is not None:
            await self.llm.close()

    def clear_context(self) -> None:
        """Clear the translation context history."""
        self._translation_history.clear()
        logger.info("[Hub] Context history cleared")

    def _get_valid_context(self) -> list[ContextEntry]:
        """Get context entries within time window and max entries limit."""
        now = self.clock.now()
        # Filter by time window and limit to max entries
        valid = [
            entry
            for entry in self._translation_history[-self.context_max_entries :]
            if (now - entry.timestamp) < self.context_time_window_s
        ]
        return valid

    def _format_context_for_llm(self, context: list[ContextEntry]) -> str:
        """Format context entries as a string for LLM prompt."""
        if not context:
            return ""
        lines = []
        for entry in context:
            lines.append(f'- "{entry.text}"')
        return "\n".join(lines)

    async def handle_vad_event(self, event: VadEvent) -> None:
        # Start typing indicator when speech begins
        if isinstance(event, SpeechStart):
            self.osc.send_typing(True)

        # Record start time for E2E latency tracking (from speech end)
        if isinstance(event, SpeechEnd):
            self._utterance_start_times[event.utterance_id] = self.clock.now()

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
            await self.ui_events.put(
                UIEvent(type=UIEventType.SESSION_STATE_CHANGED, payload=event.state)
            )
            return

        if isinstance(event, STTErrorEvent):
            await self.ui_events.put(
                UIEvent(type=UIEventType.ERROR, payload=event.message, source="Mic")
            )
            return

        if isinstance(event, STTPartialEvent):
            logger.debug(
                f"[Hub] STT Partial: '{event.transcript.text[:50]}...' id={str(event.transcript.utterance_id)[:8]}"
            )
            await self._handle_transcript(event.transcript, is_final=False, source="Mic")
            return

        if isinstance(event, STTFinalEvent):
            await self._handle_transcript(event.transcript, is_final=True, source="Mic")
            if self.llm is None or not self.translation_enabled:
                logger.info(
                    f"[Hub] Skipping translation (llm={self.llm is not None}, enabled={self.translation_enabled})"
                )
                await self._enqueue_osc(
                    event.transcript.utterance_id,
                    transcript_text=event.transcript.text,
                    translation_text=None,
                )
            else:
                await self._ensure_translation(event.transcript)
            return

    async def _handle_transcript(
        self, transcript: Transcript, *, is_final: bool, source: str | None
    ) -> None:
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
            # Get valid context for this translation
            valid_context = self._get_valid_context()
            now = self.clock.now()

            # Log context information
            logger.info(
                f"[Hub] Context: {len(valid_context)} entries within {self.context_time_window_s}s window"
            )
            for i, entry in enumerate(valid_context):
                age = now - entry.timestamp
                logger.info(f'[Hub] Context[{i}]: "{entry.text}" ({age:.1f}s ago)')

            # Add current text to context history at REQUEST time
            self._translation_history.append(ContextEntry(text=text, timestamp=now))
            if len(self._translation_history) > self.context_max_entries:
                self._translation_history.pop(0)

            # Format context for LLM
            context_str = self._format_context_for_llm(valid_context)

            # Substitute language placeholders in system prompt
            formatted_prompt = self.system_prompt
            formatted_prompt = formatted_prompt.replace(
                "${sourceName}", get_llm_language_name(self.source_language)
            )
            formatted_prompt = formatted_prompt.replace(
                "${targetName}", get_llm_language_name(self.target_language)
            )

            translation = await self.llm.translate(
                utterance_id=utterance_id,
                text=text,
                system_prompt=formatted_prompt,
                source_language=self.source_language,
                target_language=self.target_language,
                context=context_str,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"[Hub] Translation failed: {exc}")
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
        await self._enqueue_osc(
            utterance_id, transcript_text=text, translation_text=translation.text
        )

    async def _enqueue_osc(
        self, utterance_id: UUID, *, transcript_text: str, translation_text: str | None
    ) -> None:
        if translation_text is None:
            merged = transcript_text
        else:
            merged = f"{transcript_text} ({translation_text})"

        msg = OSCMessage(utterance_id=utterance_id, text=merged, created_at=self.clock.now())

        # Calculate and log E2E latency (includes hangover time)
        start_time = self._utterance_start_times.pop(utterance_id, None)
        if start_time is not None:
            processing_latency = self.clock.now() - start_time
            total_e2e = processing_latency + self.hangover_s
            logger.info(
                f"[Hub] OSC enqueue: '{merged[:50]}...' id={str(utterance_id)[:8]} (Latency: {total_e2e:.2f}s)"
            )
        else:
            logger.info(f"[Hub] OSC enqueue: '{merged[:50]}...' id={str(utterance_id)[:8]}")

        self.osc.enqueue(msg)

        # Stop typing indicator after message is sent
        self.osc.send_typing(False)

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
